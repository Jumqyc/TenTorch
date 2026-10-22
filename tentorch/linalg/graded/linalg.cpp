// The factorizations of the engine: singular value, rank revealing and their spectra.
#include "linalg.h"

#include "../../tensors/diagram.h"

#include <pybind11/complex.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <map>
#include <stdexcept>


namespace sym::engine {
namespace {

// The leg numbers of the tensor in the requested arrangement: the left group first, in order, then
// the right group. Throws when the two groups do not partition the legs.
std::vector<int64_t> leg_order(const TensorMap& tensor, const std::vector<int64_t>& left,
                               const std::vector<int64_t>& right) {
  const int64_t n = tensor.space.size();
  std::vector<int64_t> order;
  order.reserve(left.size() + right.size());
  for (const std::vector<int64_t>* group : {&left, &right}) {
    for (const int64_t leg : *group) {
      if (leg < 0 || leg >= n) {
        throw std::invalid_argument("a split names leg " + std::to_string(leg) + " of a tensor with " +
                                    std::to_string(n) + " legs");
      }
      order.push_back(leg);
    }
  }
  if (static_cast<int64_t>(order.size()) != n) {
    throw std::invalid_argument("a split has to name every leg of the tensor exactly once");
  }
  std::vector<int64_t> sorted = order;
  std::sort(sorted.begin(), sorted.end());
  for (int64_t at = 0; at < n; ++at) {
    if (sorted.at(at) != at) {
      throw std::invalid_argument("a split has to name every leg of the tensor exactly once");
    }
  }
  return order;
}

// The side every leg of the tensor sits on: false for the codomain, true for the domain.
std::vector<bool> leg_sides(const TensorMap& tensor) {
  std::vector<bool> sides(tensor.space.size(), false);
  for (std::size_t at = 0; at < tensor.space.domain.size(); ++at) {
    sides.at(tensor.space.codomain.size() + at) = true;
  }
  return sides;
}

// The legs in their current order, as the one-based positions an arrangement takes.
std::vector<int64_t> natural_order(int64_t size) {
  std::vector<int64_t> order(size);
  for (int64_t at = 0; at < size; ++at) order.at(at) = at + 1;
  return order;
}

// The map a split factorizes, and the device its blocks live on: the legs the split names first
// become the codomain and the rest the domain, in the order the split names them.
struct Prepared {
  TensorMap source;
  c10::Device device;
};

Prepared prepare(const TensorMap& tensor, const std::vector<int64_t>& left,
                 const std::vector<int64_t>& right) {
  const std::vector<int64_t> order = leg_order(tensor, left, right);
  const c10::Device device = tensor.blocks.empty() ? at::TensorOptions().device()
                                                   : tensor.blocks.begin()->second.device();
  bool arranged = static_cast<int64_t>(tensor.space.codomain.size()) == static_cast<int64_t>(left.size());
  for (std::size_t at = 0; at < order.size(); ++at) {
    if (order.at(at) != static_cast<int64_t>(at) + 1) arranged = false;
  }
  if (arranged) return Prepared{tensor, device};
  std::vector<int64_t> wanted;
  for (const int64_t at : order) wanted.push_back(at + 1);
  return Prepared{arrange_tensor(tensor, static_cast<int64_t>(left.size()), wanted, device), device};
}

// The singular values of every charge, in the order the bond leg lists the charges.
struct Part {
  Sector charge;
  at::Tensor u;
  at::Tensor values;
  at::Tensor vh;
};

// Which singular values survive the tolerance and the bond limit, per part.
std::vector<at::Tensor> keep_flags(const std::vector<Part>& parts, std::optional<int64_t> max_bond,
                                   double tol) {
  std::vector<at::Tensor> flags;
  for (const Part& part : parts) {
    flags.push_back(part.values > tol);
  }
  if (max_bond.has_value() && !parts.empty()) {
    const at::Tensor flat = at::cat([&parts] {
      std::vector<at::Tensor> values;
      for (const Part& part : parts) values.push_back(part.values);
      return values;
    }());
    const at::Tensor top =
        at::argsort(flat, /*dim=*/0, /*descending=*/true).slice(0, 0, *max_bond);
    at::Tensor allowed = at::zeros_like(flat.to(at::kBool));
    allowed.index_put_({top}, at::ones({}, allowed.options()));
    int64_t offset = 0;
    for (std::size_t at = 0; at < parts.size(); ++at) {
      const int64_t count = parts.at(at).values.size(0);
      flags.at(at) = flags.at(at).logical_and(allowed.slice(0, offset, offset + count));
      offset += count;
    }
  }
  return flags;
}

}  // namespace

// The split of a tensor: the leading legs become the codomain of the first factor and the remaining
// legs the domain of the second, joined by a bond leg that carries the kept copies of every charge. A
// tensor whose legs all lie on one side is first turned into a map, so every block is a matrix and
// factorises on its own.
Split svd(const TensorMap& tensor, const std::vector<int64_t>& left,
          const std::vector<int64_t>& right, std::optional<int64_t> max_bond, double tol,
          int64_t absorb) {
  const auto [source, device] = prepare(tensor, left, right);
  const std::shared_ptr<FusionCategory> group = source.space.codomain.front().group();
  std::vector<Part> parts;
  for (const auto& [charge, block] : source.blocks) {
    auto [u, values, vh] = at::linalg_svd(block, /*full_matrices=*/false);
    parts.push_back({charge, u, values, vh});
  }
  const std::vector<at::Tensor> flags = keep_flags(parts, max_bond, tol);
  // The bond leg of the first factor points in, the one of the second points out; both store the kept
  // copies of the charge of the block they came from, so each carries the space the other one does not.
  std::vector<std::pair<Sector, int64_t>> bond;
  std::vector<at::Tensor> spectrum;
  for (std::size_t at = 0; at < parts.size(); ++at) {
    const at::Tensor index = flags.at(at).nonzero().squeeze(-1);
    if (index.size(0) == 0) continue;
    bond.emplace_back(parts.at(at).charge, index.size(0));
    spectrum.push_back(at::index_select(parts.at(at).values, 0, index));
  }
  Split split;
  split.first.space.group = group;
  split.first.space.codomain = source.space.codomain;
  split.first.space.domain = {leg_of(group, bond, -1)};
  split.first.qtotal = group->trivial_sector();
  split.second.space.group = group;
  split.second.space.codomain = {leg_of(group, bond, 1)};
  split.second.space.domain = source.space.domain;
  split.second.qtotal = tensor.qtotal;
  std::map<Sector, at::Tensor> kept;
  for (std::size_t at = 0; at < parts.size(); ++at) {
    const at::Tensor index = flags.at(at).nonzero().squeeze(-1);
    if (index.size(0) == 0) continue;
    kept.emplace(parts.at(at).charge, at::index_select(parts.at(at).values, 0, index));
    split.first.blocks.emplace(parts.at(at).charge,
                               at::index_select(parts.at(at).u, 1, index).to(device).contiguous());
    split.second.blocks.emplace(parts.at(at).charge,
                                at::index_select(parts.at(at).vh, 0, index).to(device).contiguous());
  }
  // The values are put on a factor here when a caller wants to keep them out of its own arithmetic;
  // the bond is the last index of the first factor and the first index of the second.
  if (absorb == 1) {
    for (auto& [charge, block] : split.first.blocks) {
      block = block * kept.at(charge).reshape({1, -1});
    }
  } else if (absorb == 2) {
    for (auto& [charge, block] : split.second.blocks) {
      block = kept.at(charge).reshape({-1, 1}) * block;
    }
  }
  if (!split.first.blocks.empty()) {
    auto flatten = [&device](TensorMap factor) {
      return arrange_tensor(factor, 0, natural_order(static_cast<int64_t>(factor.space.size())), device);
    };
    split.first = flatten(split.first);
  }
  split.singular = spectrum.empty()
                       ? at::zeros({0}, at::TensorOptions().dtype(at::kDouble).device(device))
                       : at::cat(spectrum).to(at::kDouble);
  return split;
}

std::pair<TensorMap, TensorMap> qr(const TensorMap& tensor, const std::vector<int64_t>& left,
                                   const std::vector<int64_t>& right) {
  const auto [source, device] = prepare(tensor, left, right);
  const std::shared_ptr<FusionCategory> group = source.space.codomain.front().group();
  std::vector<std::pair<Sector, at::Tensor>> reduced;
  std::vector<std::pair<Sector, int64_t>> bond;
  std::vector<at::Tensor> second_blocks;
  for (const auto& [charge, block] : source.blocks) {
    auto [q, r] = at::linalg_qr(block, "reduced");
    reduced.emplace_back(charge, q);
    bond.emplace_back(charge, q.size(1));
    second_blocks.push_back(r);
  }
  TensorMap first;
  first.space.group = group;
  first.space.codomain = source.space.codomain;
  first.space.domain = {leg_of(group, bond, -1)};
  first.qtotal = group->trivial_sector();
  TensorMap second;
  second.space.group = group;
  second.space.codomain = {leg_of(group, bond, 1)};
  second.space.domain = source.space.domain;
  second.qtotal = tensor.qtotal;
  for (std::size_t at = 0; at < reduced.size(); ++at) {
    first.blocks.emplace(reduced.at(at).first, reduced.at(at).second.to(device).contiguous());
    second.blocks.emplace(reduced.at(at).first, second_blocks.at(at).to(device).contiguous());
  }
  return {first, second};
}

at::Tensor svdvals(const TensorMap& tensor, const std::vector<int64_t>& left,
                   const std::vector<int64_t>& right) {
  const auto [source, device] = prepare(tensor, left, right);
  std::vector<at::Tensor> values;
  for (const auto& [charge, block] : source.blocks) values.push_back(at::linalg_svdvals(block));
  return values.empty() ? at::zeros({0}, at::TensorOptions().dtype(at::kDouble).device(device))
                        : at::cat(values).to(at::kDouble);
}

void bind_split(py::module_& m) {
  m.def("engine_svd", [](const std::shared_ptr<FusionCategory> group, const py::dict& tensor,
                         const std::vector<int64_t>& left, const std::vector<int64_t>& right,
                         std::optional<int64_t> max_bond, double tol) {
    const Split split = svd(tensor_of(group, tensor), left, right, max_bond, tol);
    return py::make_tuple(tensor_as(split.first), split.singular, tensor_as(split.second));
  }, py::arg("group"), py::arg("tensor"), py::arg("left"), py::arg("right"),
     py::arg("max_bond") = py::none(), py::arg("tol") = 0.0);
  m.def("engine_qr", [](const std::shared_ptr<FusionCategory> group, const py::dict& tensor,
                        const std::vector<int64_t>& left, const std::vector<int64_t>& right) {
    const auto split = qr(tensor_of(group, tensor), left, right);
    return py::make_tuple(tensor_as(split.first), tensor_as(split.second));
  }, py::arg("group"), py::arg("tensor"), py::arg("left"), py::arg("right"));
  m.def("engine_svdvals", [](const std::shared_ptr<FusionCategory> group, const py::dict& tensor,
                             const std::vector<int64_t>& left, const std::vector<int64_t>& right) {
    return svdvals(tensor_of(group, tensor), left, right);
  }, py::arg("group"), py::arg("tensor"), py::arg("left"), py::arg("right"));
}

}  // namespace sym::engine


namespace sym {

namespace {

c10::Device& default_device() {
  static c10::Device device(at::kCPU);
  return device;
}

at::ScalarType& default_scalar() {
  static at::ScalarType scalar = at::kComplexDouble;
  return scalar;
}

}  // namespace

// The module-level factorizations the python layer calls: each takes the graded tensor of the
// tensor layer and hands back tensors of the same kind.
void bind_linalg(py::module_& m) {
  m.def("set_defaults", [](const std::string& device, const std::string& dtype) {
    default_device() = c10::Device(device);
    const std::string name = dtype.substr(dtype.rfind('.') + 1);
    if (name == "complex128" || name == "complex64") {
      default_scalar() = name == "complex128" ? at::kComplexDouble : at::kComplexFloat;
    } else if (name == "float64" || name == "float32") {
      default_scalar() = name == "float64" ? at::kDouble : at::kFloat;
    } else {
      throw std::invalid_argument("set_defaults: unknown dtype " + dtype);
    }
  });
  m.def("default_device", [] { return default_device(); });
  m.def("default_scalar", [] { return default_scalar(); });
  m.def("svd", [](const engine::Tensor& tensor, const std::vector<int64_t>& left,
                  const std::vector<int64_t>& right, const py::object& max_bond, double tol,
                  int64_t absorb) {
    const std::optional<int64_t> bond =
        max_bond.is_none() ? std::optional<int64_t>{} : std::optional<int64_t>{max_bond.cast<int64_t>()};
    engine::Split split = engine::svd(tensor.map(), left, right, bond, tol, absorb);
    return py::make_tuple(engine::Tensor(std::move(split.first)), split.singular,
                          engine::Tensor(std::move(split.second)));
  }, py::arg("tensor"), py::arg("left"), py::arg("right"), py::arg("max_bond") = py::none(),
     py::arg("tol") = 0.0, py::arg("absorb") = 0);
  m.def("qr", [](const engine::Tensor& tensor, const std::vector<int64_t>& left,
                 const std::vector<int64_t>& right) {
    // A = Q R with Q orthonormal in its rows: the singular values are folded into R, so Q is the
    // first factor of the SVD and R carries the rest.
    engine::Split split = engine::svd(tensor.map(), left, right, std::nullopt, 0.0, 2);
    return py::make_tuple(engine::Tensor(std::move(split.first)),
                          engine::Tensor(std::move(split.second)));
  }, py::arg("tensor"), py::arg("left"), py::arg("right"));
  m.def("rq", [](const engine::Tensor& tensor, const std::vector<int64_t>& left,
                 const std::vector<int64_t>& right) {
    // A = R Q with Q orthonormal in its rows: here the singular values go to R, which is the
    // first factor.
    engine::Split split = engine::svd(tensor.map(), left, right, std::nullopt, 0.0, 1);
    return py::make_tuple(engine::Tensor(std::move(split.first)),
                          engine::Tensor(std::move(split.second)));
  }, py::arg("tensor"), py::arg("left"), py::arg("right"));
  m.def("svdvals", [](const engine::Tensor& tensor, const std::vector<int64_t>& left,
                      const std::vector<int64_t>& right) {
    return engine::svdvals(tensor.map(), left, right);
  }, py::arg("tensor"), py::arg("left"), py::arg("right"));
  m.def("norm", [](const engine::Tensor& tensor) { return engine::norm(tensor.map()); });
  m.def("inner", [](const engine::Tensor& left, const engine::Tensor& right) {
    return engine::inner(left.map(), right.map());
  });
  m.def("absorb", [](const engine::Tensor& tensor, const at::Tensor& values) {
    // Fold the values of a split into the factor whose first codomain leg is the bond: the leg
    // keeps the space it carries, so the second factor stays a factor of the same tensor.
    return tensor.absorb(values);
  }, py::arg("tensor"), py::arg("values"));
  m.def("trace", [](const engine::Tensor& tensor) {
    // Closes the legs pairwise and hands back the scalar they leave as a tensor with no leg.
    const engine::TensorMap& source = tensor.map();
    const std::complex<double> total = engine::trace(source);
    engine::TensorMap scalar;
    scalar.space.group = source.space.group;
    scalar.qtotal = source.space.group->trivial_sector();
    scalar.blocks[scalar.qtotal] =
        at::scalar_tensor(at::Scalar(c10::complex<double>(total.real(), total.imag())),
                          at::TensorOptions().dtype(at::kComplexDouble)).reshape({1, 1});
    return engine::Tensor(std::move(scalar));
  });
}

}  // namespace sym
