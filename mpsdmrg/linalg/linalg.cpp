// The factorizations of the engine: singular value, rank revealing and their spectra.
#include "linalg.h"

#include "../tensor/diagram.h"

#include <pybind11/complex.h>
#include <pybind11/stl.h>

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

// Two factors of one part: the first keeps the columns, the second the rows, at the kept positions.
struct Factors {
  at::Tensor first;
  at::Tensor second;
};

// The tensor of a split whose codomain is the two given sides and whose bond leg carries the kept
// copies of every charge, the bond of the second factor pointing the other way.
TensorMap assemble(const std::shared_ptr<FusionCategory>& group, const std::vector<Leg>& codomain,
                   const std::vector<Leg>& domain, const std::vector<Part>& parts,
                   const std::vector<at::Tensor>& kept,
                   const std::vector<std::pair<Sector, int64_t>>& bond_sectors, bool first,
                   const c10::Device& device) {
  TensorMap tensor;
  tensor.space.group = group;
  if (first) {
    tensor.space.codomain = codomain;
    std::vector<std::pair<Sector, int64_t>> dualised;
    for (const auto& [sector, copies] : bond_sectors) {
      dualised.emplace_back(group->dual_sector(sector), copies);
    }
    tensor.space.domain = {leg_of(group, dualised, -1)};
  } else {
    tensor.space.codomain = {leg_of(group, bond_sectors, 1)};
    tensor.space.domain = domain;
  }
  for (std::size_t at = 0; at < parts.size(); ++at) {
    if (kept.at(at).size(0) == 0) continue;
    const at::Tensor& piece = first ? parts.at(at).u : parts.at(at).vh;
    tensor.blocks.emplace(parts.at(at).charge,
                          at::index_select(piece, first ? 1 : 0, kept.at(at)).to(device));
  }
  return tensor;
}

}  // namespace

Split svd(const TensorMap& tensor, const std::vector<int64_t>& left,
          const std::vector<int64_t>& right, std::optional<int64_t> max_bond, double tol) {
  // The legs are named from zero on the python side and from one by the moves.
  std::vector<int64_t> order = leg_order(tensor, left, right);
  for (int64_t& leg : order) ++leg;
  const c10::Device device = tensor.blocks.empty() ? at::TensorOptions().device()
                                                  : tensor.blocks.begin()->second.device();
  const TensorMap permuted = permute_tensor(tensor, static_cast<int64_t>(left.size()), order, device);
  std::vector<Part> parts;
  for (const auto& [charge, block] : permuted.blocks) {
    auto [u, values, vh] = at::linalg_svd(block, /*full_matrices=*/false);
    parts.push_back({charge, u, values, vh});
  }
  const std::vector<at::Tensor> flags = keep_flags(parts, max_bond, tol);
  std::vector<at::Tensor> kept;
  std::vector<std::pair<Sector, int64_t>> bond_sectors;
  std::vector<at::Tensor> spectrum;
  for (std::size_t at = 0; at < parts.size(); ++at) {
    const at::Tensor index = flags.at(at).nonzero().squeeze(-1);
    kept.push_back(index);
    if (index.size(0) == 0) continue;
    bond_sectors.emplace_back(parts.at(at).charge, index.size(0));
    spectrum.push_back(at::index_select(parts.at(at).values, 0, index));
  }
  Split split;
  split.first = assemble(tensor.space.group, permuted.space.codomain, permuted.space.domain, parts,
                         kept, bond_sectors, /*first=*/true, device);
  split.second = assemble(tensor.space.group, permuted.space.codomain, permuted.space.domain, parts,
                          kept, bond_sectors, /*first=*/false, device);
  split.singular = spectrum.empty() ? at::zeros({0}, at::TensorOptions().dtype(at::kDouble).device(device))
                                    : at::cat(spectrum).to(at::kDouble);
  return split;
}

std::pair<TensorMap, TensorMap> qr(const TensorMap& tensor, const std::vector<int64_t>& left,
                                   const std::vector<int64_t>& right) {
  // The legs are named from zero on the python side and from one by the moves.
  std::vector<int64_t> order = leg_order(tensor, left, right);
  for (int64_t& leg : order) ++leg;
  const c10::Device device = tensor.blocks.empty() ? at::TensorOptions().device()
                                                  : tensor.blocks.begin()->second.device();
  const TensorMap permuted = permute_tensor(tensor, static_cast<int64_t>(left.size()), order, device);
  std::vector<Part> parts;
  for (const auto& [charge, block] : permuted.blocks) {
    auto [q, r] = at::linalg_qr(block, "reduced");
    parts.push_back({charge, q, at::zeros({q.size(1)}, block.options()), r});
  }
  std::vector<at::Tensor> kept;
  std::vector<std::pair<Sector, int64_t>> bond_sectors;
  for (const Part& part : parts) {
    kept.push_back(at::arange(part.u.size(1), part.u.options().dtype(at::kLong)));
    if (part.u.size(1) > 0) bond_sectors.emplace_back(part.charge, part.u.size(1));
  }
  return {assemble(tensor.space.group, permuted.space.codomain, permuted.space.domain, parts, kept,
                   bond_sectors, /*first=*/true, device),
          assemble(tensor.space.group, permuted.space.codomain, permuted.space.domain, parts, kept,
                   bond_sectors, /*first=*/false, device)};
}

at::Tensor svdvals(const TensorMap& tensor, const std::vector<int64_t>& left,
                   const std::vector<int64_t>& right) {
  // The legs are named from zero on the python side and from one by the moves.
  std::vector<int64_t> order = leg_order(tensor, left, right);
  for (int64_t& leg : order) ++leg;
  const c10::Device device = tensor.blocks.empty() ? at::TensorOptions().device()
                                                  : tensor.blocks.begin()->second.device();
  const TensorMap permuted = permute_tensor(tensor, static_cast<int64_t>(left.size()), order, device);
  std::vector<at::Tensor> values;
  for (const auto& [charge, block] : permuted.blocks) values.push_back(at::linalg_svdvals(block));
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
  // The engine has one route: every contraction is the block multiplication of the tree basis.
  m.def("dense_path", [] { return false; });
  m.def("set_dense_path", [](bool enabled) {
    if (enabled) {
      throw std::invalid_argument("set_dense_path: the engine has no dense route to force");
    }
  }, py::arg("enabled") = true);

  m.def("svd", [](const engine::Tensor& tensor, const std::vector<int64_t>& left,
                  const std::vector<int64_t>& right, const py::object& max_bond, double tol) {
    const std::optional<int64_t> bond =
        max_bond.is_none() ? std::optional<int64_t>{} : std::optional<int64_t>{max_bond.cast<int64_t>()};
    engine::Split split = engine::svd(tensor.map(), left, right, bond, tol);
    return py::make_tuple(engine::Tensor(std::move(split.first)), split.singular,
                          engine::Tensor(std::move(split.second)));
  }, py::arg("tensor"), py::arg("left"), py::arg("right"), py::arg("max_bond") = py::none(),
     py::arg("tol") = 0.0);
  m.def("qr", [](const engine::Tensor& tensor, const std::vector<int64_t>& left,
                 const std::vector<int64_t>& right) {
    auto [first, second] = engine::qr(tensor.map(), left, right);
    return py::make_tuple(engine::Tensor(std::move(first)), engine::Tensor(std::move(second)));
  }, py::arg("tensor"), py::arg("left"), py::arg("right"));
  m.def("rq", [](const engine::Tensor& tensor, const std::vector<int64_t>& left,
                 const std::vector<int64_t>& right) {
    auto [first, second] = engine::qr(tensor.map(), left, right);
    return py::make_tuple(engine::Tensor(std::move(first)), engine::Tensor(std::move(second)));
  }, py::arg("tensor"), py::arg("left"), py::arg("right"));
  m.def("svdvals", [](const engine::Tensor& tensor, const std::vector<int64_t>& left,
                      const std::vector<int64_t>& right) {
    return engine::svdvals(tensor.map(), left, right);
  }, py::arg("tensor"), py::arg("left"), py::arg("right"));
  m.def("norm", [](const engine::Tensor& tensor) { return engine::norm(tensor.map()); });
  m.def("inner", [](const engine::Tensor& left, const engine::Tensor& right) {
    return engine::inner(left.map(), right.map());
  });
  m.def("trace", [](const engine::Tensor& tensor) {
    // A trace reads the legs by position: the first half points out, the second half in.
    const int64_t legs = static_cast<int64_t>(tensor.map().space.size());
    std::vector<int64_t> identity(legs);
    for (int64_t at = 0; at < legs; ++at) identity.at(at) = at + 1;
    const engine::TensorMap& source = tensor.map();
    const c10::Device device =
        source.blocks.empty() ? at::TensorOptions().device() : source.blocks.begin()->second.device();
    const engine::TensorMap map = engine::permute_tensor(source, legs / 2, identity, device);
    std::complex<double> total = 0.0;
    for (const auto& [charge, block] : map.blocks) {
      if (block.dim() != 2 || block.size(0) != block.size(1)) {
        throw std::invalid_argument("trace: a traced tensor has to be a square map");
      }
      const at::Tensor value = at::trace(block).to(at::kCPU, at::kComplexDouble);
      const c10::complex<double> entry = value.item<c10::complex<double>>();
      total += std::complex<double>(entry.real(), entry.imag());
    }
    // A traced tensor has no leg left, so the trace is the tensor with a single scalar block.
    engine::TensorMap scalar;
    scalar.space.group = map.space.group;
    scalar.qtotal = map.space.group->trivial_sector();
    scalar.blocks[scalar.qtotal] =
        at::scalar_tensor(at::Scalar(c10::complex<double>(total.real(), total.imag())),
                          at::TensorOptions().dtype(at::kComplexDouble)).reshape({1, 1});
    return engine::Tensor(std::move(scalar));
  });
}

}  // namespace sym
