// The block-sparse tensor of the engine: a graded space, the map over it, the contraction, its
// algebra and the python bindings.
#include "tensor.h"

#include "../symmetry/cat/symbol.h"
#include "diagram.h"
#include "../linalg/linalg.h"

#include <pybind11/complex.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace sym::engine {
namespace {

// Where one summand of a leg lives along that leg's full index, which is the concatenation of its
// summands.
struct Entry {
  Sector sector;
  int64_t offset;
  int64_t dim;
  int64_t multiplicity;
};

std::vector<Entry> entries(const Leg& leg) {
  std::vector<Entry> out;
  int64_t offset = 0;
  for (int64_t at = 0; at < leg.nblocks(); ++at) {
    const float dim = static_cast<float>(leg.block_dim(at));
    out.push_back(Entry{leg.sectors().at(at), offset, static_cast<int64_t>(dim), leg.multiplicity(at)});
    offset += static_cast<int64_t>(dim) * leg.multiplicity(at);
  }
  return out;
}

const Entry& entry_of(const std::vector<Entry>& entries, const Sector& sector) {
  for (const Entry& entry : entries)
    if (entry.sector == sector) return entry;
  throw std::invalid_argument("a tree carries a sector its leg cannot hold");
}

// Where a tree's stored sector lives on its leg: a tree records the sector the leg's space carries,
// which is the sector of the summand it belongs to.
const Entry& entry_of(const Leg& leg, const std::vector<Entry>& entries, const Sector& stored) {
  return entry_of(entries, stored);
}

// The sector a vertex fuses to.
const Sector& outcome_of(const FusionTree& tree, int64_t vertex) {
  return vertex + 1 == tree.size() - 1 ? tree.coupled : tree.inner.at(vertex);
}

}  // namespace

Leg leg_of(std::shared_ptr<FusionCategory> group,
           const std::vector<std::pair<Sector, int64_t>>& summands, int64_t qconj) {
  std::vector<Sector> sectors;
  std::vector<int64_t> slices{0};
  for (const auto& [sector, multiplicity] : summands) {
    sectors.push_back(sector);
    slices.push_back(slices.back() + multiplicity * group->sector_dim(sector));
  }
  return Leg(std::move(group), std::move(sectors), std::move(slices), qconj);
}

std::vector<LegCharge> leg_charges(const Leg& leg) {
  std::vector<LegCharge> out;
  for (const Sector& sector : leg.sectors()) out.push_back(LegCharge{sector, leg.qconj() == -1});
  return out;
}

bool Space::operator==(const Space& other) const {
  return codomain == other.codomain && domain == other.domain;
}

std::vector<LegCharge> TensorMap::codomain_charges() const {
  std::vector<LegCharge> out;
  for (const Leg& leg : space.codomain)
    for (const LegCharge& charge : leg_charges(leg)) out.push_back(charge);
  return out;
}

std::vector<LegCharge> TensorMap::domain_charges() const {
  std::vector<LegCharge> out;
  for (const Leg& leg : space.domain)
    for (const LegCharge& charge : leg_charges(leg)) out.push_back(charge);
  return out;
}

std::vector<FusionTreePair> TensorMap::pairs(const Sector& coupled) const {
  std::vector<std::vector<LegCharge>> out_choices;
  std::vector<std::vector<LegCharge>> in_choices;
  for (const Leg& leg : space.codomain) out_choices.push_back(leg_charges(leg));
  for (const Leg& leg : space.domain) in_choices.push_back(leg_charges(leg));
  std::vector<FusionTreePair> out;
  for (const FusionTreePair& pair : tree_pairs(*space.group, out_choices, in_choices))
    if (pair.codomain.coupled == coupled) out.push_back(pair);
  return out;
}

at::Tensor tree_tensor(const FusionCategory& group, const FusionTree& tree,
                       const c10::Device& device, bool dress) {
  check_tree(group, tree);
  const auto options = at::TensorOptions().dtype(group.fusion_dtype()).device(device);
  const int64_t size = tree.size();
  // An empty tree is the vector the coupled sector spans, not a scalar: its norm is that dimension.
  if (size == 0) return at::ones({group.sector_dim(tree.coupled)}, options);
  // A tree records the sector every leg carries on its space; the fusion runs on the sector the leg
  // points at, which is the conjugate one for a leg that carries duals.
  const auto physical = [&](int64_t axis) {
    return tree.isdual.at(axis) ? group.dual_sector(tree.sectors.at(axis)) : tree.sectors.at(axis);
  };
  at::Tensor out;
  if (size == 1) {
    // A one-leg tree couples nothing, so its tensor is the identity from the leg to the charge.
    out = at::eye(group.sector_dim(physical(0)), options);
  } else {
    // One Clebsch-Gordan tensor per vertex, joined along the coupling the previous vertex produced:
    // the last axis of `out` is always the coupling that the next vertex consumes.
    const at::Tensor first = group.fusion_tensor(
      physical(0), physical(1), outcome_of(tree, 0), device, tree.vertices.at(0));
    out = first.permute({1, 2, 0});
    for (int64_t vertex = 1; vertex + 1 < size; ++vertex) {
      const at::Tensor table = group.fusion_tensor(outcome_of(tree, vertex - 1), physical(vertex + 1),
                                                   outcome_of(tree, vertex), device,
                                                   tree.vertices.at(vertex));
      const int64_t left = table.size(1);
      const int64_t right = table.size(2);
      const int64_t next = table.size(0);
      std::vector<int64_t> shape(out.sizes().begin(), out.sizes().end() - 1);
      shape.push_back(right);
      shape.push_back(next);
      out = at::matmul(out.reshape({-1, left}),
                       table.permute({1, 2, 0}).reshape({left, right * next}))
              .reshape(shape);
    }
  }
  // The tree tensor is a unit vector: the tree basis of a side is orthonormal, so a contraction of
  // two tensors is the sum of the products of their block entries.
  const double norm = (out * out.conj()).sum().item<c10::complex<double>>().real();
  out = out / std::sqrt(norm);
  // A leg that carries duals stores its index in the conjugate basis, so its duality matrix dresses
  // it, exactly as it does the component array of a stored tree. The tree basis itself is left
  // undressed, which is what keeps it orthonormal.
  for (int64_t axis = 0; dress && axis < size; ++axis) {
    if (!tree.isdual.at(axis)) continue;
    const at::Tensor matrix = group.duality_matrix(tree.sectors.at(axis), device).to(out.scalar_type());
    out = at::tensordot(out, matrix, {axis}, {1}).movedim(-1, axis);
  }
  return out;
}

SideLayout side_layout(const std::vector<Leg>& legs, const std::vector<FusionTree>& trees) {
  SideLayout layout;
  layout.trees = trees;
  layout.base.push_back(0);
  std::vector<std::vector<Entry>> table;
  for (const Leg& leg : legs) table.push_back(entries(leg));
  for (const FusionTree& tree : trees) {
    std::vector<int64_t> mults;
    int64_t rows = 1;
    for (std::size_t axis = 0; axis < legs.size(); ++axis) {
      const Entry& entry = entry_of(legs.at(axis), table.at(axis), tree.sectors.at(axis));
      mults.push_back(entry.multiplicity);
      rows *= entry.multiplicity;
    }
    layout.mults.push_back(std::move(mults));
    layout.base.push_back(layout.base.back() + rows);
  }
  return layout;
}

std::pair<int64_t, std::vector<int64_t>> row_parts(const SideLayout& layout, int64_t row) {
  if (row < 0 || row >= layout.rows()) {
    throw std::out_of_range("a block row " + std::to_string(row) + " is out of range");
  }
  int64_t tree = static_cast<int64_t>(layout.base.size()) - 2;
  while (layout.base.at(tree) > row) --tree;
  const std::vector<int64_t>& mults = layout.mults.at(tree);
  std::vector<int64_t> parts(mults.size(), 0);
  int64_t rest = row - layout.base.at(tree);
  for (std::size_t axis = mults.size(); axis-- > 0;) {
    parts.at(axis) = mults.at(axis) == 0 ? 0 : rest % mults.at(axis);
    rest = mults.at(axis) == 0 ? 0 : rest / mults.at(axis);
  }
  return {tree, parts};
}

int64_t row_index(const SideLayout& layout, int64_t tree, const std::vector<int64_t>& mults) {
  const std::vector<int64_t>& limits = layout.mults.at(tree);
  int64_t row = layout.base.at(tree);
  int64_t stride = 1;
  for (std::size_t axis = limits.size(); axis-- > 0;) {
    if (mults.at(axis) < 0 || mults.at(axis) >= limits.at(axis)) {
      throw std::out_of_range("a multiplicity index is out of range for its leg");
    }
    row += mults.at(axis) * stride;
    stride *= limits.at(axis);
  }
  return row;
}

}  // namespace sym::engine

#include <pybind11/complex.h>

#include <map>
#include <stdexcept>

namespace sym::engine {
namespace {

// The trees of one side, in the order the rows of a block follow, keyed by the charge they couple to.
std::map<Sector, std::vector<FusionTree>> side_trees(const FusionCategory& group,
                                                     const std::vector<Leg>& legs) {
  std::vector<std::vector<LegCharge>> choices;
  for (const Leg& leg : legs) choices.push_back(leg_charges(leg));
  return trees_by_charge(group, choices);
}

// The index of every tree of one charge, so that a scattered entry can find its row or column.
std::map<FusionTree, int64_t> index_of(const std::vector<FusionTree>& trees) {
  std::map<FusionTree, int64_t> index;
  for (int64_t at = 0; at < static_cast<int64_t>(trees.size()); ++at) index.emplace(trees.at(at), at);
  return index;
}

// One charge of the destination, gathered flat: row * columns + column.
struct Gathered {
  int64_t rows = 0;
  int64_t columns = 0;
  std::map<int64_t, std::complex<double>> entries;
};

}  // namespace

TensorMap permute_tensor(const TensorMap& tensor, int64_t keep, const std::vector<int64_t>& order,
                         const c10::Device& device) {
  const FusionCategory& group = *tensor.space.group;
  const int64_t n = tensor.space.size();
  // A tensor without legs has nothing to permute, and its blocks keep the charge they carry, which
  // the legs cannot express.
  if (n == 0) {
    if (!order.empty()) throw std::invalid_argument("permute_tensor: a tensor without legs takes no order");
    return tensor;
  }
  if (static_cast<int64_t>(order.size()) != n) {
    throw std::invalid_argument("permute_tensor: the order names " + std::to_string(order.size()) +
                                " legs of a tensor with " + std::to_string(n));
  }
  if (keep < 0 || keep > n) {
    throw std::invalid_argument("permute_tensor: the codomain cannot keep " + std::to_string(keep) +
                                " of " + std::to_string(n) + " legs");
  }

  // The destination legs follow the requested order; a leg that changes side changes its direction.
  TensorMap out;
  out.space.group = tensor.space.group;
  out.qtotal = tensor.qtotal;
  const int64_t codomain_size = static_cast<int64_t>(tensor.space.codomain.size());
  // A leg that crosses from one side to the other is read in the conjugate basis, which costs the
  // negative eigenvector of its duality matrix.
  std::vector<double> crossing(static_cast<size_t>(n), 1.0);
  for (int64_t at = 0; at < n; ++at) {
    const int64_t leg = order.at(at) - 1;
    if (leg < 0 || leg >= n) {
      throw std::invalid_argument("permute_tensor: the order names a leg the tensor does not carry");
    }
    const bool from_domain = leg >= codomain_size;
    Leg moved = from_domain ? tensor.space.domain.at(leg - codomain_size)
                            : tensor.space.codomain.at(leg);
    const bool into_domain = at >= keep;
    if (from_domain != into_domain) {
      const double sign = 1.0;
      moved = moved.conj();
      crossing.at(static_cast<size_t>(at)) = sign;
    }
    (into_domain ? out.space.domain : out.space.codomain).push_back(moved);
  }
  double switched = 1.0;
  for (const double sign : crossing) switched *= sign;

  const std::map<Sector, std::vector<FusionTree>> source_out =
      side_trees(group, tensor.space.codomain);
  const std::map<Sector, std::vector<FusionTree>> source_in =
      side_trees(group, tensor.space.domain);
  const std::map<Sector, std::vector<FusionTree>> target_out =
      side_trees(group, out.space.codomain);
  const std::map<Sector, std::vector<FusionTree>> target_in =
      side_trees(group, out.space.domain);
  // A side without legs has one tree whatever the charge is: the empty tree, which couples to the
  // unit, so the charge of the block is the one the other side names.
  const auto trees_on = [](const std::vector<Leg>& legs,
                           const std::map<Sector, std::vector<FusionTree>>& trees,
                           const Sector& charge) {
    if (legs.empty()) {
      FusionTree empty;
      empty.coupled = charge;
      return std::vector<FusionTree>{empty};
    }
    const auto found = trees.find(charge);
    if (found == trees.end()) return std::vector<FusionTree>{};
    return found->second;
  };
  std::map<Sector, SideLayout> out_layouts;
  std::map<Sector, SideLayout> in_layouts;
  std::map<Sector, std::map<FusionTree, int64_t>> out_indices;
  std::map<Sector, std::map<FusionTree, int64_t>> in_indices;
  const auto layout_out = [&](const Sector& charge) -> const SideLayout& {
    const auto found = out_layouts.find(charge);
    if (found != out_layouts.end()) return found->second;
    const std::vector<FusionTree> trees = trees_on(out.space.codomain, target_out, charge);
    out_indices.emplace(charge, index_of(trees));
    return out_layouts.emplace(charge, side_layout(out.space.codomain, trees)).first->second;
  };
  const auto layout_in = [&](const Sector& charge) -> const SideLayout& {
    const auto found = in_layouts.find(charge);
    if (found != in_layouts.end()) return found->second;
    const std::vector<FusionTree> trees = trees_on(out.space.domain, target_in, charge);
    in_indices.emplace(charge, index_of(trees));
    return in_layouts.emplace(charge, side_layout(out.space.domain, trees)).first->second;
  };

  std::map<Sector, Gathered> gathered;
  for (const auto& [charge, block] : tensor.blocks) {
    const std::vector<FusionTree> out_trees = trees_on(tensor.space.codomain, source_out, charge);
    const std::vector<FusionTree> in_trees = trees_on(tensor.space.domain, source_in, charge);
    if (out_trees.empty() || in_trees.empty()) continue;
    const SideLayout source_rows = side_layout(tensor.space.codomain, out_trees);
    const SideLayout source_columns = side_layout(tensor.space.domain, in_trees);
    const at::Tensor values = block.to(at::kCPU, at::kComplexDouble).contiguous();
    const c10::complex<double>* data = values.data_ptr<c10::complex<double>>();
    const int64_t columns = source_columns.rows();
    for (int64_t row = 0; row < source_rows.rows(); ++row) {
      const auto [out_tree, out_mults] = row_parts(source_rows, row);
      for (int64_t column = 0; column < columns; ++column) {
        const c10::complex<double> stored = data[row * columns + column];
        if (stored.real() == 0.0 && stored.imag() == 0.0) continue;
        const auto [in_tree, in_mults] = row_parts(source_columns, column);
        // The multiplicity index of every leg of the source, in leg order.
        std::vector<int64_t> mults = out_mults;
        mults.insert(mults.end(), in_mults.begin(), in_mults.end());
        const FusionTreePair source{out_trees.at(out_tree), in_trees.at(in_tree)};
        for (const PairTerm& term : permute_pair(group, source, keep, order, device)) {
          // A side without legs couples to the unit, so the charge of the block is the one the other
          // side names.
          const Sector target_charge = out.space.codomain.empty() ? term.pair.domain.coupled
                                                                  : term.pair.codomain.coupled;
          const SideLayout& target_rows = layout_out(target_charge);
          const SideLayout& target_columns = layout_in(target_charge);
          const auto tree_row = out_indices.at(target_charge).find(term.pair.codomain);
          const auto tree_column = in_indices.at(target_charge).find(term.pair.domain);
          if (tree_row == out_indices.at(target_charge).end() ||
              tree_column == in_indices.at(target_charge).end()) {
            continue;
          }
          std::vector<int64_t> dest_out;
          for (int64_t at = 0; at < keep; ++at) dest_out.push_back(mults.at(order.at(at) - 1));
          std::vector<int64_t> dest_in;
          for (int64_t at = keep; at < n; ++at) dest_in.push_back(mults.at(order.at(at) - 1));
          Gathered& cell = gathered[target_charge];
          if (std::getenv("PERMUTE_TRACE") != nullptr) {
            std::fprintf(stderr, "gather charge %s rows %ld columns %ld codomain %zu domain %zu\n",
                         sector_repr(target_charge).c_str(), layout_out(target_charge).rows(),
                         layout_in(target_charge).rows(), out.space.codomain.size(),
                         out.space.domain.size());
          }
          cell.rows = target_rows.rows();
          cell.columns = target_columns.rows();
          const int64_t dest_row = row_index(target_rows, tree_row->second, dest_out);
          const int64_t dest_column = row_index(target_columns, tree_column->second, dest_in);
          cell.entries[dest_row * cell.columns + dest_column] +=
              std::complex<double>(stored.real(), stored.imag()) * term.coefficient * switched;
        }
      }
    }
  }

  for (const auto& [charge, cell] : gathered) {
    at::Tensor block = at::zeros({cell.rows, cell.columns},
                                 at::TensorOptions().dtype(at::kComplexDouble).device(at::kCPU));
    std::vector<int64_t> rows;
    std::vector<int64_t> columns;
    std::vector<std::complex<double>> values;
    for (const auto& [flat, value] : cell.entries) {
      rows.push_back(flat / cell.columns);
      columns.push_back(flat % cell.columns);
      values.push_back(value);
    }
    std::vector<c10::complex<double>> wrapped;
    wrapped.reserve(values.size());
    for (const std::complex<double>& value : values) {
      wrapped.emplace_back(value.real(), value.imag());
    }
    block.index_put_({at::tensor(rows, at::kLong), at::tensor(columns, at::kLong)},
                     at::tensor(wrapped, at::TensorOptions().dtype(at::kComplexDouble)));
    out.blocks.emplace(charge, block.to(device));
  }
  return out;
}

TensorMap contract(const TensorMap& left, const TensorMap& right) {
  if (std::getenv("EINSUM_TRACE") != nullptr) {
    std::fprintf(stderr, "contract left %zu|%zu right %zu|%zu\n", left.space.codomain.size(),
                 left.space.domain.size(), right.space.codomain.size(), right.space.domain.size());
  }
  if (left.space.domain.size() != right.space.codomain.size()) {
    throw std::invalid_argument("contract: the two tensors glue a different number of legs");
  }
  TensorMap out;
  out.space.group = left.space.group;
  out.space.codomain = left.space.codomain;
  out.space.domain = right.space.domain;
  for (const auto& [charge, block] : left.blocks) {
    const auto found = right.blocks.find(charge);
    if (found == right.blocks.end()) continue;
    if (std::getenv("EINSUM_TRACE") != nullptr) {
      std::fprintf(stderr, "contract %s left %ld,%ld right %ld,%ld\n", sector_repr(charge).c_str(),
                   block.size(0), block.size(1), found->second.size(0), found->second.size(1));
    }
    out.blocks.emplace(charge, at::matmul(block, found->second));
  }
  return out;
}

namespace {

// The legs Python hands in: each as its summands and the direction it points.
std::vector<Leg> legs_of(const std::shared_ptr<FusionCategory>& group, const py::list& data) {
  std::vector<Leg> legs;
  for (const py::handle& entry : data) {
    const py::tuple pair = entry.cast<py::tuple>();
    legs.push_back(leg_of(group, pair[0].cast<std::vector<std::pair<Sector, int64_t>>>(),
                          pair[1].cast<int64_t>()));
  }
  return legs;
}

}  // namespace

c10::Device device_of(const py::object& value) {
  if (value.is_none()) return at::TensorOptions().device();
  return value.cast<c10::Device>();
}

// A tensor as Python describes it: its two sides of legs and one block per charge.
TensorMap tensor_of(const std::shared_ptr<FusionCategory>& group, const py::dict& data) {
  TensorMap tensor;
  tensor.space.group = group;
  tensor.space.codomain = legs_of(group, data["codomain"].cast<py::list>());
  tensor.space.domain = legs_of(group, data["domain"].cast<py::list>());
  tensor.qtotal = data.contains("qtotal") ? data["qtotal"].cast<Sector>() : group->trivial_sector();
  for (const py::handle& item : data["blocks"].cast<py::list>()) {
    const py::tuple entry = item.cast<py::tuple>();
    tensor.blocks.emplace(entry[0].cast<Sector>(), entry[1].cast<at::Tensor>());
  }
  return tensor;
}

// The Python description of a tensor, its legs in the format the bindings take.
py::dict tensor_as(const TensorMap& tensor) {
  py::dict data;
  for (const std::vector<Leg>* legs : {&tensor.space.codomain, &tensor.space.domain}) {
    py::list described;
    for (const Leg& leg : *legs) {
      py::list summands;
      for (int64_t at = 0; at < leg.nblocks(); ++at) {
        summands.append(py::make_tuple(py::tuple(py::cast(leg.sector(at))), leg.multiplicity(at)));
      }
      described.append(py::make_tuple(summands, leg.qconj()));
    }
    data[legs == &tensor.space.codomain ? "codomain" : "domain"] = described;
  }
  py::list blocks;
  for (const auto& [charge, block] : tensor.blocks) blocks.append(py::make_tuple(py::tuple(py::cast(charge)), block));
  data["blocks"] = blocks;
  data["qtotal"] = py::tuple(py::cast(tensor.qtotal));
  return data;
}

void bind_contract(py::module_& m) {
  m.def("engine_permute_tensor", [](const std::shared_ptr<FusionCategory> group, const py::dict& tensor,
                                    int64_t keep, const std::vector<int64_t>& order,
                                    const py::object& device) {
    return tensor_as(permute_tensor(tensor_of(group, tensor), keep, order, device_of(device)));
  }, py::arg("group"), py::arg("tensor"), py::arg("keep"), py::arg("order"),
     py::arg("device") = py::none());
  m.def("engine_contract", [](const std::shared_ptr<FusionCategory> group, const py::dict& left,
                              const py::dict& right) {
    return tensor_as(contract(tensor_of(group, left), tensor_of(group, right)));
  }, py::arg("group"), py::arg("left"), py::arg("right"));
}

}  // namespace sym::engine

#include <stdexcept>


namespace sym::engine {
namespace {

// The sum of the squared moduli of every entry of a block.
double squared_norm(const at::Tensor& block) {
  return at::norm(block).item<double>() * at::norm(block).item<double>();
}

}  // namespace

TensorMap conj(const TensorMap& tensor) {
  TensorMap out;
  out.space = tensor.space;
  out.qtotal = tensor.qtotal;
  for (const auto& [charge, block] : tensor.blocks) out.blocks.emplace(charge, at::conj(block));
  return out;
}

TensorMap adjoint(const TensorMap& tensor) {
  TensorMap out;
  out.space.group = tensor.space.group;
  out.space.codomain = tensor.space.domain;
  out.space.domain = tensor.space.codomain;
  out.qtotal = tensor.space.group->dual_sector(tensor.qtotal);
  for (Leg& leg : out.space.codomain) leg = leg.conj();
  for (Leg& leg : out.space.domain) leg = leg.conj();
  for (const auto& [charge, block] : tensor.blocks) {
    out.blocks.emplace(charge, at::conj(block).transpose(0, 1).contiguous());
  }
  return out;
}

double norm(const TensorMap& tensor) {
  double total = 0.0;
  for (const auto& [charge, block] : tensor.blocks) total += squared_norm(block);
  return std::sqrt(total);
}

std::complex<double> inner(const TensorMap& left, const TensorMap& right) {
  if (!(left.space.codomain == right.space.codomain && left.space.domain == right.space.domain)) {
    throw std::invalid_argument("inner: the two tensors do not cover the same space");
  }
  std::complex<double> total(0.0, 0.0);
  for (const auto& [charge, block] : left.blocks) {
    const auto found = right.blocks.find(charge);
    if (found == right.blocks.end()) continue;
    const at::Tensor product = at::sum(block * at::conj(found->second));
    const c10::complex<double> value = product.item<c10::complex<double>>();
    total += std::complex<double>(value.real(), value.imag());
  }
  return total;
}

void bind_algebra(py::module_& m) {
  m.def("engine_conj", [](const std::shared_ptr<FusionCategory> group, const py::dict& tensor) {
    return tensor_as(conj(tensor_of(group, tensor)));
  }, py::arg("group"), py::arg("tensor"));
  m.def("engine_adjoint", [](const std::shared_ptr<FusionCategory> group, const py::dict& tensor) {
    return tensor_as(adjoint(tensor_of(group, tensor)));
  }, py::arg("group"), py::arg("tensor"));
  m.def("engine_norm", [](const std::shared_ptr<FusionCategory> group, const py::dict& tensor) {
    return norm(tensor_of(group, tensor));
  }, py::arg("group"), py::arg("tensor"));
  m.def("engine_inner", [](const std::shared_ptr<FusionCategory> group, const py::dict& left,
                           const py::dict& right) {
    const std::complex<double> value = inner(tensor_of(group, left), tensor_of(group, right));
    return py::make_tuple(value.real(), value.imag());
  }, py::arg("group"), py::arg("left"), py::arg("right"));
}

}  // namespace sym::engine

#include <pybind11/complex.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>

namespace sym::engine {
namespace {

std::shared_ptr<FusionCategory> group_of(const std::vector<Leg>& codomain, const std::vector<Leg>& domain) {
  if (!codomain.empty()) return codomain.front().group();
  if (!domain.empty()) return domain.front().group();
  throw std::invalid_argument("a tensor needs at least one leg to fix its category");
}

std::vector<std::vector<LegCharge>> charges_of(const std::vector<Leg>& legs) {
  std::vector<std::vector<LegCharge>> out;
  for (const Leg& leg : legs) out.push_back(leg_charges(leg));
  return out;
}

}  // namespace

const TensorMap& Tensor::map() const {
  if (!map_) throw std::logic_error("this tensor is a bare matrix without graded legs");
  return *map_;
}

TensorMap& Tensor::map() {
  if (!map_) throw std::logic_error("this tensor is a bare matrix without graded legs");
  return *map_;
}

Tensor Tensor::filled(const std::vector<Leg>& codomain, const std::vector<Leg>& domain,
                      const Sector& qtotal, int kind, at::ScalarType dtype,
                      const c10::Device& device) {
  const auto group = group_of(codomain, domain);
  TensorMap out;
  out.space.group = group;
  out.space.codomain = codomain;
  out.space.domain = domain;
  out.qtotal = qtotal;
  const std::vector<std::vector<LegCharge>> out_charges = charges_of(codomain);
  const std::vector<std::vector<LegCharge>> in_charges = charges_of(domain);
  const auto options = at::TensorOptions().dtype(dtype).device(device);
  std::vector<FusionTreePair> pairs = tree_pairs(*group, out_charges, in_charges);

  std::set<Sector> charges;
  for (const FusionTreePair& pair : pairs) {
    // A one-sided tensor is a state, so it carries exactly the total charge it is labelled with.
    if ((domain.empty() || codomain.empty()) && pair.codomain.coupled != qtotal) continue;
    charges.insert(pair.codomain.coupled);
  }
  for (const Sector& charge : charges) {
    std::vector<FusionTree> codomain_trees;
    std::vector<FusionTree> domain_trees;
    for (const FusionTreePair& pair : pairs) {
      if (pair.codomain.coupled != charge) continue;
      if (std::find(codomain_trees.begin(), codomain_trees.end(), pair.codomain) == codomain_trees.end())
        codomain_trees.push_back(pair.codomain);
      if (std::find(domain_trees.begin(), domain_trees.end(), pair.domain) == domain_trees.end())
        domain_trees.push_back(pair.domain);
    }
    const int64_t rows = side_layout(codomain, codomain_trees).rows();
    const int64_t columns = side_layout(domain, domain_trees).rows();
    at::Tensor block;
    if (kind == 0) block = at::zeros({rows, columns}, options);
    else if (kind == 1) block = at::ones({rows, columns}, options);
    else if (device.is_cpu()) block = at::rand({rows, columns}, options);
    else block = at::rand({rows, columns}, options.device(at::kCPU)).to(device);
    out.blocks.emplace(charge, std::move(block));
  }
  return Tensor(std::move(out));
}

Tensor Tensor::from_dense(const at::Tensor& data, std::optional<std::vector<Leg>> legs,
                          std::optional<Sector> qtotal) {
  if (!legs) return Tensor(data);
  const std::vector<Leg>& given = *legs;
  const std::shared_ptr<FusionCategory> group = group_of(given, {});
  for (const Leg& leg : given) {
    for (int64_t at = 0; at < leg.nblocks(); ++at) {
      if (group->sector_dim(leg.sector(at)) != 1) {
        throw std::invalid_argument(
            "a component array is only the block data of a tensor whose sectors are all "
            "one-dimensional; a symmetry-adapted leg is built in the tree basis");
      }
    }
  }
  TensorMap out;
  out.space.group = group;
  out.space.codomain = given;
  out.qtotal = qtotal.value_or(group->trivial_sector());
  const auto trees = trees_by_charge(*group, charges_of(given));
  const auto where = trees.find(out.qtotal);
  const int64_t rows = where == trees.end() ? 0 : side_layout(given, where->second).rows();
  if (data.numel() != rows) {
    throw std::invalid_argument("the component array does not hold the rows of the tree basis: " +
                                std::to_string(data.numel()) + " against " + std::to_string(rows));
  }
  out.blocks.emplace(out.qtotal, data.reshape({rows, 1}));
  return Tensor(std::move(out));
}

std::shared_ptr<FusionCategory> Tensor::group() const {
  // A bare matrix belongs to the category with no symmetry.
  if (!map_) return no_symmetry();
  return map().space.group;
}

std::vector<Leg> Tensor::legs() const {
  // A bare matrix carries no symmetry, so every axis is a leg of that dimension alone.
  if (!map_) {
    std::vector<Leg> plain;
    for (int64_t axis = 0; axis < plain_.dim(); ++axis) {
      plain.push_back(Leg::trivial(plain_.size(axis)));
    }
    return plain;
  }
  const TensorMap& tensor = map();
  std::vector<Leg> out = tensor.space.codomain;
  out.insert(out.end(), tensor.space.domain.begin(), tensor.space.domain.end());
  return out;
}

Sector Tensor::qtotal() const {
  // A bare matrix carries no category, so its total charge is the empty one.
  if (!map_) return Sector{};
  return map().qtotal;
}

std::vector<at::Tensor> Tensor::blocks() const {
  if (!map_) return {plain_};
  std::vector<at::Tensor> out;
  for (const auto& [charge, block] : map().blocks) out.push_back(block);
  return out;
}

std::vector<Sector> Tensor::charges() const {
  std::vector<Sector> out;
  if (!map_) return out;
  for (const auto& [charge, block] : map().blocks) out.push_back(charge);
  return out;
}

std::vector<int64_t> Tensor::shape() const {
  if (!map_) return plain_.sizes().vec();
  std::vector<int64_t> out;
  for (const Leg& leg : map().space.codomain) out.push_back(leg.dim());
  for (const Leg& leg : map().space.domain) out.push_back(leg.dim());
  return out;
}

at::ScalarType Tensor::dtype() const {
  if (!map_) return plain_.scalar_type();
  for (const auto& [charge, block] : map().blocks) return block.scalar_type();
  return at::kComplexDouble;
}

c10::Device Tensor::device() const {
  if (!map_) return plain_.device();
  for (const auto& [charge, block] : map().blocks) return block.device();
  return at::TensorOptions().device();
}

bool Tensor::is_dense() const {
  if (!map_) return true;
  for (const Leg& leg : legs()) {
    for (int64_t at = 0; at < leg.nblocks(); ++at) {
      if (leg.group()->sector_dim(leg.sector(at)) != 1) return false;
    }
  }
  return true;
}

std::complex<double> Tensor::item() const {
  if (!map_) {
    if (plain_.numel() != 1) throw std::invalid_argument("item needs a tensor of one entry");
    return std::complex<double>(plain_.item<double>(), 0.0);
  }
  if (!map().space.codomain.empty() || !map().space.domain.empty()) {
    throw std::invalid_argument("item needs a tensor without legs");
  }
  std::complex<double> total(0.0, 0.0);
  for (const auto& [charge, block] : map().blocks) {
    const c10::complex<double> value = block.reshape({-1}).select(0, 0).item<c10::complex<double>>();
    total += std::complex<double>(value.real(), value.imag());
  }
  return total;
}

Tensor Tensor::clone() const {
  if (!map_) return Tensor(plain_.clone());
  TensorMap out = map();
  for (auto& [charge, block] : out.blocks) block = block.clone();
  return Tensor(std::move(out));
}

Tensor Tensor::detach() const {
  if (!map_) return Tensor(plain_.detach());
  TensorMap out = map();
  for (auto& [charge, block] : out.blocks) block = block.detach();
  return Tensor(std::move(out));
}

Tensor Tensor::conj() const {
  if (!map_) return Tensor(at::conj(plain_));
  return Tensor(engine::conj(map()));
}

Tensor Tensor::to(const py::kwargs& options) const {
  at::TensorOptions wanted;
  if (options.contains("dtype")) wanted = wanted.dtype(options["dtype"].cast<at::ScalarType>());
  if (options.contains("device")) wanted = wanted.device(options["device"].cast<c10::Device>());
  if (!map_) return Tensor(plain_.to(wanted));
  TensorMap out = map();
  for (auto& [charge, block] : out.blocks) block = block.to(wanted);
  return Tensor(std::move(out));
}

Tensor Tensor::combine_legs(const std::vector<std::vector<int64_t>>& groups) const {
  throw std::logic_error("combine_legs: not implemented");
}


Tensor& Tensor::operator+=(const Tensor& other) {
  if (!map_ || !other.map_) throw std::logic_error("a bare matrix does not add to a graded tensor");
  for (const auto& [charge, block] : other.map().blocks) {
    const auto found = map_->blocks.find(charge);
    if (found == map_->blocks.end()) map_->blocks.emplace(charge, block.clone());
    else found->second = found->second + block;
  }
  return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
  if (!map_ || !other.map_) throw std::logic_error("a bare matrix does not subtract a graded tensor");
  for (const auto& [charge, block] : other.map().blocks) {
    const auto found = map_->blocks.find(charge);
    if (found == map_->blocks.end()) map_->blocks.emplace(charge, -block);
    else found->second = found->second - block;
  }
  return *this;
}

Tensor& Tensor::operator*=(const std::complex<double>& scalar) {
  if (!map_) {
    plain_ = plain_ * at::scalar_to_tensor(c10::complex<double>(scalar.real(), scalar.imag()));
    return *this;
  }
  for (auto& [charge, block] : map_->blocks) {
    block = block * at::scalar_to_tensor(c10::complex<double>(scalar.real(), scalar.imag()));
  }
  return *this;
}

Tensor& Tensor::operator/=(const std::complex<double>& scalar) {
  if (!map_) {
    plain_ = plain_ / at::scalar_to_tensor(c10::complex<double>(scalar.real(), scalar.imag()));
    return *this;
  }
  for (auto& [charge, block] : map_->blocks) {
    block = block / at::scalar_to_tensor(c10::complex<double>(scalar.real(), scalar.imag()));
  }
  return *this;
}

Tensor Tensor::operator-() const {
  if (!map_) return Tensor(-plain_);
  TensorMap out = map();
  for (auto& [charge, block] : out.blocks) block = -block;
  return Tensor(std::move(out));
}

Tensor einsum(const std::string& spec, const std::vector<Tensor>& operands) {
  const std::size_t arrow = spec.find("->");
  const std::string sides = arrow == std::string::npos ? spec : spec.substr(0, arrow);
  const std::string wanted = arrow == std::string::npos ? std::string() : spec.substr(arrow + 2);
  std::vector<std::string> groups;
  std::string current;
  for (const char letter : sides) {
    if (letter == ',') {
      groups.push_back(current);
      current.clear();
    } else {
      current.push_back(letter);
    }
  }
  groups.push_back(current);
  if (groups.size() != operands.size()) {
    throw std::invalid_argument("einsum: the specification names " + std::to_string(groups.size()) +
                                " operands, " + std::to_string(operands.size()) + " were given");
  }
  for (const Tensor& operand : operands) {
    if (!operand.graded()) throw std::invalid_argument("einsum: a bare matrix carries no legs to contract");
  }
  // The labels that are summed over, in one order for every operand, so that the two sides of a
  // contraction meet in the same order.
  std::string summed;
  for (const std::string& group : groups) {
    for (const char letter : group) {
      if (wanted.find(letter) != std::string::npos) continue;
      if (summed.find(letter) == std::string::npos) summed.push_back(letter);
    }
  }
  std::sort(summed.begin(), summed.end());
  // A kept label comes out on the side it went in on: a leg of a codomain stays in the codomain, so
  // that a contraction of maps is again a map.
  std::map<char, int> side;
  for (std::size_t at = 0; at < operands.size(); ++at) {
    const int64_t codomain_size = static_cast<int64_t>(operands.at(at).map().space.codomain.size());
    for (std::size_t where = 0; where < groups.at(at).size(); ++where) {
      const char letter = groups.at(at).at(where);
      if (wanted.find(letter) == std::string::npos) continue;
      side[letter] = static_cast<int64_t>(where) < codomain_size ? 0 : 1;
    }
  }
  const auto device = at::TensorOptions().device();
  // The operands are contracted in the order that always joins two tensors through shared labels: a
  // product of three or more indices is a chain, and grouping it greedily keeps every step a real
  // contraction instead of an outer product.
  std::vector<std::string> pending = groups;
  std::vector<Tensor> waiting = operands;
  Tensor accumulated;
  std::string labels;
  {
    accumulated = waiting.front();
    labels = pending.front();
    waiting.erase(waiting.begin());
    pending.erase(pending.begin());
  }
  while (!waiting.empty()) {
    std::size_t chosen = 0;
    std::size_t most = 0;
    for (std::size_t at = 0; at < pending.size(); ++at) {
      std::size_t common = 0;
      for (const char letter : pending.at(at))
        if (labels.find(letter) != std::string::npos) ++common;
      if (common > most) {
        most = common;
        chosen = at;
      }
    }
    if (most == 0) {
      throw std::invalid_argument(
          "einsum: the operands do not form a connected contraction, so a step would be an outer "
          "product of tensors at different charges");
    }
    const std::string next = pending.at(chosen);
    std::string shared;
    for (const char letter : labels)
      if (next.find(letter) != std::string::npos) shared.push_back(letter);
    const auto device_order = [](const std::string& from, const std::string& letters) {
      std::vector<int64_t> order;
      for (const char letter : letters) order.push_back(static_cast<int64_t>(from.find(letter)) + 1);
      return order;
    };
    std::string left_kept;
    for (const char letter : labels)
      if (shared.find(letter) == std::string::npos) left_kept.push_back(letter);
    std::string right_kept;
    for (const char letter : next)
      if (shared.find(letter) == std::string::npos) right_kept.push_back(letter);
    // The shared labels of the first tensor go to its domain and those of the second to its
    // codomain, which is where a contraction meets them; the remaining legs stay on their side.
    std::string left_order = left_kept + shared;
    std::string right_order = shared + right_kept;
    Tensor left = Tensor(permute_tensor(accumulated.map(), left_kept.size(),
                                        device_order(labels, left_order), device));
    Tensor right = Tensor(permute_tensor(waiting.at(chosen).map(), shared.size(),
                                         device_order(next, right_order), device));
    accumulated = Tensor(engine::contract(left.map(), right.map()));
    labels = left_kept + right_kept;
    waiting.erase(waiting.begin() + static_cast<int64_t>(chosen));
    pending.erase(pending.begin() + static_cast<int64_t>(chosen));
  }
  // The contracted tensor carries the kept labels in the order the operands named them; the output
  // fixes their final order.
  if (wanted.empty()) return accumulated;
  std::vector<int64_t> order;
  int64_t keep_out = 0;
  for (const int want_codomain : {0, 1}) {
    for (const char letter : wanted) {
      if (side.at(letter) != want_codomain) continue;
      const std::size_t where = labels.find(letter);
      if (where == std::string::npos) {
        throw std::invalid_argument("einsum: the output names a label no operand carries");
      }
      order.push_back(static_cast<int64_t>(where) + 1);
      if (want_codomain == 0) ++keep_out;
    }
  }
  if (order.size() != labels.size()) {
    throw std::invalid_argument("einsum: the output drops a label that no other operand takes");
  }
  return Tensor(permute_tensor(accumulated.map(), keep_out, order, device));
}


namespace {

// The legs of a constructor call: either the plain shape of a bare matrix or the graded legs.
struct Constructor {
  std::vector<Leg> codomain;
  std::vector<Leg> domain;
  Sector qtotal;
};

Constructor graded(const py::object& legs, const py::object& qtotal, int64_t codomain) {
  if (legs.is_none()) throw std::invalid_argument("a graded tensor needs its legs");
  const std::vector<Leg> all = legs.cast<std::vector<Leg>>();
  if (codomain < 0 || codomain > static_cast<int64_t>(all.size())) {
    throw std::invalid_argument("the number of codomain legs is out of range");
  }
  Constructor out;
  out.codomain.assign(all.begin(), all.begin() + codomain);
  out.domain.assign(all.begin() + codomain, all.end());
  out.qtotal = qtotal.is_none() ? all.front().group()->trivial_sector() : qtotal.cast<Sector>();
  return out;
}

Tensor filled_of(const py::args& shape, const py::kwargs& options, int kind) {
  py::object legs = py::none();
  py::object qtotal = py::none();
  if (options.contains("legs")) legs = py::reinterpret_borrow<py::object>(options["legs"]);
  if (options.contains("qtotal")) qtotal = py::reinterpret_borrow<py::object>(options["qtotal"]);
  const at::ScalarType dtype =
      options.contains("dtype") ? options["dtype"].cast<at::ScalarType>() : at::kComplexDouble;
  const c10::Device device =
      options.contains("device") ? options["device"].cast<c10::Device>() : at::TensorOptions().device();
  if (legs.is_none()) {
    std::vector<int64_t> dims;
    for (const py::handle& entry : shape) dims.push_back(entry.cast<int64_t>());
    const auto options_of = at::TensorOptions().dtype(dtype).device(device);
    at::Tensor plain;
    if (kind == 0) plain = at::zeros(dims, options_of);
    else if (kind == 1) plain = at::ones(dims, options_of);
    else if (device.is_cpu()) plain = at::rand(dims, options_of);
    else plain = at::rand(dims, options_of.device(at::kCPU)).to(device);
    return Tensor(plain);
  }
  // A caller may name the number of codomain legs; otherwise every leg points in, the way a state or
  // an operator is written as a list of legs, and the einsum driver moves the glued ones where they
  // are needed.
  const int64_t codomain = options.contains("codomain") ? options["codomain"].cast<int64_t>() : 0;
  const Constructor built = graded(legs, qtotal, codomain);
  return Tensor::filled(built.codomain, built.domain, built.qtotal, kind, dtype, device);
}

}  // namespace

void bind_tensor(py::module_& m) {
  py::class_<Tensor>(m, "Tensor")
      .def(py::init([](const std::vector<Sector>& charges, const std::vector<at::Tensor>& blocks,
                       const std::vector<Leg>& legs, const py::object& qtotal, int64_t codomain) {
        TensorMap map;
        map.space.group = legs.front().group();
        map.space.codomain.assign(legs.begin(), legs.begin() + codomain);
        map.space.domain.assign(legs.begin() + codomain, legs.end());
        map.qtotal = qtotal.is_none() ? map.space.group->trivial_sector() : qtotal.cast<Sector>();
        for (std::size_t at = 0; at < charges.size(); ++at) map.blocks.emplace(charges.at(at), blocks.at(at));
        return Tensor(std::move(map));
      }),
           py::arg("charges"), py::arg("blocks"), py::arg("legs"), py::arg("qtotal") = py::none(),
           py::arg("codomain") = -1)
      .def_static("zeros", [](const py::args& shape, const py::kwargs& options) { return filled_of(shape, options, 0); })
      .def_static("ones", [](const py::args& shape, const py::kwargs& options) { return filled_of(shape, options, 1); })
      .def_static("rand", [](const py::args& shape, const py::kwargs& options) { return filled_of(shape, options, 2); })
      .def_static("from_torch",
                  [](const at::Tensor& data, const py::object& legs, const py::object& qtotal) {
                    if (legs.is_none()) return Tensor(data);
                    return Tensor::from_dense(data, legs.cast<std::vector<Leg>>(),
                                              qtotal.is_none() ? std::optional<Sector>{}
                                                               : std::optional<Sector>{qtotal.cast<Sector>()});
                  },
                  py::arg("data"), py::arg("legs") = py::none(), py::arg("qtotal") = py::none())
      .def_property_readonly("blocks", &Tensor::blocks)
      .def_property_readonly("qdata", &Tensor::charges)
      .def_property_readonly("charges", [](const Tensor& tensor) {
        py::list out;
        for (const Sector& charge : tensor.charges()) out.append(py::tuple(py::cast(charge)));
        return out;
      })
      .def_property_readonly("legs", &Tensor::legs)
      .def_property_readonly("qtotal", [](const Tensor& tensor) {
        return py::tuple(py::cast(tensor.qtotal()));
      })
      .def_property_readonly("group", &Tensor::group)
      .def_property_readonly("shape", [](const Tensor& tensor) {
        return py::tuple(py::cast(tensor.shape()));
      })
      .def_property_readonly("dtype", &Tensor::dtype)
      .def_property_readonly("device", &Tensor::device)
      .def_property_readonly("is_dense", &Tensor::is_dense)
      .def("numel", [](const Tensor& tensor) {
        int64_t total = 0;
        for (const at::Tensor& block : tensor.blocks()) total += block.numel();
        return total;
      })
      .def("element_size", [](const Tensor& tensor) { return tensor.blocks().front().element_size(); })
      .def("item", &Tensor::item)
      .def("clone", &Tensor::clone)
      .def("detach", &Tensor::detach)
      .def("conj", &Tensor::conj)
      .def("to", [](const Tensor& tensor, const py::kwargs& options) { return tensor.to(options); })
      .def("combine_legs", &Tensor::combine_legs)
      .def("__len__", [](const Tensor& tensor) { return static_cast<int64_t>(tensor.blocks().size()); })
      .def("__getitem__", [](const Tensor& tensor, int64_t index) {
        if (tensor.graded()) {
          throw std::invalid_argument("indexing a graded tensor is not supported");
        }
        return Tensor(tensor.plain().select(0, index));
      })
      .def("abs", [](const Tensor& tensor) { return std::abs(tensor.item()); })
      .def("__abs__", [](const Tensor& tensor) { return std::abs(tensor.item()); })
      .def("__neg__", [](const Tensor& tensor) { return -tensor; })
      .def("__add__", [](const Tensor& left, const Tensor& right) { Tensor out = left.clone(); out += right; return out; })
      .def("__radd__", [](const Tensor& left, const Tensor& right) { Tensor out = right.clone(); out += left; return out; })
      .def("__sub__", [](const Tensor& left, const Tensor& right) { Tensor out = left.clone(); out -= right; return out; })
      .def("__mul__", [](const Tensor& left, const std::complex<double>& scalar) { Tensor out = left.clone(); out *= scalar; return out; })
      .def("__rmul__", [](const Tensor& left, const std::complex<double>& scalar) { Tensor out = left.clone(); out *= scalar; return out; })
      .def("__truediv__", [](const Tensor& left, const std::complex<double>& scalar) { Tensor out = left.clone(); out /= scalar; return out; })
      .def("__iadd__", [](Tensor& left, const Tensor& right) -> Tensor& { left += right; return left; })
      .def("__imul__", [](Tensor& left, const std::complex<double>& scalar) -> Tensor& { left *= scalar; return left; })
      .def("__itruediv__", [](Tensor& left, const std::complex<double>& scalar) -> Tensor& { left /= scalar; return left; })
      .def("__repr__", [](const Tensor& tensor) {
        std::string text = "Tensor(legs ";
        for (const int64_t dim : tensor.shape()) text += std::to_string(dim) + ",";
        text += " blocks " + std::to_string(tensor.blocks().size()) + ")";
        return text;
      });

  // The python layer calls the contraction with one argument per operand, not with a list.
  const auto contract_given = [](const std::string& spec, const py::args& operands) {
    std::vector<Tensor> given;
    for (const py::handle& operand : operands) given.push_back(operand.cast<Tensor>());
    if (std::getenv("EINSUM_SPEC") != nullptr) {
      std::fprintf(stderr, "einsum '%s':", spec.c_str());
      for (const Tensor& one : given)
        std::fprintf(stderr, " %zu|%zu", one.graded() ? one.map().space.codomain.size() : 0,
                     one.graded() ? one.map().space.domain.size() : 0);
      std::fprintf(stderr, "\n");
    }
    return einsum(spec, given);
  };
  m.def("cached_einsum", contract_given, py::arg("spec"));
  m.def("engine_einsum", contract_given, py::arg("spec"));
}

}  // namespace sym::engine

#include <string>
#include <utility>
#include <vector>

namespace sym::engine {
namespace {


// A leg handed in from Python: its summands as (sector, multiplicity) and the direction it points.
struct LegDesc {
  std::vector<std::pair<Sector, int64_t>> sectors;
  int64_t qconj = 1;
};

LegDesc leg_desc(const py::tuple& data) {
  LegDesc desc;
  desc.sectors = data[0].cast<std::vector<std::pair<Sector, int64_t>>>();
  desc.qconj = data[1].cast<int64_t>();
  return desc;
}

std::vector<std::vector<LegCharge>> charges_of(const FusionCategory& group,
                                               const std::vector<LegDesc>& descs) {
  std::vector<std::vector<LegCharge>> out;
  for (const LegDesc& desc : descs) {
    std::vector<LegCharge> one;
    for (const auto& [sector, multiplicity] : desc.sectors)
      one.push_back(LegCharge{sector, desc.qconj == -1});
    out.push_back(std::move(one));
  }
  return out;
}

// A map from a charge to its trees, with a tuple key so that Python can hash it.
py::dict keyed(const std::map<Sector, std::vector<FusionTree>>& trees) {
  py::dict out;
  for (const auto& [charge, list] : trees) {
    std::vector<std::string> names;
    for (const FusionTree& tree : list) names.push_back(repr(tree));
    out[py::tuple(py::cast(charge))] = names;
  }
  return out;
}

}  // namespace

void bind_engine(py::module_& m) {
  // Every tree of one side, grouped by charge: the enumeration a block layout is built from.
  m.def("engine_trees", [](const std::shared_ptr<FusionCategory> group, const py::list& legs) {
    std::vector<LegDesc> descs;
    for (const py::handle& leg : legs) descs.push_back(leg_desc(leg.cast<py::tuple>()));
    return keyed(trees_by_charge(*group, charges_of(*group, descs)));
  });

  // Every tree pair of a two-sided tensor: the block enumeration, laid out row-major per charge.
  m.def("engine_pairs", [](const std::shared_ptr<FusionCategory> group, const py::list& codomain,
                           const py::list& domain) {
    std::vector<LegDesc> out_descs;
    std::vector<LegDesc> in_descs;
    for (const py::handle& leg : codomain) out_descs.push_back(leg_desc(leg.cast<py::tuple>()));
    for (const py::handle& leg : domain) in_descs.push_back(leg_desc(leg.cast<py::tuple>()));
    std::map<Sector, std::vector<std::string>> out;
    for (const FusionTreePair& pair :
         tree_pairs(*group, charges_of(*group, out_descs), charges_of(*group, in_descs))) {
      out[pair.codomain.coupled].push_back(repr(pair));
    }
    return out;
  });

  // The same enumeration as `engine_pairs`, but with each pair as the dict the moves take.
  m.def("engine_pair_list", [](const std::shared_ptr<FusionCategory> group, const py::list& codomain,
                               const py::list& domain) {
    std::vector<LegDesc> out_descs;
    std::vector<LegDesc> in_descs;
    for (const py::handle& leg : codomain) out_descs.push_back(leg_desc(leg.cast<py::tuple>()));
    for (const py::handle& leg : domain) in_descs.push_back(leg_desc(leg.cast<py::tuple>()));
    const auto tree_as = [](const FusionTree& tree) {
      py::dict data;
      data["sectors"] = tree.sectors;
      data["isdual"] = tree.isdual;
      data["inner"] = tree.inner;
      data["vertices"] = tree.vertices;
      data["coupled"] = py::tuple(py::cast(tree.coupled));
      return data;
    };
    py::list out;
    for (const FusionTreePair& pair :
         tree_pairs(*group, charges_of(*group, out_descs), charges_of(*group, in_descs))) {
      py::dict data;
      data["codomain"] = tree_as(pair.codomain);
      data["domain"] = tree_as(pair.domain);
      out.append(data);
    }
    return out;
  });

  // The component array of one tree of one side, selected by its position in the enumeration.

  // Lay a tensor out as its component array: the sum of every block entry times its tree tensor.

  bind_duality(m);
  bind_braid(m);
  bind_contract(m);
  bind_split(m);
  bind_tensor(m);
  bind_algebra(m);
}

}  // namespace sym::engine
