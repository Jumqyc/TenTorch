// The block-sparse tensor of the engine: a graded space, the map over it, the contraction, its
// algebra and the python bindings.
#include "tensor.h"

#include "../symmetry/cat/symbol.h"
#include "diagram.h"
#include "../linalg/graded/linalg.h"

#include <pybind11/complex.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace sym::engine {
namespace {

std::vector<std::vector<LegCharge>> charges_of(const std::vector<Leg>& legs);

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
  // The blocks of a leg stand in the order of their sectors, the order every other route gives
  // them, so two legs that carry the same sectors compare equal after a split or a fusion.
  std::vector<std::size_t> order(summands.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&summands](std::size_t a, std::size_t b) {
    return summands.at(a).first < summands.at(b).first;
  });
  std::vector<Sector> sectors;
  std::vector<int64_t> slices{0};
  for (const std::size_t at : order) {
    const auto& [sector, multiplicity] = summands.at(at);
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
  // An empty tree is the unit vector the coupled sector spans, so a block entry is the coefficient of
  // the basis vector and a singular value is a singular value of the state, not of it times a dimension.
  if (size == 0) {
    const double dimension = static_cast<double>(group.sector_dim(tree.coupled));
    return at::ones({group.sector_dim(tree.coupled)}, options) / std::sqrt(dimension);
  }
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
#include <numeric>
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

// The roots a side fuses to; a side without legs fuses to the unit alone.
std::vector<Sector> roots_of(const FusionCategory& group,
                             const std::map<Sector, std::vector<FusionTree>>& trees) {
  std::vector<Sector> roots;
  for (const auto& [root, list] : trees) roots.push_back(root);
  if (roots.empty()) roots.push_back(group.trivial_sector());
  return roots;
}

// The rows (or columns) of a block keyed `key`: every tree of this side whose root can meet a root of
// the other side at the key, laid out one root after another.
struct KeyLayout {
  std::map<Sector, std::pair<SideLayout, int64_t>> parts;
  int64_t rows = 0;
};

// The rows (or columns) of a block keyed `key`: the trees of this side that fuse to the key.
KeyLayout key_layout(const std::vector<Leg>& legs,
                     const std::map<Sector, std::vector<FusionTree>>& mine, const Sector& key) {
  KeyLayout layout;
  if (legs.empty()) {
    // A side without legs has one tree whatever the key is, and that tree couples to the key.
    FusionTree empty;
    empty.coupled = key;
    const SideLayout part = side_layout(legs, std::vector<FusionTree>{empty});
    layout.parts.emplace(key, std::pair<SideLayout, int64_t>{part, 0});
    layout.rows = part.rows();
    return layout;
  }
  for (const auto& [root, trees] : mine) {
    // A block collects the trees of both sides that fuse to the same sector, which is the key.
    if (root != key) continue;
    SideLayout part = side_layout(legs, trees);
    layout.parts.emplace(root, std::pair<SideLayout, int64_t>{part, layout.rows});
    layout.rows += part.rows();
  }
  return layout;
}

std::pair<FusionTree, std::vector<int64_t>> key_parts(const KeyLayout& layout, int64_t row) {
  if (row < 0 || row >= layout.rows) throw std::out_of_range("a row is out of range for its block");
  for (auto back = layout.parts.rbegin(); back != layout.parts.rend(); ++back) {
    if (row < back->second.second) continue;
    const SideLayout& part = back->second.first;
    const auto [tree, mults] = row_parts(part, row - back->second.second);
    return {part.trees.at(tree), mults};
  }
  throw std::out_of_range("a row is out of range for its block");
}

int64_t key_row(const KeyLayout& layout, const FusionTree& tree, const std::vector<int64_t>& mults) {
  const auto found = layout.parts.find(tree.coupled);
  if (found == layout.parts.end()) {
    std::string roots;
    for (const auto& [root, part] : layout.parts) roots += " " + sector_repr(root);
    throw std::out_of_range("the tree " + repr(tree) + " does not belong to this block, whose roots are" +
                            roots);
  }
  const auto where = std::find(found->second.first.trees.begin(), found->second.first.trees.end(), tree);
  if (where == found->second.first.trees.end())
    throw std::out_of_range("the tree does not belong to this block");
  return found->second.second + row_index(found->second.first, where - found->second.first.trees.begin(), mults);
}

}  // namespace

// The legs of a tensor, rearranged so that the codomain carries `keep` of them: a leg that crosses
// changes its direction, and the trees are re-expressed in the basis of the other side. This is the
// move that lets two tensors meet on one side of each.
TensorMap repartition_tensor(const TensorMap& tensor, int64_t keep, const c10::Device& device) {
  const FusionCategory& group = *tensor.space.group;
  const int64_t total = tensor.space.size();
  if (keep < 0 || keep > total) {
    throw std::out_of_range("repartition_tensor: a tensor of " + std::to_string(total) +
                            " legs cannot carry " + std::to_string(keep) + " of them on one side");
  }
  if (tensor.space.codomain.empty() && tensor.space.domain.empty()) return tensor;

  TensorMap out;
  out.space.group = tensor.space.group;
  out.qtotal = tensor.qtotal;
  out.space.codomain = tensor.space.codomain;
  out.space.domain = tensor.space.domain;
  // A leg that crosses sides keeps the space it indexes: the side fixes the direction of its charges,
  // so the stored sector dualises with it and the sector the leg points at stays what it was.
  const auto crossed = [&group](const Leg& leg) {
    std::vector<Sector> sectors;
    sectors.reserve(leg.sectors().size());
    for (const Sector& sector : leg.sectors()) sectors.push_back(group.dual_sector(sector));
    return Leg(leg.group(), sectors, leg.slices(), -leg.qconj());
  };
  while (static_cast<int64_t>(out.space.codomain.size()) > keep) {
    const Leg moved = crossed(out.space.codomain.back());
    out.space.codomain.pop_back();
    out.space.domain.push_back(moved);
  }
  while (static_cast<int64_t>(out.space.codomain.size()) < keep) {
    const Leg moved = crossed(out.space.domain.back());
    out.space.domain.pop_back();
    out.space.codomain.push_back(moved);
  }

  const auto source_out = side_trees(group, tensor.space.codomain);
  const auto source_in = side_trees(group, tensor.space.domain);
  const auto target_out = side_trees(group, out.space.codomain);
  const auto target_in = side_trees(group, out.space.domain);
  std::map<Sector, KeyLayout> rows_of_key;
  std::map<Sector, KeyLayout> columns_of_key;
  const auto rows_at = [&](const Sector& key) -> const KeyLayout& {
    const auto found = rows_of_key.find(key);
    if (found != rows_of_key.end()) return found->second;
    return rows_of_key.emplace(key, key_layout(out.space.codomain, target_out, key)).first->second;
  };
  const auto columns_at = [&](const Sector& key) -> const KeyLayout& {
    const auto found = columns_of_key.find(key);
    if (found != columns_of_key.end()) return found->second;
    return columns_of_key.emplace(key, key_layout(out.space.domain, target_in, key)).first->second;
  };

  // The multiplicity of every leg follows it across, so both sides keep one entry per leg in order.
  std::vector<int64_t> out_mults;
  std::vector<int64_t> in_mults;
  std::map<Sector, std::map<std::pair<int64_t, int64_t>, std::complex<double>>> cells;
  std::map<Sector, std::pair<KeyLayout, KeyLayout>> layouts;
  // The tree transformation depends on the source pair alone, so each pair is moved once and its
  // terms are reused for every entry the block holds.
  std::map<FusionTreePair, std::vector<PairTerm>> moves;
  for (const auto& [root, block] : tensor.blocks) {
    const Sector source_domain_root =
        tensor.space.domain.empty()
            ? group.trivial_sector()
            : (tensor.space.codomain.empty() ? tensor.qtotal
                                             : partner_root(group, root, tensor.qtotal));
    const KeyLayout source_rows = key_layout(tensor.space.codomain, source_out, root);
    const KeyLayout source_columns = key_layout(tensor.space.domain, source_in, source_domain_root);
    if (source_rows.rows == 0 || source_columns.rows == 0) continue;
    const at::Tensor values = block.to(at::kCPU, at::kComplexDouble).contiguous();
    const c10::complex<double>* data = values.data_ptr<c10::complex<double>>();
    for (int64_t row = 0; row < source_rows.rows; ++row) {
      const auto [out_tree, row_mults] = key_parts(source_rows, row);
      for (int64_t column = 0; column < source_columns.rows; ++column) {
        const auto [in_tree, column_mults] = key_parts(source_columns, column);
        const c10::complex<double> entry = data[row * source_columns.rows + column];
        if (entry.real() == 0.0 && entry.imag() == 0.0) continue;
        const FusionTreePair source_pair{out_tree, in_tree};
        const auto known = moves.find(source_pair);
        const std::vector<PairTerm>& terms =
            known != moves.end()
                ? known->second
                : moves.emplace(source_pair, repartition(group, source_pair, keep, tensor.qtotal, device)).first->second;
        for (const PairTerm& term : terms) {
          const Sector key = term.pair.codomain.coupled;
          auto placed = layouts.find(key);
          if (placed == layouts.end()) {
            placed = layouts
                         .emplace(key, std::make_pair(rows_at(key), columns_at(term.pair.domain.coupled)))
                         .first;
          }
          // The legs the move hands to the other side keep their multiplicities, in the order the
          // move appends them: the last leg of the crowded side first.
          std::vector<int64_t> moved_out(row_mults);
          std::vector<int64_t> moved_in(column_mults);
          while (static_cast<int64_t>(moved_out.size()) > keep) {
            moved_in.push_back(moved_out.back());
            moved_out.pop_back();
          }
          while (static_cast<int64_t>(moved_out.size()) < keep) {
            moved_out.push_back(moved_in.back());
            moved_in.pop_back();
          }
          const int64_t where_row = key_row(placed->second.first, term.pair.codomain, moved_out);
          const int64_t where_column = key_row(placed->second.second, term.pair.domain, moved_in);
          cells[key][std::make_pair(where_row, where_column)] +=
              std::complex<double>(entry.real(), entry.imag()) * term.coefficient;
        }
      }
    }
  }
  const at::TensorOptions options =
      tensor.blocks.empty()
          ? at::TensorOptions().dtype(at::kComplexDouble)
          : at::TensorOptions().dtype(at::kComplexDouble).device(tensor.blocks.begin()->second.device());
  for (const auto& [key, entries] : cells) {
    const auto& [cell_rows, cell_columns] = layouts.at(key);
    // The cells are filled entry by entry on the host, then one copy puts the block on its device.
    at::Tensor block = at::zeros({cell_rows.rows, cell_columns.rows},
                                 at::TensorOptions().dtype(at::kComplexDouble));
    auto block_entries = block.accessor<c10::complex<double>, 2>();
    for (const auto& [where, value] : entries) {
      block_entries[where.first][where.second] = c10::complex<double>(value.real(), value.imag());
    }
    out.blocks.emplace(key, options.device().is_cpu() ? block : block.to(options.device()));
  }
  // The bends hand a leg over at the end of the receiving side, so the tail they built is reversed;
  // reversing it back keeps the leg order of the caller, and then the two operands of a contraction
  // glue leg by leg.
  const int64_t moved = std::llabs(keep - static_cast<int64_t>(tensor.space.codomain.size()));
  if (moved < 2) return out;
  const int64_t out_size = static_cast<int64_t>(out.space.codomain.size());
  const int64_t in_size = static_cast<int64_t>(out.space.domain.size());
  std::vector<int64_t> wanted(static_cast<std::size_t>(out_size + in_size));
  for (int64_t at = 0; at < out_size + in_size; ++at) wanted.at(at) = at + 1;
  const bool grew = out_size > static_cast<int64_t>(tensor.space.codomain.size());
  const int64_t base = grew ? out_size - moved : out_size + in_size - moved;
  for (int64_t at = 0; at < moved; ++at) wanted.at(base + at) = base + moved - at;
  return permute_tensor(out, keep, wanted, device);
}

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
  for (int64_t at = 0; at < n; ++at) {
    const int64_t leg = order.at(at) - 1;
    if (leg < 0 || leg >= n) {
      throw std::invalid_argument("permute_tensor: the order names a leg the tensor does not carry");
    }
    const bool from_domain = leg >= codomain_size;
    Leg moved = from_domain ? tensor.space.domain.at(leg - codomain_size)
                            : tensor.space.codomain.at(leg);
    const bool into_domain = at >= keep;
    // A permutation keeps every leg on its own side; turning a leg around is the adjoint of the whole
    // tensor, which the caller does before permuting.
    if (from_domain != into_domain) {
      throw std::invalid_argument(
          "permute_tensor: a permutation cannot move a leg to the other side; adjoint the tensor "
          "instead");
    }
    (into_domain ? out.space.domain : out.space.codomain).push_back(moved);
  }
  const double switched = 1.0;

  const std::map<Sector, std::vector<FusionTree>> source_out = side_trees(group, tensor.space.codomain);
  const std::map<Sector, std::vector<FusionTree>> source_in = side_trees(group, tensor.space.domain);
  const std::map<Sector, std::vector<FusionTree>> target_out = side_trees(group, out.space.codomain);
  const std::map<Sector, std::vector<FusionTree>> target_in = side_trees(group, out.space.domain);

  // The rows and columns of a destination block depend on its key alone, so they are built once.
  std::map<Sector, KeyLayout> rows_of_key;
  std::map<Sector, KeyLayout> columns_of_key;
  const auto rows_at = [&](const Sector& key) -> const KeyLayout& {
    const auto found = rows_of_key.find(key);
    if (found != rows_of_key.end()) return found->second;
    return rows_of_key
        .emplace(key, key_layout(out.space.codomain, target_out, key))
        .first->second;
  };
  const auto columns_at = [&](const Sector& key) -> const KeyLayout& {
    const auto found = columns_of_key.find(key);
    if (found != columns_of_key.end()) return found->second;
    return columns_of_key
        .emplace(key, key_layout(out.space.domain, target_in, key))
        .first->second;
  };

  std::map<Sector, Gathered> gathered;
  // The permutation of a pair of trees is the same for every entry of the block, so it is done once.
  std::map<FusionTreePair, std::vector<PairTerm>> moves;
  for (const auto& [root, block] : tensor.blocks) {
    // A block is fixed by its codomain root; the domain root is the sector the two sides fuse through.
    const Sector source_domain_root =
        tensor.space.domain.empty()
            ? group.trivial_sector()
            : (tensor.space.codomain.empty() ? tensor.qtotal
                                             : partner_root(group, root, tensor.qtotal));
    const KeyLayout source_rows = key_layout(tensor.space.codomain, source_out, root);
    const KeyLayout source_columns = key_layout(tensor.space.domain, source_in, source_domain_root);
    if (source_rows.rows == 0 || source_columns.rows == 0) continue;
    const at::Tensor values = block.to(at::kCPU, at::kComplexDouble).contiguous();
    const c10::complex<double>* data = values.data_ptr<c10::complex<double>>();
    const int64_t columns = source_columns.rows;
    if (block.numel() != source_rows.rows * columns) {
      throw std::invalid_argument("permute_tensor: a block does not hold the rows of the trees of its charge");
    }
    for (int64_t row = 0; row < source_rows.rows; ++row) {
      const auto [out_tree, out_mults] = key_parts(source_rows, row);
      for (int64_t column = 0; column < columns; ++column) {
        const c10::complex<double> stored = data[row * columns + column];
        if (stored.real() == 0.0 && stored.imag() == 0.0) continue;
        const auto [in_tree, in_mults] = key_parts(source_columns, column);
        // The multiplicity index of every leg of the source, in leg order.
        std::vector<int64_t> mults = out_mults;
        mults.insert(mults.end(), in_mults.begin(), in_mults.end());
        const FusionTreePair source{out_tree, in_tree};
        const auto known = moves.find(source);
        const std::vector<PairTerm>& terms =
            known != moves.end()
                ? known->second
                : moves.emplace(source, permute_pair(group, source, keep, order, device)).first->second;
        for (const PairTerm& term : terms) {
          // A bend changes the coupled sector the pair fuses to, so the term names its own block.
          const Sector target_charge = root;
          const KeyLayout& target_rows = rows_at(target_charge);
          const KeyLayout& target_columns = columns_at(source_domain_root);
          std::vector<int64_t> dest_out;
          for (int64_t at = 0; at < keep; ++at) dest_out.push_back(mults.at(order.at(at) - 1));
          std::vector<int64_t> dest_in;
          for (int64_t at = keep; at < n; ++at) dest_in.push_back(mults.at(order.at(at) - 1));
          Gathered& cell = gathered[target_charge];
          cell.rows = target_rows.rows;
          cell.columns = target_columns.rows;
          const int64_t dest_row = key_row(target_rows, term.pair.codomain, dest_out);
          const int64_t dest_column = key_row(target_columns, term.pair.domain, dest_in);
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

// The legs from `drop` on, as a tree of their own: they are left-nested among themselves, so their
// couplings are the tail of the couplings of the whole tree.
FusionTree glue_suffix(const FusionTree& tree, int64_t drop) {
  const int64_t begin = std::max<int64_t>(std::min<int64_t>(drop, tree.sectors.size()), 0);
  FusionTree out;
  out.sectors.assign(tree.sectors.begin() + begin, tree.sectors.end());
  out.isdual.assign(tree.isdual.begin() + begin, tree.isdual.end());
  const int64_t couplings = std::min<int64_t>(std::max<int64_t>(begin - 1, 0), tree.inner.size());
  out.inner.assign(tree.inner.begin() + couplings, tree.inner.end());
  out.vertices.assign(tree.vertices.begin() + std::min<int64_t>(begin, tree.vertices.size()),
                      tree.vertices.end());
  out.coupled = out.sectors.empty() ? Sector{}
                : out.sectors.size() == 1 ? out.sectors.front()
                                          : out.inner.back();
  return out;
}

// The sector a leg physically carries, whichever way its side points.
Sector physical_of(const FusionCategory& group, const FusionTree& tree, int64_t axis) {
  return tree.isdual.at(axis) ? group.dual_sector(tree.sectors.at(axis)) : tree.sectors.at(axis);
}

// The name of a glue tree over `mults`: two operands meet when their glue legs carry the same sectors,
// the same directions and the same couplings.
std::string glue_name(const FusionCategory& group, const FusionTree& glue,
                      const std::vector<int64_t>& mults) {
  std::string key;
  for (int64_t axis = 0; axis < static_cast<int64_t>(glue.sectors.size()); ++axis) {
    for (const int64_t component : physical_of(group, glue, axis))
      key += std::to_string(component) + ",";
    // The two glued trees are the same legs seen from opposite sides, so their duality flags differ
    // by construction and only the physical sectors and the multiplicities identify the pair.
    key += "." + std::to_string(mults.at(axis)) + ";";
  }
  for (const Sector& coupling : glue.inner) {
    for (const int64_t component : coupling) key += std::to_string(component) + ",";
    key += "/";
  }
  for (const int64_t vertex : glue.vertices) key += std::to_string(vertex) + ",";
  return key;
}

// Two trees over two leg lists, as the trees over the joined list: the junction couples the root of
// the first tree with the first leg of the second, and every allowed coupling is a separate basis
// vector of the joined side.
std::vector<FusionTree> join_trees(const FusionCategory& group, const FusionTree& first,
                                   const FusionTree& second) {
  if (first.sectors.empty()) return {second};
  if (second.sectors.empty()) return {first};
  const Sector first_root = physical_of(group, first, first.sectors.size() - 1);
  const Sector second_leg = physical_of(group, second, 0);
  std::vector<FusionTree> joined;
  for (const Sector& junction : group.fusion_outcomes(first_root, second_leg)) {
    FusionTree tree;
    tree.sectors = first.sectors;
    tree.sectors.insert(tree.sectors.end(), second.sectors.begin(), second.sectors.end());
    tree.isdual = first.isdual;
    tree.isdual.insert(tree.isdual.end(), second.isdual.begin(), second.isdual.end());
    tree.vertices = first.vertices;
    tree.vertices.insert(tree.vertices.end(), second.vertices.begin(), second.vertices.end());
    tree.inner = first.inner;
    tree.inner.push_back(junction);
    tree.inner.insert(tree.inner.end(), second.inner.begin(), second.inner.end());
    tree.coupled = second.coupled;
    joined.push_back(tree);
  }
  return joined;
}

// The name of a glue leg: its physical sectors and their multiplicities. A dual leg lists the sectors
// of its dual, so the name is taken in the same direction on both sides of a contraction; two legs with
// the same name have the same tree basis, and their blocks are multiplied row by row.
std::string leg_name(const Leg& leg) {
  std::vector<std::string> entries;
  for (int64_t at = 0; at < leg.nblocks(); ++at) {
    const Sector sector =
        leg.qconj() < 0 ? leg.group()->dual_sector(leg.sector(at)) : leg.sector(at);
    std::string entry;
    for (const int64_t component : sector) entry += std::to_string(component) + ",";
    entry += "x" + std::to_string(leg.multiplicity(at)) + ";";
    entries.push_back(entry);
  }
  // The blocks of a leg and of its dual are listed in opposite order, so the name is built from the
  // sorted entries: two legs that carry the same sectors with the same multiplicities agree.
  std::sort(entries.begin(), entries.end());
  std::string key;
  for (const std::string& entry : entries) key += entry;
  return key;
}

// The name of the leg that carries the dual space, which is what a glue leg meets in a contraction:
// the two sides of a glue carry dual spaces, and `leg_name` alone finds the slot only when every
// sector is self-dual.
std::string dual_name(const Leg& leg) {
  std::vector<std::string> entries;
  for (int64_t at = 0; at < leg.nblocks(); ++at) {
    const Sector sector =
        leg.qconj() > 0 ? leg.group()->dual_sector(leg.sector(at)) : leg.sector(at);
    std::string entry;
    for (const int64_t component : sector) entry += std::to_string(component) + ",";
    entry += "x" + std::to_string(leg.multiplicity(at)) + ";";
    entries.push_back(entry);
  }
  std::sort(entries.begin(), entries.end());
  std::string key;
  for (const std::string& entry : entries) key += entry;
  return key;
}

// The glue is the last `left_glue` legs of the left operand's domain and the first `right_glue` legs of
// the right operand's codomain: the shape of a composition. Each operand is repartitioned so that the
// glue is the whole of that side, which leaves every block a matrix over the glue alone, and one
// matrix product per charge pairs them; the tree bases of the two glue sides coincide, so no weight is
// needed.
TensorMap contract(const TensorMap& left, int64_t left_glue, const TensorMap& right, int64_t right_glue,
                   const c10::Device& device) {
  if (left_glue < 0 || left_glue != right_glue) {
    throw std::invalid_argument("contract: the two operands glue a different number of legs");
  }
  if (left_glue > left.space.size() || right_glue > right.space.size()) {
    throw std::invalid_argument("contract: the glue is longer than the tensor it cuts");
  }
  const TensorMap one = repartition_tensor(left, left.space.size() - left_glue, device);
  TensorMap two = repartition_tensor(right, right_glue, device);
  // The two glue sides are the same legs seen from opposite sides, and a block multiplies row by row,
  // so the right operand is permuted within its codomain until its glue follows the order of the left.
  if (one.space.domain.size() != two.space.codomain.size()) {
    throw std::invalid_argument("contract: the two operands do not glue the same legs");
  }
  bool dual_glue = false;
  if (!one.space.domain.empty()) {
    // Two glue legs may look alike, so every name remembers the slots it fills and each right leg takes
    // the next free one. A leg meets the dual of its partner, so the fallback looks the dual space up.
    std::map<std::string, std::vector<int64_t>> slots;
    std::map<std::string, std::vector<int64_t>> dual_slots;
    for (int64_t at = 0; at < static_cast<int64_t>(one.space.domain.size()); ++at) {
      slots[leg_name(one.space.domain.at(at))].push_back(at);
      dual_slots[dual_name(one.space.domain.at(at))].push_back(at);
    }
    std::vector<bool> taken(one.space.domain.size(), false);
    const auto take = [&](const std::map<std::string, std::vector<int64_t>>& by_name,
                          const std::string& name) -> int64_t {
      const auto found = by_name.find(name);
      if (found == by_name.end()) return -1;
      for (const int64_t slot : found->second)
        if (!taken.at(slot)) return slot;
      return -1;
    };
    std::vector<int64_t> back(two.space.codomain.size(), -1);
    for (int64_t at = 0; at < static_cast<int64_t>(two.space.codomain.size()); ++at) {
      const Leg& leg = two.space.codomain.at(at);
      int64_t slot = take(slots, leg_name(leg));
      if (slot < 0) {
        slot = take(dual_slots, leg_name(leg));
        dual_glue = true;
      }
      if (slot < 0) {
        throw std::invalid_argument("contract: the two operands glue different legs");
      }
      taken.at(slot) = true;
      back.at(slot) = at;
    }
    std::vector<int64_t> order;
    for (const int64_t at : back) order.push_back(at + 1);
    for (int64_t at = 0; at < static_cast<int64_t>(two.space.domain.size()); ++at) {
      order.push_back(static_cast<int64_t>(two.space.codomain.size()) + at + 1);
    }
    two = permute_tensor(two, static_cast<int64_t>(two.space.codomain.size()), order, device);
  }

  TensorMap out;
  out.space.group = left.space.group;
  out.qtotal = left.qtotal;
  out.space.codomain = one.space.codomain;
  out.space.domain = two.space.domain;
  for (const auto& [charge, block] : one.blocks) {
    auto other = two.blocks.find(charge);
    if (other == two.blocks.end() && dual_glue) other = two.blocks.find(left.space.group->dual_sector(charge));
    if (other == two.blocks.end()) continue;
    out.blocks.emplace(charge, at::matmul(block, other->second));
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
  m.def("engine_repartition_tensor", [](const std::shared_ptr<FusionCategory> group,
                                        const py::dict& tensor, int64_t keep,
                                        const py::object& device) {
    return tensor_as(repartition_tensor(tensor_of(group, tensor), keep, device_of(device)));
  }, py::arg("group"), py::arg("tensor"), py::arg("keep"), py::arg("device") = py::none());
  m.def("engine_contract", [](const std::shared_ptr<FusionCategory> group, const py::dict& left,
                              int64_t left_glue, const py::dict& right, int64_t right_glue,
                              const py::object& device) {
    return tensor_as(contract(tensor_of(group, left), left_glue, tensor_of(group, right), right_glue,
                              device_of(device)));
  }, py::arg("group"), py::arg("left"), py::arg("left_glue"), py::arg("right"),
     py::arg("right_glue"), py::arg("device") = py::none());
}

}  // namespace sym::engine

#include <stdexcept>


namespace sym::engine {
namespace {

// The sum of the squared moduli of every entry of a block.
double squared_norm(const at::Tensor& block) {
  const double size = at::norm(block).item<double>();
  return size * size;
}

}  // namespace

TensorMap adjoint(const TensorMap& tensor) {
  TensorMap out;
  out.space.group = tensor.space.group;
  out.space.codomain = tensor.space.domain;
  out.space.domain = tensor.space.codomain;
  out.qtotal = tensor.space.group->dual_sector(tensor.qtotal);
  for (Leg& leg : out.space.codomain) leg = leg.conj();
  for (Leg& leg : out.space.domain) leg = leg.conj();
  // Turning a tensor around keeps the charge of its legs, so every block keeps the sector it is
  // named by and only its rows and columns trade places.
  for (const auto& [root, block] : tensor.blocks) {
    out.blocks.emplace(root, at::conj(block).transpose(0, 1).contiguous());
  }
  // The legs of the result stand in the reverse order of the legs of the argument, as in the dense
  // transpose, so the first leg of the bra is the dual of the last leg of the ket. Each side is
  // reversed inside itself, which keeps every leg on the side the adjoint already gave it.
  const int64_t total = static_cast<int64_t>(out.space.size());
  if (total == 0) return out;
  const int64_t codomain = static_cast<int64_t>(out.space.codomain.size());
  std::vector<int64_t> order(static_cast<std::size_t>(total), 0);
  for (int64_t at = 0; at < total; ++at) {
    order.at(at) = at < codomain ? codomain - at : total - (at - codomain);
  }
  const c10::Device device = out.blocks.empty() ? at::TensorOptions().device()
                                               : out.blocks.begin()->second.device();
  return arrange_tensor(out, codomain, order, device);
}

double norm(const TensorMap& tensor) {
  double total = 0.0;
  for (const auto& [charge, block] : tensor.blocks) total += squared_norm(block);
  return std::sqrt(total);
}

std::complex<double> trace(const TensorMap& tensor) {
  const int64_t legs = static_cast<int64_t>(tensor.space.size());
  const FusionCategory& group = *tensor.space.group;
  std::complex<double> total = 0.0;
  if (legs == 2 && tensor.space.codomain.empty()) {
    // Closing a diagram pairs each leg with the conjugate of the other.
    if (!(tensor.space.domain.at(0) == tensor.space.domain.at(1)) &&
        !(tensor.space.domain.at(0) == tensor.space.domain.at(1).conj())) {
      throw std::invalid_argument("trace: the two closed legs have to be conjugate spaces");
    }
    const auto listed = side_trees(group, tensor.space.domain);
    for (const auto& [charge, block] : tensor.blocks) {
      const auto found = listed.find(charge);
      if (found == listed.end()) continue;
      const std::vector<FusionTree>& trees = found->second;
      if (trees.empty()) continue;
      const SideLayout layout = side_layout(tensor.space.domain, trees);
      const at::Tensor values = block.to(at::kCPU, at::kComplexDouble).contiguous();
      const c10::complex<double>* data = values.data_ptr<c10::complex<double>>();
      for (int64_t row = 0; row < layout.rows(); ++row) {
        const auto [tree, mults] = row_parts(layout, row);
        if (mults.size() != 2 || mults.at(0) != mults.at(1)) continue;
        const c10::complex<double> entry = data[row];
        total += std::complex<double>(entry.real(), entry.imag());
      }
    }
    return total;
  }
  const std::vector<int64_t> identity = [](int64_t size) {
    std::vector<int64_t> out(size);
    for (int64_t at = 0; at < size; ++at) out.at(at) = at + 1;
    return out;
  }(legs);
  const c10::Device device =
      tensor.blocks.empty() ? at::TensorOptions().device() : tensor.blocks.begin()->second.device();
  for (const auto& [charge, block] :
       permute_tensor(tensor, legs / 2, identity, device).blocks) {
    if (block.dim() != 2 || block.size(0) != block.size(1)) {
      throw std::invalid_argument("trace: a traced tensor has to be a square map");
    }
    const c10::complex<double> entry =
        at::trace(block).to(at::kCPU, at::kComplexDouble).item<c10::complex<double>>();
    total += std::complex<double>(entry.real(), entry.imag());
  }
  return total;
}

namespace {

// The same object written with another split of its sides: the legs agree in order, so the tensor is
// arranged onto the split of `into` before the entries are combined.
TensorMap aligned(const TensorMap& tensor, const TensorMap& into, const c10::Device& device) {
  const int64_t keep = static_cast<int64_t>(into.space.codomain.size());
  if (static_cast<int64_t>(tensor.space.codomain.size()) == keep) return tensor;
  std::vector<int64_t> order;
  for (int64_t at = 0; at < static_cast<int64_t>(tensor.space.size()); ++at) order.push_back(at + 1);
  return arrange_tensor(tensor, keep, order, device);
}

}  // namespace

std::complex<double> inner(const TensorMap& left, const TensorMap& right) {
  if (!(left.space.codomain == right.space.codomain && left.space.domain == right.space.domain)) {
    // Which side a leg sits on is a choice of basis for the same object, so the right operand is
    // brought onto the split of the left one before the entries are compared.
    if (left.space.size() != right.space.size()) {
      throw std::invalid_argument("inner: the two tensors do not cover the same space");
    }
    const c10::Device device = left.blocks.empty() ? at::TensorOptions().device()
                                                   : left.blocks.begin()->second.device();
    const TensorMap moved = aligned(right, left, device);
    if (!(left.space.codomain == moved.space.codomain && left.space.domain == moved.space.domain)) {
      throw std::invalid_argument("inner: the two tensors do not cover the same space");
    }
    return inner(left, moved);
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

// The domain root that goes with a codomain root at a given total charge: the block of a tensor is
// fixed by the root of its codomain side, and the domain root is the sector the two sides fuse
// through, which the total charge then pins down.
Sector partner_root(const FusionCategory& group, const Sector& codomain_root, const Sector& qtotal) {
  std::vector<Sector> valid;
  for (const Sector& candidate : group.fusion_outcomes(group.dual_sector(codomain_root), qtotal)) {
    const std::vector<Sector> fused = group.fusion_outcomes(codomain_root, candidate);
    if (std::find(fused.begin(), fused.end(), qtotal) != fused.end()) valid.push_back(candidate);
  }
  if (valid.size() != 1) {
    throw std::invalid_argument(
        "a block is fixed by the root of its codomain, and more than one domain root fuses with it "
        "to the total charge of this tensor");
  }
  return valid.front();
}

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
  const auto out_trees = side_trees(*group, codomain);
  const auto in_trees = side_trees(*group, domain);
  const std::vector<Sector> out_roots = roots_of(*group, out_trees);
  const std::vector<Sector> in_roots = roots_of(*group, in_trees);
  const auto options = at::TensorOptions().dtype(dtype).device(device);

  // A block is fixed by the root of the codomain side; the domain root is the sector the two sides
  // fuse through at the total charge.
  for (const Sector& root : out_roots) {
    // A side without legs cannot say which root the other side pairs with, so the total charge fixes
    // it: an empty domain makes the codomain root the charge itself, an empty codomain carries it on
    // the domain side.
    Sector partner;
    if (domain.empty()) {
      if (root != qtotal) continue;
      partner = group->trivial_sector();
    } else if (codomain.empty()) {
      if (root != group->trivial_sector()) continue;
      partner = qtotal;
    } else {
      partner = partner_root(*group, root, qtotal);
    }
    if (std::find(in_roots.begin(), in_roots.end(), partner) == in_roots.end()) continue;
    const KeyLayout rows = key_layout(codomain, out_trees, root);
    const KeyLayout columns = key_layout(domain, in_trees, partner);
    if (rows.rows == 0 || columns.rows == 0) continue;
    const Sector key = codomain.empty() ? partner : root;
    at::Tensor block;
    if (kind == 0) block = at::zeros({rows.rows, columns.rows}, options);
    else if (kind == 1) block = at::ones({rows.rows, columns.rows}, options);
    else if (device.is_cpu()) block = at::rand({rows.rows, columns.rows}, options);
    else block = at::rand({rows.rows, columns.rows}, options.device(at::kCPU)).to(device);
    out.blocks.emplace(key, std::move(block));
  }
  return Tensor(std::move(out));
}

std::pair<int64_t, int64_t> Tensor::slot(const Sector& charge, const std::vector<int64_t>& blocks,
                                         const std::vector<int64_t>& members) const {
  const TensorMap& tensor = map();
  const std::vector<Leg>& out_legs = tensor.space.codomain;
  const std::vector<Leg>& in_legs = tensor.space.domain;
  const int64_t total = static_cast<int64_t>(out_legs.size() + in_legs.size());
  if (static_cast<int64_t>(blocks.size()) != total || static_cast<int64_t>(members.size()) != total) {
    throw std::invalid_argument("an entry names one block and one member for every leg");
  }
  if (tensor.blocks.find(charge) == tensor.blocks.end()) {
    throw std::invalid_argument("the tensor carries no block at that charge");
  }
  const FusionCategory& group = *tensor.space.group;
  // The tree of one side is fixed by the sectors the chosen blocks carry and by the charge the
  // side couples to, and the row follows from the member inside every block.
  const auto side_row = [&group](const std::vector<Leg>& legs, const int64_t offset,
                                 const Sector& coupled, const std::vector<int64_t>& blocks,
                                 const std::vector<int64_t>& members) {
    std::vector<Sector> sectors;
    std::vector<bool> isdual;
    std::vector<int64_t> mults;
    std::vector<std::vector<LegCharge>> choices;
    for (const Leg& leg : legs) choices.push_back(leg_charges(leg));
    for (std::size_t at = 0; at < legs.size(); ++at) {
      const Leg& leg = legs.at(at);
      const int64_t index = blocks.at(offset + static_cast<int64_t>(at));
      if (index < 0 || index >= leg.nblocks()) {
        throw std::invalid_argument("an entry names a block its leg does not carry");
      }
      sectors.push_back(leg.sector(index));
      isdual.push_back(leg.qconj() < 0);
      mults.push_back(members.at(offset + static_cast<int64_t>(at)));
    }
    if (legs.empty()) return int64_t{0};
    const std::map<Sector, std::vector<FusionTree>> trees = trees_by_charge(group, choices);
    const auto where = trees.find(coupled);
    if (where == trees.end()) {
      throw std::invalid_argument("that charge is no coupling of the side");
    }
    const SideLayout layout = side_layout(legs, where->second);
    const FusionTree wanted{sectors, isdual, {}, std::vector<int64_t>(sectors.size() - 1, 0), coupled};
    const auto index = index_of(layout.trees).find(wanted);
    if (index == index_of(layout.trees).end()) {
      throw std::invalid_argument("those blocks span no tree of that charge");
    }
    return row_index(layout, index->second, mults);
  };
  const Sector partner = out_legs.empty() || in_legs.empty()
                             ? tensor.qtotal
                             : partner_root(group, charge, tensor.qtotal);
  return {side_row(out_legs, 0, charge, blocks, members),
          side_row(in_legs, static_cast<int64_t>(out_legs.size()), partner, blocks, members)};
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
  if (!map_) return Tensor(at::conj(plain_).transpose(-1, -2));
  return Tensor(engine::adjoint(map()));
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

Tensor Tensor::absorb(const at::Tensor& values) const {
  if (!map_) throw std::logic_error("absorb: a plain tensor carries no bond leg to scale");
  TensorMap out = map();
  if (out.space.codomain.size() != 1) {
    throw std::invalid_argument(
        "absorb: the tensor needs exactly one codomain leg, the bond a split made");
  }
  const Leg& bond = out.space.codomain.front();
  std::vector<int64_t> offsets(static_cast<std::size_t>(bond.nblocks()) + 1, 0);
  for (int64_t qi = 0; qi < bond.nblocks(); ++qi) {
    offsets.at(qi + 1) = offsets.at(qi) + bond.multiplicity(qi);
  }
  if (values.numel() != offsets.back()) {
    throw std::invalid_argument("absorb: the values hold " + std::to_string(values.numel()) +
                                " entries, not one per copy of the bond leg (" +
                                std::to_string(offsets.back()) + ")");
  }
  // A block is keyed by the charge of the bond leg it came from, so its rows are the copies of
  // that block of the leg and take that block's values.
  const FusionCategory& group = *out.space.group;
  for (auto& [charge, block] : out.blocks) {
    int64_t qi = 0;
    while (qi < bond.nblocks() &&
           !((bond.qconj() < 0 ? group.dual_sector(bond.sector(qi)) : bond.sector(qi)) == charge)) {
      ++qi;
    }
    if (qi == bond.nblocks()) {
      throw std::invalid_argument("absorb: a block carries a charge the bond leg does not");
    }
    const at::Tensor weights = values.narrow(0, offsets.at(qi), block.size(0));
    block = block * weights.to(block.scalar_type()).view({-1, 1});
  }
  return Tensor(std::move(out));
}

Tensor& Tensor::operator+=(const Tensor& other) {
  if (!map_ || !other.map_) throw std::logic_error("a bare matrix does not add to a graded tensor");
  for (const auto& [charge, block] : aligned(other.map(), *map_, other.device()).blocks) {
    const auto found = map_->blocks.find(charge);
    if (found == map_->blocks.end()) map_->blocks.emplace(charge, block.clone());
    else found->second = found->second + block;
  }
  return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
  if (!map_ || !other.map_) throw std::logic_error("a bare matrix does not subtract a graded tensor");
  for (const auto& [charge, block] : aligned(other.map(), *map_, other.device()).blocks) {
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

// Rearranges the legs of a tensor into a wanted sequence and a wanted split between its sides, so
// that the component array is only permuted: a leg that crosses changes its direction, and both
// sides are re-expressed in each other's tree basis.
// Args:
//   tensor: a tensor map.
//   keep: the number of legs the codomain carries afterwards.
//   order: the legs in their wanted order, as one-based positions in `tensor`.
//   device: where the associator is read from.
// Returns:
//   The same map over rearranged legs.
// Raises:
//   std::invalid_argument: if `keep` or `order` does not describe an arrangement of the legs.
TensorMap arrange_tensor(const TensorMap& tensor, int64_t keep, const std::vector<int64_t>& order,
                         const c10::Device& device) {
  const int64_t total = tensor.space.size();
  if (keep < 0 || keep > total) {
    throw std::invalid_argument("arrange_tensor: a tensor of " + std::to_string(total) +
                                " legs cannot carry " + std::to_string(keep) + " of them");
  }
  if (static_cast<int64_t>(order.size()) != total) {
    throw std::invalid_argument("arrange_tensor: the order names " + std::to_string(order.size()) +
                                " legs of a tensor with " + std::to_string(total));
  }
  std::vector<bool> seen(static_cast<std::size_t>(total), false);
  std::vector<bool> on_codomain(static_cast<std::size_t>(total), false);
  for (int64_t at = 0; at < total; ++at) {
    const int64_t leg = order.at(at) - 1;
    if (leg < 0 || leg >= total || seen.at(leg)) {
      throw std::invalid_argument("arrange_tensor: the order is not a permutation of the legs");
    }
    seen.at(leg) = true;
    on_codomain.at(leg) = at < keep;
  }
  if (total == 0) return tensor;

  // The legs in the order they stand in now, as positions in `tensor`.
  std::vector<int64_t> now(static_cast<std::size_t>(total));
  for (int64_t at = 0; at < total; ++at) now.at(at) = at;
  const int64_t codomain = static_cast<int64_t>(tensor.space.codomain.size());
  TensorMap out = tensor;
  // Hand the codomain legs that have to leave over, standing last in the codomain.
  std::vector<int64_t> stay;
  std::vector<int64_t> leave;
  for (int64_t at = 0; at < codomain; ++at) {
    (on_codomain.at(now.at(at)) ? stay : leave).push_back(at);
  }
  if (!leave.empty()) {
    std::vector<int64_t> within;
    for (const int64_t at : stay) within.push_back(at + 1);
    for (const int64_t at : leave) within.push_back(at + 1);
    for (int64_t at = codomain; at < total; ++at) within.push_back(at + 1);
    out = repartition_tensor(permute_tensor(out, codomain, within, device),
                             codomain - static_cast<int64_t>(leave.size()), device);
    // A repartition appends the leaving block to the end of the domain.
    std::vector<int64_t> next;
    for (const int64_t at : stay) next.push_back(now.at(at));
    for (int64_t at = codomain; at < total; ++at) next.push_back(now.at(at));
    for (const int64_t at : leave) next.push_back(now.at(at));
    now = next;
  }
  // Hand the domain legs that have to join over, standing last in the domain.
  const int64_t kept = codomain - static_cast<int64_t>(leave.size());
  std::vector<int64_t> enter;
  std::vector<int64_t> rest;
  for (int64_t at = kept; at < total; ++at) {
    (on_codomain.at(now.at(at)) ? enter : rest).push_back(at);
  }
  if (!enter.empty()) {
    std::vector<int64_t> within;
    for (int64_t at = 0; at < kept; ++at) within.push_back(at + 1);
    for (const int64_t at : rest) within.push_back(at + 1);
    for (const int64_t at : enter) within.push_back(at + 1);
    out = repartition_tensor(permute_tensor(out, kept, within, device),
                             kept + static_cast<int64_t>(enter.size()), device);
    // A repartition appends the joining block to the end of the codomain.
    std::vector<int64_t> next;
    for (int64_t at = 0; at < kept; ++at) next.push_back(now.at(at));
    for (const int64_t at : enter) next.push_back(now.at(at));
    for (const int64_t at : rest) next.push_back(now.at(at));
    now = next;
  }
  // Both sides carry the legs they should; only the order within them is left to fix.
  std::vector<int64_t> position(static_cast<std::size_t>(total), 0);
  for (int64_t at = 0; at < total; ++at) position.at(now.at(at)) = at + 1;
  std::vector<int64_t> wanted;
  for (const int64_t leg : order) wanted.push_back(position.at(leg - 1));
  return permute_tensor(out, keep, wanted, device);
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
  // The operands carry the place the contraction runs on; an empty block set cannot tell, and then
  // the default place is the only answer.
  const auto device = operands.front().device();
  std::vector<std::string> pending = groups;
  std::vector<Tensor> waiting = operands;
  Tensor accumulated = waiting.front();
  std::string labels = pending.front();
  waiting.erase(waiting.begin());
  pending.erase(pending.begin());
  const auto order_of = [](const std::string& from, const std::string& letters) {
    std::vector<int64_t> order;
    for (const char letter : letters) order.push_back(static_cast<int64_t>(from.find(letter)) + 1);
    return order;
  };
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
    std::string left_kept;
    for (const char letter : labels)
      if (shared.find(letter) == std::string::npos) left_kept.push_back(letter);
    std::string right_kept;
    for (const char letter : next)
      if (shared.find(letter) == std::string::npos) right_kept.push_back(letter);
    // The contraction glues the end of the left domain to the start of the right codomain, so each
    // operand is arranged with its glue on that side and its kept legs on the other.
    Tensor left = Tensor(arrange_tensor(accumulated.map(), static_cast<int64_t>(left_kept.size()),
                                        order_of(labels, left_kept + shared), device));
    Tensor right = Tensor(arrange_tensor(waiting.at(chosen).map(), static_cast<int64_t>(shared.size()),
                                         order_of(next, shared + right_kept), device));
    accumulated = Tensor(engine::contract(left.map(), static_cast<int64_t>(shared.size()),
                                          right.map(), static_cast<int64_t>(shared.size()), device));
    labels = left_kept + right_kept;
    waiting.erase(waiting.begin() + static_cast<int64_t>(chosen));
    pending.erase(pending.begin() + static_cast<int64_t>(chosen));
  }
  // The output keeps the order the caller wrote. A label that entered on a codomain stays there
  // while it is still at the front of that order; the first label that entered on a domain ends
  // the run, and keep_out is the length of the run.
  if (wanted.empty()) return accumulated;
  std::vector<int64_t> order;
  int64_t keep_out = 0;
  bool leading = true;
  for (const char letter : wanted) {
    const std::size_t where = labels.find(letter);
    if (where == std::string::npos) {
      throw std::invalid_argument("einsum: the output names a label no operand carries");
    }
    order.push_back(static_cast<int64_t>(where) + 1);
    if (leading && side.at(letter) == 0) {
      ++keep_out;
    } else {
      leading = false;
    }
  }
  if (order.size() != labels.size()) {
    throw std::invalid_argument("einsum: the output drops a label that no other operand takes");
  }
  return Tensor(arrange_tensor(accumulated.map(), keep_out, order, device));
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
  // An empty list of legs is a tensor without a symmetry, the same as naming no legs at all.
  if (!legs.is_none() && py::len(legs) == 0) legs = py::none();
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
      .def("map", [](const Tensor& tensor) { return tensor_as(tensor.map()); })
      .def(py::init([](const std::vector<Sector>& charges, const std::vector<at::Tensor>& blocks,
                       const std::vector<Leg>& legs, const py::object& qtotal, int64_t codomain) {
        if (legs.empty()) throw std::invalid_argument("a tensor of charges needs legs");
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
      .def(py::init([](const py::dict& blocks, const py::object& named,
                       const py::object& qtotal, int64_t codomain) {
        const std::vector<Leg> legs =
            named.is_none() ? std::vector<Leg>{} : named.cast<std::vector<Leg>>();
        if (legs.empty()) {
          // with no legs the one block of the trivial charge is the whole tensor, a bare matrix
          if (blocks.size() != 1) throw std::invalid_argument("a tensor without legs holds one block");
          return Tensor(blocks.begin()->second.cast<at::Tensor>());
        }
        TensorMap map;
        map.space.group = legs.front().group();
        if (codomain < 0) codomain = static_cast<int64_t>(legs.size());
        map.space.codomain.assign(legs.begin(), legs.begin() + codomain);
        map.space.domain.assign(legs.begin() + codomain, legs.end());
        map.qtotal = qtotal.is_none() ? map.space.group->trivial_sector() : qtotal.cast<Sector>();
        for (const auto& item : blocks) {
          map.blocks.emplace(item.first.cast<Sector>(), item.second.cast<at::Tensor>());
        }
        return Tensor(std::move(map));
      }),
           py::arg("blocks"), py::arg("legs") = py::none(), py::arg("qtotal") = py::none(),
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
      .def_property_readonly("blocks", [](const Tensor& tensor) {
        // The blocks as a mapping from the charge they couple to.  The torch tensors are the ones
        // the engine holds, so writing an entry writes into the tensor.
        py::dict out;
        if (!tensor.graded()) {
          out[py::tuple()] = tensor.plain();
          return out;
        }
        for (const auto& [charge, block] : tensor.map().blocks) {
          out[py::tuple(py::cast(charge))] = block;
        }
        return out;
      })
      .def("to_dict", [](const Tensor& tensor) {
        // The whole tensor as plain Python: its legs, the total charge, where the codomain ends and
        // one torch tensor per coupled charge.
        py::dict out;
        if (!tensor.graded()) {
          out["legs"] = py::list();
          out["qtotal"] = py::tuple();
          out["codomain"] = 0;
          py::dict plain;
          plain[py::tuple()] = tensor.plain();
          out["blocks"] = plain;
          return out;
        }
        out["legs"] = py::cast(tensor.legs());
        out["qtotal"] = py::tuple(py::cast(tensor.qtotal()));
        out["codomain"] = static_cast<int64_t>(tensor.map().space.codomain.size());
        py::dict blocks;
        for (const auto& [charge, block] : tensor.map().blocks) {
          blocks[py::tuple(py::cast(charge))] = block;
        }
        out["blocks"] = blocks;
        return out;
      })
      .def("set_block", [](Tensor& tensor, const py::tuple& charge, const at::Tensor& block) {
        // One block, keyed by the charge the two sides couple to.  The trees of that charge fix the
        // shape a block of it has, so a block of another shape has no place in the tensor.
        TensorMap& map = tensor.map();
        const Sector key = charge.cast<Sector>();
        const auto codomain = trees_by_charge(*map.space.group, map.codomain_charges());
        const auto domain = trees_by_charge(*map.space.group, map.domain_charges());
        const auto out = codomain.find(key);
        const auto in = domain.find(key);
        const int64_t rows =
            out == codomain.end() ? 0 : side_layout(map.space.codomain, out->second).rows();
        const int64_t columns =
            in == domain.end() ? 0 : side_layout(map.space.domain, in->second).rows();
        if (rows == 0 || columns == 0) {
          throw std::invalid_argument("set_block: this charge is no coupling of the two sides");
        }
        if (block.dim() != 2 || block.size(0) != rows || block.size(1) != columns) {
          throw std::invalid_argument("set_block: the block of this charge is " +
                                      std::to_string(rows) + "x" + std::to_string(columns) +
                                      ", so a block of " + std::to_string(block.numel()) +
                                      " entries is not the one it wants");
        }
        map.blocks[key] = block;
      }, py::arg("charge"), py::arg("block"))
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
      .def("slot", [](const Tensor& tensor, const py::tuple& charge, const std::vector<int64_t>& blocks,
                      const std::vector<int64_t>& members) {
        return tensor.slot(py::cast<Sector>(charge), blocks, members);
      })
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
