// The splittings of the engine: one fused charge at a time, the tensor is cut into a pair
// of factors that share a new leg, exactly as mpsdmrg/symmetry/linalg.py does it.
#include "linalg.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace sym {
namespace {

// The running start of every block dimension: the slice of each state of a charge.
Shape running(const std::vector<int64_t>& dims) {
  Shape out{0};
  for (int64_t dim : dims) out.push_back(out.back() + dim);
  return out;
}

// The Python spelling of a tuple of integers.
std::string int_tuple(const std::vector<int64_t>& values) {
  if (values.empty()) return "()";
  std::string out = "(";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out += ", ";
    out += std::to_string(values[i]);
  }
  return out + (values.size() == 1 ? ",)" : ")");
}

// The Python spelling of a tuple of legs.
std::string legs_repr(const std::vector<Leg>& legs) {
  if (legs.empty()) return "()";
  std::string out = "(";
  for (std::size_t i = 0; i < legs.size(); ++i) {
    if (i != 0) out += ", ";
    out += legs[i].repr();
  }
  return out + (legs.size() == 1 ? ",)" : ")");
}

// The charge a split of the first ``left`` legs fuses to.
//
// Raises:
//   NotImplementedError: If ``left`` is not a prefix of the legs, which a non-abelian group
//     cannot fuse a partial tree for.
Sector split_charge(const std::shared_ptr<FusionCategory>& group, const std::vector<Leg>& legs,
                    const BlockKey& key, const Sector& qtotal,
                    const std::vector<int64_t>& left) {
  if (left.empty() || left.size() == legs.size()) return qtotal;
  bool prefix = true;
  for (std::size_t i = 0; i < left.size() && prefix; ++i) {
    prefix = left[i] == static_cast<int64_t>(i);
  }
  if (!prefix) {
    throw not_implemented_error("A " + group->name() +
                                " split needs the first legs in order, not " + int_tuple(left) +
                                ".");
  }
  return FusionTree::from_blocks(group, legs, key, qtotal).chain().at(left.size() - 1);
}

// The weight the isometry factor of a split at ``charge`` carries: the two factors hold one
// fusion vertex fewer than the tensor they rebuild, so the isometry is weighted by the
// square root of the quantum dimension of that vertex.
double split_scale(const std::shared_ptr<FusionCategory>& group, const Sector& charge) {
  return std::sqrt(group->quantum_dim(charge));
}

// The blocks of the two factors of one fused charge, one block per distinct state.
struct FactorBlocks {
  std::vector<at::Tensor> first_blocks;
  std::vector<at::Tensor> second_blocks;
  std::vector<BlockKey> first_qdata;
  std::vector<BlockKey> second_qdata;
};

// One block of each factor per distinct state: the rows of ``first`` and the columns of
// ``second`` are the sector's left and right index.
//
// Args:
//   kept: the columns of ``first`` and rows of ``second`` to keep, all when null.
FactorBlocks factor_blocks(const Tensor& t, const std::vector<int64_t>& left,
                           const std::vector<int64_t>& right, const SectorMatrix& part,
                           const at::Tensor& first, const at::Tensor& second,
                           const at::Tensor* kept, int64_t rank) {
  const Shape left_slices = running(part.left_dims), right_slices = running(part.right_dims);
  const at::Tensor first_kept = kept != nullptr ? first.index_select(1, *kept) : first;
  const at::Tensor second_kept = kept != nullptr ? second.index_select(0, *kept) : second;
  const int64_t k = first_kept.size(1);
  FactorBlocks out;
  for (std::size_t lc = 0; lc < part.left.size(); ++lc) {
    const BlockKey& combo = part.left[lc];
    const int64_t row = left_slices[lc], rdim = part.left_dims[lc];
    Shape dims;
    for (std::size_t i = 0; i < left.size(); ++i) {
      dims.push_back(t.legs()[left[i]].multiplicity(combo.legs[i]));
    }
    dims.push_back(k);
    out.first_blocks.push_back(first_kept.slice(0, row, row + rdim).reshape(dims));
    BlockKey key;
    for (std::size_t i = 0; i < left.size(); ++i) key.legs.push_back(combo.legs[i]);
    key.legs.push_back(rank);
    key.inner = combo.inner;
    key.copies = combo.copies;
    out.first_qdata.push_back(key);
  }
  for (std::size_t rc = 0; rc < part.right.size(); ++rc) {
    const BlockKey& combo = part.right[rc];
    const int64_t col = right_slices[rc], cdim = part.right_dims[rc];
    Shape dims{k};
    for (std::size_t i = 0; i < right.size(); ++i) {
      dims.push_back(t.legs()[right[i]].multiplicity(combo.legs[i]));
    }
    out.second_blocks.push_back(second_kept.slice(1, col, col + cdim).reshape(dims));
    BlockKey key;
    key.legs.push_back(rank);
    for (int64_t leg : combo.legs) key.legs.push_back(leg);
    // The conjugated new leg carries its charge un-dualled, so the first coupling of the
    // right factor is the one that follows the left legs in the chain of the split tensor.
    key.inner = combo.inner;
    key.copies = combo.copies;
    out.second_qdata.push_back(key);
  }
  return out;
}

}  // namespace

void check_partition(const Tensor& t, const std::vector<int64_t>& left,
                     const std::vector<int64_t>& right) {
  Shape flat = left;
  flat.insert(flat.end(), right.begin(), right.end());
  std::sort(flat.begin(), flat.end());
  bool ok = flat.size() == t.legs().size();
  for (std::size_t i = 0; ok && i < flat.size(); ++i) ok = flat[i] == static_cast<int64_t>(i);
  if (!ok) {
    throw value_error("Legs " + int_tuple(left) + " and " + int_tuple(right) +
                      " do not partition a " + std::to_string(t.legs().size()) +
                      "-leg tensor.");
  }
}

std::vector<SectorMatrix> sector_matrices(const Tensor& t, const std::vector<int64_t>& left,
                                          const std::vector<int64_t>& right) {
  const std::shared_ptr<FusionCategory>& group = t.group();
  const std::vector<Leg>& legs = t.legs();
  const std::size_t cut = left.size() > 0 ? left.size() - 1 : 0;

  // Every block whose left legs fuse to the same charge forms one matrix.
  std::map<Sector, std::vector<int64_t>> by_charge;
  for (std::size_t bi = 0; bi < t.qdata().size(); ++bi) {
    const BlockKey& q = t.qdata()[bi];
    by_charge[split_charge(group, legs, q, t.qtotal(), left)].push_back(static_cast<int64_t>(bi));
  }

  std::vector<SectorMatrix> out;
  for (const auto& [charge, bis] : by_charge) {
    std::map<BlockKey, int64_t> left_at, right_at;
    std::vector<BlockKey> left_states, right_states;
    std::map<int64_t, std::pair<BlockKey, BlockKey>> states;
    for (int64_t bi : bis) {
      const BlockKey& q = t.qdata()[bi];
      BlockKey left_state, right_state;
      for (int64_t l : left) left_state.legs.push_back(q.legs[l]);
      for (int64_t r : right) right_state.legs.push_back(q.legs[r]);
      for (std::size_t i = 0; i < cut && i < q.inner.size(); ++i) {
        left_state.inner.push_back(q.inner[i]);
      }
      for (std::size_t i = cut; i < q.inner.size(); ++i) right_state.inner.push_back(q.inner[i]);
      // A group with multiplicity labels a state by the fusion copy of every vertex; the new
      // vertex of the first factor couples the cut charge with its own dual, so it is unique.
      for (std::size_t i = 0; i < cut && i < q.copies.size(); ++i) {
        left_state.copies.push_back(q.copies[i]);
      }
      if (!q.copies.empty()) left_state.copies.push_back(0);
      for (std::size_t i = cut; i < q.copies.size(); ++i) {
        right_state.copies.push_back(q.copies[i]);
      }
      states[bi] = {left_state, right_state};
      if (left_at.find(left_state) == left_at.end()) {
        left_at[left_state] = static_cast<int64_t>(left_states.size());
        left_states.push_back(left_state);
      }
      if (right_at.find(right_state) == right_at.end()) {
        right_at[right_state] = static_cast<int64_t>(right_states.size());
        right_states.push_back(right_state);
      }
    }
    std::vector<int64_t> left_dims, right_dims;
    for (const BlockKey& state : left_states) {
      int64_t dim = 1;
      for (std::size_t i = 0; i < left.size(); ++i) {
        dim *= legs[left[i]].multiplicity(state.legs[i]);
      }
      left_dims.push_back(dim);
    }
    for (const BlockKey& state : right_states) {
      int64_t dim = 1;
      for (std::size_t i = 0; i < right.size(); ++i) {
        dim *= legs[right[i]].multiplicity(state.legs[i]);
      }
      right_dims.push_back(dim);
    }
    const Shape left_slices = running(left_dims), right_slices = running(right_dims);
    at::Tensor matrix = at::zeros({left_slices.back(), right_slices.back()},
                                  t.blocks().front().options());
    for (int64_t bi : bis) {
      const auto& state = states[bi];
      const int64_t lc = left_at[state.first], rc = right_at[state.second];
      const int64_t rdim = left_dims[lc], cdim = right_dims[rc];
      Shape order = left;
      order.insert(order.end(), right.begin(), right.end());
      const at::Tensor block = t.blocks()[bi].permute(order).reshape({rdim, cdim});
      matrix.slice(0, left_slices[lc], left_slices[lc] + rdim)
          .slice(1, right_slices[rc], right_slices[rc] + cdim)
          .copy_(block);
    }
    out.push_back(SectorMatrix{charge, matrix, left_states, right_states, left_dims, right_dims,
                               t.qdata()[bis.front()]});
  }
  return out;
}

std::pair<Tensor, Tensor> assemble_split(
    const Tensor& t, const std::vector<int64_t>& left, const std::vector<int64_t>& right,
    const std::vector<SectorMatrix>& parts,
    const std::vector<std::pair<at::Tensor, at::Tensor>>& factors,
    const std::vector<Sector>& charges, const std::optional<std::vector<at::Tensor>>& kept,
    const Sector& first_qtotal, const Sector& second_qtotal) {
  // The new leg holds one sector per charge that keeps at least one singular value, one entry
  // per kept component of the sector.
  std::vector<std::pair<Sector, int64_t>> entries;
  for (std::size_t i = 0; i < charges.size(); ++i) {
    const at::Tensor* k = kept ? &(*kept)[i] : nullptr;
    if (k != nullptr && k->size(0) == 0) continue;
    const int64_t multiplicity = k != nullptr ? k->size(0) : factors[i].first.size(1);
    entries.emplace_back(charges[i], multiplicity * t.group()->sector_dim(charges[i]));
  }
  const std::pair<Leg, std::vector<int64_t>> built = Leg::from_blocks(t.group(), entries, -1);
  const Leg new_leg = built.first;
  const std::vector<int64_t> &rank = built.second;
  std::map<Sector, int64_t> block_of_charge;
  for (std::size_t i = 0; i < entries.size(); ++i) block_of_charge[entries[i].first] = rank[i];

  std::vector<at::Tensor> first_blocks, second_blocks;
  std::vector<BlockKey> first_qdata, second_qdata;
  for (std::size_t i = 0; i < parts.size(); ++i) {
    const at::Tensor* k = kept ? &(*kept)[i] : nullptr;
    if (k != nullptr && k->size(0) == 0) continue;
    const FactorBlocks blocks = factor_blocks(t, left, right, parts[i], factors[i].first,
                                              factors[i].second, k, block_of_charge[charges[i]]);
    first_blocks.insert(first_blocks.end(), blocks.first_blocks.begin(),
                        blocks.first_blocks.end());
    first_qdata.insert(first_qdata.end(), blocks.first_qdata.begin(), blocks.first_qdata.end());
    second_blocks.insert(second_blocks.end(), blocks.second_blocks.begin(),
                         blocks.second_blocks.end());
    second_qdata.insert(second_qdata.end(), blocks.second_qdata.begin(),
                        blocks.second_qdata.end());
  }
  std::vector<Leg> first_legs;
  for (int64_t l : left) first_legs.push_back(t.legs()[l]);
  first_legs.push_back(new_leg);
  std::vector<Leg> second_legs{new_leg.conj()};
  for (int64_t r : right) second_legs.push_back(t.legs()[r]);
  return {Tensor(first_blocks, first_qdata, first_legs, first_qtotal, t.group()),
          Tensor(second_blocks, second_qdata, second_legs, second_qtotal, t.group())};
}

std::tuple<Tensor, at::Tensor, Tensor> svd(const Tensor& t, const std::vector<int64_t>& left,
                                           const std::vector<int64_t>& right,
                                           std::optional<int64_t> max_bond, double tol,
                                           const std::string& direction) {
  if (direction != "left" && direction != "right") {
    throw value_error("A split is prepared for \"left\" or \"right\", not '" + direction + "'.");
  }
  check_partition(t, left, right);
  const std::vector<SectorMatrix> parts = sector_matrices(t, left, right);

  std::vector<std::tuple<at::Tensor, at::Tensor, at::Tensor>> factors;
  for (const SectorMatrix& part : parts) {
    auto [u, s, vh] = at::linalg_svd(part.matrix, /*full_matrices=*/false);
    factors.emplace_back(u, s, vh);
  }
  std::vector<at::Tensor> values;
  for (const auto& factor : factors) values.push_back(std::get<1>(factor));
  const at::Tensor flat = at::cat(values);
  at::Tensor selected = flat > tol;
  if (max_bond.has_value()) {
    const at::Tensor top =
        at::argsort(flat, /*dim=*/0, /*descending=*/true).slice(0, 0, *max_bond);
    at::Tensor allowed = at::zeros_like(selected);
    allowed.index_put_({top}, at::ones({}, selected.options()));
    selected = selected.logical_and(allowed);
  }
  std::vector<at::Tensor> kept;
  int64_t offset = 0;
  for (const auto& factor : factors) {
    const int64_t count = std::get<1>(factor).size(0);
    kept.push_back(selected.slice(0, offset, offset + count).nonzero().squeeze(-1));
    offset += count;
  }
  std::vector<std::pair<at::Tensor, at::Tensor>> weighted;
  std::vector<Sector> charges;
  for (std::size_t i = 0; i < parts.size(); ++i) {
    const double scale = split_scale(t.group(), parts[i].charge);
    const at::Tensor& u_mat = std::get<0>(factors[i]);
    const at::Tensor& vh_mat = std::get<2>(factors[i]);
    weighted.push_back(direction == "left" ? std::make_pair(u_mat, vh_mat * scale)
                                           : std::make_pair(u_mat * scale, vh_mat));
    charges.push_back(parts[i].charge);
  }
  auto [u, vh] = assemble_split(t, left, right, parts, weighted, charges, kept,
                                t.group()->trivial_sector(), t.qtotal());
  std::vector<at::Tensor> singular;
  for (std::size_t i = 0; i < factors.size(); ++i) {
    if (kept[i].size(0) > 0) singular.push_back(std::get<1>(factors[i]).index_select(0, kept[i]));
  }
  const at::Tensor singular_values =
      singular.empty() ? at::zeros({0}, t.blocks().front().options()) : at::cat(singular);
  return {u, singular_values, vh};
}

std::pair<Tensor, Tensor> qr(const Tensor& t, const std::vector<int64_t>& left,
                             const std::vector<int64_t>& right) {
  check_partition(t, left, right);
  const std::vector<SectorMatrix> parts = sector_matrices(t, left, right);
  std::vector<std::pair<at::Tensor, at::Tensor>> factors;
  std::vector<Sector> charges;
  for (const SectorMatrix& part : parts) {
    auto [q_mat, r_mat] = at::linalg_qr(part.matrix);
    factors.emplace_back(q_mat * split_scale(t.group(), part.charge), r_mat);
    charges.push_back(part.charge);
  }
  return assemble_split(t, left, right, parts, factors, charges, std::nullopt,
                        t.group()->trivial_sector(), t.qtotal());
}

std::pair<Tensor, Tensor> rq(const Tensor& t, const std::vector<int64_t>& left,
                             const std::vector<int64_t>& right) {
  check_partition(t, left, right);
  const std::vector<SectorMatrix> parts = sector_matrices(t, left, right);
  std::vector<std::pair<at::Tensor, at::Tensor>> factors;
  std::vector<Sector> charges;
  for (const SectorMatrix& part : parts) {
    auto [q_mat, r_mat] = at::linalg_qr(part.matrix.mH());
    factors.emplace_back(r_mat.mH(), q_mat.mH() * split_scale(t.group(), part.charge));
    charges.push_back(part.charge);
  }
  // The first factor couples its legs to the trivial sector, the second to the total charge.
  return assemble_split(t, left, right, parts, factors, charges, std::nullopt,
                        t.group()->trivial_sector(), t.qtotal());
}

at::Tensor svdvals(const Tensor& t, const std::vector<int64_t>& left,
                   const std::vector<int64_t>& right) {
  check_partition(t, left, right);
  const std::vector<SectorMatrix> parts = sector_matrices(t, left, right);
  std::vector<at::Tensor> values;
  for (const SectorMatrix& part : parts) values.push_back(at::linalg_svdvals(part.matrix));
  return at::cat(values);
}

at::Tensor norm(const Tensor& t, const std::optional<std::vector<int64_t>>& dim) {
  if (t.is_dense() || t.legs().empty()) {
    return dim.has_value()
               ? at::linalg_norm(t.blocks().front(), std::nullopt, at::IntArrayRef(*dim))
               : at::linalg_norm(t.blocks().front());
  }
  if (!dim.has_value()) {
    at::Tensor total = at::zeros({}, t.blocks().front().options().dtype(at::kDouble));
    for (const at::Tensor& block : t.blocks()) total = total + at::sum(block.abs().pow(2));
    return at::sqrt(total);
  }
  std::vector<int64_t> keep;
  for (int64_t i = 0; i < static_cast<int64_t>(t.legs().size()); ++i) {
    if (std::find(dim->begin(), dim->end(), i) == dim->end()) keep.push_back(i);
  }
  if (!t.group()->is_abelian()) {
    if (keep.size() != 1) {
      throw not_implemented_error("Reducing the legs of a " + t.group()->name() +
                                  " tensor keeps one leg, not " +
                                  std::to_string(keep.size()) + ".");
    }
    // One block covers one tree state of the kept leg and its dense expansion carries the
    // Clebsch-Gordan factor of that sector.
    std::vector<at::Tensor> reduced;
    for (std::size_t bi = 0; bi < t.blocks().size(); ++bi) {
      const BlockKey& q = t.qdata()[bi];
      const double charge_dim =
          std::sqrt(t.group()->quantum_dim(
              t.legs()[keep.front()].sector(q.legs[keep.front()])));
      reduced.push_back(at::sqrt(at::sum(t.blocks()[bi].abs().pow(2), *dim)) / charge_dim);
    }
    return at::cat(reduced);
  }
  Shape out_shape;
  for (int64_t i : keep) out_shape.push_back(t.legs()[i].dim());
  // An abelian block is already the dense component of its window, so the squares scatter
  // into the kept legs and take the square root once.
  at::Tensor out = at::zeros(out_shape, at::real(t.blocks().front()).options());
  for (std::size_t bi = 0; bi < t.blocks().size(); ++bi) {
    const BlockKey& q = t.qdata()[bi];
    at::Tensor window = out;
    for (std::size_t i = 0; i < keep.size(); ++i) {
      window = window.slice(static_cast<int64_t>(i), t.legs()[keep[i]].slices()[q.legs[keep[i]]],
                            t.legs()[keep[i]].slices()[q.legs[keep[i]] + 1]);
    }
    window.add_(at::sum(t.blocks()[bi].abs().pow(2), *dim));
  }
  return at::sqrt(out);
}

at::Tensor inner(const Tensor& a, const Tensor& b) {
  if (a.is_dense() && b.is_dense()) {
    return at::sum(a.blocks().front().conj() * b.blocks().front());
  }
  if (a.legs() != b.legs()) {
    throw value_error("An inner product needs equal legs, not " + legs_repr(a.legs()) + " and " +
                      legs_repr(b.legs()) + ".");
  }
  std::map<BlockKey, at::Tensor> table;
  for (std::size_t i = 0; i < b.qdata().size(); ++i) table[b.qdata()[i]] = b.blocks()[i];
  at::Tensor total = at::zeros({}, a.blocks().front().options());
  for (std::size_t i = 0; i < a.qdata().size(); ++i) {
    const auto it = table.find(a.qdata()[i]);
    if (it != table.end()) total = total + at::sum(a.blocks()[i].conj() * it->second);
  }
  return total;
}

at::Tensor trace(const Tensor& t) {
  if (t.is_dense()) return at::trace(t.blocks().front());
  if (t.legs().size() != 2 || t.legs()[0].slices() != t.legs()[1].slices()) {
    throw not_implemented_error("A trace needs two legs with equal block boundaries, not " +
                                legs_repr(t.legs()) + ".");
  }
  // A reduced block is the tree data of the dual pair, so its own diagonal is the contribution of
  // that tree to the invariant contraction of the pair.
  at::Tensor total = at::zeros({}, t.blocks().front().options());
  for (const at::Tensor& block : t.blocks()) {
    if (block.size(0) == block.size(1)) total = total + at::einsum("aa->", {block});
  }
  return total;
}

}  // namespace sym

namespace sym {

// Bind the ``linalg`` names of this module to python.
void bind_linalg(py::module_& m) {
    m.def("svd", &sym::svd, py::arg("t"), py::arg("left"), py::arg("right"),
          py::arg("max_bond") = py::none(), py::arg("tol") = 0.0,
          py::arg("direction") = "right",
          "split a tensor as u s vh, one fused charge at a time");
    m.def("qr", &sym::qr, py::arg("t"), py::arg("left"), py::arg("right"),
          "split a tensor as q r with the isometry on the left");
    m.def("rq", &sym::rq, py::arg("t"), py::arg("left"), py::arg("right"),
          "split a tensor as r q with the isometry on the right");
    m.def("svdvals", &sym::svdvals, py::arg("t"), py::arg("left"), py::arg("right"),
          "the singular values of a partition, by fused charge");
    m.def("norm", &sym::norm, py::arg("t"), py::arg("dim") = py::none(),
          "the Frobenius norm of a tensor, reduced over dim");
    m.def("inner", &sym::inner, py::arg("a"), py::arg("b"),
          "the inner product of two tensors over all legs");
    m.def("trace", &sym::trace, py::arg("t"), "the trace of a two-leg tensor");
}

}  // namespace sym
