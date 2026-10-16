// The splittings of the engine: one fused charge at a time, the tensor is cut into a pair
// of factors that share a new leg, exactly as mpsdmrg/symmetry/linalg.py does it.
#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "../symmetry/tree/tree.h"
#include "../tensor/tensor.h"

namespace mpsdmrg {

// One dense matrix per fused charge of the left legs.
//
// ``left`` holds the row state of every charge block: the charge-block index of the legs on
// the left side followed by the couplings of their fusion tree, so a row and a column
// recover the whole key of a block.
struct SectorMatrix {
  Sector charge;
  at::Tensor matrix;
  std::vector<BlockKey> left;
  std::vector<BlockKey> right;
  std::vector<int64_t> left_dims;
  std::vector<int64_t> right_dims;
  BlockKey qdata;
};

// The legs ``left`` and ``right`` must cover every leg index exactly once.
// Raises:
//   ValueError: If they do not.
void check_partition(const Tensor& t, const std::vector<int64_t>& left,
                     const std::vector<int64_t>& right);

// One dense matrix per fused charge of the left legs, sorted by charge.
std::vector<SectorMatrix> sector_matrices(const Tensor& t, const std::vector<int64_t>& left,
                                          const std::vector<int64_t>& right);

// The two factors of a split, assembled from the per-charge factor matrices: one block per
// distinct state of each side, sharing the new leg.
std::pair<Tensor, Tensor> assemble_split(
    const Tensor& t, const std::vector<int64_t>& left, const std::vector<int64_t>& right,
    const std::vector<SectorMatrix>& parts,
    const std::vector<std::pair<at::Tensor, at::Tensor>>& factors,
    const std::vector<Sector>& charges, const std::optional<std::vector<at::Tensor>>& kept,
    const Sector& first_qtotal, const Sector& second_qtotal);

// Factor the tensor between its left and right legs as u s vh, one fused charge at a time.
//
// The singular values of every charge are truncated jointly to ``max_bond`` and ``tol``, and
// ``direction`` chooses which factor carries the fusion vertex the pair does not hold.
// Args:
//   t: the tensor to split.
//   left: leg positions forming the matrix rows, in order.
//   right: leg positions forming the matrix columns, in order.
//   max_bond: maximum number of singular values to keep; all if missing.
//   tol: discard singular values not greater than this.
//   direction: ``left`` or ``right``.
// Returns:
//   ``(u, s, vh)`` with ``s`` a plain 1-D torch tensor ordered by fused charge.
// Raises:
//   ValueError: If ``direction`` is neither ``left`` nor ``right``.
std::tuple<Tensor, at::Tensor, Tensor> svd(const Tensor& t, const std::vector<int64_t>& left,
                                           const std::vector<int64_t>& right,
                                           std::optional<int64_t> max_bond, double tol,
                                           const std::string& direction);

// Factor the tensor as q r with the isometry on the left, one fused charge at a time.
std::pair<Tensor, Tensor> qr(const Tensor& t, const std::vector<int64_t>& left,
                             const std::vector<int64_t>& right);

// Factor the tensor as r q with the isometry on the right, one fused charge at a time.
std::pair<Tensor, Tensor> rq(const Tensor& t, const std::vector<int64_t>& left,
                             const std::vector<int64_t>& right);

// The singular values of the matrix the partition forms, by fused charge.
at::Tensor svdvals(const Tensor& t, const std::vector<int64_t>& left,
                   const std::vector<int64_t>& right);

// The Frobenius norm of the tensor, reduced over ``dim``; every leg when it is missing.
// Raises:
//   NotImplementedError: If the group is not abelian and the reduction keeps more than one
//     leg, since one block covers one tree state of the kept leg.
at::Tensor norm(const Tensor& t, const std::optional<std::vector<int64_t>>& dim);

// The inner product of two tensors over all legs.
// Raises:
//   ValueError: If the two tensors do not share their legs.
at::Tensor inner(const Tensor& a, const Tensor& b);

// The trace of a two-leg tensor.
// Raises:
//   NotImplementedError: If the two legs do not share their block boundaries.
at::Tensor trace(const Tensor& t);

// Bind the names of this module to python.
void bind_linalg(py::module_& m);

}  // namespace mpsdmrg
