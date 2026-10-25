// The decompositions of a tensor along a split of its legs.
#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "../../tensors/tensor.h"

namespace sym
{

  /**
   * @brief The factors of a split: a tensor over the codomain and a bond, the singular values, and a
   * tensor over the bond and the domain.
   * @details `first` carries the charge of the split tensor, `second` carries the trivial charge and
   * `first` contracted with `second` over the bond gives the split tensor back.
   */
  struct Split
  {
    Tensor first;
    at::Tensor values;
    Tensor second;
  };

  /**
   * @brief The singular value decomposition of `tensor`, with every codomain leg on the left and
   * every domain leg on the right.
   * @param tensor The tensor to split.
   * @param max_bond The most multiplets the bond may carry, over all of its sectors; no cap when
   * omitted.
   * @param tol The singular values at or below it are dropped.
   * @param absorb 0 keeps the values in `values` alone, 1 multiplies them into `first`, 2 into
   * `second`, 3 splits them as square roots into both, so that `first` contracted with `second` is
   * the tensor for 1, 2 and 3.
   * @return The two factors and the singular values, one entry per copy of the bond in the order of
   * its blocks, zero where a block supplied fewer values than its sector holds.
   * @throws std::invalid_argument When `absorb` is not 0, 1, 2 or 3, or when `max_bond` is below
   * one.
   * @details The charge is conserved, so a block of the tensor is block diagonal and each of its
   * blocks is decomposed on its own.  The bond is one leg whose multiplicity is counted per sector,
   * so every block that lands in one sector shares that count: the values of the sector are ranked
   * together and the blocks are laid out in that order along the bond.
   */
  Split svd(const Tensor &tensor,
            std::optional<int64_t> max_bond = std::nullopt,
            double tol = 0.0,
            int64_t absorb = 0);

  /**
   * @brief The same decomposition along a split the caller names.
   * @param tensor The tensor to split.
   * @param left The legs that go on the left, numbered over the codomain legs followed by the domain
   * legs; a leg of either side can be named, and the legs are brought to the side they are named for
   * before the split.
   * @param max_bond The most multiplets the bond may carry, over all of its sectors.
   * @param tol The singular values at or below it are dropped.
   * @param absorb Where the values are multiplied in, as in `svd`.
   * @return The factors of the named split.
   * @throws std::invalid_argument When a leg number is out of range or named twice.
   */
  Split svd(const Tensor &tensor,
            const std::vector<int64_t> &left,
            std::optional<int64_t> max_bond = std::nullopt,
            double tol = 0.0,
            int64_t absorb = 0);

  /**
   * @brief The singular values of `tensor`, with every codomain leg on the left and every domain leg
   * on the right, without truncation.
   * @param tensor The tensor to decompose.
   * @return The values, one entry per copy of the bond of the untruncated split.
   */
  at::Tensor svdvals(const Tensor &tensor);

  /**
   * @brief The singular values of the split the caller names, without truncation.
   * @param tensor The tensor to decompose.
   * @param left The legs that go on the left, as in `svd`.
   * @return The values, one entry per copy of the bond.
   */
  at::Tensor svdvals(const Tensor &tensor, const std::vector<int64_t> &left);

  /**
   * @brief Two factors of a tensor along the split between its two sides.
   */
  struct Decomposition
  {
    Tensor first;
    Tensor second;
  };

  /**
   * @brief The QR decomposition of `tensor`, with every codomain leg on the left and every domain leg
   * on the right.
   * @param tensor The tensor to split.
   * @return `first` over the codomain and a bond, whose columns are orthonormal, and `second` over
   * the bond and the domain, upper triangular inside every block.
   * @details The charge is conserved, so every block is decomposed on its own.  The bond carries the
   * domain root of the block it came from, so its multiplets are counted per sector over the blocks
   * that share one: the columns of a block land at its own offset, and the rest of the block is zero.
   */
  Decomposition qr(const Tensor &tensor);

  /**
   * @brief The same decomposition along a split the caller names.
   * @param tensor The tensor to split.
   * @param left The legs that go on the left, as in `svd`.
   * @return The two factors of the named split.
   */
  Decomposition qr(const Tensor &tensor, const std::vector<int64_t> &left);

  /**
   * @brief The mirror of `qr`: `first` over the codomain and a bond, `second` over the bond and the
   * domain with orthonormal rows.
   * @param tensor The tensor to split.
   * @return The two factors, whose product over the bond gives the tensor.
   * @details `second` is the orthonormal factor and `first` is the transpose of the QR factor of the
   * transposed block, so it is lower triangular; the triangular factor sits on the side of the legs
   * it came from, which is the opposite of `qr`.
   */
  Decomposition rq(const Tensor &tensor);

  /**
   * @brief The same decomposition along a split the caller names.
   * @param tensor The tensor to split.
   * @param left The legs that go on the left, as in `svd`.
   * @return The two factors of the named split.
   */
  Decomposition rq(const Tensor &tensor, const std::vector<int64_t> &left);

} // namespace sym
