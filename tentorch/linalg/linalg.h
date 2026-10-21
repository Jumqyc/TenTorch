// The factorizations of the engine: singular value, rank revealing and their spectra.
#pragma once

#include <optional>
#include <utility>

#include "../tensor/tensor.h"

#ifndef MPSDMRG_ENGINE_SPLIT_H
#define MPSDMRG_ENGINE_SPLIT_H
/**
 * The splittings of a block-sparse tensor of the engine.
 *
 * ``svd(tensor, left, right)`` factors a tensor over two leg groups, the new bond leg being the last
 * leg of the first factor and the first leg of the second factor. ``qr`` and ``svdvals`` follow the
 * same leg convention. Every factor stays in the orthonormal tree basis of the engine, so no
 * Clebsch-Gordan scale enters the factors.
 */

#include <optional>
#include <utility>

namespace sym {

// The module-level factorizations the python layer calls.
void bind_linalg(py::module_& m);

}  // namespace sym

namespace sym::engine {

/// The three pieces of a singular value decomposition.
struct Split {
  TensorMap first;          ///< the legs of ``left`` and then the new bond, as its last leg
  at::Tensor singular;      ///< one value per copy kept of every charge of the new bond
  TensorMap second;         ///< the new bond as its first leg, then the legs of ``right``
};

/**
 * Singular value decomposition of a tensor over two leg groups.
 * Args:
 *   tensor: the tensor to split.
 *   left: the legs of the first factor, in order; 1-based indices into the leg list.
 *   right: the legs of the second factor, in order; 1-based indices into the leg list.
 *   max_bond: the largest number of singular values to keep; all of them when unset.
 *   tol: singular values at or below this are dropped.
 *   absorb: 0 leaves the values out of the factors, 1 multiplies the first factor's bond leg by
 *     them, 2 multiplies the second factor's bond leg.
 * Returns:
 *   Split: the two factors and the values; a factor keeps the sides its legs had in the input.
 * Raises:
 *   std::invalid_argument: if the two groups do not partition the legs of the tensor.
 */
Split svd(const TensorMap& tensor, const std::vector<int64_t>& left,
          const std::vector<int64_t>& right, std::optional<int64_t> max_bond, double tol,
          int64_t absorb = 0);

/**
 * QR decomposition of a tensor over two leg groups.
 * Args:
 *   tensor: the tensor to split.
 *   left: the legs of the first factor, in order.
 *   right: the legs of the second factor, in order.
 * Returns:
 *   pair: the orthonormal factor and the upper triangular factor.
 * Raises:
 *   std::invalid_argument: if the two groups do not partition the legs of the tensor.
 */
std::pair<TensorMap, TensorMap> qr(const TensorMap& tensor, const std::vector<int64_t>& left,
                                   const std::vector<int64_t>& right);

/**
 * The singular values of a tensor split over two leg groups, without the factors.
 * Args:
 *   tensor: the tensor to split.
 *   left: the legs of the first group, in order.
 *   right: the legs of the second group, in order.
 * Returns:
 *   at::Tensor: one value per copy of every charge, in the order the bond leg lists them.
 * Raises:
 *   std::invalid_argument: if the two groups do not partition the legs of the tensor.
 */
at::Tensor svdvals(const TensorMap& tensor, const std::vector<int64_t>& left,
                   const std::vector<int64_t>& right);

/// Expose the splittings to Python.
void bind_split(py::module_& m);

}  // namespace sym::engine

#endif
