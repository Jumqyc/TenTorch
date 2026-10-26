// The comparison of two tensors up to a tolerance, and the scalar product they carry.
#pragma once

#include <complex>

#include "../../tensors/tensor.h"

namespace sym
{

  /**
   * @brief Whether two tensors hold the same map, up to a tolerance.
   * @param first The first tensor.
   * @param second The second tensor.
   * @param rtol The relative tolerance, which bounds the error of an entry by a fraction of its own
   * size and so is what a large entry is judged by.
   * @param atol The absolute tolerance, which bounds the error of an entry by a fixed size and so is
   * what an entry near zero is judged by.
   * @return Whether the two tensors are over the same category, carry the same total charge, have
   * the same legs on both sides, hold the same block keys with the same shapes, and satisfy
   * $|a - b| \le \text{atol} + \text{rtol}\,|b|$ entry by entry.
   * @details The two tolerances cover each other's blind spot: a relative bound alone calls every
   * pair of tiny values equal, an absolute bound alone calls every pair of large ones different.
   */
  bool is_close(const Tensor &first, const Tensor &second, double rtol = 1e-10, double atol = 1e-12);

  /**
   * @brief The scalar product of two tensors.
   * @param first The first tensor, of any total charge.
   * @param second The second tensor; it holds the same legs on both sides as `first`.
   * @return $\sum_{\text{blocks}} \overline{a} \, b$, summed entry by entry over the blocks the two
   * tensors share; zero when they carry different total charges.
   * @throws std::invalid_argument When the two do not sit over one category, or their legs differ.
   * @details Both tensors hold the same trees over the same legs, and two different trees are
   * orthogonal, so one tree meets only itself and the scalar product is the sum over the entries of
   * the blocks.  A tree weighs the dimension of its root, $\|\text{tree}\|^2 = d_r$, which this sum
   * leaves out: it is the reduced scalar product, short one factor of the dimension of the root the
   * two sides close to compared with the component array of the same tensor.
   */
  std::complex<double> inner(const Tensor &first, const Tensor &second);

  /**
   * @brief The Frobenius norm of a tensor.
   * @param tensor The tensor.
   * @return $\sqrt{\langle t | t \rangle}$, which is the square root of the sum of the squared
   * magnitudes of the block entries.
   */
  double norm(const Tensor &tensor);

} // namespace sym
