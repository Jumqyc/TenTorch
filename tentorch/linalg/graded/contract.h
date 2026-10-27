// The contraction of two tensors over pairs of legs they share.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "../../tensors/tensor.h"

namespace sym
{

  /**
   * @brief Refuse two charges that close to `charge` in more than one way.
   * @param cat The category, which decides the fusion.
   * @param above The charge on the codomain side.
   * @param below The charge on the domain side, which is dualised here.
   * @param charge The charge the two are read to close to.
   * @param subject What the two charges are, as a message opens.
   * @param what The operation that cannot recouple the closings, as a message ends.
   * @throws std::invalid_argument When there is more than one closing.
   * @details A pair of roots that closes in more than one way carries a copy index per closing, so a
   * product of blocks would have to mix the copies through a recoupling this engine does not do; the
   * condition is written once here because every step that closes two roots has to refuse it.
   */
  void require_single_closing(const FusionCategory &cat,
                              const Sector &above,
                              const Sector &below,
                              const Sector &charge,
                              const std::string &subject = "the roots",
                              const std::string &what = "a contraction");

  /**
   * @brief The two operands of a contraction brought to the layout the block product reads.
   * @param left The tensor whose unpaired legs survive on the codomain.
   * @param right The tensor whose unpaired legs survive on the domain.
   * @param left_legs The legs of `left` that are summed, numbered as `legs()` numbers them.
   * @param right_legs The legs of `right` that are summed, one per entry of `left_legs`.
   * @return `left` with the summed legs last on the domain, `right` with them first on the codomain,
   * both in the order the pairs are given; a side that already sits that way is returned unchanged.
   * @throws std::invalid_argument When the two tensors are over different categories or scalar
   * types, when the lists differ in length, when a number is out of range or named twice, or when
   * the legs of a pair name different spaces.
   * @details `contract` calls this itself.  A caller that sums a charge that leaves more than one
   * charge needs it out of its own loop, so that the two tensors are laid out once for all of them.
   */
  std::pair<Tensor, Tensor> align(const Tensor &left,
                                  const Tensor &right,
                                  const std::vector<int64_t> &left_legs,
                                  const std::vector<int64_t> &right_legs);

  /**
   * @brief The contraction of `left` with `right` over the pairs of legs the two lists name.
   * @param left The tensor whose unpaired legs survive on the codomain.
   * @param right The tensor whose unpaired legs survive on the domain.
   * @param left_legs The legs of `left` that are summed, numbered as `legs()` numbers them.
   * @param right_legs The legs of `right` that are summed, one per entry of `left_legs`; the pair at
   * one position names one space.
   * @param charge The total charge of the product, one of the outcomes of the two charges.
   * @return The tensor with the unpaired legs of `left` on the codomain, the unpaired legs of
   * `right` on the domain, both in the order they already had, and the charge `charge`.
   * @throws std::invalid_argument When the two tensors are over different categories or scalar
   * types, when the lists differ in length, when a number is out of range or named twice, when the
   * legs of a pair name different spaces, when `charge` is not an outcome of the two charges, or
   * when a charge closes a pair of roots in more than one way.
   * @details The summed legs are brought to the last legs of the domain of `left` and the first legs
   * of the codomain of `right`, in the order the pairs are given, which is the layout a block
   * product needs; a tensor that already sits that way is not rewritten.  A leg left out of a pair
   * stays on the side it was on, so an empty list of pairs is the outer product of two tensors read
   * with `left` above and `right` below.
   */
  Tensor contract(const Tensor &left,
                  const Tensor &right,
                  const std::vector<int64_t> &left_legs,
                  const std::vector<int64_t> &right_legs,
                  const Sector &charge);

  /**
   * @brief The same contraction with `right` neutral, which is the case a chain of tensors mostly
   * meets; the product carries the charge of `left`.
   * @param left The tensor whose codomain survives; the product carries its charge.
   * @param right The tensor whose domain survives, which has to be neutral.
   * @param left_legs The legs of `left` that are summed.
   * @param right_legs The legs of `right` that are summed, one per entry of `left_legs`.
   * @return The tensor with the unpaired legs of `left`, the unpaired legs of `right` and the charge
   * of `left`.
   * @throws std::invalid_argument When `right` does not carry the trivial charge, or for every
   * reason the charged form above raises.
   */
  Tensor contract(const Tensor &left,
                  const Tensor &right,
                  const std::vector<int64_t> &left_legs,
                  const std::vector<int64_t> &right_legs);

} // namespace sym
