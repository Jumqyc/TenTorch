// The contraction of a chain of tensors, one pair at a time, with the charge of every partial product
// named.
#pragma once

#include <string>
#include <vector>

#include "../../tensors/tensor.h"

namespace sym
{

  /**
   * @brief Contracts the operands as an Einstein sum does, summing every label that does not appear
   * in the output and leaving the output labels on the codomain.
   * @param spec The equation, for instance `"abc,bcd,de->ae"`; the legs of an operand map to its
   * labels in the order of `Tensor::legs`, that is the codomain first and the domain after.
   * @param operands The tensors, one per comma separated group of the left hand side.
   * @param charge The total charge of the result, one of the ways the operand charges fuse.
   * @return The tensor whose codomain holds the output labels in the order of the equation, with the
   * charge `charge`.
   * @throws std::invalid_argument When the equation is malformed, when a label names a leg that is
   * summed in more than one tensor, when two operands of a step share no leg, when no fusion of the
   * operand charges gives `charge`, or when `charge` closes a pair of roots in more than one way.
   * @details The operands are contracted left to right.  A step brings the shared legs to the domain
   * of the accumulator and the codomain of the next operand, which is a recoupling the contraction
   * hides, and then multiplies the blocks.  A charge that several intermediate charges can reach is
   * the sum of those paths, so the result holds every tree of the charges, not just one.
   */
  Tensor einsum(const std::string &spec, const std::vector<Tensor> &operands, const Sector &charge);

  /**
   * @brief The same sum with the charge worked out, which is possible when every fusion path of the
   * operand charges ends on one charge.
   * @param spec The equation.
   * @param operands The tensors.
   * @return The tensor whose codomain holds the output labels, with that charge.
   * @throws std::invalid_argument When the fusion paths end on more than one charge, in which case
   * the charge has to be named, or for every reason the form above raises.
   */
  Tensor einsum(const std::string &spec, const std::vector<Tensor> &operands);

} // namespace sym
