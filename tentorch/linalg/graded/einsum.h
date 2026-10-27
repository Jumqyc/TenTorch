// The contraction of a chain of tensors, one pair at a time, with the charge of every partial product
// named.
#pragma once

#include <string>
#include <vector>

#include "../../tensors/tensor.h"

namespace sym
{

  /**
   * @brief Einsteins the operands: every label that appears twice is summed, every label that
   * appears once is left on the side the case of its label in the output declares.
   * @param spec The equation, for instance `"aBc,bCd,dE->AE"`; the legs of an operand map to its
   * labels in the order of `Tensor::legs`, that is the codomain first and the domain after, and the
   * case of a label names the side its leg has to sit on, upper case the codomain and lower case the
   * domain.
   * @param operands The tensors, one per comma separated group of the left hand side.
   * @param charge The total charge of the result, which the output legs have to reach; the Python
   * binding names this keyword `outcharge` and defaults it to the trivial charge.
   * @return The tensor whose codomain holds the upper case output labels in the order of the
   * equation and whose domain holds the lower case ones under them, with the charge `charge`.
   * @throws std::invalid_argument When the equation is malformed, when a label appears more than
   * twice, when a label appears once but is not kept in the output, when the two legs of a label
   * name different spaces, when an outer product, a contraction or a trace closes a pair of roots in
   * more than one way, when the output legs cannot close to `charge`, or when no fusion of the
   * operand charges gives `charge`.
   * @details The case of a label is a request, not a condition: a leg that sits on the other side is
   * bent over before the operands are combined, never refused.  A label that appears once is free,
   * so its request is binding and the leg is moved to the side the equation asks for; a label that
   * appears twice is summed, and there the two legs are paired through the contraction, so one of
   * them has to change side anyway and each keeps the side it already has.  A tensor whose layout
   * already answers the equation is left untouched, one whose sides hold costs a permutation of its
   * codomain alone, and only a leg that has to change side is bent.
   * A label that appears twice in one operand is a trace over that pair of legs, which
   * lose both their vertices before the operands are combined; the pair has to name one space and,
   * when both legs sit on the same side, the second is bent over with the F symbol first.  The
   * operands are then folded into an accumulator in the order that makes the products smallest, which
   * for up to twelve operands is the exact minimum over the subsets of the operands: a step whose two
   * operands share a label brings every shared leg to the domain of the accumulator and the codomain
   * of the next operand in one contraction, which is a recoupling the contraction hides, and it is
   * taken before any step whose operands share none, since an outer product keeps every leg of both
   * operands.
   * A charge that several intermediate charges can reach is the sum of those paths, so the result
   * holds every tree of the charges, not just one.
   */
  Tensor einsum(const std::string &spec, const std::vector<Tensor> &operands, const Sector &charge);

  /**
   * @brief The same sum with the trivial charge as the total charge.
   * @param spec The equation.
   * @param operands The tensors.
   * @return The tensor whose codomain holds the upper case output labels, with the trivial charge.
   * @throws std::invalid_argument When the output legs cannot close to the trivial charge, or for
   * every reason the form above raises.
   */
  Tensor einsum(const std::string &spec, const std::vector<Tensor> &operands);

} // namespace sym
