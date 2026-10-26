// The contraction of two tensors over the leading legs of one side each.
#pragma once

#include <cstdint>
#include <vector>

#include "../../tensors/tensor.h"

namespace sym
{

  /**
   * @brief The contraction of `first` with `other` over the domain legs of `first` and the codomain
   * legs of `other`, giving a tensor of the charge `charge`.
   * @param first The tensor whose codomain survives.
   * @param other The tensor whose domain survives.
   * @param first_domain The legs of the domain of `first` that are summed, in the order they meet
   * the legs of `other`.
   * @param other_codomain The legs of the codomain of `other` that are summed, one per entry of
   * `first_domain`.
   * @param charge The total charge of the product, one of the outcomes of the two charges.
   * @return The tensor with the codomain of `first`, the domain of `other` and the charge `charge`.
   * @throws std::invalid_argument When the two tensors are over different categories or scalar
   * types, when the two leg lists differ in length, when the lists are not every leg of the side
   * they name, when an entry is not one of the leading legs of its side, when two legs that meet
   * differ in sector or multiplicity, when `charge` is not an outcome of the two charges, or when a
   * charge closes a pair of roots in more than one way, which would need a recoupling this does not
   * do yet.
   * @details The legs that are summed come first on their side, in the same order, so that a block
   * of `first` is a block matrix whose columns run over exactly the rows of a block of `other`.
   * A block of the result is then a matrix product, because the tree basis of a side is
   * orthonormal.
   */
  Tensor contract(const Tensor &first,
                  const Tensor &other,
                  const std::vector<int64_t> &first_domain,
                  const std::vector<int64_t> &other_codomain,
                  const Sector &charge);

  /**
   * @brief The same contraction with `other` neutral, which is the case a chain of tensors mostly
   * meets; the product carries the charge of `first`.
   * @param first The tensor whose codomain survives; the product carries its charge.
   * @param other The tensor whose domain survives, which has to be neutral.
   * @param first_domain The legs of the domain of `first` that are summed.
   * @param other_codomain The legs of the codomain of `other` that are summed.
   * @return The tensor with the codomain of `first`, the domain of `other` and the charge of
   * `first`.
   * @throws std::invalid_argument When `other` does not carry the trivial charge, or for every
   * reason the charged form above raises.
   */
  Tensor contract(const Tensor &first,
                  const Tensor &other,
                  const std::vector<int64_t> &first_domain,
                  const std::vector<int64_t> &other_codomain);

} // namespace sym
