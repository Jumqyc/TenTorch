// The groups of the engine: a fusion category whose F and R symbols are the overlaps of its
// Clebsch-Gordan tensors.  The tensor engine sees the category interface only; this header is for
// the group implementations.
#pragma once

#include <cstdint>
#include <map>
#include <tuple>

#include "cat.h"

namespace sym
{

  /**
   * @brief A fusion category realized by the representations of a group.
   * @details The category supplies the Clebsch-Gordan tensor of one vertex, and this class derives
   * the associator and the braiding from it: $F$ is the overlap of the two bracketings of three
   * legs and $R$ the overlap of the two orders of two legs, both divided by the quantum dimension
   * of the coupled sector.  The pentagon and the hexagon then hold because each side of the
   * identity is the same overlap computed in two ways, so a group cannot state an $F$ or an $R$
   * that disagrees with its own vertices.
   *
   * A category without a Clebsch-Gordan tensor, such as the Fibonacci category, is not a group of
   * this kind and implements `F_symbol` and `R_symbol` itself.
   */
  class Group : public FusionCategory
  {
  public:
    using FusionCategory::FusionCategory;

    /**
     * @brief The Clebsch-Gordan tensor $C^{c}_{ab}$ of one copy, memoised.
     * @param a The sector of the first leg.
     * @param b The sector of the second leg.
     * @param c The coupled sector, one outcome of `outcomes(a, b)`.
     * @param copy The copy index in $[0, N^c_{ab})$, the second index of the rows and columns that
     * `F_symbol` and `R_symbol` address.
     * @return A tensor of shape (dim a, dim b, dim c), one entry per component of the three legs.
     * @details It is the embedding of Eq. (eq:cg-orth) of the notes: each copy is a unit vector of
     * the component space, so the whole tensor has squared norm $q_{\dim}(c)$.  The tensor of one
     * sector triple is computed once per object and then held, because the associator and the
     * braiding ask for the same vertices many times.
     */
    at::Tensor clebsch_gordan(const Sector &a, const Sector &b, const Sector &c,
                              int64_t copy) const;

    /** @brief Drop every Clebsch-Gordan tensor and every fusion table this object holds. */
    void clear_cache() const override;

    /**
     * @brief The associator $F^{d}_{abc}$, the overlap of the two bracketings of three legs.
     * @param a The first sector, the leftmost of the left tree.
     * @param b The second sector.
     * @param c The third sector.
     * @param d The outer sector both trees fuse to.
     * @return The matrix `FusionCategory::F_symbol` describes, with entry $\langle ((a \otimes b)
     * \otimes c) \to d \mid (a \otimes (b \otimes c)) \to d \rangle / q_{\dim}(d)$; empty when $d$
     * is not an outcome of $a \otimes b \otimes c$.
     */
    at::Tensor F_symbol(const Sector &a, const Sector &b, const Sector &c,
                        const Sector &d) const override;

    /**
     * @brief The braiding $R_{bc}$, the overlap of the two orders of two legs.
     * @param b The left sector before the exchange.
     * @param c The right sector before the exchange.
     * @return A square matrix, one block per outcome $e$ of `outcomes(b, c)` with $N^e_{bc}$ rows
     * and $N^e_{cb}$ columns, entry $\langle (b \otimes c) \to e \mid (c \otimes b) \to e \rangle /
     * q_{\dim}(e)$.
     * @raises RuntimeError If $b$ and $c$ fuse to some $e$ with $N^e_{bc} \neq N^e_{cb}$, which no
     * braiding can relate.
     */
    at::Tensor R_symbol(const Sector &b, const Sector &c) const override;

   protected:
    /**
     * @brief Compute the Clebsch-Gordan tensor $C^{c}_{ab}$ of one copy.
     * @param a The sector of the first leg.
     * @param b The sector of the second leg.
     * @param c The coupled sector, one outcome of `outcomes(a, b)`.
     * @param copy The copy index in $[0, N^c_{ab})$.
     * @return A tensor of shape (dim a, dim b, dim c).
     * @details The one method a concrete group implements; `clebsch_gordan` wraps it with the
     * cache, so every reader of the vertex goes through the same table.
     */
    virtual at::Tensor compute_clebsch_gordan(const Sector &a, const Sector &b, const Sector &c,
                                              int64_t copy) const = 0;

   private:
    // One vertex per (a, b, c, copy); the device is the object's, so it is not part of the key.
    mutable std::map<std::tuple<Sector, Sector, Sector, int64_t>, at::Tensor> cg_cache_;
  };

} // namespace sym
