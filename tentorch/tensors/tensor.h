// The block-sparse tensors of the engine: a map between two graded spaces, one dense matrix per
// pair of roots, and the layout that numbers the rows and columns of a block.
#pragma once

#include <ATen/ATen.h>

#include <complex>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../symmetry/cat/cat.h"
#include "../symmetry/tree/tree.h"

namespace sym
{

  /**
   * @brief The rows one side of a block spans.
   * @details The side of a block is every tree of one root, and a tree contributes one row per
   * combination of the multiplicities its legs carry: $\prod_i m_i$, where $m_i$ is the
   * multiplicity of the block that leg $i$ carries in the tree.  `offsets[t]` is the first row of
   * `trees[t]`, so `offsets.back()` is the number of rows and a row is read back by mixed-radix
   * decomposition of its offset into the tree.
   *
   * The trees are in the order `fusion_trees` gives them, so the layout is a function of the legs
   * and the root alone.
   */
  struct SideLayout
  {
    std::vector<FusionTree> trees;
    std::vector<int64_t> offsets;

    // Rows the side spans.
    int64_t rows() const { return offsets.back(); }
  };

  /**
   * @brief The layout of one side of a block.
   * @param category The category.
   * @param legs The legs of the side, in order.
   * @param root The sector the trees of the side couple to.
   * @return The trees that couple to `root` and where each one starts.
   */
  SideLayout layout_of(const FusionCategory &category,
                       const std::vector<Leg> &legs,
                       const Sector &root);

  /**
   * @brief The key of one block: the two roots and which copy of the closing vertex it carries.
   * @details A pair of roots closes to the total charge through $\operatorname{Hom}(q, r \otimes \bar s)$,
   * whose dimension is $N^{q}_{r\bar s}$.  That dimension is at least one for every allowed pair, so
   * every copy is the key of its own block: a tensor with a total charge whose pair admits several
   * closings carries several blocks for it, one per invariant on that pair.
   */
  struct BlockKey
  {
    Sector codomain;
    Sector domain;
    int64_t copy = 0;

    bool operator<(const BlockKey &other) const
    {
      return std::tie(codomain, domain, copy) < std::tie(other.codomain, other.domain, other.copy);
    }
    bool operator==(const BlockKey &other) const
    {
      return codomain == other.codomain && domain == other.domain && copy == other.copy;
    }
    bool operator!=(const BlockKey &other) const { return !(*this == other); }
  };

  /**
   * @brief The block keys of a tensor.
   * @param category The category.
   * @param codomain The legs of the codomain, in order.
   * @param domain The legs of the domain, in order.
   * @param qtotal The charge the tensor as a whole carries.
   * @return Every pair of roots $(r, s)$ with $r \otimes \bar{s} \ni q$ and, for each, one key per
   * copy of the closing vertex, the codomain root outer, in the order the fusion outcomes give them.
   * @details The two sides do not couple to the same charge: a leg stores the charge it
   * contributes, so the domain root is already the dual of the charge the domain space carries, and
   * a pair closes to $q$ exactly when the two roots fuse to it.
   */
  std::vector<BlockKey> block_keys(const FusionCategory &category,
                                   const std::vector<Leg> &codomain,
                                   const std::vector<Leg> &domain,
                                   const Sector &qtotal);

  /**
   * @brief A tensor: a map between two graded spaces.
   * @details The map is held as one dense matrix per pair of roots, keyed
   * $(r, s)$: the rows are the trees of the codomain that couple to $r$, the columns the trees of
   * the domain that couple to $s$, and an entry is the coefficient of that pair of basis vectors.
   *
   * The tree basis is orthonormal up to the dimension of its root, $\|\text{tree}\|^2 = d_r$, so
   * the coefficient is the whole entry: a contraction of two tensors is the matrix product of their
   * blocks and needs no weight.  Closing a pair of legs sums the copies the two trees share.  A
   * tree of a single leg carries no vertex of its own, so the trace of the identity on a leg is one
   * rather than the dimension of that leg: a closed diagram is the reduced sum of its coefficients,
   * one factor of a closed loop's quantum dimension away from the component array of the same
   * tensor.
   * Every block of every key is allocated, so a key the legs allow always names a matrix.
   *
   * A tensor is data.  It holds the category, the legs of the two sides, its total charge and its
   * blocks, and nothing else: the layout of a block is a function of the legs and the roots, so
   * storing it would be a second copy of the same information.
   */
  class Tensor
  {
  public:
    /**
     * @brief A tensor whose blocks are all zero.
     * @param cat The category.
     * @param codomain The legs of the codomain, in order.
     * @param domain The legs of the domain, in order.
     * @param qtotal The charge the tensor as a whole carries.
     * @param dtype The scalar type, the category's own when omitted.
     * @param device The device, the category's own when omitted.
     * @throws std::invalid_argument When a leg is over a different category.
     * @throws std::invalid_argument When the category is null.
     */
    Tensor(std::shared_ptr<const FusionCategory> cat,
           std::vector<Leg> codomain,
           std::vector<Leg> domain,
           Sector qtotal,
           std::optional<c10::ScalarType> dtype = std::nullopt,
           std::optional<c10::Device> device = std::nullopt);

    const std::shared_ptr<const FusionCategory> &cat() const { return cat_; }
    const std::vector<Leg> &codomain() const { return codomain_; }
    const std::vector<Leg> &domain() const { return domain_; }
    const Sector &qtotal() const { return qtotal_; }
    const std::map<BlockKey, at::Tensor> &blocks() const { return blocks_; }

    // Legs of both sides, codomain first.
    std::vector<Leg> legs() const;
    // Number of legs of both sides.
    int64_t nlegs() const { return static_cast<int64_t>(codomain_.size() + domain_.size()); }

    /**
     * @brief The block of one pair of roots and one copy of its closing vertex.
     * @param codomain_root The sector the codomain side couples to.
     * @param domain_root The sector the domain side couples to.
     * @param copy Which invariant of the pair the block carries, in $[0, N^{q}_{r\bar s})$; the
     * first one when omitted.
     * @return The matrix of that pair.
     * @throws std::out_of_range When the legs do not allow that pair.
     */
    const at::Tensor &block(const Sector &codomain_root, const Sector &domain_root, int64_t copy = 0) const;
    at::Tensor &block(const Sector &codomain_root, const Sector &domain_root, int64_t copy = 0);

    /**
     * @brief The layout of one codomain root.
     * @param codomain_root The sector the codomain couples to.
     * @return The trees of that root and where each one starts; the layout is built once per root
     * and then read from the cache on the tensor.
     */
    const SideLayout &rows_of(const Sector &codomain_root) const;

    /**
     * @brief The layout of one domain root.
     * @param domain_root The sector the domain couples to.
     * @return The trees of that root and where each one starts; the layout is built once per root
     * and then read from the cache on the tensor.
     */
    const SideLayout &columns_of(const Sector &domain_root) const;

    // The block keys, in the order the constructor allocates them.
    std::vector<BlockKey> keys() const;

    /**
     * @brief The same tensor with two adjacent codomain legs exchanged.
     * @param position The left of the two legs, in $[0, n - 2]$.
     * @return The tensor whose legs at `position` and `position + 1` are exchanged.
     * @throws std::invalid_argument When the position is out of range.
     * @details The charge content of the blocks does not change, because the set of trees that reach
     * a root does not change: what changes is which basis vector each tree is.  The exchange of two
     * adjacent legs is the recoupling
     * $S = F^{d}_{y c b} \cdot R^{f}_{bc} \cdot (F^{d}_{y b c})^{\dagger}$ on the pair, where $y$ is
     * the charge of the legs before it and $d$ the charge of the legs through it, applied to every
     * tree the pair leaves free.  The $R$ factor is what carries the sign: exchanging the two legs
     * of a singlet multiplies it by $R^{0}_{1/2\,1/2} = -1$.
     */
    Tensor swap(int64_t position) const;

    /**
     * @brief The same tensor with one leg bent onto the other side.
     * @param position The leg of the chosen side to move; it becomes the last leg of the other
     * side.
     * @param from_codomain Which side gives the leg: `true` moves a codomain leg onto the domain,
     * `false` moves a domain leg onto the codomain.
     * @return The same map read with that leg on the other side, so the leg stores the dual of the
     * space it indexed and the total charge does not change.
     * @throws std::invalid_argument When the position does not name a leg of that side.
     * @details The leg is brought to the end of its side and then bent across by one recoupling of
     * the vertex at the root: the tree of the side that gives the leg loses its last vertex, the
     * tree of the side that takes it gains one, and every deeper vertex is left alone, which keeps
     * the layout of the untouched legs unchanged.  A tree is a tree, so both directions run the
     * same recoupling with the roles of the two trees exchanged.
     */
    Tensor move_across(int64_t position, bool from_codomain = true) const;

    /**
     * @brief The same tensor with its codomain legs in another order.
     * @param order The new order: `order[j]` is the codomain leg that moves to position `j`.
     * @return The permuted tensor.
     * @throws std::invalid_argument When `order` is not a permutation of the codomain legs.
     */
    Tensor permute(const std::vector<int64_t> &order) const;

    /**
     * @brief The same tensor with the named legs on the codomain.
     * @param codomain The legs to put there, numbered over the codomain legs followed by the domain
     * legs, listed in the order they are to appear.
     * @return The tensor with exactly those legs on its codomain, in that order, and the other legs
     * on its domain in their original relative order.
     * @throws std::invalid_argument When a number is out of range or named twice.
     * @details Only the codomain can be permuted, so every leg is first bent onto the codomain,
     * ordered there, and the legs that are not named are bent back last to first, which lands them
     * on the domain in their original relative order.
     */
    Tensor repartition(const std::vector<int64_t> &codomain) const;

    /**
     * @brief The conjugate of the tensor, read as the bra.
     * @return The tensor whose blocks are conjugated, whose codomain holds the legs the domain held
     * and the other way round, and whose total charge is the dual charge.
     * @details A leg stores the charge of the space it indexes and not the side it sits on, so
     * moving a leg to the other side is what conjugates the space: the same leg object then meets
     * itself in a contraction, which is how $\langle A | B\rangle$ pairs the indices without a
     * duality matrix.  The block key words swap along with the roots, so the block over $(r, s)$
     * becomes the block over $(s, r)$, and $N^{\bar q}_{s \bar r} = N^{q}_{r \bar s}$ copies the
     * other way.  Both sides lay their rows out through the same tree order, so the block transposes
     * as it conjugates.
     */
    Tensor conj() const;

    /**
     * @brief Copies this tensor with storage of its own.
     * @return A tensor with the same legs and charge whose blocks share no memory with this one.
     * @details The copy constructor shares the block tensors, which is what a read-only view
     * wants; an in-place update needs this instead.
     */
    Tensor clone() const;

    /**
     * @brief Adds `other` into this tensor, block by block.
     * @param other The tensor to add, with the same legs and the same charge.
     * @return This tensor.
     * @throws std::invalid_argument When `other` is over another category, or does not carry the
     * same legs and the same charge.
     * @details The tree basis of a side is fixed by the legs and the root, so two tensors with the
     * same legs hold their coefficients over one basis and a sum is a sum of coefficients.
     */
    Tensor &operator+=(const Tensor &other);

    /**
     * @brief Subtracts `other` from this tensor, block by block.
     * @param other The tensor to subtract, with the same legs and the same charge.
     * @return This tensor.
     * @throws std::invalid_argument When `other` is over another category, or does not carry the
     * same legs and the same charge.
     */
    Tensor &operator-=(const Tensor &other);

    /**
     * @brief Scales every entry of this tensor.
     * @param scalar The factor.
     * @return This tensor.
     * @details A tensor that carries a charge is a vector in a space of several sectors at once,
     * so a factor applies to every block: it is not a physical symmetry operation.
     */
    Tensor &operator*=(double scalar);

    /**
     * @brief Scales every entry of this tensor by a complex factor.
     * @param scalar The factor.
     * @return This tensor.
     * @throws std::invalid_argument When a block holds real entries and the factor is not real.
     */
    Tensor &operator*=(std::complex<double> scalar);

    /**
     * @brief Divides every entry of this tensor.
     * @param scalar The divisor.
     * @return This tensor.
     * @throws std::invalid_argument When `scalar` is zero.
     */
    Tensor &operator/=(double scalar);

    at::ScalarType dtype() const { return dtype_; }
    c10::Device device() const { return device_; }

  private:
    std::shared_ptr<const FusionCategory> cat_;
    std::vector<Leg> codomain_;
    std::vector<Leg> domain_;
    Sector qtotal_;
    // The layouts of the two sides, one per root: a layout is a function of the legs and the root,
    // so every block that shares a root shares the layout as well.
    mutable std::map<Sector, SideLayout> codomain_layouts_;
    mutable std::map<Sector, SideLayout> domain_layouts_;
    at::ScalarType dtype_ = at::kComplexDouble;
    c10::Device device_ = c10::Device(c10::DeviceType::CPU);
    std::map<BlockKey, at::Tensor> blocks_;
  };

} // namespace sym
