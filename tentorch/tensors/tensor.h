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
#include "layout.h"

namespace sym
{

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
     * @brief The same tensor with `above` on the codomain and `below` on the domain.
     * @param above The legs to read as codomain, in the order they are to appear; a leg is numbered
     * over the codomain legs followed by the domain legs, as `legs()` numbers them.
     * @param below The legs to read as domain, in the order they are to appear.  The two lists name
     * every leg exactly once.
     * @return The same entries read with that layout.
     * @throws std::invalid_argument When a number is out of range, is named twice, or is left out.
     * @details A leg stores the space it indexes and not the side it sits on, so a leg that crosses
     * keeps both its value and the total charge: only the basis of the trees holding it changes.
     * The codomain is the side whose legs can be ordered, so every leg is bent onto the codomain,
     * the order there is reached by exchanging neighbours, and the legs that end on the domain are
     * taken off its end last named first, which is what lands them in the order they were named in.
     * A layout whose sides already hold the right legs costs those exchanges alone.
     *
     * A bend is one $F$ recoupling at the root of the two trees it touches and an exchange is
     * $F^{d}_{y c b} \, R^{f}_{bc} \, (F^{d}_{y b c})^{\dagger}$ on the pair, where $y$ is the charge
     * of the legs before the pair and $d$ the charge of those through it; either is applied to every
     * tree the step leaves free, so the whole chain costs one pass per step but a single output
     * tensor, whose buffers are reused from one step to the next.
     */
    Tensor place(const std::vector<int64_t> &above, const std::vector<int64_t> &below) const;

    /**
     * @brief The same tensor with `above` on the codomain and every other leg on the domain.
     * @param above The legs to read as codomain, in the order they are to appear.
     * @return The same entries read with that layout, the legs `above` leaves out on the domain in
     * their original relative order.
     * @throws std::invalid_argument When a number is out of range or is named twice.
     */
    Tensor place(const std::vector<int64_t> &above) const;

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

  /**
   * @brief The positions in `legs()` of every leg of the codomain, in order.
   * @param tensor The tensor.
   * @return `0, 1, ..., codomain().size() - 1`.
   */
  std::vector<int64_t> codomain_legs(const Tensor &tensor);

  /**
   * @brief The positions in `legs()` of every leg of the domain, in order.
   * @param tensor The tensor.
   * @return `codomain().size(), ..., nlegs() - 1`.
   */
  std::vector<int64_t> domain_legs(const Tensor &tensor);

} // namespace sym
