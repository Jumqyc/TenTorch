// The block-sparse tensors of the engine: a map between two graded spaces, one dense matrix per
// pair of roots, and the layout that numbers the rows and columns of a block.
#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
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
   * @brief The block keys of a tensor.
   * @param category The category.
   * @param codomain The legs of the codomain, in order.
   * @param domain The legs of the domain, in order.
   * @param qtotal The charge the tensor as a whole carries.
   * @return Every pair of roots $(r, s)$ with $r \otimes s \ni q$, the codomain root outer, in the
   * order the fusion outcomes of the two sides give them.
   * @details The two sides do not couple to the same charge: a leg stores the charge it
   * contributes, so the domain root is already the dual of the charge the domain space carries, and
   * a pair closes to $q$ exactly when the two roots fuse to it.
   */
  std::vector<std::pair<Sector, Sector>> block_keys(const FusionCategory &category,
                                                    const std::vector<Leg> &codomain,
                                                    const std::vector<Leg> &domain,
                                                    const Sector &qtotal);

  /**
   * @brief A tensor: a map between two graded spaces.
   * @details The map is held as one dense matrix per pair of roots, keyed
   * $(r, s)$: the rows are the trees of the codomain that couple to $r$, the columns the trees of
   * the domain that couple to $s$, and an entry is the coefficient of that pair of basis vectors.
   *
   * The tree basis is orthonormal, so the coefficient is the whole entry: a contraction of two
   * tensors is the matrix product of their blocks and needs no weight, and a closed diagram is a
   * sum of products of coefficients.  Every block of every key is allocated, so a key the legs allow
   * always names a matrix.
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
    const std::map<std::pair<Sector, Sector>, at::Tensor> &blocks() const { return blocks_; }

    // Legs of both sides, codomain first.
    std::vector<Leg> legs() const;
    // Number of legs of both sides.
    int64_t size() const { return static_cast<int64_t>(codomain_.size() + domain_.size()); }

    /**
     * @brief The block of one pair of roots.
     * @param codomain_root The sector the codomain side couples to.
     * @param domain_root The sector the domain side couples to.
     * @return The matrix of that pair.
     * @throws std::out_of_range When the legs do not allow that pair.
     */
    const at::Tensor &block(const Sector &codomain_root, const Sector &domain_root) const;
    at::Tensor &block(const Sector &codomain_root, const Sector &domain_root);

    // The rows of one codomain root, and the columns of one domain root.
    SideLayout rows_of(const Sector &codomain_root) const;
    SideLayout columns_of(const Sector &domain_root) const;

    // The block keys, in the order the constructor allocates them.
    std::vector<std::pair<Sector, Sector>> keys() const;

    at::ScalarType dtype() const { return blocks_.empty() ? at::kComplexDouble : blocks_.begin()->second.scalar_type(); }
    c10::Device device() const { return blocks_.empty() ? cat_->device() : blocks_.begin()->second.device(); }

    std::string repr() const;

    bool operator==(const Tensor &other) const;
    bool operator!=(const Tensor &other) const { return !(*this == other); }

  private:
    std::shared_ptr<const FusionCategory> cat_;
    std::vector<Leg> codomain_;
    std::vector<Leg> domain_;
    Sector qtotal_;
    std::map<std::pair<Sector, Sector>, at::Tensor> blocks_;
  };

} // namespace sym
