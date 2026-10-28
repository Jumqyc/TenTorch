// The layout of one side of a block and the blocks a pair of roots admits: which fusion trees
// couple to a root, where each one starts, and which copies of a closing vertex a tensor holds.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
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
   * @brief The block of a leg that carries one sector.
   * @param leg The leg, whose blocks hold their sectors sorted.
   * @param sector The sector to look up.
   * @return The index of the block that carries `sector`.
   * @throws std::invalid_argument When the leg carries no block in that sector.
   */
  int64_t block_of(const Leg &leg, const Sector &sector);

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

} // namespace sym
