// The charge-adapted legs of the engine and the fusion trees over them: what a leg carries, which
// trees one side admits, and the pairs of trees that key a tensor block.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../cat/cat.h"

namespace sym
{

  /**
   * @brief Whether two categories are the same symmetry.
   * @param a One category.
   * @param b Another category.
   * @return True when both name the same object.
   * @details The engine runs in one interpreter and a category is a single object there, so identity is the address.  A leg stores the name of its category when it is pickled and not the category itself, so an unpickled leg points at that same object and stays fusible with the live one.
   */
  inline bool same_group(const FusionCategory &a, const FusionCategory &b) {return &a == &b;}

  /**
   * @brief One side of a tensor: the graded space $\bigoplus_c (V_c \otimes \mathbb{C}^{m_c})$, one
   * block per sector $c$ with multiplicity $m_c$.
   * @details The blocks are held sorted by sector, so two legs over the same space hold identical
   * storage and compare equal without a search.
   *
   * The sectors a leg stores are the charges of the space it indexes, so a leg reads the same
   * whichever side of a tensor it sits on and two legs over one space are one value.  The arrow is
   * not stored: a domain leg carries a bra index, so the charge it contributes to the fusion is the
   * dual of what it stores, and that dual is taken where the total charge is compared.
   *
   * `dim` counts components and `multiplets` counts copies.  A block of multiplicity $m$ in sector
   * $c$ contributes $m \dim c$ components and $m$ multiplets; a bond dimension is counted in
   * multiplets, which is the quantity a truncation is expressed in.
   */
  class Leg
  {
  public:
    /**
     * @brief A leg from the sector of every block and its multiplicity.
     * @param cat The category the sectors belong to.
     * @param sectors The sector of every block, sorted here and required to be distinct.
     * @param multiplicities The multiplicity of every block, one per sector, each positive.
     * @throws std::invalid_argument When the two lists differ in length, a multiplicity is not
     * positive, a sector repeats, or the category gives a sector no components.
     */
    Leg(std::shared_ptr<const FusionCategory> cat,
        std::vector<Sector> sectors,
        std::vector<int64_t> multiplicities);

    /**
     * @brief One block in the trivial sector, carrying `multiplets` copies of it.
     * @param multiplets The number of copies, which is the number of components because the trivial
     * sector has one component.
     * @param cat The category.
     * @return The leg.
     */
    static Leg trivial(int64_t multiplets, std::shared_ptr<const FusionCategory> cat);

    /**
     * @brief A leg from `(sector, multiplicity)` entries, together with where each entry landed.
     * @param cat The category.
     * @param entries The blocks, in any order.
     * @return The leg, whose blocks are sorted by sector, and for entry `k` the block index that
     * entry ended up at.
     */
    static std::pair<Leg, std::vector<int64_t>> from_blocks(
        std::shared_ptr<const FusionCategory> cat,
        const std::vector<std::pair<Sector, int64_t>> &entries);

    const std::shared_ptr<const FusionCategory> &cat() const { return cat_; }
    const std::vector<Sector> &sectors() const { return sectors_; }
    const std::vector<int64_t> &multiplicities() const { return multiplicities_; }

    // Number of charge blocks on the leg.
    int64_t nblocks() const { return static_cast<int64_t>(sectors_.size()); }
    // Sector of one block.
    const Sector &sector(int64_t block) const { return sectors_.at(static_cast<size_t>(block)); }
    // Multiplicity of one block.
    int64_t multiplicity(int64_t block) const { return multiplicities_.at(static_cast<size_t>(block)); }
    // Copies of the whole leg, $\sum_c m_c$.
    int64_t multiplets() const;

    /**
     * @brief The graded tensor product with another leg.
     * @param other The other leg, over the same category.
     * @return The leg whose block in sector $c$ has multiplicity
     * $\sum_{a,b} m_a m_b N^c_{ab}$.
     * @throws std::invalid_argument When the two legs are over different categories.
     */
    Leg fuse(const Leg &other) const;

    /**
     * @brief The same space with the arrow of every leg reversed.
     * @return The leg whose sectors are the duals of this one's.
     */
    Leg conj() const;

    /**
     * @brief The leg cut down to at most `max_multiplets` multiplets.
     * @param max_multiplets The largest number of multiplets to keep.
     * @return The leg itself when it is small enough, and otherwise its blocks in sector order up to
     * the last one the cut reaches, that one truncated.
     */
    Leg capped(int64_t max_multiplets) const;

    bool operator==(const Leg &other) const;
    bool operator!=(const Leg &other) const { return !(*this == other); }

    /**
     * @brief A hash consistent with `operator==`.
     * @return The hash of the name of the category and of every sector and multiplicity.
     * @details The name is hashed rather than the address, so that the hash of a leg does not change
     * between runs.  Two categories sharing a name collide, which a hash allows.
     */
    std::size_t hash() const;

  private:
    std::shared_ptr<const FusionCategory> cat_;
    std::vector<Sector> sectors_;
    std::vector<int64_t> multiplicities_;
  };

  /**
   * @brief One left-nested fusion order of a side, $((l_0 l_1) l_2) \cdots$.
   * @details A side of $n$ legs has $n - 1$ vertices, one per fusion, and the last vertex produces
   * the coupled charge.  `inner[k]` is the outcome of vertex $k$, for $k < n - 2$, and
   * `copies[k]` is the vertex of that fusion the tree realises, in $[0, N^{c_k}_{c_{k-1} l_k})$.
   *
   * The sectors are the charges the legs contribute, so a tree never carries a direction flag and
   * `validate` reads the tree alone.  A tree holds no category: it is data, it copies and compares
   * without a virtual call, and the category is an argument of every function that reads it.
   *
   * The side with no legs is the unit.  Its tree has no sector, no coupling and no copy, and it
   * couples to the trivial charge: a tensor with one side empty has the blocks of its other side.
   */
  struct FusionTree
  {
    std::vector<Sector> sectors;
    std::vector<Sector> inner;
    std::vector<int64_t> copies;
    Sector coupled;

    // Number of legs on the side.
    int64_t size() const { return static_cast<int64_t>(sectors.size()); }

    /**
     * @brief The couplings $c_0, \ldots, c_{n-1}$, where $c_k$ couples legs $0$ to $k$.
     * @return $n$ entries: the first leg, then `inner`, then `coupled`.
     */
    std::vector<Sector> chain() const;

    bool operator==(const FusionTree &other) const;
    bool operator!=(const FusionTree &other) const { return !(*this == other); }
    bool operator<(const FusionTree &other) const;

    std::size_t hash() const;
  };

  /**
   * @brief Check a tree against a category.
   * @param category The category.
   * @param tree The tree.
   * @throws std::invalid_argument When the tree has the wrong number of couplings or copies, when a
   * vertex fuses to a charge the category does not allow, or when a copy index is out of range.
   * @details One check for the whole engine.  It is not a constructor: the engine builds trees in
   * inner loops and validates them where a mistake would be made.
   */
  void validate(const FusionCategory &category, const FusionTree &tree);

  /**
   * @brief The charges a side can couple to.
   * @param category The category.
   * @param legs The legs of the side, in order.
   * @return Every sector some tree over the side couples to, in the order the fusion outcomes of the
   * category produce them.  The side with no legs couples to the trivial charge only.
   */
  std::vector<Sector> reachable_charges(const FusionCategory &category, const std::vector<Leg> &legs);

  /**
   * @brief Every tree over a side that couples to one charge.
   * @param category The category.
   * @param legs The legs of the side, in order; every block of every leg is a choice.
   * @param coupled The charge the trees couple to.
   * @return The trees, in the order the blocks of the legs and the fusion outcomes are explored, so
   * that the order is a function of the category data alone.
   */
  std::vector<FusionTree> fusion_trees(const FusionCategory &category,
                                       const std::vector<Leg> &legs,
                                       const Sector &coupled);

  // The sector as it is written in a message, `(0,)` for one component and `(0, 0)` for two.
  std::string render(const Sector &sector);
  std::string render_list(const std::vector<Sector> &sectors);
  std::string render_copies(const std::vector<int64_t> &values);

} // namespace sym
