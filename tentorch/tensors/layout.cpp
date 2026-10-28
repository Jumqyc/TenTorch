#include "layout.h"

#include <algorithm>

namespace sym
{

  int64_t block_of(const Leg &leg, const Sector &sector)
  {
    const std::vector<Sector> &sectors = leg.sectors();
    const auto found = std::lower_bound(sectors.begin(), sectors.end(), sector);
    if (found == sectors.end() || *found != sector)
      throw std::invalid_argument("a leg of " + std::to_string(leg.nblocks()) +
                                  " blocks carries none in sector " + render(sector));
    return found - sectors.begin();
  }

  SideLayout layout_of(const FusionCategory &category, const std::vector<Leg> &legs, const Sector &root)
  {
    SideLayout layout;
    layout.trees = fusion_trees(category, legs, root);
    layout.offsets.push_back(0);
    for (const FusionTree &tree : layout.trees)
    {
      // One row per combination of the multiplicities the tree's legs carry.
      int64_t rows = 1;
      for (size_t axis = 0; axis < legs.size(); ++axis)
        rows *= legs.at(axis).multiplicity(block_of(legs.at(axis), tree.sectors.at(axis)));
      layout.offsets.push_back(layout.offsets.back() + rows);
    }
    return layout;
  }

  std::vector<BlockKey> block_keys(const FusionCategory &category,
                                   const std::vector<Leg> &codomain,
                                   const std::vector<Leg> &domain,
                                   const Sector &qtotal)
  {
    const std::vector<Sector> out_roots = reachable_charges(category, codomain);
    const std::vector<Sector> in_roots = reachable_charges(category, domain);
    std::vector<BlockKey> keys;
    for (const Sector &r : out_roots)
      for (const Sector &s : in_roots)
        for (int64_t copy = 0; copy < category.n_symbol(r, category.dual(s), qtotal); ++copy)
          keys.push_back(BlockKey{r, s, copy});
    return keys;
  }

} // namespace sym
