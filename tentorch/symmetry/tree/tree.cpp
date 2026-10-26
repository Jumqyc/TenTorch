// The legs, the fusion trees over them and the enumeration of the blocks of a tensor.
#include "tree.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sym
{

  // The sector as it is written in a message, `(0,)` for one component and `(0, 0)` for two.
  std::string render(const Sector &sector)
  {
    std::string out = "(";
    for (int64_t i = 0; i < sector.size(); ++i)
    {
      if (i != 0)
      {
        out += ", ";
      }
      out += std::to_string(sector[i]);
    }
    if (sector.size() == 1)
    {
      out += ",";
    }
    return out + ")";
  }

  std::string render_list(const std::vector<Sector> &sectors)
  {
    std::string out = "[";
    for (size_t i = 0; i < sectors.size(); ++i)
    {
      if (i != 0)
      {
        out += ", ";
      }
      out += render(sectors[i]);
    }
    return out + "]";
  }

  std::string render_copies(const std::vector<int64_t> &values)
  {
    std::string out = "[";
    for (size_t i = 0; i < values.size(); ++i)
    {
      if (i != 0)
      {
        out += ", ";
      }
      out += std::to_string(values[i]);
    }
    return out + "]";
  }

  namespace
  {
    void hash_combine(std::size_t &seed, std::size_t value)
    {
      seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    }

    std::size_t hash_sector(const Sector &sector)
    {
      std::size_t seed = 0;
      for (int64_t value : sector)
      {
        hash_combine(seed, std::hash<int64_t>{}(value));
      }
      return seed;
    }

    // The blocks of a leg in the order the sector sorts, which is the order the leg stores.
    std::vector<int64_t> sector_order(const std::vector<Sector> &sectors)
    {
      std::vector<int64_t> order(sectors.size());
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(),
                [&sectors](int64_t a, int64_t b) { return sectors[a] < sectors[b]; });
      return order;
    }
  } // namespace

  Leg::Leg(std::shared_ptr<const FusionCategory> cat,
           std::vector<Sector> sectors,
           std::vector<int64_t> multiplicities)
      : cat_(std::move(cat)), sectors_(std::move(sectors)), multiplicities_(std::move(multiplicities))
  {
    if (cat_ == nullptr)
    {
      throw std::invalid_argument("A leg needs a category.");
    }
    if (sectors_.size() != multiplicities_.size())
    {
      throw std::invalid_argument("A leg needs one multiplicity per sector, got " +
                                  std::to_string(multiplicities_.size()) + " for " +
                                  std::to_string(sectors_.size()) + " sectors.");
    }
    for (int64_t multiplicity : multiplicities_)
    {
      if (multiplicity <= 0)
      {
        throw std::invalid_argument("A block multiplicity must be positive, got " +
                                    std::to_string(multiplicity) + ".");
      }
    }
    const std::vector<int64_t> order = sector_order(sectors_);
    std::vector<Sector> sorted_sectors;
    std::vector<int64_t> sorted_multiplicities;
    sorted_sectors.reserve(order.size());
    sorted_multiplicities.reserve(order.size());
    for (int64_t block : order)
    {
      sorted_sectors.push_back(std::move(sectors_[static_cast<size_t>(block)]));
      sorted_multiplicities.push_back(multiplicities_[static_cast<size_t>(block)]);
    }
    sectors_ = std::move(sorted_sectors);
    multiplicities_ = std::move(sorted_multiplicities);
    for (size_t block = 1; block < sectors_.size(); ++block)
    {
      if (sectors_[block] == sectors_[block - 1])
      {
        throw std::invalid_argument("A leg cannot hold the sector " + render(sectors_[block]) + " twice.");
      }
    }
    for (const Sector &sector : sectors_)
    {
      if (cat_->q_dim(sector) <= 0.0)
      {
        throw std::invalid_argument("The sector " + render(sector) + " has no components in " +
                                    cat_->name() + ".");
      }
    }
  }

  Leg Leg::trivial(int64_t multiplets, std::shared_ptr<const FusionCategory> cat)
  {
    if (multiplets <= 0)
    {
      throw std::invalid_argument("A trivial leg needs at least one component, got " +
                                  std::to_string(multiplets) + ".");
    }
    if (cat == nullptr)
    {
      throw std::invalid_argument("A trivial leg needs a category.");
    }
    return Leg(cat, {cat->trivial()}, {multiplets});
  }

  std::pair<Leg, std::vector<int64_t>> Leg::from_blocks(
      std::shared_ptr<const FusionCategory> cat,
      const std::vector<std::pair<Sector, int64_t>> &entries)
  {
    std::vector<Sector> sectors;
    std::vector<int64_t> multiplicities;
    sectors.reserve(entries.size());
    multiplicities.reserve(entries.size());
    for (const auto &entry : entries)
    {
      sectors.push_back(entry.first);
      multiplicities.push_back(entry.second);
    }
    Leg leg(cat, std::move(sectors), std::move(multiplicities));
    std::vector<int64_t> rank(entries.size());
    for (size_t k = 0; k < entries.size(); ++k)
    {
      const auto found = std::lower_bound(leg.sectors().begin(), leg.sectors().end(), entries[k].first);
      rank[k] = static_cast<int64_t>(found - leg.sectors().begin());
    }
    return {std::move(leg), std::move(rank)};
  }

  int64_t Leg::multiplets() const
  {
    return std::accumulate(multiplicities_.begin(), multiplicities_.end(), int64_t{0});
  }

  Leg Leg::fuse(const Leg &other) const
  {
    if (cat_ != other.cat_)
    {
      throw std::invalid_argument("Cannot fuse a leg of " + cat_->name() + " with one of " +
                                  other.cat_->name() + ".");
    }
    std::map<Sector, int64_t> fused;
    for (int64_t first = 0; first < nblocks(); ++first)
    {
      for (int64_t second = 0; second < other.nblocks(); ++second)
      {
        const int64_t weight = multiplicity(first) * other.multiplicity(second);
        for (const Outcome &outcome : cat_->outcomes(sector(first), other.sector(second)))
        {
          fused[outcome.sector] += weight * outcome.multiplicity;
        }
      }
    }
    std::vector<Sector> sectors;
    std::vector<int64_t> multiplicities;
    sectors.reserve(fused.size());
    multiplicities.reserve(fused.size());
    for (const auto &entry : fused)
    {
      sectors.push_back(entry.first);
      multiplicities.push_back(entry.second);
    }
    return Leg(cat_, std::move(sectors), std::move(multiplicities));
  }

  Leg Leg::conj() const
  {
    std::vector<Sector> duals;
    duals.reserve(sectors_.size());
    for (const Sector &sector : sectors_)
    {
      duals.push_back(cat_->dual(sector));
    }
    return Leg(cat_, std::move(duals), multiplicities_);
  }

  Leg Leg::capped(int64_t max_multiplets) const
  {
    if (max_multiplets <= 0)
    {
      throw std::invalid_argument("A cap must keep at least one multiplet, got " +
                                  std::to_string(max_multiplets) + ".");
    }
    std::vector<Sector> sectors;
    std::vector<int64_t> multiplicities;
    int64_t kept = 0;
    for (int64_t block = 0; block < nblocks() && kept < max_multiplets; ++block)
    {
      const int64_t take = std::min(multiplicity(block), max_multiplets - kept);
      sectors.push_back(sector(block));
      multiplicities.push_back(take);
      kept += take;
    }
    return Leg(cat_, std::move(sectors), std::move(multiplicities));
  }

  bool Leg::operator==(const Leg &other) const
  {
    return cat_ == other.cat_ && sectors_ == other.sectors_ && multiplicities_ == other.multiplicities_;
  }

  std::size_t Leg::hash() const
  {
    std::size_t seed = std::hash<std::string>{}(cat_->name());
    for (size_t block = 0; block < sectors_.size(); ++block)
    {
      hash_combine(seed, hash_sector(sectors_[block]));
      hash_combine(seed, std::hash<int64_t>{}(multiplicities_[block]));
    }
    return seed;
  }

  std::vector<Sector> FusionTree::chain() const
  {
    if (sectors.size() <= 1)
    {
      return sectors;
    }
    std::vector<Sector> out;
    out.reserve(sectors.size());
    out.push_back(sectors[0]);
    for (const Sector &inner_sector : inner)
    {
      out.push_back(inner_sector);
    }
    out.push_back(coupled);
    return out;
  }

  bool FusionTree::operator==(const FusionTree &other) const
  {
    return sectors == other.sectors && inner == other.inner && copies == other.copies &&
           coupled == other.coupled;
  }

  bool FusionTree::operator<(const FusionTree &other) const
  {
    if (sectors != other.sectors)
    {
      return sectors < other.sectors;
    }
    if (inner != other.inner)
    {
      return inner < other.inner;
    }
    if (copies != other.copies)
    {
      return copies < other.copies;
    }
    return coupled < other.coupled;
  }

  std::size_t FusionTree::hash() const
  {
    std::size_t seed = 0;
    for (const Sector &sector : sectors)
    {
      hash_combine(seed, hash_sector(sector));
    }
    for (const Sector &inner_sector : inner)
    {
      hash_combine(seed, hash_sector(inner_sector));
    }
    for (int64_t copy : copies)
    {
      hash_combine(seed, std::hash<int64_t>{}(copy));
    }
    hash_combine(seed, hash_sector(coupled));
    return seed;
  }

  void validate(const FusionCategory &category, const FusionTree &tree)
  {
    const int64_t legs = tree.size();
    if (legs <= 1)
    {
      if (!tree.inner.empty())
      {
        throw std::invalid_argument("A tree of " + std::to_string(legs) +
                                    " legs has no inner coupling, got " + render_list(tree.inner) + ".");
      }
      if (!tree.copies.empty())
      {
        throw std::invalid_argument("A tree of " + std::to_string(legs) + " legs has no vertex, got " +
                                    render_copies(tree.copies) + ".");
      }
      if (legs == 1 && tree.coupled != tree.sectors[0])
      {
        throw std::invalid_argument("A tree of one leg couples to " + render(tree.sectors[0]) +
                                    ", not to " + render(tree.coupled) + ".");
      }
      return;
    }
    if (static_cast<int64_t>(tree.inner.size()) != legs - 2)
    {
      throw std::invalid_argument("A tree of " + std::to_string(legs) + " legs has " +
                                  std::to_string(legs - 2) + " inner couplings, got " +
                                  std::to_string(tree.inner.size()) + ".");
    }
    if (static_cast<int64_t>(tree.copies.size()) != legs - 1)
    {
      throw std::invalid_argument("A tree of " + std::to_string(legs) + " legs has " +
                                  std::to_string(legs - 1) + " vertices, got " +
                                  std::to_string(tree.copies.size()) + ".");
    }
    for (int64_t vertex = 0; vertex + 1 < legs; ++vertex)
    {
      const Sector &left = vertex == 0 ? tree.sectors[0] : tree.inner[static_cast<size_t>(vertex - 1)];
      const Sector &right = tree.sectors[static_cast<size_t>(vertex + 1)];
      const Sector &out = vertex + 2 == legs ? tree.coupled : tree.inner[static_cast<size_t>(vertex)];
      const int64_t copies = category.n_symbol(left, right, out);
      if (copies == 0)
      {
        throw std::invalid_argument("The vertex " + render(left) + " x " + render(right) + " -> " +
                                    render(out) + " is not allowed by " + category.name() + ".");
      }
      if (tree.copies[static_cast<size_t>(vertex)] < 0 || tree.copies[static_cast<size_t>(vertex)] >= copies)
      {
        throw std::invalid_argument("Vertex " + std::to_string(vertex) + " has " + std::to_string(copies) +
                                    " copies, got copy " +
                                    std::to_string(tree.copies[static_cast<size_t>(vertex)]) + ".");
      }
    }
  }

  std::vector<Sector> reachable_charges(const FusionCategory &category, const std::vector<Leg> &legs)
  {
    std::vector<Sector> charges{category.trivial()};
    for (const Leg &leg : legs)
    {
      std::vector<Sector> next;
      std::set<Sector> seen;
      for (const Sector &charge : charges)
      {
        for (int64_t block = 0; block < leg.nblocks(); ++block)
        {
          for (const Outcome &outcome : category.outcomes(charge, leg.sector(block)))
          {
            if (seen.insert(outcome.sector).second)
            {
              next.push_back(outcome.sector);
            }
          }
        }
      }
      charges = std::move(next);
    }
    return charges;
  }

  std::vector<FusionTree> fusion_trees(const FusionCategory &category,
                                       const std::vector<Leg> &legs,
                                       const Sector &coupled)
  {
    const int64_t count = static_cast<int64_t>(legs.size());
    if (count == 0)
    {
      // The empty side is the unit, which couples to the trivial charge and to nothing else.
      return coupled == category.trivial() ? std::vector<FusionTree>{FusionTree{{}, {}, {}, coupled}}
                                           : std::vector<FusionTree>{};
    }
    std::vector<FusionTree> out;
    FusionTree tree;
    tree.sectors.resize(static_cast<size_t>(count));
    tree.inner.resize(static_cast<size_t>(count >= 2 ? count - 2 : 0));
    tree.copies.resize(static_cast<size_t>(count >= 2 ? count - 1 : 0));
    tree.coupled = coupled;

    // `previous` is the coupling of the legs already chosen, the trivial charge before the first.
    std::function<void(int64_t, const Sector &)> walk = [&](int64_t leg, const Sector &previous)
    {
      if (leg == 0)
      {
        for (int64_t block = 0; block < legs[0].nblocks(); ++block)
        {
          tree.sectors[0] = legs[0].sector(block);
          if (count == 1)
          {
            if (tree.sectors[0] == coupled)
            {
              out.push_back(tree);
            }
          }
          else
          {
            walk(1, tree.sectors[0]);
          }
        }
        return;
      }
      for (int64_t block = 0; block < legs[static_cast<size_t>(leg)].nblocks(); ++block)
      {
        const Sector &sector = legs[static_cast<size_t>(leg)].sector(block);
        tree.sectors[static_cast<size_t>(leg)] = sector;
        for (const Outcome &outcome : category.outcomes(previous, sector))
        {
          if (leg + 1 == count && outcome.sector != coupled)
          {
            continue;
          }
          if (leg + 1 < count)
          {
            tree.inner[static_cast<size_t>(leg - 1)] = outcome.sector;
          }
          for (int64_t copy = 0; copy < outcome.multiplicity; ++copy)
          {
            tree.copies[static_cast<size_t>(leg - 1)] = copy;
            if (leg + 1 == count)
            {
              out.push_back(tree);
            }
            else
            {
              walk(leg + 1, outcome.sector);
            }
          }
        }
      }
    };
    walk(0, category.trivial());
    return out;
  }

} // namespace sym
