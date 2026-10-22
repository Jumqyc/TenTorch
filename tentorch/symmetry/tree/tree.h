// The fusion tree of a graded space: one side of a tensor, its legs, their dualities and the
// couplings between them.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../cat/cat.h"

#include <map>
#include <string>
#include <utility>
#include <vector>


namespace sym::engine {

// One leg of a fusion tree: the sector the leg carries, which is its own sector when the leg points
// along the tensor and the dual sector when it points against it.
struct LegCharge {
  Sector sector;
  bool isdual;

  bool operator==(const LegCharge& other) const {
    return sector == other.sector && isdual == other.isdual;
  }
  bool operator<(const LegCharge& other) const {
    return sector != other.sector ? sector < other.sector : isdual < other.isdual;
  }
};

// One side of a tensor: its legs, the coupling of every prefix of them, and the copy realised at
// every fusion vertex.
//
//   legs      ``sectors[i]`` is the sector leg ``i`` carries in the direction it points, so a leg
//             written in the conjugate basis already carries its dual sector and ``isdual[i]``.
//   inner     one coupling per prefix that is not the whole side: ``inner[k]`` couples the legs up
//             to and including leg ``k + 1``, so there are ``sectors.size() - 2`` of them.
//   vertices  the copy realised at every fusion vertex, ``sectors.size() - 1`` of them.
//   coupled   the charge of the whole side.
struct FusionTree {
  std::vector<Sector> sectors;
  std::vector<bool> isdual;
  std::vector<Sector> inner;
  std::vector<int64_t> vertices;
  Sector coupled;

  int64_t size() const { return static_cast<int64_t>(sectors.size()); }
  bool operator==(const FusionTree& other) const;
  bool operator!=(const FusionTree& other) const { return !(*this == other); }
  bool operator<(const FusionTree& other) const;
};

// A tensor block: the two sides of the map, which have to carry the same coupled charge.
struct FusionTreePair {
  FusionTree codomain;
  FusionTree domain;

  bool operator==(const FusionTreePair& other) const {
    return codomain == other.codomain && domain == other.domain;
  }
  bool operator!=(const FusionTreePair& other) const { return !(*this == other); }
  bool operator<(const FusionTreePair& other) const {
    return codomain != other.codomain ? codomain < other.codomain : domain < other.domain;
  }
};

// A tree written out as the fields a probe compares.
std::string repr(const FusionTree& tree);
std::string repr(const FusionTreePair& pair);

// The structural contract of a tree, which every move preserves. Throws std::invalid_argument on a
// tree a move built wrong, so a mistake surfaces where it is made rather than in a result.
void check_tree(const FusionTree& tree);

// The same, with the fusion structure: every coupling has to be an outcome of the pair below it and
// every vertex copy has to be realised by the category.
void check_tree(const FusionCategory& group, const FusionTree& tree);

// Every fusion tree over ``legs``, grouped by its coupled charge and enumerated in the order the
// category's fusion outcomes come in, so the order is a function of the category data alone.
// Every tree of a side, where each leg may carry any of the sectors its space lists.
std::map<Sector, std::vector<FusionTree>> trees_by_charge(
    const FusionCategory& group, const std::vector<std::vector<LegCharge>>& choices);

std::map<Sector, std::vector<FusionTree>> trees_by_charge(const FusionCategory& group,
                                                          const std::vector<LegCharge>& legs);

// Every tree pair whose two sides carry the same coupled charge: the block enumeration of a tensor
// with ``codomain`` and ``domain`` as its legs. Pairs of one charge come out codomain-major.
std::vector<FusionTreePair> tree_pairs(const FusionCategory& group,
                                       const std::vector<LegCharge>& codomain,
                                       const std::vector<LegCharge>& domain);

// The pairs of two sides whose legs may each carry any of several sectors.
std::vector<FusionTreePair> tree_pairs(const FusionCategory& group,
                                       const std::vector<std::vector<LegCharge>>& codomain,
                                       const std::vector<std::vector<LegCharge>>& domain);

// The tree pairs of one coupled charge, in the same order ``tree_pairs`` uses.
std::vector<FusionTreePair> tree_pairs_of(const std::vector<FusionTree>& codomain,
                                          const std::vector<FusionTree>& domain);

// The copy a fusion vertex realises, which is what sizes a block: the number of pairs of one charge.
int64_t vertex_copies(const FusionCategory& group, const FusionTree& tree, int64_t vertex);

}  // namespace sym::engine

// The charge-adapted legs and fusion trees of the engine: the block structure of a leg, the
// tree that fixes how its sectors fuse, and the dense Clebsch-Gordan expansion of a tree.
#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../cat/cat.h"

namespace sym {

// Charge-block key of one block: the block index on every leg, then the coupled sector of
// every inner vertex of the block's fusion tree.
struct BlockKey {
  std::vector<int64_t> legs;
  std::vector<Sector> inner;
  // The realized copy of every fusion vertex, one entry per vertex, empty when the key
  // predates the copies or the group couples every vertex at most once.
  std::vector<int64_t> copies;

  bool operator==(const BlockKey& other) const {
    return legs == other.legs && inner == other.inner && copies == other.copies;
  }
  bool operator!=(const BlockKey& other) const { return !(*this == other); }
  bool operator<(const BlockKey& other) const {
    return legs != other.legs ? legs < other.legs
                              : (inner != other.inner ? inner < other.inner : copies < other.copies);
  }
};

class Leg;
class FusionTree;

// Raise Python's AssertionError, the exception type the Python API asserts with.
[[noreturn]] void assertion_failed(const std::string& message);

// Whether two groups describe the same symmetry.
inline bool same_group(const FusionCategory& a, const FusionCategory& b) { return a.is_same_symmetry(b); }

// The charge structure of one leg: its group, one block per sector and the block boundaries.
// The blocks are kept sorted by sector, so equal legs hold identically laid out storage.
class Leg {
 public:
  // Args:
  //   group: the group of the charges.
  //   sectors: the sector of every block, all valid and distinct.
  //   slices: the block boundaries, len(sectors) + 1 of them, starting at zero and strictly
  //       increasing.
  //   qconj: +1 if the leg carries its charges, -1 if it carries their duals.
  Leg(std::shared_ptr<FusionCategory> group, std::vector<Sector> sectors, std::vector<int64_t> slices,
      int64_t qconj = 1);

  // One block of `dim` indices in the trivial sector.
  // Args:
  //   dim: number of indices on the leg.
  //   group: the group of the leg; the group without symmetry when it is null.
  //   qconj: +1 if the leg carries its charges, -1 if it carries their duals.
  static Leg trivial(int64_t dim, std::shared_ptr<FusionCategory> group = nullptr, int64_t qconj = 1);

  // A leg from `(sector, block dimension)` entries, sorted by sector, together with, for
  // entry `k`, the block index it ends up at.
  static std::pair<Leg, std::vector<int64_t>> from_blocks(
      std::shared_ptr<FusionCategory> group, const std::vector<std::pair<Sector, int64_t>>& entries,
      int64_t qconj = 1);

  // Number of indices on the leg.
  int64_t dim() const { return slices_.back(); }
  // Number of charge blocks on the leg.
  int64_t nblocks() const { return static_cast<int64_t>(sectors_.size()); }
  // Dimension of one charge block: the sector dimension times its multiplicity.
  int64_t block_dim(int64_t qi) const;
  // Multiplicity of one charge block on the leg.
  int64_t multiplicity(int64_t qi) const;
  // Number of multiplets on the leg: the sum of its block multiplicities.  This is the bond
  // dimension of the QSpace convention, while `dim()` counts the states instead, every
  // multiplet holding `sector_dim` of them.
  int64_t multiplets() const;
  // Sector of one charge block; the stored sector dualised for a leg that carries duals.
  Sector sector(int64_t qi, bool dual = false) const;
  // The leg carrying every outcome of fusing this leg's sectors with `other`'s, one block
  // per outcome, its multiplicity the n-symbol weighted sum of the pair multiplicities.
  Leg fuse(const Leg& other) const;
  // The same leg with the direction of its charges reversed.
  Leg conj() const;

  const std::shared_ptr<FusionCategory>& group() const { return group_; }
  const std::vector<Sector>& sectors() const { return sectors_; }
  const std::vector<int64_t>& slices() const { return slices_; }
  int64_t qconj() const { return qconj_; }

  std::string repr() const;
  bool operator==(const Leg& other) const;
  bool operator!=(const Leg& other) const { return !(*this == other); }
  std::size_t hash() const;

 private:
  std::shared_ptr<FusionCategory> group_;
  std::vector<Sector> sectors_;
  std::vector<int64_t> slices_;
  int64_t qconj_ = 1;
};

// One left-nested fusion order ``((l0 l1) l2) ...`` of a tensor's legs: the sector of every
// leg, the coupled sector of every inner vertex and the sector they all fuse to.
class FusionTree {
 public:
  // Args:
  //   group: the group of the sectors.
  //   sectors: the sector of every leg, in leg order, dualised for a leg carrying duals.
  //   inner: the coupled sector of every inner vertex, len(sectors) - 2 of them.
  //   coupled: the sector every leg fuses to.
  FusionTree(std::shared_ptr<FusionCategory> group, std::vector<Sector> sectors,
             std::vector<Sector> inner, Sector coupled, std::vector<int64_t> copies = {});

  // The tree stored in one charge-block key: one sector per leg, dualised where the leg
  // carries duals, and the couplings that follow the leg indices in the key.
  static FusionTree from_blocks(const std::shared_ptr<FusionCategory>& group,
                                const std::vector<Leg>& legs, const BlockKey& key,
                                const Sector& qtotal);

  const std::shared_ptr<FusionCategory>& group() const { return group_; }
  const std::vector<Sector>& sectors() const { return sectors_; }
  const std::vector<Sector>& inner() const { return inner_; }
  const Sector& coupled() const { return coupled_; }
  const std::vector<int64_t>& copies() const { return copies_; }

  // The couplings ``c_0, ..., c_{n-1}``, where ``c_k`` couples legs 0 to k.
  std::vector<Sector> chain() const;

  std::string repr() const;
  bool operator==(const FusionTree& other) const;
  bool operator!=(const FusionTree& other) const { return !(*this == other); }
  std::size_t hash() const;

 private:
  std::shared_ptr<FusionCategory> group_;
  std::vector<Sector> sectors_;
  std::vector<Sector> inner_;
  Sector coupled_;
  std::vector<int64_t> copies_;
};

// Every left-nested fusion of `sectors` into `coupled`, one tree per allowed combination of
// the prefix couplings, in the order the couplings are explored.
std::vector<FusionTree> fusion_trees(const std::shared_ptr<FusionCategory>& group,
                                     const std::vector<Sector>& sectors, const Sector& coupled);

// A sector as Python renders it, `(0,)`, and a sequence of them, `((0,), (2,))`.
std::string sector_repr(const Sector& sector);
std::string sectors_repr(const std::vector<Sector>& sectors);

// The shape algebra a model layer needs without naming a leg: the leg of a bare dimension, the
// leg of one multiplet, a bond leg from (sector, dimension) entries, the fusion of two legs, and
// a leg capped to the largest number of multiplets that fits.
Leg site_leg(const std::shared_ptr<FusionCategory>& group, int64_t dim, int64_t qconj = 1);
Leg multiplet_leg(const std::shared_ptr<FusionCategory>& group, const Sector& sector,
                  int64_t qconj = 1);
Leg bond_leg(const std::shared_ptr<FusionCategory>& group,
             std::vector<std::pair<Sector, int64_t>> entries, int64_t qconj = 1);
Leg fuse_legs(const Leg& first, const Leg& second);
Leg capped_leg(const Leg& leg, int64_t max_multiplets);

// Bind the names of this module to python.
void bind_tree(py::module_& m);

}  // namespace sym
