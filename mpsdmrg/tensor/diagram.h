// The tree moves of the engine: bend, fold, cycle, braid and the pair permutation.
#pragma once

#include <complex>
#include <vector>

#include "tensor.h"

// The moves that repartition a tensor map's legs between its two sides, each with the coefficient it
// costs: bending one leg past the other side, and the repeated bend that leaves a given number of
// legs on the codomain.


#include <complex>
#include <vector>

namespace sym::engine {

// A tree pair together with the coefficient that reaches it from the pair the move started at.
struct PairTerm {
  FusionTreePair pair;
  std::complex<double> coefficient;
};

// One side of a tree pair together with the coefficient that reaches it from the tree the move
// started at.
struct TreeTerm {
  FusionTree tree;
  std::complex<double> coefficient;
};

// Bends the last codomain leg onto the domain side, where it joins as the conjugate of its sector.
// This is the move that turns a leg pointing into the tensor into a leg pointing out of it.
// Args:
//   group: the category all sectors belong to.
//   pair: a tree pair with at least one codomain leg, both sides coupled to the same sector.
//   device: where the associator is read from.
// Returns:
//   One term per copy the new vertex can carry; their coefficients are the B symbol.
// Raises:
//   std::invalid_argument: if the codomain side has no leg to bend.
std::vector<PairTerm> bend_right(const FusionCategory& group, const FusionTreePair& pair,
                                 const c10::Device& device);

// Bends the last domain leg onto the codomain side, the mirror move of `bend_right`.
// Args:
//   group: the category all sectors belong to.
//   pair: a tree pair with at least one domain leg, both sides coupled to the same sector.
//   device: where the associator is read from.
// Returns:
//   One term per copy the new vertex can carry; their coefficients are the A symbol.
// Raises:
//   std::invalid_argument: if the domain side has no leg to bend.
std::vector<PairTerm> bend_left(const FusionCategory& group, const FusionTreePair& pair,
                                const c10::Device& device);

// Recouples the codomain side so that its first leg is split off: the resulting pairs carry one
// codomain leg fewer and one domain leg more, the extra one being the conjugate of that first leg.
// This is the move that changes the bracketing of a side, and with it which legs sit next to which.
// Args:
//   group: the category all sectors belong to.
//   pair: a tree pair with at least one codomain leg.
//   device: where the associator and the A symbol are read from.
// Returns:
//   One term per split-off coupling and per copy of the new vertex.
// Raises:
//   std::invalid_argument: if the codomain side has no leg, or the category couples a fusion in
//     more than one way (the move then carries a matrix rather than a coefficient).
std::vector<PairTerm> fold_right(const FusionCategory& group, const FusionTreePair& pair,
                                 const c10::Device& device);

// The mirror of `fold_right`: recouples the domain side so that its first leg is split off.
// Args:
//   group: the category all sectors belong to.
//   pair: a tree pair with at least one domain leg.
//   device: where the associator and the A symbol are read from.
// Returns:
//   One term per split-off coupling and per copy of the new vertex.
// Raises:
//   std::invalid_argument: if the domain side has no leg.
std::vector<PairTerm> fold_left(const FusionCategory& group, const FusionTreePair& pair,
                                const c10::Device& device);

// Rotates the legs of a pair by one place, which is the move that brings an arbitrary leg order
// into the one a permutation asks for.
// Args:
//   group: the category all sectors belong to.
//   pair: a tree pair to rotate.
//   device: where the associator and the symbols are read from.
// Returns:
//   The terms that reach the rotated pair.
std::vector<PairTerm> cycle_clockwise(const FusionCategory& group, const FusionTreePair& pair,
                                      const c10::Device& device);

// Rotates the legs of a pair by one place in the other direction.
// Args:
//   group: the category all sectors belong to.
//   pair: a tree pair to rotate.
//   device: where the associator and the symbols are read from.
// Returns:
//   The terms that reach the rotated pair.
std::vector<PairTerm> cycle_anticlockwise(const FusionCategory& group, const FusionTreePair& pair,
                                          const c10::Device& device);

// Splits the first codomain leg off a tree, the associator product that recouples a side.
// Args:
//   group: the category all sectors belong to.
//   longer: the tree to recouple, with at least one leg.
//   shorter: a tree over the legs of `longer` except the first, at the coupling the split leaves.
//   device: where the associator is read from.
// Returns:
//   The coefficient of that recoupling, zero when the two trees are not related by it.
std::complex<double> multi_associator(const FusionCategory& group, const FusionTree& longer,
                                      const FusionTree& shorter, const c10::Device& device);

// Splits the first leg off a tree: every tree over the remaining legs that the split can reach.
// Args:
//   group: the category all sectors belong to.
//   tree: the tree to split, with at least one leg.
//   device: where the associator is read from.
// Returns:
//   One term per tree over the remaining legs, with the coefficient of `multi_associator`.
// Raises:
//   std::invalid_argument: if the tree has no leg.
std::vector<TreeTerm> multi_fmove(const FusionCategory& group, const FusionTree& tree,
                                  const c10::Device& device);

// The mirror of `multi_fmove`: prepends `a` to a tree and reports every tree the fusion can reach.
// Args:
//   group: the category all sectors belong to.
//   a: the sector the new first leg carries.
//   c: the charge the new tree has to couple to.
//   tree: the tree the leg is prepended to.
//   isduala: whether the new first leg is written in the conjugate basis.
//   device: where the associator is read from.
// Returns:
//   One term per tree over the new leg list, with the coefficient of the recoupling.
std::vector<TreeTerm> multi_fmove_inv(const FusionCategory& group, const Sector& a, const Sector& c,
                                      const FusionTree& tree, bool isduala,
                                      const c10::Device& device);

// Bends legs until the codomain side carries `keep` of them.
// Args:
//   group: the category all sectors belong to.
//   pair: a tree pair to repartition.
//   keep: how many legs the codomain side is to carry, between 0 and the leg count.
//   device: where the associator is read from.
// Returns:
//   The terms that reach a pair of that shape, with the product of the bend coefficients.
// Raises:
//   std::out_of_range: if `keep` is not between 0 and the leg count.
std::vector<PairTerm> repartition(const FusionCategory& group, const FusionTreePair& pair,
                                  int64_t keep, const c10::Device& device);

// The component array of a tree pair: the two tree tensors contracted on the coupled sector they
// share, laid out as the codomain legs followed by the domain legs. This is what a coefficient of a
// move has to reproduce.
// Args:
//   group: the category all sectors belong to.
//   pair: a tree pair, both sides coupled to the same sector.
//   device: where the Clebsch-Gordan tensors are read from.
// Returns:
//   An array of shape (codomain leg dimensions..., domain leg dimensions...).
at::Tensor pair_tensor(const FusionCategory& group, const FusionTreePair& pair,
                       const c10::Device& device, bool dress = false);

// Bind the moves above, so that a check can compare them with the component arrays.
void bind_duality(py::module_& m);

}  // namespace sym::engine

// The exchange of two neighbouring legs of a tree, and the permutation of a tree pair's legs that
// is built from it. This is the move that brings the legs of a contraction into the order the
// multiplication needs.


#include <complex>
#include <vector>

namespace sym::engine {

// Exchanges the legs at zero-based positions `k` and `k + 1` of a tree, which costs a braiding
// symbol and, one vertex away from the top, an associator.
// Args:
//   group: the category all sectors belong to.
//   tree: the tree to braid, with at least two legs and `k + 1` inside it.
//   k: the zero-based position of the left leg of the exchange.
//   inverse: whether to exchange them with the inverse braiding.
//   device: where the braiding and the associator are read from.
// Returns:
//   One term per copy the recoupled vertex can carry.
// Raises:
//   std::out_of_range: if `k` is not a position whose right neighbour exists.
std::vector<TreeTerm> swap_adjacent(const FusionCategory& group, const FusionTree& tree, int64_t k,
                                    bool inverse, const c10::Device& device);

// Permutes the legs of a tree into a given order by a sequence of exchanges.
// Args:
//   group: the category all sectors belong to.
//   tree: the tree to permute.
//   order: the sectors the new tree carries, as one-based positions of the old tree, in the order
//     they are to appear.
//   device: where the braiding and the associator are read from.
// Returns:
//   The terms that reach the permuted trees, with the product of the exchange coefficients.
// Raises:
//   std::invalid_argument: if `order` is not a permutation of one to the leg count.
std::vector<TreeTerm> braid_tree(const FusionCategory& group, const FusionTree& tree,
                                 const std::vector<int64_t>& order, const c10::Device& device);

// Permutes the legs of a tree pair, which may also move legs between its two sides.
// Args:
//   group: the category all sectors belong to.
//   pair: a tree pair whose legs are numbered codomain first, then domain.
//   keep: how many legs the codomain side carries afterwards.
//   order: the legs the result carries, as one-based positions of the pair's legs, codomain first.
//   device: where the moves read their symbols from.
// Returns:
//   The terms that reach the permuted pairs.
std::vector<PairTerm> permute_pair(const FusionCategory& group, const FusionTreePair& pair,
                                   int64_t keep, const std::vector<int64_t>& order,
                                   const c10::Device& device);

// Bind the permutation moves above, so that a check can compare them with TensorKit's.
void bind_braid(py::module_& m);

}  // namespace sym::engine
