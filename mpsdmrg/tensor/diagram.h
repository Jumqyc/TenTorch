#pragma once
#include <complex>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "../symmetry/tree/tree.h"

namespace sym {

// A splitting tree in TensorKit's normal form: the sector of every leg, one duality flag per leg,
// the coupled charge of every inner vertex, the copy label of every vertex, and the sector the
// tree couples to.  The leg sectors are the ones the engine's chain already carries, dualised
// where the leg carries a dual.
struct PairTree {
  std::vector<Sector> uncoupled;
  std::vector<bool> isdual;
  std::vector<Sector> inner;
  std::vector<int64_t> vertices;
  Sector coupled;

  bool operator<(const PairTree& other) const {
    return std::tie(uncoupled, isdual, inner, vertices, coupled) <
           std::tie(other.uncoupled, other.isdual, other.inner, other.vertices, other.coupled);
  }

  bool operator==(const PairTree& other) const {
    return std::tie(uncoupled, isdual, inner, vertices, coupled) ==
           std::tie(other.uncoupled, other.isdual, other.inner, other.vertices, other.coupled);
  }
};

// How the legs of a two-operand contraction line up: the operand axes the two glue along, in the
// left operand's leg order, and the output axis of every leg of each operand, minus one for a leg
// that is glued instead of free.
struct Gluing {
  std::vector<int64_t> left;
  std::vector<int64_t> right;
  std::vector<int64_t> out_left;
  std::vector<int64_t> out_right;

  bool operator<(const Gluing& other) const {
    return std::tie(left, right, out_left, out_right) <
           std::tie(other.left, other.right, other.out_left, other.out_right);
  }

  bool operator==(const Gluing& other) const {
    return std::tie(left, right, out_left, out_right) ==
           std::tie(other.left, other.right, other.out_left, other.out_right);
  }
};

// The diagram tree of one operand: its fusion tree, on the legs whose charges it holds.
PairTree pair_tree(const std::vector<Leg>& legs, const FusionTree& tree);

// The leg alignment of a two-operand contraction, read off the label every leg carries in each
// operand and in the output.  Args:
//   left_labels, right_labels: the label of every operand leg, in leg order.
//   out_labels: the label of every output leg, in output order.
Gluing glue_legs(const std::vector<int64_t>& left_labels, const std::vector<int64_t>& right_labels,
                 const std::vector<int64_t>& out_labels);

// The categorical weight of one output tree: the pairing of the two operands glued along their
// shared legs with the output tree, evaluated from the associator, the braiding and the quantum
// dimensions alone.  The operands and the output are left-nested chains, so a tree describes each
// one completely.
//
// Args:
//   group: the fusion category the sectors belong to.
//   left, right, out: the trees of the two operands and of the output.
//   glue: the leg alignment of the three; its leg lists must be the ones the trees were built on.
//   device: the device the category symbols are built on.
//   weight: set to the weight when the block is evaluated, untouched otherwise.
// The leg alignment is an invariant of the caller: every leg of an operand is free and named by one
// output axis, or glued to one leg of the other operand carrying the opposite duality.  A layout that
// breaks that, and an output tree that does not name every free leg, throw rather than decline.  A
// recoupling that reaches no state is a zero of the weight, not a decline.
// Returns:
//   0 when the block is evaluated, otherwise the one shape the route cannot evaluate yet, so the
//   caller keeps its Clebsch-Gordan route:
//     1  a leg whose label the output drops without a partner of the other operand to glue it to
//     12 a closed diagram whose operands are not both uncharged
int diagram_weight(const std::shared_ptr<FusionCategory>& group, const PairTree& left,
                   const PairTree& right, const PairTree& out, const Gluing& glue,
                   const c10::Device& device, std::complex<double>& weight);

// Drop the cached weights, the cached diagrams and the decoded symbol entries.
void clear_weight_cache();

// The work the route did, so a probe can say where a contraction spends its time.
struct RouteStats {
  std::size_t weight_calls = 0;    // weights the callers asked for
  std::size_t weight_hits = 0;     // of those, answered from the weight table
  std::size_t diagram_misses = 0;  // operand halves built from scratch
  std::size_t entry_calls = 0;     // symbol entry maps decoded
  std::size_t entry_values = 0;    // nonzero entries those maps held
  std::size_t move_calls = 0;      // chains brought into an order
  std::size_t move_states = 0;     // states those chains produced
  std::size_t weigh_calls = 0;     // output trees weighed
  std::size_t weigh_pairs = 0;     // tree triples the weighing compared
};

// The counters since they were last reset.
RouteStats route_stats();

// Set every counter back to zero.
void reset_route_stats();

}  // namespace sym
