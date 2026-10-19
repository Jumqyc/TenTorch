#pragma once
#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

#include "../symmetry/cat/cat.h"

namespace sym {

// The categorical weight of one output tree: the pairing of the two operands glued along their
// shared legs with the output tree, evaluated from the associator, the braiding and the quantum
// dimensions alone.  The operands and the output are left-nested chains, so a leg list, the
// coupled charge of every inner vertex and the total charge describe each one completely.
//
// Args:
//   group: the fusion category the sectors belong to.
//   left_legs, right_legs: the sector of every leg of the two operands, in leg order.
//   left_inner, right_inner: the coupled charge of every inner vertex of the operand chain,
//     `legs.size() - 2` of them; the coupled charge of the last vertex is `coupled`.
//   left_coupled, right_coupled: the sector every operand leg fuses to.
//   out_legs, out_inner, out_coupled: the same three descriptions of the output chain.
//   shared_left, shared_right: the operand axes the two glue along, in the left's leg order.
//   left_dual, right_dual, out_dual: one flag per leg of each operand and of the output, 1 when
//     that leg carries the dual of its charges.
//   out_left_axis, out_right_axis: the output axis of every operand leg, -1 when it is glued
//     instead of free.
//   device: the device the category symbols are built on.
// Returns:
//   True and `weight` when the block is evaluated.  False for a shape this routine does not
//   cover (a leg both operands keep while the output names it), so the caller can fall back to
//   the dense route.
bool diagram_weight(const std::shared_ptr<FusionCategory>& group,
                    const std::vector<Sector>& left_legs, const std::vector<Sector>& left_inner,
                    const Sector& left_coupled, const std::vector<Sector>& right_legs,
                    const std::vector<Sector>& right_inner, const Sector& right_coupled,
                    const std::vector<Sector>& out_legs, const std::vector<Sector>& out_inner,
                    const Sector& out_coupled, const std::vector<int64_t>& left_copies,
                    const std::vector<int64_t>& right_copies,
                    const std::vector<int64_t>& out_copies,
                    const std::vector<int8_t>& left_dual, const std::vector<int8_t>& right_dual,
                    const std::vector<int8_t>& out_dual,
                    const std::vector<int64_t>& shared_left,
                    const std::vector<int64_t>& shared_right,
                    const std::vector<int64_t>& out_left_axis,
                    const std::vector<int64_t>& out_right_axis, const c10::Device& device,
                    std::complex<double>& weight);

// Drop the cached per-tree weights.
void clear_weight_cache();

// The guard that declined the last call to ``diagram_weight``, 0 when no guard did.
int diagram_weight_decline();

}  // namespace sym
