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
//   shared_dual: one entry per glued pair, 1 when the left operand's leg carries the dual of its
//     charges; the pair's duality matrix then sits on the left instead of the right.
//   out_left_axis, out_right_axis: the output axis of every operand leg, -1 when it is glued
//     instead of free.
//   device: the device the category symbols are built on.
// Returns:
//   True and `weight` when the block is evaluated.  False for a shape this routine does not
//   cover (a leg both operands keep, or a fusion multiplicity that pairs the F symbols into
//   matrices over copies), so the caller can fall back to the dense route.
bool diagram_weight(const std::shared_ptr<FusionCategory>& group,
                    const std::vector<Sector>& left_legs, const std::vector<Sector>& left_inner,
                    const Sector& left_coupled, const std::vector<Sector>& right_legs,
                    const std::vector<Sector>& right_inner, const Sector& right_coupled,
                    const std::vector<Sector>& out_legs, const std::vector<Sector>& out_inner,
                    const Sector& out_coupled, const std::vector<int64_t>& shared_left,
                    const std::vector<int64_t>& shared_right,
                    const std::vector<int8_t>& shared_dual,
                    const std::vector<int64_t>& out_left_axis,
                    const std::vector<int64_t>& out_right_axis, const c10::Device& device,
                    std::complex<double>& weight);

}  // namespace sym
