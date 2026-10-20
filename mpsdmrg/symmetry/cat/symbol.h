// The symbols a fusion tree costs when one of its vertices is turned upside down: the Frobenius-Schur
// phase of a sector, and the A and B symbols that carry a splitting vertex into a fusion vertex.
#pragma once

#include "cat.h"

#include <ATen/ATen.h>

#include <complex>

namespace sym {

// The phase a bent line of this sector carries: the sign of the invariant form that realises the
// duality, which is the Frobenius-Schur indicator of a self-dual sector and zero for the others.
// Args:
//   group: the category the sector belongs to.
//   a: the sector of the bent line.
//   device: unused; the sign is a property of the sector, kept for symmetry with the symbols.
// Returns:
//   The phase, a unit complex number, real.
std::complex<double> frobenius_schur_phase(const FusionCategory& group, const Sector& a,
                                           const c10::Device& device);

// The symbol that turns the vertex (a b) -> c upside down, with row `N^c_ab` and column
// `N^b_{dual(a) c}`.
// Args:
//   group: the category the sectors belong to.
//   a, b: the sectors of the two legs that meet.
//   c: the sector of the coupled leg.
//   device: where the associator is read from.
// Returns:
//   A matrix of the two multiplicity counts; all its axes are one for a multiplicity-free fusion.
at::Tensor a_symbol(const FusionCategory& group, const Sector& a, const Sector& b, const Sector& c,
                    const c10::Device& device);

// The same for the vertex a -> (b c), with row `N^c_ab` and column `N^a_{c dual(b)}`.
// Args:
//   group: the category the sectors belong to.
//   a, b: the sectors of the two legs that meet.
//   c: the sector of the coupled leg.
//   device: where the associator is read from.
// Returns:
//   A matrix of the two multiplicity counts; all its axes are one for a multiplicity-free fusion.
at::Tensor b_symbol(const FusionCategory& group, const Sector& a, const Sector& b, const Sector& c,
                    const c10::Device& device);

}  // namespace sym
