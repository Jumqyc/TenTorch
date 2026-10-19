#pragma once
#include <memory>

#include <ATen/ATen.h>

#include <array>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "../group.h"

namespace sym::su3 {

/*
 * The SU(3) representation data: the Dynkin labels (p, q) of a sector, its weights, its
 * fusion and its invariant couplings.  The weights are traceless three-vectors whose
 * coordinates are multiples of a sixth, kept as integers so that they are exact keys.
 */

// The dimension of the sector, (p + 1) (q + 1) (p + q + 2) / 2.
int64_t dimension(const Sector& sector);

// The weights of the sector as exact integer six-vectors, memoised.
const std::vector<std::array<int64_t, 3>>& weight_keys(const Sector& sector);

// The Littlewood-Richardson outcomes of first (x) second with their multiplicities.
std::map<Sector, int64_t> fusion(const Sector& first, const Sector& second);

// The invariant coupling of first (x) second -> third, in the shape (d1, d2, d3) with the
// third factor in its conjugate basis; an empty tensor when the triple does not fuse.
at::Tensor clebsch_copies(const Sector& first, const Sector& second, const Sector& third,
                          const c10::Device& device);

at::Tensor clebsch(const Sector& first, const Sector& second, const Sector& third,
                   const c10::Device& device);

// The invariant form of the sector with its conjugate in the shape (d, d).
at::Tensor metric(const Sector& sector, const c10::Device& device);

// Drop the memo tables of the representation data.
void clear_caches();

}  // namespace sym::su3

namespace sym {

class SU3Symmetry : public Group {
 public:
  SU3Symmetry();
  bool is_valid_sector(const Sector& a) const override;
  Sector dual_sector(const Sector& a) const override;
  std::vector<Sector> fusion_outcomes(const Sector& a, const Sector& b) const override;
  int64_t sector_dim(const Sector& a) const override;
  // SU(3) pairs the associators of a vertex with multiplicity into matrices over its copy spaces,
  // which the diagram weight contracts copy by copy.
  at::Tensor fusion_tensor(const Sector& a, const Sector& b, const Sector& c,
                           const c10::Device& device, int64_t copy = 0) const override;
  int64_t n_symbol(const Sector& a, const Sector& b, const Sector& c) const override;
  at::Tensor duality_matrix(const Sector& a, const c10::Device& device) const override;
  at::Tensor invariant_form(const Sector& a, const c10::Device& device) const override;
  int64_t reference_index(const Sector& a) const override;
  c10::ScalarType fusion_dtype() const override;
  double dual_sign(const Sector& sector) const override;
  double dual_phase(const Sector& sector) const override;
  bool is_same_symmetry(const FusionCategory& other) const override;
  void clear_caches() const override;
};

// The SU(3) group the engine runs on.
const std::shared_ptr<Group>& su3_symmetry();

// Bind the SU(3) representation data.
void bind_su3(py::module_& m);

}  // namespace sym
