// The SU(2) group, with its sectors labelled by twice the spin.
#pragma once

#include "../group.h"
#include "su2_cg.h"

namespace sym {

class SU2Symmetry : public Group {
 public:
  SU2Symmetry();
  bool is_valid_sector(const Sector& a) const override;
  Sector dual_sector(const Sector& a) const override;
  std::vector<Sector> fusion_outcomes(const Sector& a, const Sector& b) const override;
  int64_t sector_dim(const Sector& a) const override;
  at::Tensor fusion_tensor(const Sector& a, const Sector& b, const Sector& c,
                           const c10::Device& device, int64_t copy = 0) const override;
  double dual_sign(const Sector& sector) const override;
  double dual_phase(const Sector& sector) const override;
  at::Tensor duality_matrix(const Sector& a, const c10::Device& device) const override;
  at::Tensor invariant_form(const Sector& a, const c10::Device& device) const override;
  int64_t reference_index(const Sector& a) const override;
  c10::ScalarType fusion_dtype() const override;
  bool is_same_symmetry(const FusionCategory& other) const override;
  void clear_caches() const override;
};


// Bind SU(2) and its Clebsch-Gordan tables.
void bind_su2(py::module_& m);

}  // namespace sym
