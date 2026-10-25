// The SU(2) group, with its sectors labelled by twice the spin.
#pragma once

#include "../group.h"
#include "su2_cg.h"

namespace sym {

/**
 * @brief SU(2) in the basis of twice the spin.
 * @details Every fusion has multiplicity one, so the associator is the $6j$ symbol and the braiding
 * is the phase $(-1)^{j_1 + j_2 - J}$; both fall out of the Clebsch-Gordan tensor of the group layer.
 */
class SU2Symmetry : public Group {
 public:
  explicit SU2Symmetry(c10::Device device = default_device());

  Sector dual(Sector a) const override;
  Sector trivial() const override;
  double q_dim(Sector a) const override;
  double frobenius_schur_sign(const Sector& a) const override;
  c10::ScalarType dtype() const override;

 protected:
  std::vector<Outcome> compute_outcomes(const Sector& a, const Sector& b) const override;
  at::Tensor compute_clebsch_gordan(const Sector& a, const Sector& b, const Sector& c,
                                    int64_t copy) const override;
};

// The SU(2) group the engine runs on.
const std::shared_ptr<Group>& su2_symmetry();

// Bind SU(2) and its Clebsch-Gordan tables.

}  // namespace sym
