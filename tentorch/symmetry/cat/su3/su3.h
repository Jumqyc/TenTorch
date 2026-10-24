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

/**
 * @brief SU(3) in the basis of the Dynkin labels $(p, q)$.
 * @details The octet fuses with itself to two independent octets, so a vertex carries a copy index
 * and a recoupling move is a matrix over those copies rather than a phase.
 */
class SU3Symmetry : public Group {
 public:
  explicit SU3Symmetry(c10::Device device = default_device());

  Sector dual(Sector a) const override;
  Sector trivial() const override;
  int dim(Sector a) const override;
  c10::ScalarType dtype() const override;

 protected:
  std::vector<Outcome> compute_outcomes(const Sector& a, const Sector& b) const override;
  at::Tensor compute_clebsch_gordan(const Sector& a, const Sector& b, const Sector& c,
                                    int64_t copy) const override;
};

// The SU(3) group the engine runs on.
const std::shared_ptr<Group>& su3_symmetry();

// Bind the SU(3) representation data.
void bind_su3(py::module_& m);

}  // namespace sym
