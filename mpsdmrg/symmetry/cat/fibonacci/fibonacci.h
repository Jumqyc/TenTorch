// The Fibonacci fusion category: two simple objects, 1 and tau, with tau (x) tau = 1 + tau.
// It is the smallest category that is not a group, so it is what a categorical interface has to
// support beyond a Clebsch-Gordan calculus.
#pragma once

#include "../cat.h"

namespace sym {

class FibonacciCategory : public FusionCategory {
 public:
  FibonacciCategory();
  bool is_valid_sector(const Sector& a) const override;
  Sector dual_sector(const Sector& a) const override;
  std::vector<Sector> fusion_outcomes(const Sector& a, const Sector& b) const override;
  int64_t sector_dim(const Sector& a) const override;
  double quantum_dim(const Sector& a) const override;
  at::Tensor fusion_tensor(const Sector& a, const Sector& b, const Sector& c,
                           const c10::Device& device, int64_t copy = 0) const override;
  at::Tensor associator(const Sector& a, const Sector& b, const Sector& c, const Sector& d,
                        const Sector& e, const Sector& f,
                        const c10::Device& device) const override;
  at::Tensor braiding(const Sector& a, const Sector& b, const Sector& c,
                      const c10::Device& device) const override;
  at::Tensor duality_matrix(const Sector& a, const c10::Device& device) const override;
  int64_t reference_index(const Sector& a) const override;
  c10::ScalarType fusion_dtype() const override;
  bool is_same_symmetry(const FusionCategory& other) const override;
};

// The Fibonacci category, one singleton of the engine.
const std::shared_ptr<FusionCategory>& fibonacci_category();

// Bind the Fibonacci category.
void bind_fibonacci(py::module_& m);

}  // namespace sym
