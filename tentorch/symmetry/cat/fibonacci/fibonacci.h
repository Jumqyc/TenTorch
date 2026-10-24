// The Fibonacci category: one nontrivial sector of quantum dimension the golden ratio.
#pragma once

#include "../cat.h"

namespace sym {

/**
 * @brief The Fibonacci category, $\tau \otimes \tau = 1 \oplus \tau$.
 * @details It is not the category of representations of a group, so it has no Clebsch-Gordan
 * tensor: the dimension equation $\sum_c N^c_{ab} \dim c = \dim a \dim b$ reads $1 + x = x^2$ on
 * $\tau \otimes \tau$ and has no positive integer solution.  Its symbols are therefore given
 * directly, and the pentagon and hexagon equations fix them up to gauge.
 */
class FibonacciCategory : public FusionCategory {
 public:
  explicit FibonacciCategory(c10::Device device = default_device());

  at::Tensor F_symbol(const Sector& a, const Sector& b, const Sector& c,
                      const Sector& d) const override;
  at::Tensor R_symbol(const Sector& b, const Sector& c) const override;
  double q_dim(Sector a) const override;
  int dim(Sector a) const override;
  Sector dual(Sector a) const override;
  Sector trivial() const override;
  c10::ScalarType dtype() const override;

 protected:
  std::vector<Outcome> compute_outcomes(const Sector& a, const Sector& b) const override;
};

// The Fibonacci category the engine runs on.
const std::shared_ptr<FusionCategory>& fibonacci_category();

// Bind the Fibonacci category.
void bind_fibonacci(py::module_& m);

}  // namespace sym
