// The Fibonacci category: one nontrivial sector of quantum dimension the golden ratio.
#include "fibonacci.h"

#include <cmath>
#include <complex>
#include <cstdint>
#include <numbers>
#include <vector>

namespace sym {

namespace {

constexpr int64_t kTrivial = 0;
constexpr int64_t kTau = 1;

double golden_ratio() { return (1.0 + std::sqrt(5.0)) / 2.0; }

}  // namespace

FibonacciCategory::FibonacciCategory(c10::Device device)
    : FusionCategory("FibonacciCategory", device) {}

std::vector<Outcome> FibonacciCategory::compute_outcomes(const Sector& a, const Sector& b) const {
  if (a.at(0) == kTrivial) return {{b, 1}};
  if (b.at(0) == kTrivial) return {{a, 1}};
  return {{{kTrivial}, 1}, {{kTau}, 1}};
}

at::Tensor FibonacciCategory::F_symbol(const Sector& a, const Sector& b, const Sector& c,
                                       const Sector& d) const {
  const std::vector<FIndex> rows = f_rows(a, b, c, d);
  const std::vector<FIndex> columns = f_columns(a, b, c, d);
  const int64_t side = static_cast<int64_t>(rows.size());
  const int64_t other = static_cast<int64_t>(columns.size());
  at::Tensor value = at::zeros({side, other}, at::TensorOptions().dtype(dtype()));
  if (side == 0 || other == 0) return value.to(device());
  if (a.at(0) == kTau && b.at(0) == kTau && c.at(0) == kTau && d.at(0) == kTau) {
    // The golden matrix, the only recoupling of this category that is not the identity.
    const double phi = golden_ratio();
    const double golden[2][2] = {{1.0 / phi, 1.0 / std::sqrt(phi)},
                                 {1.0 / std::sqrt(phi), -1.0 / phi}};
    for (int64_t i = 0; i < side; ++i) {
      for (int64_t j = 0; j < other; ++j) {
        value[i][j] = golden[rows.at(static_cast<size_t>(i)).sector.at(0)]
                            [columns.at(static_cast<size_t>(j)).sector.at(0)];
      }
    }
    return value.to(device());
  }
  // Every other matrix has one entry per coupling, and each of those is the identity.
  return value.fill_(1.0).to(device());
}

at::Tensor FibonacciCategory::R_symbol(const Sector& b, const Sector& c) const {
  const std::vector<RIndex> rows = r_rows(b, c);
  const int64_t size = static_cast<int64_t>(rows.size());
  at::Tensor value = at::zeros({size, size}, at::TensorOptions().dtype(dtype()));
  for (int64_t at = 0; at < size; ++at) {
    c10::complex<double> phase(1.0, 0.0);
    if (b.at(0) == kTau && c.at(0) == kTau) {
      phase = rows.at(static_cast<size_t>(at)).sector.at(0) == kTrivial
                  ? std::polar(1.0, -4.0 * std::numbers::pi / 5.0)
                  : std::polar(1.0, 3.0 * std::numbers::pi / 5.0);
    }
    value[at][at] = phase;
  }
  return value.to(device());
}

double FibonacciCategory::q_dim(Sector a) const {
  return a.at(0) == kTau ? golden_ratio() : 1.0;
}

Sector FibonacciCategory::dual(Sector a) const { return a; }

Sector FibonacciCategory::trivial() const { return {kTrivial}; }

// $\tau$ is self-dual with a symmetric invariant form.
double FibonacciCategory::frobenius_schur_sign(const Sector&) const { return 1.0; }

c10::ScalarType FibonacciCategory::dtype() const { return at::kComplexDouble; }

const std::shared_ptr<FusionCategory>& fibonacci_category() {
  static const std::shared_ptr<FusionCategory> instance = std::make_shared<FibonacciCategory>();
  return instance;
}

}  // namespace sym
