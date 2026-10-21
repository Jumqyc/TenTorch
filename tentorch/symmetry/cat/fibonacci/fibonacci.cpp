// The Fibonacci fusion category: the fusion rules, the quantum dimensions, and the F and R symbols
// that recouple it.  Every multiplicity is one, so an amplitude is a single number.
#include "fibonacci.h"

#include <cmath>
#include <complex>
#include <memory>
#include <string>
#include <vector>

namespace sym {

namespace {

// The label of the trivial object 1, and the label of tau.
constexpr int64_t TRIVIAL = 0;
constexpr int64_t TAU = 1;

// The golden ratio, the quantum dimension of tau.
double golden_ratio() {
  return (1.0 + std::sqrt(5.0)) / 2.0;
}

at::TensorOptions options(const c10::Device& device) {
  return at::TensorOptions().dtype(at::kComplexDouble).device(device);
}

}  // namespace

FibonacciCategory::FibonacciCategory()
    : FusionCategory("FibonacciCategory", FusionStyle::general, BraidingStyle::anyonic,
                     Sector{TRIVIAL}, "Fibonacci") {}

bool FibonacciCategory::is_valid_sector(const Sector& a) const {
  return a.size() == 1 && (a[0] == TRIVIAL || a[0] == TAU);
}

// Both simple objects are self-dual.
Sector FibonacciCategory::dual_sector(const Sector& a) const {
  if (!is_valid_sector(a)) {
    throw value_error("The Fibonacci category has no sector " + std::to_string(a.at(0)) + ".");
  }
  return a;
}

std::vector<Sector> FibonacciCategory::fusion_outcomes(const Sector& a, const Sector& b) const {
  if (!is_valid_sector(a) || !is_valid_sector(b)) {
    throw value_error("The Fibonacci category fuses only the sectors 0 and 1.");
  }
  if (a[0] == TRIVIAL) {
    return {b};
  }
  if (b[0] == TRIVIAL) {
    return {a};
  }
  return {Sector{TRIVIAL}, Sector{TAU}};
}

// One index per simple object: this is the state count the dense code needs, while the dimension
// that weighs a recoupling is `quantum_dim`, which is irrational here.
int64_t FibonacciCategory::sector_dim(const Sector& a) const {
  if (!is_valid_sector(a)) {
    throw value_error("The Fibonacci category has no sector " + std::to_string(a.at(0)) + ".");
  }
  return 1;
}

double FibonacciCategory::quantum_dim(const Sector& a) const {
  if (!is_valid_sector(a)) {
    throw value_error("The Fibonacci category has no sector " + std::to_string(a.at(0)) + ".");
  }
  return a[0] == TAU ? golden_ratio() : 1.0;
}

// There is no Clebsch-Gordan calculus behind this category: it has fusion and F/R symbols only.
at::Tensor FibonacciCategory::fusion_tensor(const Sector&, const Sector&, const Sector&,
                                            const c10::Device&, int64_t) const {
  throw not_implemented_error("The Fibonacci category has no Clebsch-Gordan tensor.");
}

// F^(a b c)_d over the channels e of (a b) and f of (b c): 1 unless all four are tau, where it is
// the golden-ratio matrix [[1/phi, 1/sqrt(phi)], [1/sqrt(phi), -1/phi]].
at::Tensor FibonacciCategory::associator(const Sector& a, const Sector& b, const Sector& c,
                                         const Sector& d, const Sector& e, const Sector& f,
                                         const c10::Device& device) const {
  if (!can_fuse_to(a, b, e) || !can_fuse_to(b, c, f) || !can_fuse_to(e, c, d) ||
      !can_fuse_to(a, f, d)) {
    return at::empty({0, 0, 0, 0}, options(device));
  }
  double value = 1.0;
  if (a[0] == TAU && b[0] == TAU && c[0] == TAU && d[0] == TAU) {
    const double phi = golden_ratio();
    if (e[0] == TRIVIAL) {
      value = f[0] == TRIVIAL ? 1.0 / phi : 1.0 / std::sqrt(phi);
    } else {
      value = f[0] == TRIVIAL ? 1.0 / std::sqrt(phi) : -1.0 / phi;
    }
  }
  return at::full({1, 1, 1, 1}, c10::complex<double>(value, 0.0), options(device));
}

// R^(a b)_c: 1 unless both are tau, where the phases are exp(-4 pi i / 5) to the trivial channel
// and exp(3 pi i / 5) to tau.
at::Tensor FibonacciCategory::braiding(const Sector& a, const Sector& b, const Sector& c,
                                       const c10::Device& device) const {
  if (!can_fuse_to(a, b, c) || !can_fuse_to(b, a, c)) {
    return at::empty({0, 0}, options(device));
  }
  std::complex<double> value(1.0, 0.0);
  if (a[0] == TAU && b[0] == TAU) {
    value = std::polar(1.0, c[0] == TRIVIAL ? -4.0 * M_PI / 5.0 : 3.0 * M_PI / 5.0);
  }
  return at::full({1, 1}, c10::complex<double>(value), options(device));
}

// The reduced index of a simple object is one-dimensional, so its conjugate basis is the identity.
at::Tensor FibonacciCategory::duality_matrix(const Sector& a, const c10::Device& device) const {
  if (!is_valid_sector(a)) {
    throw value_error("The Fibonacci category has no sector " + std::to_string(a.at(0)) + ".");
  }
  return at::ones({1, 1}, options(device));
}

int64_t FibonacciCategory::reference_index(const Sector& a) const {
  if (!is_valid_sector(a)) {
    throw value_error("The Fibonacci category has no sector " + std::to_string(a.at(0)) + ".");
  }
  return 0;
}

c10::ScalarType FibonacciCategory::fusion_dtype() const {
  return at::kComplexDouble;
}

bool FibonacciCategory::is_same_symmetry(const FusionCategory& other) const {
  return other.type_name() == type_name();
}

const std::shared_ptr<FusionCategory>& fibonacci_category() {
  static const std::shared_ptr<FusionCategory> instance = std::make_shared<FibonacciCategory>();
  return instance;
}

void bind_fibonacci(py::module_& m) {
  py::class_<sym::FibonacciCategory, sym::FusionCategory, std::shared_ptr<sym::FibonacciCategory>>(
      m, "FibonacciCategory")
      .def(py::init<>());
  m.attr("fibonacci") = py::cast(sym::fibonacci_category());
}

}  // namespace sym
