#include "symbol.h"

#include <cmath>

namespace sym {

std::complex<double> frobenius_schur_phase(const FusionCategory& group, const Sector& a,
                                           const c10::Device& device) {
  // The duality matrix of a self-dual sector carries the indicator as a sign; a sector that is not
  // self-dual has none. The category reports it, and the associator cannot: the symbol of a sector
  // that couples more than once is not one number.
  (void)device;
  return {group.dual_sign(a), 0.0};
}

at::Tensor a_symbol(const FusionCategory& group, const Sector& a, const Sector& b, const Sector& c,
                    const c10::Device& device) {
  const Sector unit = group.trivial_sector();
  // A^{ab}_c = sqrt(d_a d_b / d_c) * conj(kappa_a * F(dual(a), a, b, b, unit, c))
  const at::Tensor symbol =
      group.associator(group.dual_sector(a), a, b, b, unit, c, device);
  const double scale = std::sqrt(group.quantum_dim(a) * group.quantum_dim(b) / group.quantum_dim(c));
  const std::complex<double> phase = frobenius_schur_phase(group, a, device);
  const std::complex<double> factor = scale * std::conj(phase);
  return (symbol * at::scalar_to_tensor(c10::complex<double>(factor.real(), factor.imag())))
      .reshape({group.n_symbol(a, b, c), group.n_symbol(group.dual_sector(a), c, b)});
}

at::Tensor b_symbol(const FusionCategory& group, const Sector& a, const Sector& b, const Sector& c,
                    const c10::Device& device) {
  const Sector unit = group.trivial_sector();
  // B^{ab}_c = sqrt(d_a d_b / d_c) * F(a, b, dual(b), a, c, unit)
  const at::Tensor symbol = group.associator(a, b, group.dual_sector(b), a, c, unit, device);
  const double scale = std::sqrt(group.quantum_dim(a) * group.quantum_dim(b) / group.quantum_dim(c));
  return (symbol * scale).reshape({group.n_symbol(a, b, c),
                                   group.n_symbol(c, group.dual_sector(b), a)});
}

}  // namespace sym
