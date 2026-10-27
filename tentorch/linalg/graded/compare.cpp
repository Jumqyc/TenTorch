#include "compare.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "../../symmetry/tree/tree.h"

namespace sym
{

  bool is_close(const Tensor &first, const Tensor &second, double rtol, double atol)
  {
    if (!same_group(*first.cat(), *second.cat())) return false;
    if (first.qtotal() != second.qtotal()) return false;
    if (first.codomain() != second.codomain()) return false;
    if (first.domain() != second.domain()) return false;
    const std::map<BlockKey, at::Tensor> &self = first.blocks();
    const std::map<BlockKey, at::Tensor> &other = second.blocks();
    if (self.size() != other.size()) return false;
    for (const auto &[key, block] : self)
    {
      const auto found = other.find(key);
      if (found == other.end()) return false;
      if (block.sizes() != found->second.sizes()) return false;
      if (!at::allclose(block, found->second, rtol, atol)) return false;
    }
    return true;
  }

  std::complex<double> inner(const Tensor &first, const Tensor &second)
  {
    if (!same_group(*first.cat(), *second.cat()))
      throw std::invalid_argument("a scalar product needs both tensors over one category");
    if (first.codomain() != second.codomain() || first.domain() != second.domain())
      throw std::invalid_argument("a scalar product needs both tensors with the same legs");
    if (!(first.qtotal() == second.qtotal())) return 0.0;
    at::Tensor total = at::zeros({}, at::TensorOptions().dtype(first.dtype()).device(first.device()));
    for (const auto &[key, block] : first.blocks())
      total += (block.conj() * second.blocks().at(key)).sum();
    const c10::complex<double> value = total.item().toComplexDouble();
    return std::complex<double>(value.real(), value.imag());
  }

  double norm(const Tensor &tensor) { return std::sqrt(std::abs(inner(tensor, tensor))); }

} // namespace sym
