#include "tensor.h"

#include <complex>
#include <map>
#include <stdexcept>
#include <utility>

#include "cache.h"

namespace sym
{

  namespace
  {
    // Whether two tensors live over one category with the same legs and the same charge.
    bool same_space(const Tensor &a, const Tensor &b)
    {
      return same_group(*a.cat(), *b.cat()) && a.qtotal() == b.qtotal() && a.codomain() == b.codomain() &&
             a.domain() == b.domain();
    }
  } // namespace

  Tensor::Tensor(std::shared_ptr<const FusionCategory> cat,
                 std::vector<Leg> codomain,
                 std::vector<Leg> domain,
                 Sector qtotal,
                 std::optional<c10::ScalarType> dtype,
                 std::optional<c10::Device> device)
      : cat_(std::move(cat)),
        codomain_(std::move(codomain)),
        domain_(std::move(domain)),
        qtotal_(std::move(qtotal))
  {
    if (!cat_) throw std::invalid_argument("a tensor needs a category");
    dtype_ = dtype.value_or(cat_->dtype());
    device_ = device.value_or(cat_->device());
    for (const Leg &leg : codomain_)
      if (!same_group(*leg.cat(), *cat_))
        throw std::invalid_argument("a codomain leg is over a different category");
    for (const Leg &leg : domain_)
      if (!same_group(*leg.cat(), *cat_))
        throw std::invalid_argument("a domain leg is over a different category");
    const at::TensorOptions options = at::TensorOptions().dtype(dtype_).device(device_);
    // One layout per root and not one per key: the keys of a side share it, and the layout is cached
    // on the tensor so that a later access of the same root costs a lookup.
    const std::vector<BlockKey> keys = block_keys(*cat_, codomain_, domain_, qtotal_);
    // The keys of a side share a root, and a side of a root is the same rows or columns every time,
    // so each root is looked up once.
    std::map<Sector, int64_t> rows_of_root;
    std::map<Sector, int64_t> columns_of_root;
    for (const BlockKey &key : keys)
    {
      auto rows = rows_of_root.find(key.codomain);
      if (rows == rows_of_root.end())
        rows = rows_of_root.emplace(key.codomain, TensorCaches::instance().rows(cat_, codomain_, key.codomain)).first;
      auto columns = columns_of_root.find(key.domain);
      if (columns == columns_of_root.end())
        columns = columns_of_root.emplace(key.domain, TensorCaches::instance().rows(cat_, domain_, key.domain)).first;
      blocks_.emplace(key, at::zeros({rows->second, columns->second}, options));
    }
  }

  std::vector<Leg> Tensor::legs() const
  {
    std::vector<Leg> out = codomain_;
    out.insert(out.end(), domain_.begin(), domain_.end());
    return out;
  }

  const at::Tensor &Tensor::block(const Sector &codomain_root, const Sector &domain_root, int64_t copy) const
  {
    const auto found = blocks_.find(BlockKey{codomain_root, domain_root, copy});
    if (found == blocks_.end())
      throw std::out_of_range("the tensor has no block at " + render(codomain_root) + " x " + render(domain_root) +
                              " of copy " + std::to_string(copy));
    return found->second;
  }

  at::Tensor &Tensor::block(const Sector &codomain_root, const Sector &domain_root, int64_t copy)
  {
    const auto found = blocks_.find(BlockKey{codomain_root, domain_root, copy});
    if (found == blocks_.end())
      throw std::out_of_range("the tensor has no block at " + render(codomain_root) + " x " + render(domain_root) +
                              " of copy " + std::to_string(copy));
    return found->second;
  }

  const SideLayout &Tensor::rows_of(const Sector &codomain_root) const
  {
    return TensorCaches::instance().side(*cat_, codomain_, codomain_root);
  }

  const SideLayout &Tensor::columns_of(const Sector &domain_root) const
  {
    return TensorCaches::instance().side(*cat_, domain_, domain_root);
  }

  std::vector<BlockKey> Tensor::keys() const
  {
    std::vector<BlockKey> out;
    for (const auto &[key, matrix] : blocks_) out.push_back(key);
    return out;
  }

  Tensor Tensor::conj() const
  {
    // The rows of the conjugate are the columns of the tensor: the two sides lay their rows out
    // through the same tree order, so the block turns over as it conjugates.
    Tensor out(cat_, domain_, codomain_, cat_->dual(qtotal_), dtype_, device_);
    for (const auto &[key, block] : blocks_)
      out.block(key.domain, key.codomain, key.copy).copy_(block.conj().transpose(0, 1));
    return out;
  }

  std::vector<int64_t> codomain_legs(const Tensor &tensor)
  {
    std::vector<int64_t> legs(static_cast<size_t>(tensor.codomain().size()));
    for (int64_t at = 0; at < static_cast<int64_t>(legs.size()); ++at) legs.at(static_cast<size_t>(at)) = at;
    return legs;
  }

  std::vector<int64_t> domain_legs(const Tensor &tensor)
  {
    const int64_t above = static_cast<int64_t>(tensor.codomain().size());
    std::vector<int64_t> legs(static_cast<size_t>(tensor.domain().size()));
    for (int64_t at = 0; at < static_cast<int64_t>(legs.size()); ++at)
      legs.at(static_cast<size_t>(at)) = above + at;
    return legs;
  }

  Tensor Tensor::clone() const
  {
    Tensor out(*this);
    for (auto &[key, block] : out.blocks_) block = block.clone();
    return out;
  }

  Tensor &Tensor::operator+=(const Tensor &other)
  {
    if (!same_space(*this, other))
      throw std::invalid_argument("a sum needs both tensors with the same legs and the same charge");
    for (const auto &[key, block] : other.blocks()) blocks_.at(key).add_(block);
    return *this;
  }

  Tensor &Tensor::operator-=(const Tensor &other)
  {
    if (!same_space(*this, other))
      throw std::invalid_argument("a difference needs both tensors with the same legs and the same "
                                  "charge");
    for (const auto &[key, block] : other.blocks()) blocks_.at(key).sub_(block);
    return *this;
  }

  Tensor &Tensor::operator*=(double scalar)
  {
    for (auto &[key, block] : blocks_) block.mul_(scalar);
    return *this;
  }

  Tensor &Tensor::operator*=(std::complex<double> scalar)
  {
    if (scalar.imag() == 0.0) return *this *= scalar.real();
    for (auto &[key, block] : blocks_)
    {
      if (!block.is_complex())
        throw std::invalid_argument("a block of real entries cannot take the complex factor " +
                                    std::to_string(scalar.real()) + "+" +
                                    std::to_string(scalar.imag()) + "i");
      block.mul_(c10::complex<double>(scalar.real(), scalar.imag()));
    }
    return *this;
  }

  Tensor &Tensor::operator/=(double scalar)
  {
    if (scalar == 0.0) throw std::invalid_argument("a tensor cannot be divided by zero");
    for (auto &[key, block] : blocks_) block.div_(scalar);
    return *this;
  }

} // namespace sym
