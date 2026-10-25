#include "tensor.h"

#include <algorithm>
#include <stdexcept>

namespace sym
{

  namespace
  {

    // A sector the way Python prints it, `(0,)` or `(1, 0)`.
    std::string render(const Sector &sector)
    {
      std::string out = "(";
      for (size_t at = 0; at < sector.size(); ++at)
      {
        if (at > 0) out += ", ";
        out += std::to_string(sector.at(at));
      }
      if (sector.size() == 1) out += ",";
      return out + ")";
    }

    // The block of a leg that carries `sector`, which the leg holds sorted.
    int64_t block_of(const Leg &leg, const Sector &sector)
    {
      const std::vector<Sector> &sectors = leg.sectors();
      const auto found = std::lower_bound(sectors.begin(), sectors.end(), sector);
      if (found == sectors.end() || *found != sector)
        throw std::invalid_argument("the leg " + leg.repr() + " carries no block in sector " + render(sector));
      return found - sectors.begin();
    }

  } // namespace

  SideLayout layout_of(const FusionCategory &category, const std::vector<Leg> &legs, const Sector &root)
  {
    SideLayout layout;
    layout.trees = fusion_trees(category, legs, root);
    layout.offsets.push_back(0);
    for (const FusionTree &tree : layout.trees)
    {
      // One row per combination of the multiplicities the tree's legs carry.
      int64_t rows = 1;
      for (size_t axis = 0; axis < legs.size(); ++axis)
        rows *= legs.at(axis).multiplicity(block_of(legs.at(axis), tree.sectors.at(axis)));
      layout.offsets.push_back(layout.offsets.back() + rows);
    }
    return layout;
  }

  std::vector<std::pair<Sector, Sector>> block_keys(const FusionCategory &category,
                                                    const std::vector<Leg> &codomain,
                                                    const std::vector<Leg> &domain,
                                                    const Sector &qtotal)
  {
    const std::vector<Sector> out_roots = reachable_charges(category, codomain);
    const std::vector<Sector> in_roots = reachable_charges(category, domain);
    std::vector<std::pair<Sector, Sector>> keys;
    for (const Sector &r : out_roots)
      for (const Sector &s : in_roots)
        if (category.n_symbol(r, s, qtotal) > 0) keys.emplace_back(r, s);
    return keys;
  }

  Tensor::Tensor(std::shared_ptr<const FusionCategory> cat,
                 std::vector<Leg> codomain,
                 std::vector<Leg> domain,
                 Sector qtotal,
                 std::optional<c10::ScalarType> dtype,
                 std::optional<c10::Device> device)
      : cat_(std::move(cat)), codomain_(std::move(codomain)), domain_(std::move(domain)), qtotal_(std::move(qtotal))
  {
    if (!cat_) throw std::invalid_argument("a tensor needs a category");
    for (const Leg &leg : codomain_)
      if (!same_group(*leg.cat(), *cat_))
        throw std::invalid_argument("a codomain leg is over a different category");
    for (const Leg &leg : domain_)
      if (!same_group(*leg.cat(), *cat_))
        throw std::invalid_argument("a domain leg is over a different category");
    const at::TensorOptions options = at::TensorOptions()
                                          .dtype(dtype.value_or(cat_->dtype()))
                                          .device(device.value_or(cat_->device()));
    for (const auto &[r, s] : block_keys(*cat_, codomain_, domain_, qtotal_))
      blocks_.emplace(std::make_pair(r, s),
                      at::zeros({layout_of(*cat_, codomain_, r).rows(), layout_of(*cat_, domain_, s).rows()},
                                options));
  }

  std::vector<Leg> Tensor::legs() const
  {
    std::vector<Leg> out = codomain_;
    out.insert(out.end(), domain_.begin(), domain_.end());
    return out;
  }

  const at::Tensor &Tensor::block(const Sector &codomain_root, const Sector &domain_root) const
  {
    const auto found = blocks_.find(std::make_pair(codomain_root, domain_root));
    if (found == blocks_.end())
      throw std::out_of_range("the tensor has no block at " + render(codomain_root) + " x " + render(domain_root));
    return found->second;
  }

  at::Tensor &Tensor::block(const Sector &codomain_root, const Sector &domain_root)
  {
    const auto found = blocks_.find(std::make_pair(codomain_root, domain_root));
    if (found == blocks_.end())
      throw std::out_of_range("the tensor has no block at " + render(codomain_root) + " x " + render(domain_root));
    return found->second;
  }

  SideLayout Tensor::rows_of(const Sector &codomain_root) const
  {
    return layout_of(*cat_, codomain_, codomain_root);
  }

  SideLayout Tensor::columns_of(const Sector &domain_root) const
  {
    return layout_of(*cat_, domain_, domain_root);
  }

  std::vector<std::pair<Sector, Sector>> Tensor::keys() const
  {
    std::vector<std::pair<Sector, Sector>> out;
    for (const auto &[key, matrix] : blocks_) out.push_back(key);
    return out;
  }

  std::string Tensor::repr() const
  {
    std::string out = "Tensor(" + std::to_string(size()) + " legs, q = " + render(qtotal_) + ", " +
                      std::to_string(blocks_.size()) + " blocks:";
    for (const auto &[key, matrix] : blocks_)
      out += " " + render(key.first) + " x " + render(key.second) + " " + std::to_string(matrix.size(0)) +
             "x" + std::to_string(matrix.size(1));
    return out + ")";
  }

  bool Tensor::operator==(const Tensor &other) const
  {
    if (!same_group(*cat_, *other.cat_) || qtotal_ != other.qtotal_) return false;
    if (codomain_.size() != other.codomain_.size() || domain_.size() != other.domain_.size()) return false;
    for (size_t at = 0; at < codomain_.size(); ++at)
      if (codomain_.at(at) != other.codomain_.at(at)) return false;
    for (size_t at = 0; at < domain_.size(); ++at)
      if (domain_.at(at) != other.domain_.at(at)) return false;
    if (blocks_.size() != other.blocks_.size()) return false;
    for (const auto &[key, matrix] : blocks_)
    {
      const auto found = other.blocks_.find(key);
      if (found == other.blocks_.end()) return false;
      if (matrix.sizes() != found->second.sizes() || !at::equal(matrix, found->second)) return false;
    }
    return true;
  }

} // namespace sym
