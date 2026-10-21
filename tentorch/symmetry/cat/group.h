// The groups of the engine: a fusion category whose associator and braiding come from its
// Clebsch-Gordan calculus.  The tensor engine sees the category interface only; this header is
// for the group implementations and for the bindings.
#pragma once

#include <array>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "cat.h"

namespace sym {

class Group : public FusionCategory {
 public:
  Group(std::string type_name, FusionStyle fusion_style, BraidingStyle braiding_style,
        Sector trivial_sector, std::string name);

  // The associator of a group is the overlap of the two fusion trees ((a b) c) -> d and
  // (a (b c)) -> d in its Clebsch-Gordan basis: the 6j symbol of the three sectors.
  at::Tensor associator(const Sector& a, const Sector& b, const Sector& c, const Sector& d,
                        const Sector& e, const Sector& f,
                        const c10::Device& device) const override;
  // The braiding of a group is what exchanging the two legs of one vertex costs.
  at::Tensor braiding(const Sector& a, const Sector& b, const Sector& c,
                      const c10::Device& device) const override;

  // Both tables are pure functions of their sectors, and one contraction asks for the same
  // entry many times over, so the last answers are kept per device.
  void clear_caches() const override;

 private:
  // The number of entries one table may hold before it is dropped.
  static constexpr std::size_t kCacheLimit = 1U << 16U;

  mutable std::map<std::tuple<std::array<Sector, 6>, int64_t>, at::Tensor> associator_cache_;
  mutable std::map<std::tuple<std::array<Sector, 3>, int64_t>, at::Tensor> braiding_cache_;
};

class AbelianGroup : public Group {
 public:
  AbelianGroup(std::string type_name, Sector trivial_sector, std::string name,
               BraidingStyle braiding_style = BraidingStyle::bosonic);

  // Every sector of an abelian group has quantum dimension one.
  int64_t sector_dim(const Sector& a) const override;

  // The fusion of an abelian group has one outcome, so its Clebsch-Gordan vertex is one
  // whenever the sectors fuse and does not exist otherwise.
  at::Tensor fusion_tensor(const Sector& a, const Sector& b, const Sector& c,
                           const c10::Device& device, int64_t copy = 0) const override;
  c10::ScalarType fusion_dtype() const override;

  // A one-dimensional sector needs no basis conversion, and its invariant form is the
  // one-by-one identity: the abelian vertex already carries the pairing.
  at::Tensor duality_matrix(const Sector& a, const c10::Device& device) const override;
  at::Tensor invariant_form(const Sector& a, const c10::Device& device) const override;
};

const std::shared_ptr<Group>& u1_symmetry();
const std::shared_ptr<Group>& z_n_symmetry(int64_t modulus);
const std::shared_ptr<Group>& su2_symmetry();
const std::shared_ptr<Group>& fermion_parity();

// Register the group subclass under the category interface.
void bind_group(py::module_& m);

}  // namespace sym
