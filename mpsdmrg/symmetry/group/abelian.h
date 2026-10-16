// The abelian groups: no symmetry, U(1), Z_N and fermion parity.
#pragma once

#include "group.h"

namespace sym {

class NoSymmetry : public AbelianGroup {
 public:
  NoSymmetry();
  bool is_valid_sector(const Sector& a) const override;
  Sector dual_sector(const Sector& a) const override;
  std::vector<Sector> fusion_outcomes(const Sector& a, const Sector& b) const override;
  bool is_same_symmetry(const Group& other) const override;
  std::vector<Sector> all_sectors() const override;
};

class U1Symmetry : public AbelianGroup {
 public:
  U1Symmetry();
  bool is_valid_sector(const Sector& a) const override;
  Sector dual_sector(const Sector& a) const override;
  std::vector<Sector> fusion_outcomes(const Sector& a, const Sector& b) const override;
  bool is_same_symmetry(const Group& other) const override;
};

class ZNSymmetry : public AbelianGroup {
 public:
  // Raises:
  //   value_error: If `modulus` is not positive.
  explicit ZNSymmetry(int64_t modulus);
  int64_t modulus() const { return modulus_; }
  bool is_valid_sector(const Sector& a) const override;
  Sector dual_sector(const Sector& a) const override;
  std::vector<Sector> fusion_outcomes(const Sector& a, const Sector& b) const override;
  bool is_same_symmetry(const Group& other) const override;
  std::vector<Sector> all_sectors() const override;

 private:
  int64_t modulus_;
};
class FermionParity : public AbelianGroup {
 public:
  FermionParity();
  bool is_valid_sector(const Sector& a) const override;
  Sector dual_sector(const Sector& a) const override;
  std::vector<Sector> fusion_outcomes(const Sector& a, const Sector& b) const override;
  bool is_same_symmetry(const Group& other) const override;
  std::vector<Sector> all_sectors() const override;
};

// The singletons behind the module-level names; the same object every call.

// Bind the abelian groups and their singletons.
void bind_abelian(py::module_& m);

}  // namespace sym
