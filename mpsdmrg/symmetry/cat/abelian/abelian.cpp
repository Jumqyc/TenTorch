// The abelian groups: no symmetry, U(1), Z_N and fermion parity.
#include "abelian.h"

#include <map>
#include <string>

namespace sym {

namespace {

std::string z_n_name(int64_t modulus) {
  if (modulus <= 0) {
    throw value_error("The modulus of a Z_N group must be positive, not " +
                      std::to_string(modulus) + ".");
  }
  return "Z" + std::to_string(modulus);
}

// The non-negative residue of `charge` modulo `modulus`.
int64_t residue(int64_t charge, int64_t modulus) {
  const int64_t rest = charge % modulus;
  return rest < 0 ? rest + modulus : rest;
}

}  // namespace

AbelianGroup::AbelianGroup(std::string type_name, Sector trivial_sector, std::string group_name,
                           BraidingStyle braiding_style)
    : Group(std::move(type_name), FusionStyle::single, braiding_style, std::move(trivial_sector),
            std::move(group_name)) {}

int64_t AbelianGroup::sector_dim(const Sector&) const {
  return 1;
}

at::Tensor AbelianGroup::fusion_tensor(const Sector& a, const Sector& b, const Sector& c,
                                       const c10::Device& device, int64_t copy) const {
  if (!can_fuse_to(a, b, c)) {
    throw value_error("The sectors do not fuse to the requested one.");
  }
  return at::ones({1, 1, 1}, at::TensorOptions().dtype(fusion_dtype()).device(device));
}

at::Tensor AbelianGroup::duality_matrix(const Sector&, const c10::Device& device) const {
  return at::ones({1, 1}, at::TensorOptions().dtype(fusion_dtype()).device(device));
}

at::Tensor AbelianGroup::invariant_form(const Sector&, const c10::Device& device) const {
  return at::ones({1, 1}, at::TensorOptions().dtype(fusion_dtype()).device(device));
}

c10::ScalarType AbelianGroup::fusion_dtype() const {
  return at::kDouble;
}

NoSymmetry::NoSymmetry() : AbelianGroup("NoSymmetry", {0}, "NoSymmetry") {}

bool NoSymmetry::is_valid_sector(const Sector& a) const {
  return a == Sector{0};
}

Sector NoSymmetry::dual_sector(const Sector& a) const {
  return a;
}

std::vector<Sector> NoSymmetry::fusion_outcomes(const Sector& a, const Sector&) const {
  return {a};
}

bool NoSymmetry::is_same_symmetry(const FusionCategory& other) const {
  return other.type_name() == "NoSymmetry";
}

bool NoSymmetry::is_trivial() const {
  return true;
}

U1Symmetry::U1Symmetry() : AbelianGroup("U1Symmetry", {0}, "U(1)") {}

bool U1Symmetry::is_valid_sector(const Sector& a) const {
  return a.size() == 1;
}

Sector U1Symmetry::dual_sector(const Sector& a) const {
  return {-a.at(0)};
}

std::vector<Sector> U1Symmetry::fusion_outcomes(const Sector& a, const Sector& b) const {
  return {{a.at(0) + b.at(0)}};
}

bool U1Symmetry::is_same_symmetry(const FusionCategory& other) const {
  return other.type_name() == "U1Symmetry";
}

ZNSymmetry::ZNSymmetry(int64_t modulus)
    : AbelianGroup("ZNSymmetry", {0}, z_n_name(modulus)), modulus_(modulus) {}

bool ZNSymmetry::is_valid_sector(const Sector& a) const {
  return a.size() == 1 && 0 <= a[0] && a[0] < modulus_;
}

Sector ZNSymmetry::dual_sector(const Sector& a) const {
  return {residue(-a.at(0), modulus_)};
}

std::vector<Sector> ZNSymmetry::fusion_outcomes(const Sector& a, const Sector& b) const {
  return {{residue(a.at(0) + b.at(0), modulus_)}};
}

bool ZNSymmetry::is_same_symmetry(const FusionCategory& other) const {
  return other.type_name() == "ZNSymmetry" &&
         static_cast<const ZNSymmetry&>(other).modulus_ == modulus_;
}

FermionParity::FermionParity()
    : AbelianGroup("FermionParity", {0}, "FermionParity", BraidingStyle::fermionic) {}

bool FermionParity::is_valid_sector(const Sector& a) const {
  return a == Sector{0} || a == Sector{1};
}

Sector FermionParity::dual_sector(const Sector& a) const {
  return a;
}

std::vector<Sector> FermionParity::fusion_outcomes(const Sector& a, const Sector& b) const {
  return {{residue(a.at(0) + b.at(0), 2)}};
}

bool FermionParity::is_same_symmetry(const FusionCategory& other) const {
  return other.type_name() == "FermionParity";
}

const std::shared_ptr<FusionCategory>& no_symmetry() {
  static const std::shared_ptr<FusionCategory> instance = std::make_shared<NoSymmetry>();
  return instance;
}

const std::shared_ptr<Group>& u1_symmetry() {
  static const std::shared_ptr<Group> instance = std::make_shared<U1Symmetry>();
  return instance;
}

const std::shared_ptr<Group>& z_n_symmetry(int64_t modulus) {
  static std::map<int64_t, std::shared_ptr<Group>> instances;
  std::shared_ptr<Group>& instance = instances[modulus];
  if (instance == nullptr) {
    instance = std::make_shared<ZNSymmetry>(modulus);
  }
  return instance;
}
const std::shared_ptr<Group>& fermion_parity() {
  static const std::shared_ptr<Group> instance = std::make_shared<FermionParity>();
  return instance;
}


}  // namespace sym

namespace sym {

void bind_abelian(py::module_& m) {
  py::class_<sym::AbelianGroup, sym::Group, std::shared_ptr<sym::AbelianGroup>>(
      m, "AbelianGroup");
  py::class_<sym::NoSymmetry, sym::AbelianGroup, std::shared_ptr<sym::NoSymmetry>>(
      m, "NoSymmetry")
      .def(py::init<>());
  py::class_<sym::U1Symmetry, sym::AbelianGroup, std::shared_ptr<sym::U1Symmetry>>(
      m, "U1Symmetry")
      .def(py::init<>());
  py::class_<sym::FermionParity, sym::AbelianGroup,
             std::shared_ptr<sym::FermionParity>>(m, "FermionParity")
      .def(py::init<>());
  py::class_<sym::ZNSymmetry, sym::AbelianGroup, std::shared_ptr<sym::ZNSymmetry>>(
      m, "ZNSymmetry")
      .def(py::init<int64_t>(), py::arg("N"));
  m.attr("no_symmetry") = py::cast(sym::no_symmetry());
  m.attr("u1_symmetry") = py::cast(sym::u1_symmetry());
  m.attr("z2_symmetry") = py::cast(sym::z_n_symmetry(2));
  m.attr("fermion_parity") = py::cast(sym::fermion_parity());
}

}  // namespace sym
