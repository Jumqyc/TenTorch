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
                           double num_sectors, BraidingStyle braiding_style)
    : Group(std::move(type_name), FusionStyle::single, braiding_style, std::move(trivial_sector),
            std::move(group_name), num_sectors) {}

int64_t AbelianGroup::sector_dim(const Sector&) const {
  return 1;
}

NoSymmetry::NoSymmetry() : AbelianGroup("NoSymmetry", {0}, "NoSymmetry", 1) {}

bool NoSymmetry::is_valid_sector(const Sector& a) const {
  return a == Sector{0};
}

Sector NoSymmetry::dual_sector(const Sector& a) const {
  return a;
}

std::vector<Sector> NoSymmetry::fusion_outcomes(const Sector& a, const Sector&) const {
  return {a};
}

bool NoSymmetry::is_same_symmetry(const Group& other) const {
  return other.type_name() == "NoSymmetry";
}

std::vector<Sector> NoSymmetry::all_sectors() const {
  return {{0}};
}

U1Symmetry::U1Symmetry()
    : AbelianGroup("U1Symmetry", {0}, "U(1)", std::numeric_limits<double>::infinity()) {}

bool U1Symmetry::is_valid_sector(const Sector& a) const {
  return a.size() == 1;
}

Sector U1Symmetry::dual_sector(const Sector& a) const {
  return {-a.at(0)};
}

std::vector<Sector> U1Symmetry::fusion_outcomes(const Sector& a, const Sector& b) const {
  return {{a.at(0) + b.at(0)}};
}

bool U1Symmetry::is_same_symmetry(const Group& other) const {
  return other.type_name() == "U1Symmetry";
}

ZNSymmetry::ZNSymmetry(int64_t modulus)
    : AbelianGroup("ZNSymmetry", {0}, z_n_name(modulus), modulus), modulus_(modulus) {}

bool ZNSymmetry::is_valid_sector(const Sector& a) const {
  return a.size() == 1 && 0 <= a[0] && a[0] < modulus_;
}

Sector ZNSymmetry::dual_sector(const Sector& a) const {
  return {residue(-a.at(0), modulus_)};
}

std::vector<Sector> ZNSymmetry::fusion_outcomes(const Sector& a, const Sector& b) const {
  return {{residue(a.at(0) + b.at(0), modulus_)}};
}

bool ZNSymmetry::is_same_symmetry(const Group& other) const {
  return other.type_name() == "ZNSymmetry" &&
         static_cast<const ZNSymmetry&>(other).modulus_ == modulus_;
}

std::vector<Sector> ZNSymmetry::all_sectors() const {
  std::vector<Sector> sectors;
  sectors.reserve(modulus_);
  for (int64_t a = 0; a < modulus_; ++a) {
    sectors.push_back({a});
  }
  return sectors;
}
FermionParity::FermionParity()
    : AbelianGroup("FermionParity", {0}, "FermionParity", 2, BraidingStyle::fermionic) {}

bool FermionParity::is_valid_sector(const Sector& a) const {
  return a == Sector{0} || a == Sector{1};
}

Sector FermionParity::dual_sector(const Sector& a) const {
  return a;
}

std::vector<Sector> FermionParity::fusion_outcomes(const Sector& a, const Sector& b) const {
  return {{residue(a.at(0) + b.at(0), 2)}};
}

bool FermionParity::is_same_symmetry(const Group& other) const {
  return other.type_name() == "FermionParity";
}

std::vector<Sector> FermionParity::all_sectors() const {
  return {{0}, {1}};
}

const std::shared_ptr<Group>& no_symmetry() {
  static const std::shared_ptr<Group> instance = std::make_shared<NoSymmetry>();
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
      .def(py::init<int64_t>(), py::arg("N"))
      .def_property_readonly("N", &sym::ZNSymmetry::modulus);
  m.attr("no_symmetry") = py::cast(sym::no_symmetry());
  m.attr("u1_symmetry") = py::cast(sym::u1_symmetry());
  m.attr("z2_symmetry") = py::cast(sym::z_n_symmetry(2));
  m.attr("z3_symmetry") = py::cast(sym::z_n_symmetry(3));
  m.attr("z4_symmetry") = py::cast(sym::z_n_symmetry(4));
  m.attr("z5_symmetry") = py::cast(sym::z_n_symmetry(5));
  m.attr("z6_symmetry") = py::cast(sym::z_n_symmetry(6));
  m.attr("z7_symmetry") = py::cast(sym::z_n_symmetry(7));
  m.attr("z8_symmetry") = py::cast(sym::z_n_symmetry(8));
  m.attr("z9_symmetry") = py::cast(sym::z_n_symmetry(9));
  m.attr("fermion_parity") = py::cast(sym::fermion_parity());
}

}  // namespace sym
