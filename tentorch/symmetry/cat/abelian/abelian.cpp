// The abelian groups: no symmetry, U(1), Z_N and fermion parity.
#include "abelian.h"

#include <pybind11/stl.h>

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace sym {

namespace {

// The residue of `charge` in [0, modulus), so that every sector is the one the group names.
int64_t residue(int64_t charge, int64_t modulus) {
  const int64_t reduced = charge % modulus;
  return reduced < 0 ? reduced + modulus : reduced;
}

// The name of the group, and the one place a non-positive modulus is rejected.
std::string z_n_name(int64_t modulus) {
  if (modulus <= 0) {
    throw std::invalid_argument("The modulus of Z_N must be positive, not " +
                                std::to_string(modulus) + ".");
  }
  return "Z_" + std::to_string(modulus);
}

}  // namespace

std::vector<Outcome> AbelianGroup::compute_outcomes(const Sector& a, const Sector& b) const {
  return {{combine(a, b), 1}};
}

Sector AbelianGroup::dual(Sector a) const { return inverse(a); }

Sector AbelianGroup::trivial() const { return {0}; }

int AbelianGroup::dim(Sector) const { return 1; }

c10::ScalarType AbelianGroup::dtype() const { return at::kDouble; }

at::Tensor AbelianGroup::compute_clebsch_gordan(const Sector& a, const Sector& b, const Sector& c,
                                        int64_t copy) const {
  TORCH_CHECK(copy == 0, "An abelian fusion couples at most once, not ", copy + 1, " times.");
  TORCH_CHECK(combine(a, b) == c, "The sectors do not fuse to the requested one.");
  return at::ones({1, 1, 1}, at::TensorOptions().dtype(dtype()).device(device()));
}

NoSymmetry::NoSymmetry(c10::Device device) : AbelianGroup("NoSymmetry", device) {}

Sector NoSymmetry::combine(const Sector&, const Sector&) const { return {0}; }

Sector NoSymmetry::inverse(const Sector&) const { return {0}; }

U1Symmetry::U1Symmetry(c10::Device device) : AbelianGroup("U(1)", device) {}

Sector U1Symmetry::combine(const Sector& a, const Sector& b) const {
  return {a.at(0) + b.at(0)};
}

Sector U1Symmetry::inverse(const Sector& a) const { return {-a.at(0)}; }

ZNSymmetry::ZNSymmetry(int64_t modulus, c10::Device device)
    : AbelianGroup(z_n_name(modulus), device), modulus_(modulus) {}

Sector ZNSymmetry::combine(const Sector& a, const Sector& b) const {
  return {residue(a.at(0) + b.at(0), modulus_)};
}

Sector ZNSymmetry::inverse(const Sector& a) const { return {residue(-a.at(0), modulus_)}; }

FermionParity::FermionParity(c10::Device device) : FusionCategory("FermionParity", device) {}

std::vector<Outcome> FermionParity::compute_outcomes(const Sector& a, const Sector& b) const {
  return {{{residue(a.at(0) + b.at(0), 2)}, 1}};
}

at::Tensor FermionParity::F_symbol(const Sector& a, const Sector& b, const Sector& c,
                                   const Sector& d) const {
  // One vertex per triple, so the associator is the single entry 1 exactly when the outer sector is
  // the one the three legs fuse to, and an empty matrix otherwise.
  const int64_t side = residue(a.at(0) + b.at(0) + c.at(0), 2) == d.at(0) ? 1 : 0;
  return at::ones({side, side}, at::TensorOptions().dtype(dtype()).device(device()));
}

at::Tensor FermionParity::R_symbol(const Sector& b, const Sector& c) const {
  // The braiding is the Koszul sign: a fermion passes another fermion with -1, and nothing else
  // picks up a sign.
  const double sign = b.at(0) == 1 && c.at(0) == 1 ? -1.0 : 1.0;
  return at::full({1, 1}, sign, at::TensorOptions().dtype(dtype()).device(device()));
}

double FermionParity::q_dim(Sector) const { return 1.0; }

int FermionParity::dim(Sector) const { return 1; }

Sector FermionParity::dual(Sector a) const { return a; }

Sector FermionParity::trivial() const { return {0}; }

c10::ScalarType FermionParity::dtype() const { return at::kDouble; }

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
  std::shared_ptr<Group>& kept = instances[modulus];
  if (!kept) {
    kept = std::make_shared<ZNSymmetry>(modulus);
  }
  return kept;
}

const std::shared_ptr<FusionCategory>& fermion_parity() {
  static const std::shared_ptr<FusionCategory> instance = std::make_shared<FermionParity>();
  return instance;
}

void bind_abelian(py::module_& m) {
  py::class_<AbelianGroup, Group, std::shared_ptr<AbelianGroup>>(m, "AbelianGroup");

  py::class_<NoSymmetry, AbelianGroup, std::shared_ptr<NoSymmetry>>(m, "NoSymmetry")
      .def(py::init<>());

  py::class_<U1Symmetry, AbelianGroup, std::shared_ptr<U1Symmetry>>(m, "U1Symmetry")
      .def(py::init<>());

  py::class_<ZNSymmetry, AbelianGroup, std::shared_ptr<ZNSymmetry>>(m, "ZNSymmetry")
      .def(py::init<int64_t>(), py::arg("modulus"))
      .def_property_readonly("modulus", &ZNSymmetry::modulus);

  py::class_<FermionParity, FusionCategory, std::shared_ptr<FermionParity>>(m, "FermionParity")
      .def(py::init<>());

  m.attr("no_symmetry") = py::cast(no_symmetry());
  m.attr("u1_symmetry") = py::cast(u1_symmetry());
  m.attr("fermion_parity") = py::cast(fermion_parity());
  m.def("z_n_symmetry", &z_n_symmetry, py::arg("modulus"),
        "the Z_N group of the given modulus, the same object every call");
}

}  // namespace sym
