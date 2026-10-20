// The categorical interface: the fusion every category shares, and the questions a category
// without the extra structure answers by refusing.
#include "cat.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace sym {

FusionCategory::FusionCategory(std::string type_name, FusionStyle fusion_style,
                               BraidingStyle braiding_style, Sector trivial_sector,
                               std::string category_name)
    : fusion_style_(fusion_style),
      braiding_style_(braiding_style),
      trivial_sector_(std::move(trivial_sector)),
      name_(std::move(category_name)),
      type_name_(std::move(type_name)) {}

Sector FusionCategory::fuse(const std::vector<Sector>& sectors) const {
  if (!is_abelian()) {
    throw value_error("Category " + name_ + " is not abelian; fusion has no single outcome.");
  }
  Sector fused = trivial_sector_;
  for (const Sector& sector : sectors) {
    fused = fusion_outcomes(fused, sector).front();
  }
  return fused;
}

bool FusionCategory::can_fuse_to(const Sector& a, const Sector& b, const Sector& c) const {
  const std::vector<Sector> outcomes = fusion_outcomes(a, b);
  return std::find(outcomes.begin(), outcomes.end(), c) != outcomes.end();
}

int64_t FusionCategory::n_symbol(const Sector& a, const Sector& b, const Sector& c) const {
  return can_fuse_to(a, b, c) ? 1 : 0;
}

double FusionCategory::quantum_dim(const Sector& a) const {
  return static_cast<double>(sector_dim(a));
}

at::Tensor FusionCategory::invariant_form(const Sector&, const c10::Device&) const {
  throw not_implemented_error("Category " + name_ + " does not define an invariant form.");
}

at::Tensor FusionCategory::fusion_tensor(const Sector&, const Sector&, const Sector&,
                                         const c10::Device&, int64_t) const {
  throw not_implemented_error("Category " + name_ + " has no Clebsch-Gordan tensor.");
}

at::Tensor FusionCategory::associator(const Sector&, const Sector&, const Sector&, const Sector&,
                                      const Sector&, const Sector&, const c10::Device&) const {
  throw not_implemented_error("Category " + name_ + " has no associator.");
}

at::Tensor FusionCategory::braiding(const Sector&, const Sector&, const Sector&,
                                    const c10::Device&) const {
  throw not_implemented_error("Category " + name_ + " has no braiding.");
}

double FusionCategory::dual_sign(const Sector&) const {
  return 1.0;
}

double FusionCategory::dual_phase(const Sector&) const {
  return 1.0;
}

double FusionCategory::conjugation_phase(const Sector&) const {
  return 1.0;
}

int64_t FusionCategory::reference_index(const Sector&) const {
  throw not_implemented_error("Category " + name_ + " does not define a reference component.");
}

c10::ScalarType FusionCategory::fusion_dtype() const {
  throw not_implemented_error("Category " + name_ + " does not define a fusion tensor.");
}

at::Tensor FusionCategory::duality_matrix(const Sector&, const c10::Device&) const {
  throw not_implemented_error("Category " + name_ + " does not define its conjugate basis.");
}

}  // namespace sym

namespace sym {

void bind_category(py::module_& m) {
  py::enum_<sym::FusionStyle> fusion_style(m, "FusionStyle");
  fusion_style.value("single", sym::FusionStyle::single)
      .value("multiple_unique", sym::FusionStyle::multiple_unique)
      .value("general", sym::FusionStyle::general);
  py::enum_<sym::BraidingStyle> braiding_style(m, "BraidingStyle");
  braiding_style.value("bosonic", sym::BraidingStyle::bosonic)
      .value("fermionic", sym::BraidingStyle::fermionic)
      .value("anyonic", sym::BraidingStyle::anyonic);

  // Hand back the registered members, not a fresh instance of the same value, so that the
  // ``is`` comparisons of the checks hold.
  const py::object single = fusion_style.attr("single");
  const py::object multiple_unique = fusion_style.attr("multiple_unique");
  const py::object general = fusion_style.attr("general");
  const py::object bosonic = braiding_style.attr("bosonic");
  const py::object fermionic = braiding_style.attr("fermionic");
  const py::object anyonic = braiding_style.attr("anyonic");

  py::class_<sym::FusionCategory, std::shared_ptr<sym::FusionCategory>>(m, "FusionCategory")
      .def_property_readonly("fusion_style", [single, multiple_unique, general](const sym::FusionCategory& self) {
        const sym::FusionStyle style = self.fusion_style();
        return style == sym::FusionStyle::single ? single
             : (style == sym::FusionStyle::multiple_unique ? multiple_unique : general);
      })
      .def_property_readonly("braiding_style", [bosonic, fermionic, anyonic](const sym::FusionCategory& self) {
        const sym::BraidingStyle style = self.braiding_style();
        return style == sym::BraidingStyle::bosonic ? bosonic
             : (style == sym::BraidingStyle::fermionic ? fermionic : anyonic);
      })
      .def_property_readonly("trivial_sector", [](const sym::FusionCategory& self) {
        return sym::to_python(self.trivial_sector());
      })
      .def_property_readonly("type_name", &sym::FusionCategory::type_name)
      .def_property_readonly("name", &sym::FusionCategory::name)
      .def_property_readonly("is_abelian", &sym::FusionCategory::is_abelian)
      .def_property_readonly("is_trivial", &sym::FusionCategory::is_trivial)
      .def("dual_sector", [](const sym::FusionCategory& self, const sym::Sector& a) {
        return sym::to_python(self.dual_sector(a));
      }, py::arg("a"))
      .def("fusion_outcomes",
           [](const sym::FusionCategory& self, const sym::Sector& a, const sym::Sector& b) {
             return sym::to_python(self.fusion_outcomes(a, b));
           }, py::arg("a"), py::arg("b"))
      .def("sector_dim", &sym::FusionCategory::sector_dim, py::arg("a"))
      .def("quantum_dim", &sym::FusionCategory::quantum_dim, py::arg("a"))
      .def("invariant_form",
           [](const sym::FusionCategory& self, const sym::Sector& sector,
              const c10::Device& device) { return self.invariant_form(sector, device); },
           py::arg("sector"), py::arg("device"))
      .def("fuse", [](const sym::FusionCategory& self, py::args sectors) {
        std::vector<sym::Sector> parsed;
        parsed.reserve(sectors.size());
        for (const py::handle& sector : sectors) {
          parsed.push_back(sector.cast<sym::Sector>());
        }
        return sym::to_python(self.fuse(parsed));
      })
      .def("fusion_tensor",
           [](const sym::FusionCategory& self, const sym::Sector& a, const sym::Sector& b,
              const sym::Sector& c, const c10::Device& device, int64_t copy) {
             return self.fusion_tensor(a, b, c, device, copy);
           }, py::arg("a"), py::arg("b"), py::arg("c"), py::arg("device"), py::arg("copy") = 0)
      .def("associator",
           [](const sym::FusionCategory& self, const sym::Sector& a, const sym::Sector& b,
              const sym::Sector& c, const sym::Sector& d, const sym::Sector& e,
              const sym::Sector& f, const c10::Device& device) {
             return self.associator(a, b, c, d, e, f, device);
           }, py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"), py::arg("e"), py::arg("f"),
           py::arg("device"))
      .def("braiding",
           [](const sym::FusionCategory& self, const sym::Sector& a, const sym::Sector& b,
              const sym::Sector& c, const c10::Device& device) {
             return self.braiding(a, b, c, device);
           }, py::arg("a"), py::arg("b"), py::arg("c"), py::arg("device"))
      .def("dual_phase",
           [](const sym::FusionCategory& self, const sym::Sector& sector) {
             return self.dual_phase(sector);
           }, py::arg("sector"))
      .def("dual_sign",
           [](const sym::FusionCategory& self, const sym::Sector& sector) {
             return self.dual_sign(sector);
           }, py::arg("sector"))
      .def("conjugation_phase",
           [](const sym::FusionCategory& self, const sym::Sector& sector) {
             return self.conjugation_phase(sector);
           }, py::arg("sector"))
      .def("__eq__", [](const sym::FusionCategory& self, const py::object& other) -> py::object {
        if (!py::isinstance<sym::FusionCategory>(other)) {
          return py::reinterpret_borrow<py::object>(Py_NotImplemented);
        }
        return py::bool_(self.is_same_symmetry(other.cast<const sym::FusionCategory&>()));
      })
      .def("__hash__", [](const sym::FusionCategory& self) {
        return py::hash(py::make_tuple(self.type_name(), self.name()));
      })
      .def("__repr__", &sym::FusionCategory::name)
      .def("__reduce__", [](const sym::FusionCategory& self) {
        return py::make_tuple(py::module_::import("mpsdmrg.ext").attr("_restore_group"),
                              py::make_tuple(self.name()));
      }, "the name of the singleton group, so that a group survives a pickle round trip");
}

}  // namespace sym
