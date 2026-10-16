// The base group: the fusion every group shares, and the questions a group without extra
// structure answers by refusing.
#include "group.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace sym {

Group::Group(std::string type_name, FusionStyle fusion_style, BraidingStyle braiding_style,
             Sector trivial_sector, std::string group_name, double num_sectors)
    : fusion_style_(fusion_style),
      braiding_style_(braiding_style),
      trivial_sector_(std::move(trivial_sector)),
      group_name_(std::move(group_name)),
      type_name_(std::move(type_name)),
      num_sectors_(num_sectors) {}

Sector Group::fuse(const std::vector<Sector>& sectors) const {
  if (!is_abelian()) {
    throw value_error("Group " + group_name_ + " is not abelian; fusion has no single outcome.");
  }
  Sector fused = trivial_sector_;
  for (const Sector& sector : sectors) {
    fused = fusion_outcomes(fused, sector).front();
  }
  return fused;
}

bool Group::can_fuse_to(const Sector& a, const Sector& b, const Sector& c) const {
  const std::vector<Sector> outcomes = fusion_outcomes(a, b);
  return std::find(outcomes.begin(), outcomes.end(), c) != outcomes.end();
}

int64_t Group::n_symbol(const Sector& a, const Sector& b, const Sector& c) const {
  return can_fuse_to(a, b, c) ? 1 : 0;
}

at::Tensor Group::fusion_tensor(const Sector&, const Sector&, const Sector&,
                                const c10::Device&, int64_t) const {
  throw not_implemented_error("Group " + group_name_ + " has no Clebsch-Gordan tensor.");
}

std::vector<Sector> Group::all_sectors() const {
  throw not_implemented_error("Group " + group_name_ + " does not list its sectors.");
}

double Group::dual_sign(const Sector&) const {
  return 1.0;
}

double Group::dual_phase(const Sector&) const {
  return 1.0;
}

Sector Group::sector_of_dimension(int64_t) const {
  throw not_implemented_error("Group " + group_name_ + " does not name a sector by its dimension.");
}

int64_t Group::reference_index(const Sector&) const {
  throw not_implemented_error("Group " + group_name_ + " does not define a reference component.");
}

c10::ScalarType Group::fusion_dtype() const {
  throw not_implemented_error("Group " + group_name_ + " does not define a fusion tensor.");
}

at::Tensor Group::invariant_form(const Sector&, const c10::Device&) const {
  throw not_implemented_error("Group " + group_name_ + " does not define an invariant form.");
}

at::Tensor Group::duality_matrix(const Sector&, const c10::Device&) const {
  throw not_implemented_error("Group " + group_name_ + " does not define its conjugate basis.");
}

}  // namespace sym

namespace sym {

void bind_group(py::module_& m) {
  py::enum_<sym::FusionStyle> fusion_style(m, "FusionStyle");
  fusion_style.value("single", sym::FusionStyle::single)
      .value("multiple_unique", sym::FusionStyle::multiple_unique)
      .value("general", sym::FusionStyle::general);
  py::enum_<sym::BraidingStyle> braiding_style(m, "BraidingStyle");
  braiding_style.value("bosonic", sym::BraidingStyle::bosonic)
      .value("fermionic", sym::BraidingStyle::fermionic);

  // Hand back the registered members, not a fresh instance of the same value, so that the
  // ``is`` comparisons of the checks hold.
  const py::object single = fusion_style.attr("single");
  const py::object multiple_unique = fusion_style.attr("multiple_unique");
  const py::object general = fusion_style.attr("general");
  const py::object bosonic = braiding_style.attr("bosonic");
  const py::object fermionic = braiding_style.attr("fermionic");

  py::class_<sym::Group, std::shared_ptr<sym::Group>>(m, "Group")
      .def_property_readonly("fusion_style", [single, multiple_unique, general](const sym::Group& self) {
        const sym::FusionStyle style = self.fusion_style();
        return style == sym::FusionStyle::single ? single
             : (style == sym::FusionStyle::multiple_unique ? multiple_unique : general);
      })
      .def_property_readonly("braiding_style", [bosonic, fermionic](const sym::Group& self) {
        return self.braiding_style() == sym::BraidingStyle::bosonic ? bosonic : fermionic;
      })
      .def_property_readonly("trivial_sector", [](const sym::Group& self) {
        return sym::to_python(self.trivial_sector());
      })
      .def_property_readonly("type_name", &sym::Group::type_name)
      .def_property_readonly("group_name", &sym::Group::group_name)
      .def_property_readonly("is_abelian", &sym::Group::is_abelian)
      .def("dual_sector", [](const sym::Group& self, const sym::Sector& a) {
        return sym::to_python(self.dual_sector(a));
      }, py::arg("a"))
      .def("fusion_outcomes",
           [](const sym::Group& self, const sym::Sector& a, const sym::Sector& b) {
             return sym::to_python(self.fusion_outcomes(a, b));
           }, py::arg("a"), py::arg("b"))
      .def("sector_dim", &sym::Group::sector_dim, py::arg("a"))
      .def("fuse", [](const sym::Group& self, py::args sectors) {
        std::vector<sym::Sector> parsed;
        parsed.reserve(sectors.size());
        for (const py::handle& sector : sectors) {
          parsed.push_back(sector.cast<sym::Sector>());
        }
        return sym::to_python(self.fuse(parsed));
      })
      .def("fusion_tensor",
           [](const sym::Group& self, const sym::Sector& a, const sym::Sector& b,
              const sym::Sector& c, const c10::Device& device, int64_t copy) {
             return self.fusion_tensor(a, b, c, device, copy);
           }, py::arg("a"), py::arg("b"), py::arg("c"), py::arg("device"), py::arg("copy") = 0)
      .def("sector_of_dimension",
           [](const sym::Group& self, int64_t dim) {
             return sym::to_python(self.sector_of_dimension(dim));
           }, py::arg("dim"))
      .def("invariant_form",
           [](const sym::Group& self, const sym::Sector& sector,
              const c10::Device& device) { return self.invariant_form(sector, device); },
           py::arg("sector"), py::arg("device"))
      .def("dual_phase",
           [](const sym::Group& self, const sym::Sector& sector) {
             return self.dual_phase(sector);
           }, py::arg("sector"))
      .def("dual_sign",
           [](const sym::Group& self, const sym::Sector& sector) {
             return self.dual_sign(sector);
           }, py::arg("sector"))
      .def("__eq__", [](const sym::Group& self, const py::object& other) -> py::object {
        if (!py::isinstance<sym::Group>(other)) {
          return py::reinterpret_borrow<py::object>(Py_NotImplemented);
        }
        return py::bool_(self.is_same_symmetry(other.cast<const sym::Group&>()));
      })
      .def("__hash__", [](const sym::Group& self) {
        return py::hash(py::make_tuple(self.type_name(), self.group_name()));
      })
      .def("__repr__", &sym::Group::group_name)
      .def("__reduce__", [](const sym::Group& self) {
        return py::make_tuple(py::module_::import("mpsdmrg.ext").attr("_restore_group"),
                              py::make_tuple(self.group_name()));
      }, "the name of the singleton group, so that a group survives a pickle round trip");
}

}  // namespace sym
