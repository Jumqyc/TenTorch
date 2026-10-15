// The base group: the fusion every group shares, and the questions a group without extra
// structure answers by refusing.
#include "group.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace mpsdmrg {

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
                                const c10::Device&) const {
  throw not_implemented_error("Group " + group_name_ + " has no Clebsch-Gordan tensor.");
}

std::vector<Sector> Group::all_sectors() const {
  throw not_implemented_error("Group " + group_name_ + " does not list its sectors.");
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

at::Tensor Group::duality_matrix(const Sector&, const c10::Device&) const {
  throw not_implemented_error("Group " + group_name_ + " does not define its conjugate basis.");
}

}  // namespace mpsdmrg

namespace mpsdmrg {

void bind_group(py::module_& m) {
  py::enum_<mpsdmrg::FusionStyle> fusion_style(m, "FusionStyle");
  fusion_style.value("single", mpsdmrg::FusionStyle::single)
      .value("multiple_unique", mpsdmrg::FusionStyle::multiple_unique);
  py::enum_<mpsdmrg::BraidingStyle> braiding_style(m, "BraidingStyle");
  braiding_style.value("bosonic", mpsdmrg::BraidingStyle::bosonic)
      .value("fermionic", mpsdmrg::BraidingStyle::fermionic);

  // Hand back the registered members, not a fresh instance of the same value, so that the
  // ``is`` comparisons of the checks hold.
  const py::object single = fusion_style.attr("single");
  const py::object multiple_unique = fusion_style.attr("multiple_unique");
  const py::object bosonic = braiding_style.attr("bosonic");
  const py::object fermionic = braiding_style.attr("fermionic");

  py::class_<mpsdmrg::Group, std::shared_ptr<mpsdmrg::Group>>(m, "Group")
      .def_property_readonly("fusion_style", [single, multiple_unique](const mpsdmrg::Group& self) {
        return self.fusion_style() == mpsdmrg::FusionStyle::single ? single : multiple_unique;
      })
      .def_property_readonly("braiding_style", [bosonic, fermionic](const mpsdmrg::Group& self) {
        return self.braiding_style() == mpsdmrg::BraidingStyle::bosonic ? bosonic : fermionic;
      })
      .def_property_readonly("trivial_sector", [](const mpsdmrg::Group& self) {
        return mpsdmrg::to_python(self.trivial_sector());
      })
      .def_property_readonly("type_name", &mpsdmrg::Group::type_name)
      .def_property_readonly("group_name", &mpsdmrg::Group::group_name)
      .def_property_readonly("num_sectors", [](const mpsdmrg::Group& self) -> py::object {
        if (std::isinf(self.num_sectors())) {
          return py::float_(self.num_sectors());
        }
        return py::int_(static_cast<int64_t>(self.num_sectors()));
      })
      .def_property_readonly("sector_ind_len", &mpsdmrg::Group::sector_ind_len)
      .def_property_readonly("is_abelian", &mpsdmrg::Group::is_abelian)
      .def("is_valid_sector", &mpsdmrg::Group::is_valid_sector, py::arg("a"))
      .def("dual_sector", [](const mpsdmrg::Group& self, const mpsdmrg::Sector& a) {
        return mpsdmrg::to_python(self.dual_sector(a));
      }, py::arg("a"))
      .def("fusion_outcomes",
           [](const mpsdmrg::Group& self, const mpsdmrg::Sector& a, const mpsdmrg::Sector& b) {
             return mpsdmrg::to_python(self.fusion_outcomes(a, b));
           }, py::arg("a"), py::arg("b"))
      .def("sector_dim", &mpsdmrg::Group::sector_dim, py::arg("a"))
      .def("is_same_symmetry", &mpsdmrg::Group::is_same_symmetry, py::arg("other"))
      .def("fuse", [](const mpsdmrg::Group& self, py::args sectors) {
        std::vector<mpsdmrg::Sector> parsed;
        parsed.reserve(sectors.size());
        for (const py::handle& sector : sectors) {
          parsed.push_back(sector.cast<mpsdmrg::Sector>());
        }
        return mpsdmrg::to_python(self.fuse(parsed));
      })
      .def("can_fuse_to", &mpsdmrg::Group::can_fuse_to, py::arg("a"), py::arg("b"), py::arg("c"))
      .def("n_symbol", &mpsdmrg::Group::n_symbol, py::arg("a"), py::arg("b"), py::arg("c"))
      .def("fusion_tensor",
           [](const mpsdmrg::Group& self, const mpsdmrg::Sector& a, const mpsdmrg::Sector& b,
              const mpsdmrg::Sector& c, const c10::Device& device) {
             return self.fusion_tensor(a, b, c, device);
           }, py::arg("a"), py::arg("b"), py::arg("c"), py::arg("device"))
      .def("all_sectors", [](const mpsdmrg::Group& self) {
        if (std::isinf(self.num_sectors())) {
          throw mpsdmrg::value_error("Group " + self.group_name() +
                                     " has infinitely many sectors.");
        }
        return mpsdmrg::to_python(self.all_sectors());
      })
      .def("sector_of_dimension",
           [](const mpsdmrg::Group& self, int64_t dim) {
             return mpsdmrg::to_python(self.sector_of_dimension(dim));
           }, py::arg("dim"))
      .def("fusion_dtype", [](const mpsdmrg::Group& self) { return self.fusion_dtype(); })
      .def("reference_index",
           [](const mpsdmrg::Group& self, const mpsdmrg::Sector& sector) {
             return self.reference_index(sector);
           }, py::arg("sector"))
      .def("duality_matrix",
           [](const mpsdmrg::Group& self, const mpsdmrg::Sector& sector,
              const c10::Device& device) { return self.duality_matrix(sector, device); },
           py::arg("sector"), py::arg("device"))
      .def("__eq__", [](const mpsdmrg::Group& self, const py::object& other) -> py::object {
        if (!py::isinstance<mpsdmrg::Group>(other)) {
          return py::reinterpret_borrow<py::object>(Py_NotImplemented);
        }
        return py::bool_(self.is_same_symmetry(other.cast<const mpsdmrg::Group&>()));
      })
      .def("__hash__", [](const mpsdmrg::Group& self) {
        return py::hash(py::make_tuple(self.type_name(), self.group_name()));
      })
      .def("__repr__", &mpsdmrg::Group::group_name)
      .def("__reduce__", [](const mpsdmrg::Group& self) {
        return py::make_tuple(py::module_::import("mpsdmrg.symmetry.cpp").attr("_restore_group"),
                              py::make_tuple(self.group_name()));
      }, "the name of the singleton group, so that a group survives a pickle round trip");
}

}  // namespace mpsdmrg
