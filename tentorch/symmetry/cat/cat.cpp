// The categorical interface: the fusion every category shares, and the two questions that follow
// from it so that no category answers them twice.
#include "cat.h"

#include <pybind11/stl.h>

#include <string>
#include <utility>
#include <vector>

namespace sym
{

  c10::Device default_device()
  {
    return at::globalContext().hasCUDA() ? c10::Device(c10::kCUDA) : c10::Device(c10::kCPU);
  }

  FusionCategory::FusionCategory(std::string name, c10::Device device)
      : name_(std::move(name)), device_(device) {}

  const std::string &FusionCategory::name() const { return name_; }

  c10::Device FusionCategory::device() const { return device_; }

  const std::vector<Outcome> &FusionCategory::outcomes(const Sector &a, const Sector &b) const
  {
    const auto by_first = outcomes_cache_.find(a);
    if (by_first != outcomes_cache_.end())
    {
      const auto by_second = by_first->second.find(b);
      if (by_second != by_first->second.end())
      {
        return by_second->second;
      }
    }
    return outcomes_cache_[a].emplace(b, compute_outcomes(a, b)).first->second;
  }

  void FusionCategory::clear_cache() const { outcomes_cache_.clear(); }

  // The multiplicity of `a (x) b -> c`: how many distinct vertices couple the pair into `c`, and
  // zero when they do not couple to it at all.
  int64_t FusionCategory::n_symbol(const Sector &a, const Sector &b, const Sector &c) const
  {
    for (const Outcome &outcome : outcomes(a, b))
    {
      if (outcome.sector == c)
      {
        return outcome.multiplicity;
      }
    }
    return 0;
  }

  // Whether `a (x) b` couples to `c` at all.
  bool FusionCategory::can_fuse_to(const Sector &a, const Sector &b, const Sector &c) const
  {
    return n_symbol(a, b, c) > 0;
  }

  // Bind the fusion outcome and the base category.  A concrete category is bound next to its own
  // implementation; what lives here is the part of the interface all of them share.
  void bind_category(py::module_ &m)
  {
    py::class_<Outcome>(m, "Outcome")
        .def_readonly("sector", &Outcome::sector)
        .def_readonly("multiplicity", &Outcome::multiplicity)
        .def("__repr__", [](const Outcome &outcome)
             {
               return "Outcome(sector=" +
                      py::cast<std::string>(py::repr(to_python(outcome.sector))) +
                      ", multiplicity=" + std::to_string(outcome.multiplicity) + ")";
             });

    py::class_<FusionCategory, std::shared_ptr<FusionCategory>>(m, "FusionCategory")
        .def_property_readonly("name", &FusionCategory::name)
        .def("outcomes", [](const FusionCategory &category, const Sector &a, const Sector &b)
             {
               py::list out;
               for (const Outcome &outcome : category.outcomes(a, b))
               {
                 out.append(py::cast(outcome));
               }
               return out;
             })
        .def("n_symbol", &FusionCategory::n_symbol)
        .def("can_fuse_to", &FusionCategory::can_fuse_to)
        .def("F_symbol", &FusionCategory::F_symbol)
        .def("R_symbol", &FusionCategory::R_symbol)
        .def("q_dim", &FusionCategory::q_dim)
        .def("dim", &FusionCategory::dim)
        .def("dual", &FusionCategory::dual)
        .def("trivial", &FusionCategory::trivial)
        .def("dtype", &FusionCategory::dtype)
        .def("device", &FusionCategory::device);
  }

} // namespace sym
