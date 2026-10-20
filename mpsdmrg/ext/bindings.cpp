// The python bindings of the symmetry engine: every class and free function the package
// exposes, in a translation unit of its own, so a binding edit recompiles only this file.
#include <torch/extension.h>
#include <pybind11/stl.h>

#include "../symmetry/cat/group.h"
#include "../symmetry/cat/abelian/abelian.h"
#include "../symmetry/cat/fibonacci/fibonacci.h"
#include "../symmetry/cat/su2/su2.h"
#include "../symmetry/cat/su3/su3.h"
#include "../linalg/linalg.h"
#include "../tensor/tensor.h"
#include "../tensor/diagram.h"

namespace sym {

// Drop every table and cache the engine keeps. They are read-only, so this only costs a rebuild.
void clear_caches() {
  no_symmetry()->clear_caches();
  u1_symmetry()->clear_caches();
  fermion_parity()->clear_caches();
  su2_symmetry()->clear_caches();
  su3_symmetry()->clear_caches();
  fibonacci_category()->clear_caches();
  clear_weight_cache();
#ifdef DENSEPATH
  clear_stored_dense();
  clear_tree_caches();
#endif
}

}  // namespace sym

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::register_exception_translator([](std::exception_ptr failure) {
    try {
      if (failure) {
        std::rethrow_exception(failure);
      }
    } catch (const sym::value_error& error) {
      PyErr_SetString(PyExc_ValueError, error.what());
    } catch (const sym::not_implemented_error& error) {
      PyErr_SetString(PyExc_NotImplementedError, error.what());
    }
  });

  m.def("clear_caches", &sym::clear_caches, "drop every table and cache the engine keeps");

  sym::bind_category(m);
  sym::bind_group(m);
  sym::bind_abelian(m);
  sym::bind_fibonacci(m);
  sym::bind_su2(m);
  sym::bind_su3(m);
  sym::bind_tree(m);
  m.def("dense_path", &sym::dense_path, "Whether a contraction is forced onto the dense route.");
#ifdef DENSEPATH
  m.def("set_dense_path", &sym::set_dense_path, "Force the dense route for debugging.", py::arg("enabled"));
#endif
  sym::bind_tensor(m);
  sym::bind_linalg(m);
  py::class_<sym::RouteStats>(m, "RouteStats")
      .def_readonly("weight_calls", &sym::RouteStats::weight_calls)
      .def_readonly("weight_hits", &sym::RouteStats::weight_hits)
      .def_readonly("diagram_misses", &sym::RouteStats::diagram_misses)
      .def_readonly("entry_calls", &sym::RouteStats::entry_calls)
      .def_readonly("entry_values", &sym::RouteStats::entry_values)
      .def_readonly("move_calls", &sym::RouteStats::move_calls)
      .def_readonly("move_states", &sym::RouteStats::move_states)
      .def_readonly("weigh_calls", &sym::RouteStats::weigh_calls)
      .def_readonly("weigh_pairs", &sym::RouteStats::weigh_pairs);
  m.def("route_stats", &sym::route_stats, "the work the diagram route did since the last reset");
  m.def("reset_route_stats", &sym::reset_route_stats, "set every route counter back to zero");
}
