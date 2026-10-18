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

namespace sym {

// Drop every table and cache the engine keeps. They are read-only, so this only costs a rebuild.
void clear_caches() {
  no_symmetry()->clear_caches();
  u1_symmetry()->clear_caches();
  fermion_parity()->clear_caches();
  su2_symmetry()->clear_caches();
  su3_symmetry()->clear_caches();
  fibonacci_category()->clear_caches();
  clear_stored_dense();
  clear_tree_caches();
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
  sym::bind_tensor(m);
  sym::bind_linalg(m);
}
