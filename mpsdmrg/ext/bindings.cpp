// The python bindings of the symmetry engine: every class and free function the package
// exposes, in a translation unit of its own, so a binding edit recompiles only this file.
#include <torch/extension.h>
#include <pybind11/stl.h>

#include "../symmetry/group/group.h"
#include "../symmetry/group/abelian.h"
#include "../symmetry/group/su2.h"
#include "../symmetry/group/su3.h"
#include "../linalg/linalg.h"
#include "../tensor/tensor.h"

namespace mpsdmrg {

// Drop every table and cache the engine keeps. They are read-only, so this only costs a rebuild.
void clear_caches() {
  su2::clear_tables();
  clear_stored_dense();
  clear_tree_caches();
}

}  // namespace mpsdmrg

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::register_exception_translator([](std::exception_ptr failure) {
    try {
      if (failure) {
        std::rethrow_exception(failure);
      }
    } catch (const mpsdmrg::value_error& error) {
      PyErr_SetString(PyExc_ValueError, error.what());
    } catch (const mpsdmrg::not_implemented_error& error) {
      PyErr_SetString(PyExc_NotImplementedError, error.what());
    }
  });

  m.def("clear_caches", &mpsdmrg::clear_caches, "drop every table and cache the engine keeps");

  mpsdmrg::bind_group(m);
  mpsdmrg::bind_abelian(m);
  mpsdmrg::bind_su2(m);
  mpsdmrg::bind_su3(m);
  mpsdmrg::bind_tree(m);
  mpsdmrg::bind_tensor(m);
  mpsdmrg::bind_linalg(m);
}
