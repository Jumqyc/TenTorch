// The python bindings of the symmetry engine: every class and free function the package
// exposes, in a translation unit of its own, so a binding edit recompiles only this file.
#include <torch/extension.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

#include "../symmetry/cat/group.h"
#include "../symmetry/cat/abelian/abelian.h"
#include "../symmetry/cat/fibonacci/fibonacci.h"
#include "../symmetry/cat/su2/su2.h"
#include "../symmetry/cat/su3/su3.h"
#include "../linalg/graded/linalg.h"
#include "../tensors/tensor.h"
#include "../tensors/diagram.h"

namespace sym {

// Drop every table and cache the engine keeps. They are read-only, so this only costs a rebuild.
void clear_caches() {
  su2::clear_tables();
  su3::clear_caches();
}

}  // namespace sym

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("clear_caches", &sym::clear_caches, "drop every table and cache the engine keeps");

  sym::bind_category(m);
  sym::bind_group(m);
  sym::bind_abelian(m);
  sym::bind_fibonacci(m);
  sym::bind_su2(m);
  sym::bind_su3(m);
  sym::bind_tree(m);
  sym::engine::bind_engine(m);
  sym::engine::bind_split(m);
  sym::bind_linalg(m);
}
