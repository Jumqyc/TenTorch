// The python bindings of the symmetry engine: every class and free function the package
// exposes, in a translation unit of its own, so a binding edit recompiles only this file.
#include <torch/extension.h>
#include <pybind11/stl.h>

#include "groups/group.h"
#include "groups/abelian.h"
#include "groups/su2.h"
#include "groups/su3.h"
#include "linalg.h"
#include "tensor.h"

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
  py::class_<mpsdmrg::Leg, std::shared_ptr<mpsdmrg::Leg>>(m, "Leg")
      .def(py::init<std::shared_ptr<mpsdmrg::Group>, std::vector<mpsdmrg::Sector>,
                    std::vector<int64_t>, int64_t>(),
           py::arg("group"), py::arg("sectors"), py::arg("slices"), py::arg("qconj") = 1)
      .def_static("trivial", &mpsdmrg::Leg::trivial, py::arg("dim"), py::arg("group") = nullptr,
                  py::arg("qconj") = 1)
      .def_static(
          "from_blocks",
          [](std::shared_ptr<mpsdmrg::Group> group,
             const std::vector<std::pair<mpsdmrg::Sector, int64_t>>& entries, int64_t qconj) {
            auto [leg, rank] = mpsdmrg::Leg::from_blocks(std::move(group), entries, qconj);
            return py::make_tuple(py::cast(leg), py::tuple(py::cast(rank)));
          },
          py::arg("group"), py::arg("entries"), py::arg("qconj") = 1)
      .def_property_readonly("group", &mpsdmrg::Leg::group)
      .def_property_readonly("sectors", [](const mpsdmrg::Leg& leg) {
        return mpsdmrg::to_python(leg.sectors());
      })
      .def_property_readonly("slices", [](const mpsdmrg::Leg& leg) {
        return py::tuple(py::cast(leg.slices()));
      })
      .def_property_readonly("qconj", &mpsdmrg::Leg::qconj)
      .def_property_readonly("dim", &mpsdmrg::Leg::dim)
      .def_property_readonly("nblocks", &mpsdmrg::Leg::nblocks)
      .def("sector",
           [](const mpsdmrg::Leg& leg, int64_t qi, bool dual) {
             return mpsdmrg::to_python(leg.sector(qi, dual));
           },
           py::arg("qi"), py::arg("dual") = false)
      .def("block_dim", &mpsdmrg::Leg::block_dim, py::arg("qi"))
      .def("multiplicity", &mpsdmrg::Leg::multiplicity, py::arg("qi"))
      .def("fuse", &mpsdmrg::Leg::fuse, py::arg("other"))
      .def("conj", &mpsdmrg::Leg::conj)
      .def("__eq__",
           [](const mpsdmrg::Leg& leg, py::object other) -> py::object {
             if (!py::isinstance<mpsdmrg::Leg>(other)) {
               return py::reinterpret_borrow<py::object>(Py_NotImplemented);
             }
             return py::cast(leg == other.cast<const mpsdmrg::Leg&>());
           })
      .def("__hash__", &mpsdmrg::Leg::hash)
      .def("__repr__", &mpsdmrg::Leg::repr)
      .def("__reduce__", [](const mpsdmrg::Leg& self) {
        return py::make_tuple(py::type::of<mpsdmrg::Leg>(),
                              py::make_tuple(self.group(), mpsdmrg::to_python(self.sectors()),
                                             py::cast(self.slices()), self.qconj()));
      }, "the constructor arguments, so that a leg survives a pickle round trip");

  py::class_<mpsdmrg::FusionTree, std::shared_ptr<mpsdmrg::FusionTree>>(m, "FusionTree")
      .def(py::init<std::shared_ptr<mpsdmrg::Group>, std::vector<mpsdmrg::Sector>,
                    std::vector<mpsdmrg::Sector>, mpsdmrg::Sector>(),
           py::arg("group"), py::arg("sectors"), py::arg("inner"), py::arg("coupled"))
      .def_static(
          "from_blocks",
          [](const std::shared_ptr<mpsdmrg::Group>& group, const std::vector<mpsdmrg::Leg>& legs,
             const py::tuple& qdata, const mpsdmrg::Sector& qtotal) {
            mpsdmrg::BlockKey key;
            for (py::size_t i = 0; i < qdata.size(); ++i) {
              if (static_cast<int64_t>(i) < static_cast<int64_t>(legs.size())) {
                key.legs.push_back(qdata[i].cast<int64_t>());
              } else {
                key.inner.push_back(qdata[i].cast<mpsdmrg::Sector>());
              }
            }
            return mpsdmrg::FusionTree::from_blocks(group, legs, key, qtotal);
          },
          py::arg("group"), py::arg("legs"), py::arg("qdata"), py::arg("qtotal"))
      .def_property_readonly("group", &mpsdmrg::FusionTree::group)
      .def_property_readonly("sectors", [](const mpsdmrg::FusionTree& tree) {
        return mpsdmrg::to_python(tree.sectors());
      })
      .def_property_readonly("inner", [](const mpsdmrg::FusionTree& tree) {
        return mpsdmrg::to_python(tree.inner());
      })
      .def_property_readonly("coupled", [](const mpsdmrg::FusionTree& tree) {
        return mpsdmrg::to_python(tree.coupled());
      })
      .def_property_readonly("chain", [](const mpsdmrg::FusionTree& tree) {
        return mpsdmrg::to_python(tree.chain());
      })
      .def("to_dense_block", &mpsdmrg::FusionTree::to_dense_block, py::arg("device"))
      .def("__eq__",
           [](const mpsdmrg::FusionTree& tree, py::object other) -> py::object {
             if (!py::isinstance<mpsdmrg::FusionTree>(other)) {
               return py::reinterpret_borrow<py::object>(Py_NotImplemented);
             }
             return py::cast(tree == other.cast<const mpsdmrg::FusionTree&>());
           })
      .def("__hash__", &mpsdmrg::FusionTree::hash)
      .def("__repr__", &mpsdmrg::FusionTree::repr)
      .def("__reduce__", [](const mpsdmrg::FusionTree& self) {
        return py::make_tuple(py::type::of<mpsdmrg::FusionTree>(),
                              py::make_tuple(self.group(), mpsdmrg::to_python(self.sectors()),
                                             mpsdmrg::to_python(self.inner()),
                                             mpsdmrg::to_python(self.coupled())));
      }, "the constructor arguments, so that a tree survives a pickle round trip");

  m.def("fusion_trees",
        [](const std::shared_ptr<mpsdmrg::Group>& group,
           const std::vector<mpsdmrg::Sector>& sectors, const mpsdmrg::Sector& coupled) {
          return py::tuple(py::cast(mpsdmrg::fusion_trees(group, sectors, coupled)));
        },
        py::arg("group"), py::arg("sectors"), py::arg("coupled"));
  m.def("dual_sign", &mpsdmrg::dual_sign, py::arg("group"), py::arg("sector"));
  m.def("stored_dense", &mpsdmrg::stored_dense, py::arg("legs"), py::arg("tree"),
        py::arg("device"));

  m.def("set_defaults", &mpsdmrg::set_defaults, py::arg("device"), py::arg("dtype"),
        "the device and dtype every fresh block buffer takes");
  m.def("cached_einsum", &mpsdmrg::cached_einsum,
        "contract symmetry-adapted tensors, block by block over the shared sectors");

  m.def("svd", &mpsdmrg::svd, py::arg("t"), py::arg("left"), py::arg("right"),
        py::arg("max_bond") = py::none(), py::arg("tol") = 0.0,
        py::arg("direction") = "right",
        "split a tensor as u s vh, one fused charge at a time");
  m.def("qr", &mpsdmrg::qr, py::arg("t"), py::arg("left"), py::arg("right"),
        "split a tensor as q r with the isometry on the left");
  m.def("rq", &mpsdmrg::rq, py::arg("t"), py::arg("left"), py::arg("right"),
        "split a tensor as r q with the isometry on the right");
  m.def("svdvals", &mpsdmrg::svdvals, py::arg("t"), py::arg("left"), py::arg("right"),
        "the singular values of a partition, by fused charge");
  m.def("norm", &mpsdmrg::norm, py::arg("t"), py::arg("dim") = py::none(),
        "the Frobenius norm of a tensor, reduced over dim");
  m.def("inner", &mpsdmrg::inner, py::arg("a"), py::arg("b"),
        "the inner product of two tensors over all legs");
  m.def("trace", &mpsdmrg::trace, py::arg("t"), "the trace of a two-leg tensor");

  py::class_<mpsdmrg::Tensor>(m, "Tensor")
      .def(py::init([](const py::object& blocks, const py::object& qdata, const py::object& legs,
                       const py::object& qtotal, const py::object& group) {
             std::vector<mpsdmrg::Leg> parsed_legs = mpsdmrg::parse_legs(legs);
             std::vector<at::Tensor> parsed_blocks;
             for (auto item : py::reinterpret_borrow<py::iterable>(blocks)) {
               parsed_blocks.push_back(item.cast<at::Tensor>());
             }
             // the keys are parsed before the move: argument evaluation has no order
             std::vector<mpsdmrg::BlockKey> parsed_keys =
                 mpsdmrg::parse_keys(qdata, static_cast<int64_t>(parsed_legs.size()));
             std::optional<mpsdmrg::Sector> parsed_qtotal;
             if (!qtotal.is_none()) parsed_qtotal = qtotal.cast<mpsdmrg::Sector>();
             return mpsdmrg::Tensor(
                 std::move(parsed_blocks), std::move(parsed_keys), std::move(parsed_legs),
                 parsed_qtotal,
                 group.is_none() ? nullptr : group.cast<std::shared_ptr<mpsdmrg::Group>>());
           }),
           py::arg("blocks"), py::arg("qdata"), py::arg("legs"), py::arg("qtotal") = py::none(),
           py::arg("group") = py::none())
      .def_static("from_torch",
                  [](const at::Tensor& data, const py::object& legs, const py::object& qtotal) {
                    std::optional<std::vector<mpsdmrg::Leg>> parsed_legs;
                    if (!legs.is_none()) parsed_legs = mpsdmrg::parse_legs(legs);
                    std::optional<mpsdmrg::Sector> parsed_qtotal;
                    if (!qtotal.is_none()) parsed_qtotal = qtotal.cast<mpsdmrg::Sector>();
                    return mpsdmrg::Tensor::from_torch(data, parsed_legs, parsed_qtotal);
                  },
                  py::arg("data"), py::arg("legs") = py::none(), py::arg("qtotal") = py::none())
      .def_static("rand", &mpsdmrg::Tensor::rand)
      .def_static("ones", &mpsdmrg::Tensor::ones)
      .def_static("zeros", &mpsdmrg::Tensor::zeros)
      .def_property_readonly("blocks", &mpsdmrg::Tensor::blocks)
      .def_property_readonly("qdata", [](const mpsdmrg::Tensor& self) {
        py::tuple out(self.qdata().size());
        for (std::size_t i = 0; i < self.qdata().size(); ++i) {
          out[i] = mpsdmrg::key_to_python(self.qdata()[i]);
        }
        return out;
      })
      .def_property_readonly("legs", [](const mpsdmrg::Tensor& self) {
        py::tuple out(self.legs().size());
        for (std::size_t i = 0; i < self.legs().size(); ++i) out[i] = py::cast(self.legs()[i]);
        return out;
      })
      .def_property_readonly("qtotal",
                             [](const mpsdmrg::Tensor& self) { return mpsdmrg::to_python(self.qtotal()); })
      .def_property_readonly("group", &mpsdmrg::Tensor::group)
      .def_property_readonly("shape", [](const mpsdmrg::Tensor& self) {
        return mpsdmrg::to_python(self.shape());
      })
      .def_property_readonly("dtype", &mpsdmrg::Tensor::dtype)
      .def_property_readonly("device", &mpsdmrg::Tensor::device)
      .def_property_readonly("is_dense", &mpsdmrg::Tensor::is_dense)
      .def_property_readonly("real", &mpsdmrg::Tensor::real)
      .def("numel", &mpsdmrg::Tensor::numel)
      .def("element_size", &mpsdmrg::Tensor::element_size)
      .def("to_torch", &mpsdmrg::Tensor::to_torch)
      .def("item", &mpsdmrg::Tensor::item)
      .def("clone", &mpsdmrg::Tensor::clone)
      .def("detach", &mpsdmrg::Tensor::detach)
      .def("conj", &mpsdmrg::Tensor::conj)
      .def("to", &mpsdmrg::Tensor::to, py::arg("device") = py::none(),
           py::arg("dtype") = py::none())
      .def("combine_legs", &mpsdmrg::Tensor::combine_legs, py::arg("groups"))
      .def("__getitem__", &mpsdmrg::Tensor::getitem)
      .def("__setitem__", &mpsdmrg::Tensor::setitem)
      .def("__neg__", &mpsdmrg::Tensor::operator-)
      .def("__add__", &mpsdmrg::Tensor::add)
      .def("__radd__", &mpsdmrg::Tensor::add)
      .def("__sub__", &mpsdmrg::Tensor::sub)
      .def("__mul__", [](const mpsdmrg::Tensor& self, const py::object& other) {
        return self.scale(other, false);
      })
      .def("__rmul__", [](const mpsdmrg::Tensor& self, const py::object& other) {
        return self.scale(other, false);
      })
      .def("__truediv__", [](const mpsdmrg::Tensor& self, const py::object& other) {
        return self.scale(other, true);
      })
      .def("__iadd__", [](mpsdmrg::Tensor& self, const py::object& other) -> mpsdmrg::Tensor& {
        return self.iadd(other);
      }, py::return_value_policy::reference)
      .def("__imul__", [](mpsdmrg::Tensor& self, const py::object& other) -> mpsdmrg::Tensor& {
        return self.imul(other);
      }, py::return_value_policy::reference)
      .def("__itruediv__", [](mpsdmrg::Tensor& self, const py::object& other) -> mpsdmrg::Tensor& {
        return self.idiv(other);
      }, py::return_value_policy::reference)
      .def("__reduce__", [](const mpsdmrg::Tensor& self) {
        py::list blocks;
        for (const at::Tensor& block : self.blocks()) blocks.append(py::cast(block));
        py::tuple keys(self.qdata().size());
        for (std::size_t i = 0; i < self.qdata().size(); ++i) {
          keys[i] = mpsdmrg::key_to_python(self.qdata()[i]);
        }
        py::tuple legs(self.legs().size());
        for (std::size_t i = 0; i < self.legs().size(); ++i) legs[i] = py::cast(self.legs()[i]);
        return py::make_tuple(py::type::of<mpsdmrg::Tensor>(),
                              py::make_tuple(blocks, keys, legs,
                                             mpsdmrg::to_python(self.qtotal()), self.group()));
      }, "the constructor arguments, so that a tensor survives a pickle round trip");

}
