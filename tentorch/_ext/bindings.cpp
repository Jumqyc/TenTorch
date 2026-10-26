// The Python face of the engine: every class, free function and singleton the package exposes, in a
// translation unit of its own.  The engine itself holds no pybind11, so a binding edit recompiles
// only this file.
#include <torch/extension.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "../linalg/graded/compare.h"
#include "../linalg/graded/contract.h"
#include "../linalg/graded/einsum.h"
#include "../linalg/graded/svd.h"
#include "../symmetry/cat/abelian/abelian.h"
#include "../symmetry/cat/fibonacci/fibonacci.h"
#include "../symmetry/cat/group.h"
#include "../symmetry/cat/su2/su2.h"
#include "../symmetry/cat/su3/su3.h"
#include "../symmetry/tree/tree.h"
#include "../tensors/tensor.h"

namespace pybind11
{
  namespace detail
  {
    /**
     * @brief The sector as a Python tuple.
     * @details A sector is the key of a block, so it has to hash: a list would make
     * `{(0, 0): block}` impossible.  Loading accepts a tuple or a list, so a Python caller can write
     * either.
     */
    template <>
    struct type_caster<std::vector<int64_t>>
    {
      PYBIND11_TYPE_CASTER(std::vector<int64_t>, const_name("tuple"));

      bool load(handle source, bool)
      {
        if (!isinstance<tuple>(source) && !isinstance<list>(source)) return false;
        value.clear();
        for (const handle item : reinterpret_borrow<iterable>(source))
          value.push_back(item.cast<int64_t>());
        return true;
      }

      static handle cast(const std::vector<int64_t> &source, return_value_policy, handle)
      {
        tuple out(source.size());
        for (size_t at = 0; at < source.size(); ++at) out[at] = pybind11::cast(source[at]);
        return out.release();
      }
    };

    /**
     * @brief The sector as a Python tuple.
     * @details A sector is the key of a block, so it has to hash: a list would make
     * `{(0, 0): block}` impossible.  Loading accepts a tuple or a list, so a Python caller can write
     * either.
     */
    template <int64_t N>
    struct type_caster<sym::SectorOf<N>>
    {
      PYBIND11_TYPE_CASTER(sym::SectorOf<N>, const_name("tuple"));

      bool load(handle source, bool)
      {
        if (!isinstance<tuple>(source) && !isinstance<list>(source)) return false;
        std::vector<int64_t> charges;
        for (const handle item : reinterpret_borrow<iterable>(source))
          charges.push_back(item.cast<int64_t>());
        value = sym::SectorOf<N>(charges);
        return true;
      }

      static handle cast(const sym::SectorOf<N> &source, return_value_policy, handle)
      {
        tuple out(static_cast<size_t>(source.size()));
        for (int64_t at = 0; at < source.size(); ++at) out[static_cast<size_t>(at)] = pybind11::cast(source[at]);
        return out.release();
      }
    };
  } // namespace detail
} // namespace pybind11

namespace sym
{

  namespace
  {

    // The tensor as one line: how many legs, its charge, and the shape of every block.
    std::string tensor_repr(const Tensor &tensor)
    {
      std::string out = "Tensor(" + std::to_string(tensor.nlegs()) + " legs, q = " + render(tensor.qtotal()) +
                        ", " + std::to_string(tensor.blocks().size()) + " blocks:";
      for (const auto &[key, matrix] : tensor.blocks())
        out += " " + render(key.codomain) + " x " + render(key.domain) + " " +
               std::to_string(matrix.size(0)) + "x" + std::to_string(matrix.size(1));
      return out + ")";
    }

    // The leg as one line: every block and its multiplicity.
    std::string leg_repr(const Leg &leg)
    {
      std::string out = "Leg(" + leg.cat()->name() + ": ";
      for (int64_t block = 0; block < leg.nblocks(); ++block)
      {
        if (block != 0) out += ", ";
        out += render(leg.sector(block)) + ":" + std::to_string(leg.multiplicity(block));
      }
      return out + ")";
    }

    // The tree as one line: its legs, its inner couplings and the copies it fixed.
    std::string tree_repr(const FusionTree &tree)
    {
      return "FusionTree(" + render_list(tree.sectors) + " -> " + render_list(tree.inner) + " -> " +
             render(tree.coupled) + ", copies " + render_copies(tree.copies) + ")";
    }

    // A block key as one line.
    std::string key_repr(const BlockKey &key)
    {
      return "BlockKey(" + render(key.codomain) + ", " + render(key.domain) + ", " +
             std::to_string(key.copy) + ")";
    }

    // A tensor holding the given blocks, every other block zero.
    Tensor from_blocks(std::shared_ptr<FusionCategory> cat,
                       const std::vector<Leg> &codomain,
                       const std::vector<Leg> &domain,
                       const Sector &qtotal,
                       const py::object &blocks)
    {
      Tensor out(std::shared_ptr<const FusionCategory>(std::move(cat)), codomain, domain, qtotal);
      if (blocks.is_none()) return out;
      for (const auto &item : py::cast<py::dict>(blocks))
      {
        const BlockKey key = py::cast<BlockKey>(item.first);
        const at::Tensor block = py::cast<at::Tensor>(item.second);
        const at::Tensor &mine = out.block(key.codomain, key.domain, key.copy);
        if (block.sizes() != mine.sizes())
          throw std::invalid_argument("the block at " + key_repr(key) + " has shape " +
                                      std::to_string(block.size(0)) + "x" +
                                      std::to_string(block.size(1)) + ", not " +
                                      std::to_string(mine.size(0)) + "x" + std::to_string(mine.size(1)));
        out.block(key.codomain, key.domain, key.copy).copy_(block.to(out.dtype()).to(out.device()));
      }
      return out;
    }

    // A tensor whose blocks are drawn from the generator, so a caller seeds `torch` and gets a
    // reproducible one.
    Tensor random_tensor(std::shared_ptr<FusionCategory> cat,
                         const std::vector<Leg> &codomain,
                         const std::vector<Leg> &domain,
                         const Sector &qtotal)
    {
      Tensor out(std::shared_ptr<const FusionCategory>(std::move(cat)), codomain, domain, qtotal);
      for (const BlockKey &key : out.keys()) out.block(key.codomain, key.domain, key.copy).normal_();
      return out;
    }

    void bind_category(py::module_ &m)
    {
      py::class_<Outcome>(m, "Outcome")
          .def_readonly("sector", &Outcome::sector)
          .def_readonly("multiplicity", &Outcome::multiplicity)
          .def("__repr__", [](const Outcome &outcome)
               { return "Outcome(" + render(outcome.sector) + ", " + std::to_string(outcome.multiplicity) + ")"; });

      py::class_<FusionCategory, std::shared_ptr<FusionCategory>>(m, "FusionCategory")
          .def_property_readonly("name", &FusionCategory::name)
          .def("outcomes", &FusionCategory::outcomes, py::arg("a"), py::arg("b"))
          .def("n_symbol", &FusionCategory::n_symbol, py::arg("a"), py::arg("b"), py::arg("c"))
          .def("can_fuse_to", &FusionCategory::can_fuse_to, py::arg("a"), py::arg("b"), py::arg("c"))
          .def("F_symbol", &FusionCategory::F_symbol, py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"))
          .def("R_symbol", &FusionCategory::R_symbol, py::arg("b"), py::arg("c"))
          .def("q_dim", &FusionCategory::q_dim, py::arg("a"))
          .def("dual", &FusionCategory::dual, py::arg("a"))
          .def("frobenius_schur_sign", &FusionCategory::frobenius_schur_sign, py::arg("a"))
          .def("trivial", &FusionCategory::trivial)
          .def_property_readonly("dtype", &FusionCategory::dtype)
          .def_property_readonly("device", &FusionCategory::device)
          .def("clear_cache", &FusionCategory::clear_cache);
    }

    void bind_tree(py::module_ &m)
    {
      py::class_<Leg>(m, "Leg")
          .def(py::init([](std::shared_ptr<FusionCategory> cat, std::vector<Sector> sectors,
                           std::vector<int64_t> multiplicities)
                        { return Leg(std::shared_ptr<const FusionCategory>(std::move(cat)), std::move(sectors),
                                     std::move(multiplicities)); }),
               py::arg("cat"), py::arg("sectors"), py::arg("multiplicities"))
          .def_static("trivial", &Leg::trivial, py::arg("multiplets"), py::arg("cat"))
          .def_static("from_blocks", &Leg::from_blocks, py::arg("cat"), py::arg("entries"))
          .def_property_readonly("cat",
                                 [](const Leg &leg)
                                 { return std::const_pointer_cast<FusionCategory>(leg.cat()); })
          .def_property_readonly("sectors", &Leg::sectors)
          .def_property_readonly("multiplicities", &Leg::multiplicities)
          .def("nblocks", &Leg::nblocks)
          .def("sector", &Leg::sector, py::arg("block"))
          .def("multiplicity", &Leg::multiplicity, py::arg("block"))
          .def("multiplets", &Leg::multiplets)
          .def("fuse", &Leg::fuse, py::arg("other"))
          .def("conj", &Leg::conj)
          .def("capped", &Leg::capped, py::arg("max_multiplets"))
          .def("__repr__", &leg_repr)
          .def("__eq__", &Leg::operator==, py::is_operator())
          .def("__ne__", &Leg::operator!=, py::is_operator())
          .def("__hash__", &Leg::hash);

      py::class_<FusionTree>(m, "FusionTree")
          .def_readonly("sectors", &FusionTree::sectors)
          .def_readonly("inner", &FusionTree::inner)
          .def_readonly("copies", &FusionTree::copies)
          .def_readonly("coupled", &FusionTree::coupled)
          .def("size", &FusionTree::size)
          .def("chain", &FusionTree::chain)
          .def("__repr__", &tree_repr)
          .def("__eq__", &FusionTree::operator==, py::is_operator())
          .def("__ne__", &FusionTree::operator!=, py::is_operator())
          .def("__hash__", &FusionTree::hash);

      m.def("validate", &validate, py::arg("category"), py::arg("tree"));
      m.def("reachable_charges", &reachable_charges, py::arg("category"), py::arg("legs"));
      m.def("fusion_trees", &fusion_trees, py::arg("category"), py::arg("legs"), py::arg("coupled"));
    }

    void bind_tensor(py::module_ &m)
    {
      py::class_<BlockKey>(m, "BlockKey")
          .def(py::init([](Sector codomain, Sector domain, int64_t copy)
                        { return BlockKey{std::move(codomain), std::move(domain), copy}; }),
               py::arg("codomain"), py::arg("domain"), py::arg("copy") = 0)
          .def_readonly("codomain", &BlockKey::codomain)
          .def_readonly("domain", &BlockKey::domain)
          .def_readonly("copy", &BlockKey::copy)
          .def("__repr__", &key_repr)
          .def("__eq__", &BlockKey::operator==, py::is_operator())
          .def("__ne__", &BlockKey::operator!=, py::is_operator())
          .def("__lt__", &BlockKey::operator<, py::is_operator())
          .def("__hash__", [](const BlockKey &key) { return std::hash<std::string>{}(key_repr(key)); });

      py::class_<Tensor>(m, "Tensor")
          .def(py::init([](std::shared_ptr<FusionCategory> cat, std::vector<Leg> codomain,
                           std::vector<Leg> domain, Sector qtotal,
                           std::optional<c10::ScalarType> dtype, std::optional<c10::Device> device)
                        {
                          return Tensor(std::shared_ptr<const FusionCategory>(std::move(cat)),
                                        std::move(codomain), std::move(domain), std::move(qtotal), dtype,
                                        device);
                        }),
               py::arg("cat"), py::arg("codomain"), py::arg("domain"), py::arg("qtotal"),
               py::arg("dtype") = py::none(), py::arg("device") = py::none())
          .def_property_readonly("cat", [](const Tensor &tensor)
                                 { return std::const_pointer_cast<FusionCategory>(tensor.cat()); })
          .def_property_readonly("codomain", &Tensor::codomain)
          .def_property_readonly("domain", &Tensor::domain)
          .def_property_readonly("qtotal", &Tensor::qtotal)
          .def_property_readonly("dtype", &Tensor::dtype)
          .def_property_readonly("device", &Tensor::device)
          .def("nlegs", &Tensor::nlegs)
          .def("legs", &Tensor::legs)
          .def("nblocks", [](const Tensor &tensor) { return static_cast<int64_t>(tensor.blocks().size()); })
          .def("keys", &Tensor::keys)
          .def("conj", &Tensor::conj)
          .def("clone", &Tensor::clone)
          .def("repartition", &Tensor::repartition, py::arg("codomain"))
          .def("__iadd__", &Tensor::operator+=, py::is_operator())
          .def("__isub__", &Tensor::operator-=, py::is_operator())
          .def("__imul__",
               [](Tensor &tensor, const py::object &scalar) -> Tensor &
               {
                 if (py::isinstance<py::float_>(scalar) || py::isinstance<py::int_>(scalar))
                   return tensor *= scalar.cast<double>();
                 return tensor *= scalar.cast<std::complex<double>>();
               },
               py::is_operator())
          .def("__itruediv__", &Tensor::operator/=, py::is_operator())
          .def("__mul__",
               [](const Tensor &tensor, const py::object &scalar)
               {
                 Tensor out = tensor.clone();
                 if (py::isinstance<py::float_>(scalar) || py::isinstance<py::int_>(scalar))
                   out *= scalar.cast<double>();
                 else out *= scalar.cast<std::complex<double>>();
                 return out;
               },
               py::is_operator())
          .def("__rmul__",
               [](const Tensor &tensor, const py::object &scalar)
               {
                 Tensor out = tensor.clone();
                 if (py::isinstance<py::float_>(scalar) || py::isinstance<py::int_>(scalar))
                   out *= scalar.cast<double>();
                 else out *= scalar.cast<std::complex<double>>();
                 return out;
               },
               py::is_operator())
          .def("to_tuple", [](const Tensor &tensor)
               {
                 py::dict blocks;
                 for (const auto &[key, block] : tensor.blocks()) blocks[py::cast(key)] = block;
                 // filled here, not through `py::make_tuple`, which keeps a reference to the local
                 // dict and would hand Python a dangling one
                 py::tuple out(4);
                 out[0] = py::cast(tensor.codomain());
                 out[1] = py::cast(tensor.domain());
                 out[2] = py::cast(tensor.qtotal());
                 out[3] = blocks;
                 return out;
               })
          .def_static("from_tuple", &from_blocks, py::arg("cat"), py::arg("codomain"), py::arg("domain"),
                      py::arg("qtotal"), py::arg("blocks") = py::none())
          .def_static("random", &random_tensor, py::arg("cat"), py::arg("codomain"), py::arg("domain"),
                      py::arg("qtotal"))
          .def("__repr__", &tensor_repr);

      m.def("block_keys", &block_keys, py::arg("category"), py::arg("codomain"), py::arg("domain"),
            py::arg("qtotal"));
    }

    void bind_linalg(py::module_ &m)
    {
      m.def(
        "contract",
        [](const Tensor &first, const Tensor &other, const std::vector<int64_t> &first_domain,
           const std::vector<int64_t> &other_codomain) {
          return contract(first, other, first_domain, other_codomain);
        },
        py::arg("first"), py::arg("other"), py::arg("first_domain"), py::arg("other_codomain"));
      m.def("contract",
            py::overload_cast<const Tensor &, const Tensor &, const std::vector<int64_t> &,
                              const std::vector<int64_t> &, const Sector &>(&contract),
            py::arg("first"), py::arg("other"), py::arg("first_domain"), py::arg("other_codomain"),
            py::arg("charge"));
      m.def(
        "einsum",
        [](const std::string &spec, const py::args &args, const py::kwargs &options) {
          std::vector<Tensor> operands;
          operands.reserve(args.size());
          for (const py::handle item : args) operands.push_back(item.cast<Tensor>());
          for (const auto &item : options)
            if (std::string(py::str(item.first)) != "outcharge")
              throw std::invalid_argument("einsum does not take the keyword '" +
                                          std::string(py::str(item.first)) + "'");
          if (options.contains("outcharge") && !options["outcharge"].is_none())
            return einsum(spec, operands, options["outcharge"].cast<Sector>());
          return einsum(spec, operands);
        },
        py::arg("spec"));
      m.def("is_close", &is_close, py::arg("first"), py::arg("second"), py::arg("rtol") = 1e-10,
            py::arg("atol") = 1e-12);
      m.def("inner", &inner, py::arg("first"), py::arg("second"));
      m.def("norm", &norm, py::arg("tensor"));

      m.def("svd",
            [](const Tensor &tensor, std::optional<int64_t> max_bond, double tol, int64_t absorb)
            {
              Split split = svd(tensor, max_bond, tol, absorb);
              return py::make_tuple(split.first, split.values, split.second);
            },
            py::arg("tensor"), py::arg("max_bond") = py::none(), py::arg("tol") = 0.0, py::arg("absorb") = 0);
      m.def("svd",
            [](const Tensor &tensor, const std::vector<int64_t> &left, std::optional<int64_t> max_bond,
               double tol, int64_t absorb)
            {
              Split split = svd(tensor, left, max_bond, tol, absorb);
              return py::make_tuple(split.first, split.values, split.second);
            },
            py::arg("tensor"), py::arg("left"), py::arg("max_bond") = py::none(), py::arg("tol") = 0.0,
            py::arg("absorb") = 0);
      m.def("svdvals", py::overload_cast<const Tensor &>(&svdvals), py::arg("tensor"));
      m.def("svdvals", py::overload_cast<const Tensor &, const std::vector<int64_t> &>(&svdvals),
            py::arg("tensor"), py::arg("left"));
      m.def("qr",
            [](const Tensor &tensor)
            {
              Decomposition split = qr(tensor);
              return py::make_tuple(split.first, split.second);
            },
            py::arg("tensor"));
      m.def("qr",
            [](const Tensor &tensor, const std::vector<int64_t> &left)
            {
              Decomposition split = qr(tensor, left);
              return py::make_tuple(split.first, split.second);
            },
            py::arg("tensor"), py::arg("left"));
      m.def("rq",
            [](const Tensor &tensor)
            {
              Decomposition split = rq(tensor);
              return py::make_tuple(split.first, split.second);
            },
            py::arg("tensor"));
      m.def("rq",
            [](const Tensor &tensor, const std::vector<int64_t> &left)
            {
              Decomposition split = rq(tensor, left);
              return py::make_tuple(split.first, split.second);
            },
            py::arg("tensor"), py::arg("left"));
    }

    // Drop every table and cache the engine keeps: the vertex tables of the two groups first, then
    // the fusion table and the vertex cache every category object holds.
    void clear_caches()
    {
      su2::clear_tables();
      su3::clear_caches();
      no_symmetry()->clear_cache();
      u1_symmetry()->clear_cache();
      clear_z_n_caches();
      fermion_parity()->clear_cache();
      su2_symmetry()->clear_cache();
      su3_symmetry()->clear_cache();
      fibonacci_category()->clear_cache();
    }

  } // namespace

} // namespace sym

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("clear_caches", &sym::clear_caches, "drop every table and cache the engine keeps");

  sym::bind_category(m);
  sym::bind_tree(m);
  sym::bind_tensor(m);
  sym::bind_linalg(m);

  py::class_<sym::Group, sym::FusionCategory, std::shared_ptr<sym::Group>>(m, "Group")
      .def("clebsch_gordan", &sym::Group::clebsch_gordan, py::arg("a"), py::arg("b"), py::arg("c"),
           py::arg("copy"));
  py::class_<sym::AbelianGroup, sym::Group, std::shared_ptr<sym::AbelianGroup>>(m, "AbelianGroup");
  py::class_<sym::NoSymmetry, sym::AbelianGroup, std::shared_ptr<sym::NoSymmetry>>(m, "NoSymmetry")
      .def(py::init<>());
  py::class_<sym::U1Symmetry, sym::AbelianGroup, std::shared_ptr<sym::U1Symmetry>>(m, "U1Symmetry")
      .def(py::init<>());
  py::class_<sym::ZNSymmetry, sym::AbelianGroup, std::shared_ptr<sym::ZNSymmetry>>(m, "ZNSymmetry")
      .def(py::init<int64_t>(), py::arg("modulus"));
  py::class_<sym::FermionParity, sym::FusionCategory, std::shared_ptr<sym::FermionParity>>(m, "FermionParity")
      .def(py::init<>());
  py::class_<sym::SU2Symmetry, sym::Group, std::shared_ptr<sym::SU2Symmetry>>(m, "SU2Symmetry")
      .def(py::init<>());
  py::class_<sym::SU3Symmetry, sym::Group, std::shared_ptr<sym::SU3Symmetry>>(m, "SU3Symmetry")
      .def(py::init<>());
  py::class_<sym::FibonacciCategory, sym::FusionCategory, std::shared_ptr<sym::FibonacciCategory>>(
      m, "FibonacciCategory")
      .def(py::init<>());

  m.attr("no_symmetry") = py::cast(sym::no_symmetry());
  m.attr("u1") = py::cast(sym::u1_symmetry());
  m.def("z_n", &sym::z_n_symmetry, py::arg("modulus"));
  m.attr("fermion_parity") = py::cast(sym::fermion_parity());
  m.attr("su2") = py::cast(sym::su2_symmetry());
  m.attr("su3") = py::cast(sym::su3_symmetry());
  m.attr("fibonacci") = py::cast(sym::fibonacci_category());
}
