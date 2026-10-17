// The block-sparse tensor of the engine: one dense buffer per charge block, the einsum of
// those blocks, and the element-wise arithmetic dmrg.py drives a sweep with.
#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../symmetry/tree/tree.h"

namespace sym
{

  // The shape of a tensor: one dimension per leg.
  using Shape = std::vector<int64_t>;

  // The label alphabet of an einsum subscript, in the order free labels are drawn from.
  extern const std::string kLabels;

  // The device and dtype a block buffer defaults to, set once from ``mpsdmrg.base`` by the
  // Python loader so the engine follows the project defaults.
  void set_defaults(const std::string &device, const std::string &dtype);

  // Allocation options from a device and a dtype, a ``torch.device`` and a ``torch.dtype`` or
  // their names; the project default where an argument is None.
  at::TensorOptions block_options(const py::object &dtype = py::none(),
                                  const py::object &device = py::none());

  // The charge the blocks a key selects fuse to, over the axes ``axes`` in order.
  Sector fused_charge(const std::shared_ptr<Group> &group,
                      const std::vector<Leg> &legs,
                      const BlockKey &key,
                      const std::vector<int64_t> &axes);

  // The charge every sector of ``sectors`` fuses to.
  Sector fused_qtotal(const std::shared_ptr<Group> &group, const std::vector<Sector> &sectors);

  // Every block key of ``legs`` that can carry ``qtotal``, in the Python engine's order.
  std::vector<BlockKey> combos(const std::vector<Leg> &legs, const Sector &qtotal);

  // The shape of the block a key selects: one multiplicity per leg.
  Shape block_shape(const std::vector<Leg> &legs, const BlockKey &key);

  // The running start of every block of a leg, the windows a raw factor is cut into.
  std::vector<std::vector<int64_t>> running_offsets(const std::vector<Leg> &legs);

  // The labels of one einsum subscript: one label id per axis of every operand and of the
  // result, in the result's order.
  struct EinsumSpec
  {
    std::string equation;
    std::vector<std::vector<int64_t>> labels;
    std::vector<int64_t> out;
  };

  // Parse ``lhs->rhs``; a missing ``->`` keeps every label that appears exactly once.
  EinsumSpec parse_einsum(const std::string &spec, std::size_t operands);

  // The subscript of a sequence of labels.
  std::string label_string(const std::vector<int64_t> &labels);

  // Drop the cached CG folds and tree weights.
  void clear_tree_caches();

  // One block key as Python spells it: the leg blocks, then the coupled sectors.
  py::tuple key_to_python(const BlockKey &key);

  // The block keys of a Python ``qdata`` argument; the first ``legs`` entries of an entry are
  // the leg blocks, the rest the coupled sectors.
  std::vector<BlockKey> parse_keys(const py::object &qdata, const std::vector<Leg> &legs);

  // The legs of a Python sequence of legs.
  Leg natural_leg(const std::shared_ptr<Group> &group, int64_t dim);
  std::vector<Leg> parse_legs(const py::object &legs, const py::object &group_value = py::none());

  // The dimensions of a ``*shape`` argument: a single sequence is spread into the dimensions.
  Shape parse_shape(const py::args &shape);

  // One block-sparse tensor: the dense buffer of every charge block and the key of each.
  //
  // The keys are kept sorted, so two tensors of equal legs hold their blocks in the same
  // order.  A block of a group with one fusion outcome is keyed by its leg blocks alone; a
  // block of a group with several outcomes also carries the coupling of every inner vertex.
  class Tensor
  {
  public:
    // Args:
    //   blocks: the dense buffer of every charge block.
    //   qdata: the block key of every buffer, in the same order.
    //   legs: the legs of the tensor, every block indexed by them.
    //   qtotal: the charge of the tensor; the trivial charge when it is missing.
    //   group: the group of the charges; the group of the legs when it is missing.
    // Raises:
    //   AssertionError: If a key does not describe a block of those legs.
    // ``homogeneous`` false builds a tensor that sums several total charges: every block
    // keeps the coupled charge of its own fusion path and ``qtotal`` is not comparable.
    Tensor(std::vector<at::Tensor> blocks, std::vector<BlockKey> qdata, std::vector<Leg> legs,
           std::optional<Sector> qtotal = std::nullopt,
           std::shared_ptr<Group> group = nullptr, bool homogeneous = true);

    // The dense tensor that stores ``data``, with the legs ``legs`` or one trivial leg per
    // dimension; the buffers share storage with ``data`` when it holds one whole block.
    static Tensor from_torch(const at::Tensor &data,
                             std::optional<std::vector<Leg>> legs = std::nullopt,
                             std::optional<Sector> qtotal = std::nullopt);

    // A tensor of random values, of ones or of zeros, of the shape ``shape``, the legs
    // ``legs`` and the charge ``qtotal``; ``dtype`` and ``device`` default to the project
    // defaults.
    static Tensor rand(py::args shape, const py::kwargs &options);
    static Tensor ones(py::args shape, const py::kwargs &options);
    static Tensor zeros(py::args shape, const py::kwargs &options);

    // The engine's own buffers: a write into one of them writes into the tensor.
    const std::vector<at::Tensor> &blocks() const { return blocks_; }
    const std::vector<BlockKey> &qdata() const { return qdata_; }
    const std::vector<Leg> &legs() const { return legs_; }
    const Sector &qtotal() const;
    bool homogeneous() const { return homogeneous_; }
    const std::shared_ptr<Group> &group() const { return group_; }

    // The dimensions of the legs, and the torch dtype and device of a block buffer.
    Shape shape() const;
    py::object dtype() const;
    py::object device() const;
    // Whether a single block of the trivial group stores the whole tensor.
    bool is_dense() const;
    Tensor real() const;
    int64_t numel() const;
    int64_t element_size() const;

    // The dense tensor of the sector-concatenated basis.
    at::Tensor to_torch() const;
  /*
   * The same tensor with every leg unresolved: each dense window of a resolved leg is projected
   * onto the Clebsch-Gordan component the expansion writes, which is the exact inverse of
   * ``to_torch`` up to the multiplicity representative (the projection always fills the first).
   * Returns this tensor when no leg is resolved.
   */
  Tensor unresolve() const;

  /*
   * The same tensor with every leg resolved: each plain block is written into the dense window of
   * its charge, so the tensor can take a bare matrix on a leg of more than one component.
   */
  Tensor resolve() const;

    // The single entry of a tensor of no indices.
    py::object item() const;
    Tensor clone() const;
    Tensor detach() const;
    // The complex conjugate, with the dual legs reversed.
    Tensor conj() const;
    // The same tensor on another device or dtype.
    Tensor to(const py::object &device, const py::object &dtype) const;
    // Fuse each group of legs into one leg, in the order of ``groups``.
    Tensor combine_legs(const std::vector<std::vector<int64_t>> &groups) const;

    py::object getitem(const py::object &key) const;
    void setitem(const py::object &key, const py::object &value);

    Tensor operator-() const;
    Tensor add(const py::object &other) const;
    Tensor sub(const py::object &other) const;
    Tensor scale(const py::object &other, bool divide = false) const;
    Tensor &iadd(const py::object &other);
    Tensor &imul(const py::object &other);
    Tensor &idiv(const py::object &other);

  private:
    // The constructor of ``zeros``, ``ones`` and ``rand``: ``kind`` picks the fill.
    static Tensor filled(py::args shape, const py::kwargs &options, int kind);

    // The sum with a tensor of the same legs, a raw array or a number, negated for ``sub``.
    Tensor binary_add(const py::object &other, bool negate) const;

    // Sort the keys, assert they are distinct and that every block matches its key.
    void canonicalize();

    std::vector<at::Tensor> blocks_;
    std::vector<BlockKey> qdata_;
    std::vector<Leg> legs_;
    Sector qtotal_;
    // False when the blocks couple to different total charges, i.e. the tensor sums them.
    bool homogeneous_ = true;
    std::shared_ptr<Group> group_;
  };

  // Contract symmetry-adapted tensors with the einsum of the Python engine: block by block
  // over the shared sector of every contracted subscript.
  Tensor cached_einsum(const std::string &spec, py::args tensors);

  // The block-by-block einsum.
  Tensor blocked_einsum(const std::string &spec, const std::vector<Tensor> &tensors);

  // Bind the names of this module to python.
  void bind_tensor(py::module_ &m);

} // namespace sym
