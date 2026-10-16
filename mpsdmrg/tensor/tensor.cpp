// The block-sparse tensor of the engine and the contraction of its blocks.
#include "tensor.h"

#include <ATen/ScalarOps.h>
#include <ATen/TensorIndexing.h>
#include <ATen/core/Tensor.h>

#include <algorithm>
#include <cstddef>
#include <map>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace mpsdmrg {

const std::string kLabels = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

namespace {

// The device and dtype every freshly allocated block buffer takes.
c10::Device& default_device() {
  static c10::Device device{"cpu"};
  return device;
}

at::ScalarType& default_dtype() {
  static at::ScalarType dtype = at::kComplexDouble;
  return dtype;
}

std::string strip_torch(const std::string& name) {
  return name.rfind("torch.", 0) == 0 ? name.substr(6) : name;
}

at::ScalarType scalar_type_of_name(const std::string& name) {
  if (name == "complex128" || name == "cdouble" || name == "complex") return at::kComplexDouble;
  if (name == "complex64" || name == "cfloat") return at::kComplexFloat;
  if (name == "float64" || name == "double") return at::kDouble;
  if (name == "float32" || name == "float") return at::kFloat;
  throw value_error("The engine has no dtype " + name + ".");
}

at::ScalarType scalar_type_of(const py::object& value) {
  if (value.is_none()) return default_dtype();
  return scalar_type_of_name(strip_torch(py::cast<std::string>(py::str(value))));
}

c10::Device device_of(const py::object& value) {
  if (value.is_none()) return default_device();
  const std::string name = py::cast<std::string>(py::str(value));
  return c10::Device(name);
}

// A Python sequence of integers as a sector.
Sector sector_of(const py::object& value) {
  Sector sector;
  for (auto item : py::reinterpret_borrow<py::iterable>(value)) sector.push_back(py::cast<int64_t>(item));
  return sector;
}

// The labels of an einsum subscript as ids into ``kLabels``.
std::vector<int64_t> label_ids_of(const std::string& text) {
  std::vector<int64_t> out;
  for (char c : text) {
    const std::size_t id = kLabels.find(c);
    if (id == std::string::npos) throw value_error(std::string("Unknown einsum label '") + c + "'.");
    out.push_back(static_cast<int64_t>(id));
  }
  return out;
}

bool contains(const std::vector<int64_t>& values, int64_t value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

// Run ``body`` once per combination of ``sizes``, the last index varying fastest.
template <typename Body>
void for_each_combination(const std::vector<std::size_t>& sizes, Body&& body) {
  const std::size_t n = sizes.size();
  for (std::size_t size : sizes) {
    if (size == 0) return;
  }
  std::vector<std::size_t> index(n, 0);
  while (true) {
    body(index);
    std::size_t k = n;
    bool carry = true;
    while (k > 0 && carry) {
      --k;
      if (++index[k] < sizes[k]) {
        carry = false;
      } else {
        index[k] = 0;
      }
    }
    if (carry) return;
  }
}

// The last axis of ``tensor`` at one index, with the axis removed.
at::Tensor take_last(const at::Tensor& tensor, int64_t index) {
  std::vector<at::indexing::TensorIndex> select(tensor.dim() - 1, at::indexing::Slice());
  select.emplace_back(index);
  return at::indexing::get_item(tensor, select);
}

// One entry of a block buffer at a point, with every axis removed.
at::Tensor scalar_at(const at::Tensor& tensor, const std::vector<int64_t>& point) {
  std::vector<at::indexing::TensorIndex> index;
  for (int64_t p : point) index.emplace_back(p);
  return at::indexing::get_item(tensor, index);
}

// A python object holding a torch tensor.  ``py::isinstance`` of ``at::Tensor`` cannot tell:
// the caster of ``at::Tensor`` is not registered with pybind11's type registry, so it sees a
// raw tensor as an unknown object and the scalar branch would then cast it to a number.
bool is_tensor(const py::handle& value) { return THPVariable_Check(value.ptr()); }

// One Python index as the index type of the tensor API.
at::indexing::TensorIndex tensor_index(const py::handle& key) {
  if (key.is_none()) return at::indexing::None;
  if (py::isinstance<py::bool_>(key)) return key.cast<bool>();
  if (py::isinstance<py::int_>(key)) return key.cast<int64_t>();
  if (key.ptr() == Py_Ellipsis) return at::indexing::Ellipsis;
  if (is_tensor(key)) return key.cast<at::Tensor>();
  if (PySlice_Check(key.ptr())) {
    Py_ssize_t start = 0;
    Py_ssize_t stop = 0;
    Py_ssize_t step = 1;
    if (PySlice_Unpack(key.ptr(), &start, &stop, &step) < 0) throw py::error_already_set();
    return at::indexing::Slice(c10::SymInt(static_cast<int64_t>(start)),
                               c10::SymInt(static_cast<int64_t>(stop)),
                               c10::SymInt(static_cast<int64_t>(step)));
  }
  throw value_error("The engine has no index of this type.");
}

// The index expression of a Python key: one index or a tuple of them.
std::vector<at::indexing::TensorIndex> index_of(const py::object& key) {
  std::vector<at::indexing::TensorIndex> index;
  if (py::isinstance<py::tuple>(key)) {
    for (auto item : py::reinterpret_borrow<py::tuple>(key)) index.push_back(tensor_index(item));
  } else {
    index.push_back(tensor_index(key));
  }
  return index;
}

// A Python number as the scalar of the ATen API, which has no Python caster of its own.
c10::Scalar scalar_of(const py::object& value) {
  if (py::isinstance<py::bool_>(value)) return c10::Scalar(value.cast<bool>());
  if (py::isinstance<py::int_>(value)) return c10::Scalar(value.cast<int64_t>());
  if (PyComplex_Check(value.ptr())) {
    return c10::Scalar(c10::complex<double>(PyComplex_RealAsDouble(value.ptr()),
                                            PyComplex_ImagAsDouble(value.ptr())));
  }
  return c10::Scalar(value.cast<double>());
}

// ``torch.as_tensor``: a raw tensor is used as it is, a number becomes one element.
at::Tensor as_tensor(const py::object& value, const at::TensorOptions& options) {
  if (is_tensor(value)) return value.cast<at::Tensor>();
  return at::scalar_tensor(scalar_of(value), options);
}

// The dense contraction of the operands of one block combination, through the einsum parser.
at::Tensor reduce_blocks(const std::vector<at::Tensor>& operands,
                         const std::vector<std::vector<int64_t>>& labels,
                         const std::vector<int64_t>& out) {
  std::string equation;
  for (std::size_t i = 0; i < operands.size(); ++i) {
    if (i > 0) equation += ',';
    equation += label_string(labels[i]);
  }
  equation += "->" + label_string(out);
  return at::einsum(equation, operands);
}

// Fold the dense expansions of the operands into one buffer over the free coupled indices,
// so that the output tree is a single reduction against it.
std::pair<at::Tensor, std::vector<int64_t>> fold_operands(
    const std::vector<at::Tensor>& operands, const std::vector<std::vector<int64_t>>& labels,
    const std::vector<int64_t>& out) {
  std::vector<bool> used(kLabels.size(), false);
  for (const std::vector<int64_t>& ls : labels) {
    for (int64_t l : ls) used[l] = true;
  }
  for (int64_t l : out) used[l] = true;
  std::vector<std::vector<int64_t>> operand_labels;
  std::vector<int64_t> folded_labels = out;
  for (std::size_t i = 0; i < operands.size(); ++i) {
    // The expansion carries one axis per leg plus one for the coupled index, which no other operand
    // labels: one free label per operand keeps it while every shared label contracts away.
    int64_t free = 0;
    while (used[free]) ++free;
    used[free] = true;
    operand_labels.push_back(labels[i]);
    operand_labels.back().push_back(free);
    folded_labels.push_back(free);
  }
  return {reduce_blocks(operands, operand_labels, folded_labels), folded_labels};
}

// The overlap of one output tree with the folded expansions of the operand trees. The result
// stays a device scalar: reading it back would synchronise the stream once per tree.
at::Tensor tree_weight(const at::Tensor& output, const at::Tensor& folded,
                       const std::vector<int64_t>& out, const std::vector<int64_t>& folded_labels) {
  if (out.empty()) return folded.sum();
  return reduce_blocks({output, folded}, {out, folded_labels}, {});
}

// A 64-bit fingerprint of one equation; it only has to separate the calls of one process.
int64_t fingerprint(const std::string& text) {
  return static_cast<int64_t>(std::hash<std::string>{}(text));
}

// The code that identifies a device among the tables of the engine.
int64_t device_code(const c10::Device& device) {
  return static_cast<int64_t>(device.type()) * 1000
         + (device.has_index() ? device.index() : -1);
}

// A stable identity for a group object. The table of identities keeps the object alive, so a
// released group cannot hand its address to the next one.
int64_t group_id(const std::shared_ptr<Group>& group) {
  static std::map<std::shared_ptr<Group>, int64_t> identities;
  const auto found = identities.find(group);
  if (found != identities.end()) return found->second;
  const int64_t identity = static_cast<int64_t>(identities.size());
  identities.emplace(group, identity);
  return identity;
}

// Append the description of one fusion tree: the tree, the orientations of its legs, the group
// and the device are exactly what its CG expansion depends on. Two equal descriptions therefore
// share their expansion, their fold and their weights.
void append_tree(std::vector<int64_t>& key, const std::vector<Leg>& legs, const FusionTree& tree,
                 const c10::Device& device) {
  key.push_back(static_cast<int64_t>(legs.size()));
  key.push_back(group_id(tree.group()));
  key.push_back(device_code(device));
  for (const Leg& leg : legs) key.push_back(leg.qconj());
  for (const Sector& sector : tree.sectors()) {
    for (int64_t entry : sector) key.push_back(entry);
    key.push_back(-2);
  }
  key.push_back(-1);
  for (const Sector& inner : tree.inner()) {
    for (int64_t entry : inner) key.push_back(entry);
    key.push_back(-2);
  }
  key.push_back(-1);
  for (int64_t entry : tree.coupled()) key.push_back(entry);
  // Two trees of a group with multiplicity share their path but differ in the realized copy.
  key.push_back(-1);
  for (int64_t copy : tree.copies()) key.push_back(copy);
  key.push_back(-1);
}

// The tree-dependent work of a contraction. The same trees come back at every site and at every
// Lanczos step, so the fold of the operand expansions and the weight of an output tree are
// computed once and then handed out: both are read-only, only the block values are multiplied
// into them.
std::map<std::vector<int64_t>, std::pair<at::Tensor, std::vector<int64_t>>> folded_cache;
std::map<std::vector<int64_t>, at::Tensor> weight_cache;
constexpr std::size_t tree_cache_limit = 8192;

const std::pair<at::Tensor, std::vector<int64_t>>& cached_fold(
    const std::vector<int64_t>& key, const std::vector<at::Tensor>& operands,
    const std::vector<std::vector<int64_t>>& labels, const std::vector<int64_t>& out) {
  const auto found = folded_cache.find(key);
  if (found != folded_cache.end()) return found->second;
  if (folded_cache.size() >= tree_cache_limit) folded_cache.clear();
  return folded_cache.emplace(key, fold_operands(operands, labels, out)).first->second;
}

const at::Tensor& cached_weight(const std::vector<int64_t>& key, const at::Tensor& output,
                                const at::Tensor& folded, const std::vector<int64_t>& out,
                                const std::vector<int64_t>& folded_labels) {
  const auto found = weight_cache.find(key);
  if (found != weight_cache.end()) return found->second;
  if (weight_cache.size() >= tree_cache_limit) weight_cache.clear();
  return weight_cache.emplace(key, tree_weight(output, folded, out, folded_labels)).first->second;
}

}  // namespace

// Drop the cached folds and weights.
void clear_tree_caches() {
  folded_cache.clear();
  weight_cache.clear();
}

void set_defaults(const std::string& device, const std::string& dtype) {
  default_device() = device_of(py::str(device));
  default_dtype() = scalar_type_of_name(strip_torch(dtype));
}

at::TensorOptions block_options(const py::object& dtype, const py::object& device) {
  return at::TensorOptions().dtype(scalar_type_of(dtype)).device(device_of(device));
}

Sector fused_charge(const std::shared_ptr<Group>& group, const std::vector<Leg>& legs,
                    const BlockKey& key, const std::vector<int64_t>& axes) {
  Sector fused = group->trivial_sector();
  for (int64_t i : axes) {
    const Sector& sector = legs.at(i).sectors().at(key.legs.at(i));
    fused = group->fuse({fused, legs.at(i).qconj() == 1 ? sector : group->dual_sector(sector)});
  }
  return fused;
}

Sector fused_qtotal(const std::shared_ptr<Group>& group, const std::vector<Sector>& sectors) {
  if (sectors.empty()) throw value_error("Fusing no charges has no outcome.");
  Sector total = group->trivial_sector();
  for (const Sector& sector : sectors) {
    const std::vector<Sector> outcomes = group->fusion_outcomes(total, sector);
    if (outcomes.size() != 1) {
      throw not_implemented_error("Fusing " + sector_repr(total) + " with " + sector_repr(sector) +
                                  " has " + std::to_string(outcomes.size()) +
                                  " outcomes; the contracted tensor is a sum of them.");
    }
    total = outcomes.front();
  }
  return total;
}

std::vector<BlockKey> combos(const std::vector<Leg>& legs, const Sector& qtotal) {
  if (legs.empty()) return {BlockKey{{}, {}}};
  const std::shared_ptr<Group>& group = legs.front().group();
  std::vector<std::size_t> sizes;
  for (const Leg& leg : legs) sizes.push_back(static_cast<std::size_t>(leg.nblocks()));
  std::vector<BlockKey> out;
  for_each_combination(sizes, [&](const std::vector<std::size_t>& index) {
    BlockKey key;
    for (std::size_t i = 0; i < index.size(); ++i) key.legs.push_back(static_cast<int64_t>(index[i]));
    std::vector<Sector> sectors;
    for (std::size_t i = 0; i < legs.size(); ++i) sectors.push_back(legs[i].sector(key.legs[i], true));
    for (const FusionTree& tree : fusion_trees(group, sectors, qtotal)) {
      BlockKey filled = key;
      filled.inner = tree.inner();
      if (group->carries_multiplicity()) filled.copies = tree.copies();
      out.push_back(std::move(filled));
    }
  });
  return out;
}

Shape block_shape(const std::vector<Leg>& legs, const BlockKey& key) {
  Shape out;
  for (std::size_t i = 0; i < legs.size(); ++i) out.push_back(legs[i].multiplicity(key.legs.at(i)));
  return out;
}

std::vector<std::vector<int64_t>> running_offsets(const std::vector<Leg>& legs) {
  std::vector<std::vector<int64_t>> offsets;
  for (const Leg& leg : legs) {
    std::vector<int64_t> running{0};
    for (int64_t qi = 0; qi < leg.nblocks(); ++qi) running.push_back(running.back() + leg.multiplicity(qi));
    offsets.push_back(running);
  }
  return offsets;
}

EinsumSpec parse_einsum(const std::string& spec, std::size_t operands) {
  EinsumSpec out;
  for (char c : spec) {
    if (c != ' ') out.equation += c;
  }
  const std::size_t arrow = out.equation.find("->");
  const std::string lhs = arrow == std::string::npos ? out.equation : out.equation.substr(0, arrow);
  const std::string rhs = arrow == std::string::npos ? std::string() : out.equation.substr(arrow + 2);
  std::size_t start = 0;
  while (true) {
    const std::size_t comma = lhs.find(',', start);
    const std::size_t length = comma == std::string::npos ? std::string::npos : comma - start;
    out.labels.push_back(label_ids_of(lhs.substr(start, length)));
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  if (arrow != std::string::npos) {
    out.out = label_ids_of(rhs);
    return out;
  }
  std::vector<int64_t> counts(kLabels.size(), 0);
  for (const std::vector<int64_t>& ls : out.labels) {
    for (int64_t l : ls) ++counts[l];
  }
  for (std::size_t l = 0; l < counts.size(); ++l) {
    if (counts[l] == 1) out.out.push_back(static_cast<int64_t>(l));
  }
  (void)operands;
  return out;
}

std::string label_string(const std::vector<int64_t>& labels) {
  std::string out;
  for (int64_t l : labels) out += kLabels.at(static_cast<std::size_t>(l));
  return out;
}

py::tuple key_to_python(const BlockKey& key) {
  py::tuple out(key.legs.size() + key.inner.size() + key.copies.size());
  std::size_t position = 0;
  for (std::size_t i = 0; i < key.legs.size(); ++i) out[position++] = py::int_(key.legs[i]);
  for (std::size_t i = 0; i < key.inner.size(); ++i) out[position++] = to_python(key.inner[i]);
  for (std::size_t i = 0; i < key.copies.size(); ++i) out[position++] = py::int_(key.copies[i]);
  return out;
}

std::vector<BlockKey> parse_keys(const py::object& qdata, const std::vector<Leg>& legs) {
  const int64_t n = static_cast<int64_t>(legs.size());
  const bool multiplicity =
      !legs.empty() && legs.front().group()->carries_multiplicity() && n >= 2;
  const int64_t inner_count = n > 2 ? n - 2 : 0;
  std::vector<BlockKey> out;
  for (auto item : py::reinterpret_borrow<py::iterable>(qdata)) {
    BlockKey key;
    int64_t position = 0;
    for (auto entry : py::reinterpret_borrow<py::iterable>(item)) {
      if (position < n) {
        key.legs.push_back(py::cast<int64_t>(entry));
      } else if (position < n + inner_count) {
        key.inner.push_back(sector_of(py::reinterpret_borrow<py::object>(entry)));
      } else if (multiplicity) {
        key.copies.push_back(py::cast<int64_t>(entry));
      } else {
        assertion_failed("Block indices carry more entries than the legs of the tensor.");
      }
      ++position;
    }
    out.push_back(std::move(key));
  }
  return out;
}

std::vector<Leg> parse_legs(const py::object& legs) {
  std::vector<Leg> out;
  for (auto item : py::reinterpret_borrow<py::iterable>(legs)) out.push_back(item.cast<Leg>());
  return out;
}

Shape parse_shape(const py::args& shape) {
  Shape out;
  if (shape.size() == 1 &&
      (py::isinstance<py::tuple>(shape[0]) || py::isinstance<py::list>(shape[0]))) {
    for (auto item : py::reinterpret_borrow<py::iterable>(shape[0])) out.push_back(py::cast<int64_t>(item));
    return out;
  }
  for (auto item : shape) out.push_back(py::cast<int64_t>(item));
  return out;
}

Tensor::Tensor(std::vector<at::Tensor> blocks, std::vector<BlockKey> qdata,
               std::vector<Leg> legs, std::optional<Sector> qtotal,
               std::shared_ptr<Group> group)
    : blocks_(std::move(blocks)), qdata_(std::move(qdata)), legs_(std::move(legs)) {
  group_ = legs_.empty() ? (group ? std::move(group) : no_symmetry()) : legs_.front().group();
  qtotal_ = qtotal ? *qtotal : group_->trivial_sector();
  canonicalize();
}

void Tensor::canonicalize() {
  const std::size_t n = legs_.size();
  for (const Leg& leg : legs_) {
    if (!same_group(*leg.group(), *group_)) {
      assertion_failed("All legs of a tensor share one group, not " + leg.group()->group_name() +
                       " and " + group_->group_name() + ".");
    }
  }
  if (blocks_.size() != qdata_.size()) {
    assertion_failed(std::to_string(blocks_.size()) +
                     " blocks need as many charge-block index tuples.");
  }
  std::vector<std::size_t> order(qdata_.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](std::size_t a, std::size_t b) { return qdata_[a] < qdata_[b]; });
  std::vector<at::Tensor> blocks;
  std::vector<BlockKey> keys;
  for (std::size_t i : order) {
    blocks.push_back(blocks_[i]);
    keys.push_back(qdata_[i]);
  }
  blocks_ = std::move(blocks);
  qdata_ = std::move(keys);
  for (std::size_t i = 1; i < qdata_.size(); ++i) {
    if (qdata_[i] == qdata_[i - 1]) assertion_failed("Charge-block indices repeat a block.");
  }
  // A group that can couple a vertex in several ways stores the realized copy with the path.
  const bool multiplicity = group_->carries_multiplicity() && n >= 2;
  const int64_t key_length =
      static_cast<int64_t>(n + (n > 2 ? n - 2 : 0) + (multiplicity ? n - 1 : 0));
  for (std::size_t i = 0; i < blocks_.size(); ++i) {
    const BlockKey& key = qdata_[i];
    const int64_t entries =
        static_cast<int64_t>(key.legs.size() + key.inner.size() + key.copies.size());
    if (entries != key_length) {
      assertion_failed("Block indices need " + std::to_string(key_length) + " entries, not " +
                       std::to_string(entries) + ".");
    }
    for (std::size_t k = 0; k < n; ++k) {
      if (key.legs[k] < 0 || key.legs[k] >= legs_[k].nblocks()) {
        assertion_failed("Block indices are out of range for the legs of the tensor.");
      }
    }
    const Shape dims = block_shape(legs_, key);
    const Shape shape(blocks_[i].sizes().begin(), blocks_[i].sizes().end());
    if (shape != dims) {
      assertion_failed("Block shape does not match the legs of the tensor.");
    }
    // Every block stores its fusion path and that path must reach the total charge.
    if (n >= 2) {
      FusionTree::from_blocks(group_, legs_, key, qtotal_);
    } else if (n == 1 && legs_.front().sector(key.legs.front(), true) != qtotal_) {
      assertion_failed("A block carries another charge than the total charge of the tensor.");
    }
  }
}

Tensor Tensor::from_torch(const at::Tensor& data, std::optional<std::vector<Leg>> legs,
                          std::optional<Sector> qtotal) {
  const Shape shape(data.sizes().begin(), data.sizes().end());
  std::vector<Leg> resolved;
  if (legs) {
    resolved = *legs;
    if (!shape.empty()) {
      Shape dims;
      for (const Leg& leg : resolved) dims.push_back(leg.dim());
      if (dims != shape) {
        throw value_error("Legs of dimensions ... do not match the shape ...");
      }
    }
  } else {
    for (int64_t dim : shape) resolved.push_back(Leg::trivial(dim, no_symmetry()));
  }
  const std::shared_ptr<Group>& group = resolved.empty() ? no_symmetry() : resolved.front().group();
  const Sector total = qtotal ? *qtotal : group->trivial_sector();
  const std::vector<BlockKey> keys = combos(resolved, total);
  const bool whole = keys.size() == 1 &&
                     std::all_of(resolved.begin(), resolved.end(),
                                 [](const Leg& leg) { return leg.nblocks() == 1; });
  if (whole) return Tensor({data}, keys, resolved, total);
  std::vector<at::Tensor> blocks;
  for (const BlockKey& key : keys) {
    std::vector<at::indexing::TensorIndex> index;
    for (std::size_t k = 0; k < resolved.size(); ++k) {
      index.emplace_back(at::indexing::Slice(resolved[k].slices().at(key.legs[k]),
                                             resolved[k].slices().at(key.legs[k] + 1)));
    }
    blocks.push_back(at::indexing::get_item(data, index).clone());
  }
  return Tensor(blocks, keys, resolved, total);
}

Tensor Tensor::filled(py::args shape, const py::kwargs& options, int kind) {
  const Shape dims = parse_shape(shape);
  py::object legs_value = py::none();
  py::object group_value = py::none();
  py::object qtotal_value = py::none();
  py::object dtype_value = py::none();
  py::object device_value = py::none();
  if (options.contains("legs")) legs_value = options["legs"];
  if (options.contains("group")) group_value = options["group"];
  if (options.contains("qtotal")) qtotal_value = options["qtotal"];
  if (options.contains("dtype")) dtype_value = options["dtype"];
  if (options.contains("device")) device_value = options["device"];
  std::vector<Leg> legs;
  if (legs_value.is_none()) {
    const std::shared_ptr<Group>& group =
        group_value.is_none() ? no_symmetry() : group_value.cast<std::shared_ptr<Group>>();
    for (int64_t dim : dims) legs.push_back(Leg::trivial(dim, group));
  } else {
    legs = parse_legs(legs_value);
    if (!dims.empty()) {
      Shape sizes;
      for (const Leg& leg : legs) sizes.push_back(leg.dim());
      if (sizes != dims) throw value_error("Legs of dimensions ... do not match the shape ...");
    }
  }
  const std::shared_ptr<Group>& group =
      legs.empty() ? (group_value.is_none() ? no_symmetry() : group_value.cast<std::shared_ptr<Group>>())
                   : legs.front().group();
  const Sector total = qtotal_value.is_none() ? group->trivial_sector() : sector_of(qtotal_value);
  const at::TensorOptions storage = block_options(dtype_value, device_value);
  const std::vector<BlockKey> keys = combos(legs, total);
  std::vector<at::Tensor> blocks;
  for (const BlockKey& key : keys) {
    const Shape block = block_shape(legs, key);
    if (kind == 0) blocks.push_back(at::zeros(block, storage));
    else if (kind == 1) blocks.push_back(at::ones(block, storage));
    else blocks.push_back(at::rand(block, storage));
  }
  return Tensor(blocks, keys, legs, total);
}

Tensor Tensor::zeros(py::args shape, const py::kwargs& options) {
  return filled(std::move(shape), options, 0);
}

Tensor Tensor::ones(py::args shape, const py::kwargs& options) {
  return filled(std::move(shape), options, 1);
}

Tensor Tensor::rand(py::args shape, const py::kwargs& options) {
  return filled(std::move(shape), options, 2);
}

Shape Tensor::shape() const {
  Shape out;
  for (const Leg& leg : legs_) out.push_back(leg.dim());
  return out;
}

py::object Tensor::dtype() const {
  const at::TensorOptions options = blocks_.empty() ? block_options() : blocks_.front().options();
  return py::cast(at::empty({}, options)).attr("dtype");
}

py::object Tensor::device() const {
  return py::cast(blocks_.empty() ? default_device() : blocks_.front().device());
}

bool Tensor::is_dense() const {
  return blocks_.size() == 1 && same_group(*group_, *no_symmetry());
}

Tensor Tensor::real() const {
  std::vector<at::Tensor> blocks;
  for (const at::Tensor& block : blocks_) blocks.push_back(at::real(block));
  return Tensor(blocks, qdata_, legs_, qtotal_, group_);
}

int64_t Tensor::numel() const {
  int64_t total = 0;
  for (const at::Tensor& block : blocks_) total += block.numel();
  return total;
}

int64_t Tensor::element_size() const {
  return blocks_.empty() ? static_cast<int64_t>(block_options().dtype().itemsize())
                          : blocks_.front().element_size();
}

at::Tensor Tensor::to_torch() const {
  if (is_dense()) return blocks_.front();
  // A contraction that vanishes by symmetry carries no block at all: its dense form is zero.
  const at::TensorOptions options = blocks_.empty() ? block_options() : blocks_.front().options();
  at::Tensor out = at::zeros(shape(), options);
  if (blocks_.empty()) return out;
  if (legs_.empty()) {
    out.fill_(blocks_.front());
    return out;
  }
  if (group_->is_abelian()) {
    for (std::size_t i = 0; i < blocks_.size(); ++i) {
      at::Tensor view = out;
      for (std::size_t k = 0; k < legs_.size(); ++k) {
        view = view.slice(k, legs_[k].slices().at(qdata_[i].legs[k]),
                          legs_[k].slices().at(qdata_[i].legs[k] + 1));
      }
      view.add_(blocks_[i]);
    }
    return out;
  }
  if (qtotal_ != group_->trivial_sector()) {
    throw not_implemented_error("Expanding the multiplet of charge " + sector_repr(qtotal_) +
                                " needs one component per entry.");
  }
  for (std::size_t i = 0; i < blocks_.size(); ++i) {
    const FusionTree tree = FusionTree::from_blocks(group_, legs_, qdata_[i], qtotal_);
    const at::Tensor x = take_last(stored_dense(legs_, tree, blocks_[i].device()),
                                   group_->reference_index(tree.coupled()));
    std::vector<std::size_t> sizes;
    for (std::size_t k = 0; k < legs_.size(); ++k) {
      sizes.push_back(static_cast<std::size_t>(legs_[k].multiplicity(qdata_[i].legs[k])));
    }
    for_each_combination(sizes, [&](const std::vector<std::size_t>& index) {
      at::Tensor view = out;
      std::vector<int64_t> point;
      for (std::size_t k = 0; k < legs_.size(); ++k) {
        const int64_t step = group_->sector_dim(tree.sectors().at(k));
        const int64_t first = legs_[k].slices().at(qdata_[i].legs[k]) +
                              static_cast<int64_t>(index[k]) * step;
        view = view.slice(k, first, first + step);
        point.push_back(static_cast<int64_t>(index[k]));
      }
      view.add_(x * scalar_at(blocks_[i], point));
    });
  }
  return out;
}

py::object Tensor::item() const { return py::cast(to_torch().item()); }

Tensor Tensor::clone() const {
  std::vector<at::Tensor> blocks;
  for (const at::Tensor& block : blocks_) blocks.push_back(block.clone());
  return Tensor(blocks, qdata_, legs_, qtotal_, group_);
}

Tensor Tensor::detach() const {
  std::vector<at::Tensor> blocks;
  for (const at::Tensor& block : blocks_) blocks.push_back(block.detach());
  return Tensor(blocks, qdata_, legs_, qtotal_, group_);
}

Tensor Tensor::conj() const {
  std::vector<Leg> legs;
  for (const Leg& leg : legs_) legs.push_back(leg.conj());
  std::vector<BlockKey> qdata = qdata_;
  // Dualising the legs dualises every coupling of a block's fusion path.
  for (BlockKey& key : qdata) {
    for (Sector& coupling : key.inner) coupling = group_->dual_sector(coupling);
  }
  std::vector<at::Tensor> blocks;
  for (std::size_t i = 0; i < blocks_.size(); ++i) {
    at::Tensor block = blocks_[i].conj();
    for (std::size_t k = 0; k < legs_.size(); ++k) {
      if (legs_[k].qconj() == -1) {
        block = block * group_->dual_sign(legs_[k].sectors().at(qdata_[i].legs[k]));
      }
    }
    blocks.push_back(block);
  }
  return Tensor(blocks, qdata, legs, group_->dual_sector(qtotal_), group_);
}

Tensor Tensor::to(const py::object& device, const py::object& dtype) const {
  if (device.is_none() && dtype.is_none()) return Tensor(blocks_, qdata_, legs_, qtotal_, group_);
  const at::TensorOptions options = block_options(dtype, device);
  std::vector<at::Tensor> blocks;
  for (const at::Tensor& block : blocks_) blocks.push_back(block.to(options));
  return Tensor(blocks, qdata_, legs_, qtotal_, group_);
}

Tensor Tensor::combine_legs(const std::vector<std::vector<int64_t>>& groups) const {
  // For a non-abelian group the charge a group of legs fuses to is read off each block's own
  // fusion chain, but the merged block is not a reshape of the old one: its multiplet is the
  // coupled one, so the data would have to be projected with the group's Clebsch-Gordan
  // tensors first. Until that projection exists, the fusion of several legs is refused.
  if (!group_->is_abelian()) {
    throw not_implemented_error("Combining the legs of a non-abelian tensor is not supported.");
  }
  std::vector<int64_t> flat;
  for (const std::vector<int64_t>& group : groups) {
    for (int64_t i : group) flat.push_back(i);
  }
  std::vector<int64_t> expected;
  for (std::size_t i = 0; i < legs_.size(); ++i) expected.push_back(static_cast<int64_t>(i));
  std::vector<int64_t> sorted = flat;
  std::sort(sorted.begin(), sorted.end());
  if (sorted != expected) throw value_error("The legs of a fusion do not partition the legs.");
  if (is_dense()) {
    at::Tensor data = blocks_.front();
    if (!flat.empty()) {
      std::vector<int64_t> permutation(flat.begin(), flat.end());
      data = data.permute(permutation);
    }
    Shape dims;
    for (const std::vector<int64_t>& group : groups) {
      int64_t dim = 1;
      for (int64_t i : group) dim *= legs_[i].dim();
      dims.push_back(dim);
    }
    return Tensor::from_torch(data.reshape(dims));
  }
  std::vector<Leg> new_legs;
  std::vector<std::vector<int64_t>> ranks;
  for (const std::vector<int64_t>& group : groups) {
    std::vector<std::pair<Sector, int64_t>> entries;
    for (const BlockKey& key : qdata_) {
      int64_t dim = 1;
      for (int64_t i : group) dim *= legs_[i].block_dim(key.legs[i]);
      entries.emplace_back(fused_charge(group_, legs_, key, group), dim);
    }
    auto [leg, rank] = Leg::from_blocks(group_, entries, 1);
    new_legs.push_back(leg);
    ranks.push_back(rank);
  }
  std::vector<int64_t> local;
  for (std::size_t i = 0; i < legs_.size(); ++i) {
    local.push_back(static_cast<int64_t>(std::find(flat.begin(), flat.end(), i) - flat.begin()));
  }
  std::vector<at::Tensor> blocks;
  std::vector<BlockKey> keys;
  for (std::size_t bi = 0; bi < qdata_.size(); ++bi) {
    Shape dims;
    for (const std::vector<int64_t>& group : groups) {
      int64_t dim = 1;
      for (int64_t i : group) dim *= legs_[i].block_dim(qdata_[bi].legs[i]);
      dims.push_back(dim);
    }
    blocks.push_back(blocks_[bi].permute(local).reshape(dims));
    BlockKey key;
    std::vector<Sector> sectors;
    for (std::size_t g = 0; g < groups.size(); ++g) {
      key.legs.push_back(ranks[g][bi]);
      sectors.push_back(new_legs[g].sector(ranks[g][bi], true));
    }
    // A combined block stores the fusion path of its legs, like every other block.
    const std::vector<FusionTree> trees = fusion_trees(group_, sectors, qtotal_);
    if (trees.empty()) throw value_error("The combined legs do not fuse to the total charge.");
    key.inner = trees.front().inner();
    if (group_->carries_multiplicity()) key.copies = trees.front().copies();
    keys.push_back(key);
  }
  return Tensor(blocks, keys, new_legs, qtotal_, group_);
}

py::object Tensor::getitem(const py::object& key) const {
  if (is_dense()) return py::cast(Tensor::from_torch(at::indexing::get_item(blocks_.front(), index_of(key))));
  const at::Tensor out = at::indexing::get_item(to_torch(), index_of(key));
  if (out.dim() != 0) {
    throw not_implemented_error("Indexing a charged tensor supports scalar selections only.");
  }
  return py::cast(Tensor::from_torch(out));
}

void Tensor::setitem(const py::object& key, const py::object& value) {
  if (is_dense()) {
    // ``_data`` of the python engine: an engine tensor hands over its buffer.
    const at::Tensor data = py::isinstance<Tensor>(value)
                                ? value.cast<const Tensor&>().to_torch()
                                : (is_tensor(value)
                                       ? value.cast<at::Tensor>()
                                       : at::scalar_tensor(scalar_of(value),
                                                           blocks_.front().options()));
    at::indexing::set_item(blocks_.front(), index_of(key), data);
    return;
  }
  if (is_tensor(key)) {
    const at::Tensor mask = key.cast<at::Tensor>();
    if (mask.scalar_type() == at::kBool && mask.numel() == 1) {
      if (mask.item<bool>()) {
        for (at::Tensor& block : blocks_) block = at::zeros_like(block);
      }
      return;
    }
  }
  throw not_implemented_error("Assigning into a charged tensor supports a scalar boolean mask only.");
}

Tensor Tensor::operator-() const {
  std::vector<at::Tensor> blocks;
  for (const at::Tensor& block : blocks_) blocks.push_back(-block);
  return Tensor(blocks, qdata_, legs_, qtotal_, group_);
}

Tensor Tensor::binary_add(const py::object& other, bool negate) const {
  if (py::isinstance<Tensor>(other)) {
    const Tensor& o = other.cast<const Tensor&>();
    if (o.legs() != legs_) throw value_error("Adding tensors of different legs.");
    std::map<BlockKey, at::Tensor> table;
    for (std::size_t i = 0; i < o.qdata().size(); ++i) table.emplace(o.qdata()[i], o.blocks()[i]);
    std::vector<at::Tensor> blocks;
    std::vector<BlockKey> keys;
    for (std::size_t i = 0; i < qdata_.size(); ++i) {
      const auto found = table.find(qdata_[i]);
      if (found == table.end()) blocks.push_back(blocks_[i].clone());
      else blocks.push_back(negate ? blocks_[i] - found->second : blocks_[i] + found->second);
      keys.push_back(qdata_[i]);
    }
    for (const auto& [key, block] : table) {
      if (std::find(qdata_.begin(), qdata_.end(), key) == qdata_.end()) {
        keys.push_back(key);
        blocks.push_back(negate ? -block : block.clone());
      }
    }
    return Tensor(blocks, keys, legs_, qtotal_, group_);
  }
  at::Tensor raw = as_tensor(other, block_options());
  if (negate) raw = -raw;
  if (raw.dim() > 0) {
    std::vector<at::Tensor> blocks;
    for (const at::Tensor& block : blocks_) blocks.push_back(block + raw);
    return Tensor(blocks, qdata_, legs_, qtotal_, group_);
  }
  if (raw.eq(0).item<bool>()) return clone();
  throw not_implemented_error("A constant offset cannot be represented on a charged tensor.");
}

Tensor Tensor::add(const py::object& other) const { return binary_add(other, false); }

Tensor Tensor::sub(const py::object& other) const { return binary_add(other, true); }

Tensor Tensor::scale(const py::object& other, bool divide) const {
  if (py::isinstance<Tensor>(other)) {
    const Tensor& o = other.cast<const Tensor&>();
    if (o.legs() != legs_) throw value_error("Scaling tensors of different legs.");
    std::map<BlockKey, at::Tensor> table;
    for (std::size_t i = 0; i < o.qdata().size(); ++i) table.emplace(o.qdata()[i], o.blocks()[i]);
    std::vector<at::Tensor> blocks;
    std::vector<BlockKey> keys;
    for (std::size_t i = 0; i < qdata_.size(); ++i) {
      const auto found = table.find(qdata_[i]);
      if (found == table.end()) continue;
      blocks.push_back(divide ? blocks_[i] / found->second : blocks_[i] * found->second);
      keys.push_back(qdata_[i]);
    }
    return Tensor(blocks, keys, legs_, qtotal_, group_);
  }
  const bool tensor_factor = is_tensor(other);
  if (!tensor_factor || other.cast<at::Tensor>().numel() == 1 || other.cast<at::Tensor>().dim() == 0) {
    std::vector<at::Tensor> blocks;
    for (const at::Tensor& block : blocks_) {
      at::Tensor scaled = block;
      if (tensor_factor) {
        scaled = divide ? block / other.cast<at::Tensor>() : block * other.cast<at::Tensor>();
      } else {
        scaled = divide ? block / scalar_of(other) : block * scalar_of(other);
      }
      blocks.push_back(scaled);
    }
    return Tensor(blocks, qdata_, legs_, qtotal_, group_);
  }
  const at::Tensor factor = other.cast<at::Tensor>();
  if (factor.dim() != static_cast<int64_t>(legs_.size())) {
    throw value_error("A raw factor of rank " + std::to_string(factor.dim()) +
                      " does not broadcast over " + std::to_string(legs_.size()) + " legs.");
  }
  const std::vector<std::vector<int64_t>> offsets = running_offsets(legs_);
  std::vector<at::Tensor> blocks;
  for (std::size_t i = 0; i < blocks_.size(); ++i) {
    std::vector<at::indexing::TensorIndex> index;
    for (std::size_t k = 0; k < legs_.size(); ++k) {
      if (factor.size(static_cast<int64_t>(k)) == 1) {
        index.emplace_back(at::indexing::Slice());
      } else {
        index.emplace_back(at::indexing::Slice(offsets[k].at(qdata_[i].legs[k]),
                                               offsets[k].at(qdata_[i].legs[k] + 1)));
      }
    }
    const at::Tensor window = at::indexing::get_item(factor, index);
    blocks.push_back(divide ? blocks_[i] / window : blocks_[i] * window);
  }
  return Tensor(blocks, qdata_, legs_, qtotal_, group_);
}

Tensor& Tensor::iadd(const py::object& other) {
  if (is_dense() && !py::isinstance<Tensor>(other)) {
    blocks_.front().add_(as_tensor(other, blocks_.front().options()));
    return *this;
  }
  Tensor sum = binary_add(other, false);
  blocks_ = std::move(sum.blocks_);
  qdata_ = std::move(sum.qdata_);
  return *this;
}

Tensor& Tensor::imul(const py::object& other) {
  if (is_dense() && !py::isinstance<Tensor>(other)) {
    blocks_.front().mul_(as_tensor(other, blocks_.front().options()));
    return *this;
  }
  Tensor product = scale(other, false);
  blocks_ = std::move(product.blocks_);
  qdata_ = std::move(product.qdata_);
  return *this;
}

Tensor& Tensor::idiv(const py::object& other) {
  if (is_dense() && !py::isinstance<Tensor>(other)) {
    blocks_.front().div_(as_tensor(other, blocks_.front().options()));
    return *this;
  }
  Tensor quotient = scale(other, true);
  blocks_ = std::move(quotient.blocks_);
  qdata_ = std::move(quotient.qdata_);
  return *this;
}

Tensor blocked_einsum(const std::string& spec, const std::vector<Tensor>& tensors) {
  const EinsumSpec parsed = parse_einsum(spec, tensors.size());
  const std::vector<std::vector<int64_t>>& labels = parsed.labels;
  const std::vector<int64_t>& out = parsed.out;
  if (labels.size() != tensors.size()) {
    throw value_error(std::to_string(labels.size()) + " subscript groups do not match " +
                      std::to_string(tensors.size()) + " operands.");
  }
  const std::size_t n_operands = tensors.size();
  std::vector<int64_t> counts(kLabels.size(), 0);
  for (const std::vector<int64_t>& ls : labels) {
    for (int64_t l : ls) ++counts[l];
  }
  for (int64_t l : out) {
    if (counts[l] > 1) {
      throw not_implemented_error("Subscript '" + std::string(1, kLabels.at(l)) + "' appears " +
                                  std::to_string(counts[l]) + " times in the output.");
    }
  }
  std::vector<int64_t> summed;
  for (std::size_t l = 0; l < counts.size(); ++l) {
    if (counts[l] > 0 && !contains(out, static_cast<int64_t>(l))) summed.push_back(static_cast<int64_t>(l));
  }
  for (int64_t l : summed) {
    if (counts[l] != 2) {
      throw not_implemented_error("Subscript '" + std::string(1, kLabels.at(l)) + "' is contracted " +
                                  std::to_string(counts[l]) +
                                  " times; only a pair of operands is supported.");
    }
  }
  std::vector<std::map<int64_t, int64_t>> axes(n_operands);
  for (std::size_t o = 0; o < n_operands; ++o) {
    for (std::size_t a = 0; a < labels[o].size(); ++a) axes[o][labels[o][a]] = static_cast<int64_t>(a);
  }
  for (int64_t l : summed) {
    std::size_t first = n_operands;
    std::size_t second = n_operands;
    for (std::size_t o = 0; o < n_operands; ++o) {
      if (axes[o].count(l)) {
        if (first == n_operands) first = o;
        else second = o;
      }
    }
    const Leg& first_leg = tensors[first].legs()[axes[first][l]];
    const Leg& second_leg = tensors[second].legs()[axes[second][l]];
    if (first_leg.sectors() != second_leg.sectors() || first_leg.slices() != second_leg.slices() ||
        first_leg.qconj() != -second_leg.qconj()) {
      throw value_error("Contracted subscript '" + std::string(1, kLabels.at(l)) + "' pairs " +
                        first_leg.repr() + " with " + second_leg.repr() + ".");
    }
  }
  std::vector<std::map<std::vector<std::pair<int64_t, Sector>>, std::vector<int64_t>>> table(n_operands);
  for (std::size_t o = 0; o < n_operands; ++o) {
    const Tensor& t = tensors[o];
    for (std::size_t bi = 0; bi < t.qdata().size(); ++bi) {
      std::vector<std::pair<int64_t, Sector>> cell;
      for (int64_t l : summed) {
        const auto axis = axes[o].find(l);
        if (axis == axes[o].end()) continue;
        cell.emplace_back(l, t.legs()[axis->second].sector(t.qdata()[bi].legs[axis->second]));
      }
      table[o][cell].push_back(static_cast<int64_t>(bi));
    }
  }
  std::vector<int64_t> carrier(out.size(), -1);
  for (std::size_t i = 0; i < out.size(); ++i) {
    for (std::size_t o = 0; o < n_operands; ++o) {
      if (axes[o].count(out[i])) {
        carrier[i] = static_cast<int64_t>(o);
        break;
      }
    }
    if (carrier[i] < 0) throw value_error("No operand carries a subscript of the output.");
  }
  std::map<int64_t, std::vector<Sector>> options;
  for (int64_t l : summed) {
    for (std::size_t o = 0; o < n_operands; ++o) {
      const auto axis = axes[o].find(l);
      if (axis == axes[o].end()) continue;
      options[l] = tensors[o].legs()[axis->second].sectors();
      break;
    }
  }
  std::vector<Leg> out_legs;
  for (std::size_t i = 0; i < out.size(); ++i) {
    out_legs.push_back(tensors[carrier[i]].legs()[axes[carrier[i]][out[i]]]);
  }
  const std::shared_ptr<Group>& group = tensors.front().group();
  std::vector<Sector> totals;
  for (const Tensor& t : tensors) totals.push_back(t.qtotal());
  const Sector qtotal = fused_qtotal(group, totals);
  // A block-less operand is the zero tensor, so the contraction vanishes.
  bool zeroth = false;
  for (const Tensor& t : tensors) zeroth = zeroth || t.blocks().empty();
  if (zeroth) return Tensor({}, {}, out_legs, qtotal, group);
  const c10::Device device = tensors.front().blocks().front().device();
  // One block combination: the operands to contract, the output key it lands in and, for a
  // non-abelian group, the weight of every fusion tree its value is spread over.
  struct Combo {
    std::vector<at::Tensor> raw;
    std::vector<int64_t> key_legs;
    std::vector<Sector> out_sectors;
    std::vector<std::vector<Sector>> inners;
    std::vector<std::vector<int64_t>> copies;
    std::vector<at::Tensor> weights;
  };
  std::vector<Combo> combinations;
  std::vector<std::size_t> option_sizes;
  for (int64_t l : summed) option_sizes.push_back(options[l].size());
  for_each_combination(option_sizes, [&](const std::vector<std::size_t>& position) {
    std::map<int64_t, Sector> sectors;
    for (std::size_t k = 0; k < summed.size(); ++k) sectors[summed[k]] = options[summed[k]][position[k]];
    std::vector<std::vector<int64_t>> candidates(n_operands);
    std::vector<std::size_t> candidate_sizes;
    for (std::size_t o = 0; o < n_operands; ++o) {
      std::vector<std::pair<int64_t, Sector>> cell;
      for (int64_t l : summed) {
        if (axes[o].count(l)) cell.emplace_back(l, sectors[l]);
      }
      const auto found = table[o].find(cell);
      if (found != table[o].end()) candidates[o] = found->second;
      candidate_sizes.push_back(candidates[o].size());
    }
    for_each_combination(candidate_sizes, [&](const std::vector<std::size_t>& pick) {
      Combo combo;
      combo.raw.resize(n_operands);
      for (std::size_t o = 0; o < n_operands; ++o) {
        combo.raw[o] = tensors[o].blocks()[candidates[o][pick[o]]];
      }
      for (std::size_t i = 0; i < out.size(); ++i) {
        const int64_t o = carrier[i];
        const int64_t bi = candidates[o][pick[o]];
        combo.key_legs.push_back(tensors[o].qdata()[bi].legs[axes[o][out[i]]]);
      }
      for (std::size_t i = 0; i < out.size(); ++i) {
        combo.out_sectors.push_back(out_legs[i].sector(combo.key_legs[i], true));
      }
      // An abelian group has no Clebsch-Gordan tensor: its every tree couples the blocks
      // with weight one, so the reduced block is the block of the contracted tensor.
      if (group->is_abelian()) {
        for (const FusionTree& tree : fusion_trees(group, combo.out_sectors, qtotal)) {
          combo.inners.push_back(tree.inner());
          combo.copies.push_back(group->carries_multiplicity() ? tree.copies()
                                                               : std::vector<int64_t>{});
        }
      } else {
        std::vector<at::Tensor> operands;
        std::vector<int64_t> tree_key;
        tree_key.push_back(fingerprint(spec));
        for (std::size_t o = 0; o < n_operands; ++o) {
          const int64_t bi = candidates[o][pick[o]];
          const FusionTree tree = FusionTree::from_blocks(group, tensors[o].legs(),
                                                          tensors[o].qdata()[bi], tensors[o].qtotal());
          operands.push_back(stored_dense(tensors[o].legs(), tree, device));
          append_tree(tree_key, tensors[o].legs(), tree, device);
        }
        const auto& [folded, folded_labels] = cached_fold(tree_key, operands, labels, out);
        for (const FusionTree& tree : fusion_trees(group, combo.out_sectors, qtotal)) {
          const at::Tensor output = take_last(stored_dense(out_legs, tree, device),
                                              group->reference_index(tree.coupled()));
          std::vector<int64_t> weight_key = tree_key;
          append_tree(weight_key, out_legs, tree, device);
          combo.inners.push_back(tree.inner());
          combo.copies.push_back(group->carries_multiplicity() ? tree.copies()
                                                               : std::vector<int64_t>{});
          combo.weights.push_back(cached_weight(weight_key, output, folded, out, folded_labels));
        }
      }
      combinations.push_back(std::move(combo));
    });
  });
  // Contract the combinations in groups of equal operand shapes: one einsum call per group
  // instead of one per combination, which is what the per-block dispatch costs.
  std::map<std::string, std::vector<std::size_t>> groups;
  for (std::size_t i = 0; i < combinations.size(); ++i) {
    std::string signature;
    for (const at::Tensor& block : combinations[i].raw) {
      for (const int64_t size : block.sizes()) signature += std::to_string(size) + ",";
      signature += "|";
    }
    groups[signature].push_back(i);
  }
  int64_t batch_label = 0;
  while (batch_label < static_cast<int64_t>(kLabels.size()) &&
         (contains(out, batch_label) ||
          std::any_of(labels.begin(), labels.end(), [batch_label](const std::vector<int64_t>& group) {
            return contains(group, batch_label);
          }))) {
    ++batch_label;
  }
  std::map<BlockKey, at::Tensor> blocks;
  for (const auto& [signature, members] : groups) {
    at::Tensor batched;
    if (members.size() > 1 && batch_label < static_cast<int64_t>(kLabels.size())) {
      std::string equation;
      for (std::size_t o = 0; o < n_operands; ++o) {
        if (o > 0) equation += ",";
        equation += kLabels.at(batch_label);
        equation += label_string(labels[o]);
      }
      equation += "->";
      equation += kLabels.at(batch_label);
      equation += label_string(out);
      std::vector<at::Tensor> stacked;
      for (std::size_t o = 0; o < n_operands; ++o) {
        std::vector<at::Tensor> pieces;
        pieces.reserve(members.size());
        for (std::size_t index : members) pieces.push_back(combinations[index].raw[o]);
        stacked.push_back(at::stack(pieces, 0));
      }
      batched = at::einsum(equation, stacked);
    }
    for (std::size_t position = 0; position < members.size(); ++position) {
      const Combo& combo = combinations[members[position]];
      const at::Tensor value = batched.defined()
                                   ? batched.select(0, static_cast<int64_t>(position))
                                   : reduce_blocks(combo.raw, labels, out);
      for (std::size_t t = 0; t < combo.inners.size(); ++t) {
        BlockKey key{combo.key_legs, combo.inners[t]};
        key.copies = combo.copies[t];
        // The first term seeds the output block; later terms accumulate into it, and a
        // weighted term fuses the multiply and the add into the one kernel of addcmul_.
        const auto found = blocks.find(key);
        if (found == blocks.end()) {
          blocks.emplace(key, combo.weights.empty() ? value : value * combo.weights[t]);
        } else if (combo.weights.empty()) {
          found->second.add_(value);
        } else {
          found->second.addcmul_(value, combo.weights[t]);
        }
      }
    }
  }
  std::vector<at::Tensor> block_list;
  std::vector<BlockKey> key_list;
  for (auto& [key, block] : blocks) {
    key_list.push_back(key);
    block_list.push_back(block);
  }
  return Tensor(block_list, key_list, out_legs, qtotal, group);
}

Tensor cached_einsum(const std::string& spec, py::args tensors) {
  std::vector<Tensor> operands;
  bool any_raw = false;
  bool any_charged = false;
  for (py::handle item : tensors) {
    if (py::isinstance<Tensor>(item)) {
      Tensor tensor = item.cast<Tensor>();
      if (!tensor.is_dense()) any_charged = true;
      operands.push_back(std::move(tensor));
    } else {
      any_raw = true;
      operands.push_back(Tensor::from_torch(item.cast<at::Tensor>()));
    }
  }
  if (any_charged) {
    if (any_raw) {
      throw not_implemented_error(
          "A symmetry-adapted contraction needs every operand to be a Tensor, not a mix with raw arrays.");
    }
    return blocked_einsum(spec, operands);
  }
  std::vector<at::Tensor> raw;
  for (const Tensor& tensor : operands) raw.push_back(tensor.blocks().front());
  const EinsumSpec parsed = parse_einsum(spec, raw.size());
  return Tensor::from_torch(reduce_blocks(raw, parsed.labels, parsed.out));
}

}  // namespace mpsdmrg

namespace mpsdmrg {

// Bind the ``tensor`` names of this module to python.
void bind_tensor(py::module_& m) {
    m.def("set_defaults", &mpsdmrg::set_defaults, py::arg("device"), py::arg("dtype"),
          "the device and dtype every fresh block buffer takes");
    m.def("cached_einsum", &mpsdmrg::cached_einsum,
          "contract symmetry-adapted tensors, block by block over the shared sectors");

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
                   mpsdmrg::parse_keys(qdata, parsed_legs);
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

}  // namespace mpsdmrg
