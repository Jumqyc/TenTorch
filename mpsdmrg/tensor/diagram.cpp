#include "diagram.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace sym {
// The guard that declined the last categorical weight, 0 when the weight was evaluated.  A build
// without a dense route reports it, so a shrinking coverage names itself.
int g_weight_decline = 0;

// Which guard declined the last call to ``diagram_weight``, 0 when none did.
int diagram_weight_decline() { return g_weight_decline; }

namespace {

// The recoupling matrix of one adjacent swap: the entries of one swap are the copy label of the
// rewritten vertex and of the vertex after it, and the labels the swap turns them into, so the key
// is (old, after, fresh, after').  A label of minus one means the swap leaves that copy alone.  A
// multiplicity-free category has the single entry (0, 0, 0, 0).
using CopyEntry = std::map<std::array<int64_t, 4>, std::complex<double>>;
using SwapMatrix = std::map<Sector, std::map<Sector, CopyEntry>>;

// A chain during the permutation: its inner couplings, the coupled charge excluded, the copy label
// of every vertex (one per coupling, the coupled charge included), and its coefficient.
struct ChainState {
  std::vector<Sector> inner;
  std::vector<int64_t> copies;
  std::complex<double> value;
};

// A splitting tree in TensorKit's normal form: the sector of every leg, one duality flag per leg,
// the couplings, the copy label of every vertex, and the sector the tree couples to.  The leg
// sectors are the ones the engine's chain already carries, dualised where the leg carries a dual.
struct PairTree {
  std::vector<Sector> uncoupled;
  std::vector<int8_t> isdual;
  std::vector<Sector> inner;
  std::vector<int64_t> vertices;
  Sector coupled;
};

// One evaluated pair: the tree that stayed in the codomain, the tree the bends moved into the
// domain, and the coefficient of the whole move.
struct TreeTerm {
  PairTree kept;
  PairTree other;
  std::complex<double> value;
};

// The two associators of a swap and the braiding of the exchanged charge, contracted into the
// amplitudes of the four vertices the swap touches.
at::Tensor contract_swap(const at::Tensor& first, const at::Tensor& exchange,
                         const at::Tensor& second) {
  return at::einsum("abcd,cr,pqrd->abpq", {first, exchange, second});
}

// The nonzero entries of a copy tensor, one-based labels, as (axis labels) -> value.
CopyEntry copy_entry(const at::Tensor& tensor) {
  CopyEntry entry;
  const at::Tensor flat = tensor.contiguous();
  std::vector<int64_t> labels(static_cast<std::size_t>(flat.dim()), 1);
  while (true) {
    at::Tensor scalar = flat;
    // Selecting always drops axis zero, so the next select reads the label of the next axis.
    for (std::size_t axis = 0; axis < labels.size(); ++axis) {
      scalar = scalar.select(0, labels[axis] - 1);
    }
    const c10::complex<double> value = scalar.item<c10::complex<double>>();
    if (value.real() != 0.0 || value.imag() != 0.0) {
      std::array<int64_t, 4> key{0, 0, 0, 0};
      for (std::size_t axis = 0; axis < labels.size() && axis < 4; ++axis) {
        key[axis] = labels[axis] - 1;
      }
      entry[key] = std::complex<double>(value.real(), value.imag());
    }
    std::size_t axis = 0;
    for (; axis < labels.size(); ++axis) {
      if (++labels[axis] <= flat.size(axis)) break;
      labels[axis] = 1;
    }
    if (axis == labels.size()) break;
  }
  return entry;
}

// One adjacent swap of the legs (position, position + 1) of a chain, both couplings read off the
// associator and the braiding.  Empty when a symbol is not a scalar.
std::optional<SwapMatrix> move_matrix(
    const std::vector<Sector>& legs, const std::vector<Sector>& inner, const Sector& coupled,
    int64_t position, const std::shared_ptr<FusionCategory>& group, const c10::Device& device) {
  SwapMatrix out;
  const Sector left = legs[position];
  const Sector right = legs[position + 1];
  if (position == 0) {
    // Swapping the first two legs only relabels the copy of their vertex; the label of the vertex
    // after it is untouched, which the zero entries say.  The braiding's second axis is the label
    // the swapped vertex takes.
    const Sector old = inner.empty() ? coupled : inner[0];
    const CopyEntry entry = copy_entry(group->braiding(left, right, old, device));
    for (const auto& [labels, value] : entry) {
      out[old][old][{labels[0], -1, labels[1], -1}] += value;
    }
    return out;
  }
  const Sector before = position == 1 ? legs[0] : inner[position - 2];
  const Sector after = position < static_cast<int64_t>(inner.size()) ? inner[position] : coupled;
  for (const Sector& f : group->fusion_outcomes(left, right)) {
    const std::vector<Sector> tail = group->fusion_outcomes(before, f);
    if (std::find(tail.begin(), tail.end(), after) == tail.end()) continue;
    const at::Tensor exchange = group->braiding(left, right, f, device);
    for (const Sector& old : group->fusion_outcomes(before, left)) {
      const std::vector<Sector> old_tail = group->fusion_outcomes(old, right);
      if (std::find(old_tail.begin(), old_tail.end(), after) == old_tail.end()) continue;
      const at::Tensor first = group->associator(before, left, right, after, old, f, device);
      for (const Sector& fresh : group->fusion_outcomes(before, right)) {
        const std::vector<Sector> fresh_tail = group->fusion_outcomes(fresh, left);
        if (std::find(fresh_tail.begin(), fresh_tail.end(), after) == fresh_tail.end()) continue;
        const at::Tensor second =
            group->associator(before, right, left, after, fresh, f, device);
        const CopyEntry entry = copy_entry(contract_swap(first, exchange, second));
        for (const auto& [labels, value] : entry) {
          out[old][fresh][labels] += value;
        }
      }
    }
  }
  return out;
}

// Bring a chain into the given leg order, one adjacent swap at a time.  Empty when the order is
// not a permutation of the legs or a symbol is not a scalar.
std::optional<std::vector<ChainState>> permute_chain(
    const std::vector<Sector>& legs, const std::vector<Sector>& inner, const Sector& coupled,
    const std::vector<int64_t>& copies, const std::vector<int64_t>& order,
    const std::shared_ptr<FusionCategory>& group, const c10::Device& device) {
  if (order.size() != legs.size()) return std::nullopt;
  if (!copies.empty() && copies.size() != legs.size() - 1) return std::nullopt;
  std::vector<int64_t> arrangement(legs.size());
  for (size_t axis = 0; axis < arrangement.size(); ++axis) {
    arrangement[axis] = static_cast<int64_t>(axis);
  }
  const std::vector<int64_t> initial =
      copies.empty() ? std::vector<int64_t>(legs.size() - 1, 0) : copies;
  std::vector<ChainState> states{{inner, initial, std::complex<double>(1.0, 0.0)}};
  for (size_t wanted = 0; wanted < order.size(); ++wanted) {
    size_t at = wanted;
    while (at < arrangement.size() && arrangement[at] != order[wanted]) ++at;
    if (at >= arrangement.size()) return std::nullopt;
    while (at > wanted) {
      std::vector<Sector> current;
      for (int64_t index : arrangement) current.push_back(legs[index]);
      const int64_t position = static_cast<int64_t>(at) - 1;
      const size_t slot = position == 0 ? 0 : static_cast<size_t>(position) - 1;
      std::vector<ChainState> next;
      for (const ChainState& state : states) {
        const auto matrix = move_matrix(current, state.inner, coupled, position, group, device);
        if (!matrix.has_value()) return std::nullopt;
        // The first swap of a two-leg chain couples the two legs, so the moved state gains that
        // coupling instead of rewriting an existing one.
        const Sector old = slot < state.inner.size() ? state.inner[slot] : coupled;
        const int64_t old_copy = state.copies.at(slot);
        const int64_t after_copy = slot + 1 < state.copies.size() ? state.copies.at(slot + 1) : 1;
        const auto row = matrix->find(old);
        if (row == matrix->end()) continue;
        for (const auto& [fresh, entry] : row->second) {
          for (const auto& [labels, factor] : entry) {
            if (labels[0] != old_copy) continue;
            if (labels[1] != -1 && labels[1] != after_copy) continue;
            ChainState moved = state;
            // A two-leg chain has no inner vertex, so its swap leaves the coupling list empty.
            if (slot < moved.inner.size()) moved.inner[slot] = fresh;
            if (labels[2] != -1) moved.copies[slot] = labels[2];
            if (labels[3] != -1 && slot + 1 < moved.copies.size()) {
              moved.copies[slot + 1] = labels[3];
            }
            moved.value *= factor;
            next.push_back(std::move(moved));
          }
        }
      }
      states = std::move(next);
      std::swap(arrangement[at], arrangement[at - 1]);
      --at;
    }
  }
  return states;
}

// The empty tree: no legs, coupled to the trivial sector.
PairTree empty_tree(const std::shared_ptr<FusionCategory>& group) {
  PairTree tree;
  tree.coupled = group->trivial_sector();
  return tree;
}

// Whether two trees are the same basis element of one fusion space: the legs, their dualities, the
// couplings and the vertex labels.  The coupled sector is kept out of it, because a bend leaves the
// two trees of a pair coupled to a channel of its own.
bool same_shape(const PairTree& one, const PairTree& other) {
  return one.uncoupled == other.uncoupled && one.isdual == other.isdual &&
         one.inner == other.inner && one.vertices == other.vertices;
}

// The same identity including the coupled sector, which two trees of the same block share.
bool same_tree(const PairTree& one, const PairTree& other) {
  return same_shape(one, other) && one.coupled == other.coupled;
}

// The Frobenius-Schur phase of a sector: the sign of the associator that carries the sector, its
// dual and two units, which is how TensorKit defines it.  Bending a dual leg costs it.
double frobenius_schur_phase(const std::shared_ptr<FusionCategory>& group, const Sector& sector,
                             const c10::Device& device) {
  const Sector unit = group->trivial_sector();
  const at::Tensor symbol =
      group->associator(sector, group->dual_sector(sector), sector, sector, unit, unit, device);
  if (symbol.numel() == 0) return 1.0;
  return symbol.reshape({-1}).select(0, 0).item<c10::complex<double>>().real() < 0.0 ? -1.0 : 1.0;
}

// One bend, TensorKit's ``bendright``: the last leg of ``first`` leaves as the dual leg of
// ``second``, and both trees then couple to the channel the split vertex sat under.  The copy
// labels come from the associator of that vertex: its first axis is the label the leg leaves
// behind and its second is the label the new vertex of ``second`` takes.
std::vector<std::pair<std::pair<PairTree, PairTree>, std::complex<double>>> bend_right(
    const std::shared_ptr<FusionCategory>& group, const PairTree& first, const PairTree& second,
    const c10::Device& device) {
  std::vector<std::pair<std::pair<PairTree, PairTree>, std::complex<double>>> out;
  const int64_t n1 = static_cast<int64_t>(first.uncoupled.size());
  const int64_t n2 = static_cast<int64_t>(second.uncoupled.size());
  if (n1 == 0) return out;
  const Sector& c = first.coupled;
  const Sector a =
      n1 == 1 ? group->trivial_sector() : (n1 == 2 ? first.uncoupled[0] : first.inner.back());
  const Sector& b = first.uncoupled.back();
  double scale = std::sqrt(group->quantum_dim(b));
  if (first.isdual.back() != 0) {
    scale *= frobenius_schur_phase(group, group->dual_sector(b), device);
  }
  const at::Tensor symbol =
      group->associator(a, b, group->dual_sector(b), a, c, group->trivial_sector(), device);
  if (symbol.numel() == 0) return out;
  PairTree head;
  head.uncoupled.assign(first.uncoupled.begin(), first.uncoupled.end() - 1);
  head.isdual.assign(first.isdual.begin(), first.isdual.end() - 1);
  if (n1 > 2) head.inner.assign(first.inner.begin(), first.inner.end() - 1);
  if (n1 > 1) head.vertices.assign(first.vertices.begin(), first.vertices.end() - 1);
  head.coupled = a;
  const int64_t head_label = n1 > 1 ? first.vertices.back() : 0;
  for (const auto& [labels, value] : copy_entry(symbol)) {
    if (labels[0] != head_label) continue;
    PairTree grown;
    grown.uncoupled = second.uncoupled;
    grown.uncoupled.push_back(group->dual_sector(b));
    grown.isdual = second.isdual;
    grown.isdual.push_back(first.isdual.back() == 0 ? 1 : 0);
    if (n2 > 1) {
      grown.inner = second.inner;
      grown.inner.push_back(c);
    }
    if (n2 > 0) {
      grown.vertices = second.vertices;
      grown.vertices.push_back(labels[1]);
    }
    grown.coupled = a;
    out.push_back({{head, grown}, scale * value});
  }
  return out;
}

// Every pair a tree reaches when ``count`` of its codomain legs are bent away, with the coefficient
// of the bends.
std::vector<std::pair<std::pair<PairTree, PairTree>, std::complex<double>>> repartition(
    const std::shared_ptr<FusionCategory>& group, const PairTree& first, const PairTree& second,
    int64_t count, const c10::Device& device) {
  std::vector<std::pair<std::pair<PairTree, PairTree>, std::complex<double>>> states{
      {{first, second}, std::complex<double>(1.0, 0.0)}};
  const int64_t steps = static_cast<int64_t>(first.uncoupled.size()) - count;
  for (int64_t step = 0; step < steps && !states.empty(); ++step) {
    std::vector<std::pair<std::pair<PairTree, PairTree>, std::complex<double>>> grown;
    for (const auto& [pair, value] : states) {
      for (const auto& [moved, coefficient] : bend_right(group, pair.first, pair.second, device)) {
        grown.push_back({moved, value * coefficient});
      }
    }
    states = std::move(grown);
  }
  return states;
}

// Bring a tree into the leg order ``target`` by the braiding of every adjacent swap.
std::vector<std::pair<PairTree, std::complex<double>>> permute_tree(
    const std::shared_ptr<FusionCategory>& group, const PairTree& tree,
    const std::vector<int64_t>& target, const c10::Device& device) {
  std::vector<std::pair<PairTree, std::complex<double>>> out;
  if (tree.uncoupled.empty()) {
    out.push_back({tree, std::complex<double>(1.0, 0.0)});
    return out;
  }
  const auto states =
      permute_chain(tree.uncoupled, tree.inner, tree.coupled, tree.vertices, target, group, device);
  if (!states.has_value()) return out;
  for (const ChainState& state : *states) {
    PairTree moved;
    for (int64_t axis : target) {
      moved.uncoupled.push_back(tree.uncoupled[axis]);
      moved.isdual.push_back(tree.isdual[axis]);
    }
    moved.inner = state.inner;
    moved.vertices = state.copies;
    moved.coupled = tree.coupled;
    out.push_back({moved, state.value});
  }
  return out;
}

// One integer naming the device a cached weight belongs to.
int64_t weight_device_code(const c10::Device& device) {
  return static_cast<int64_t>(device.type()) * 1000 + device.index();
}

// The weight of a tree triple does not depend on the blocks that carry it, and a sweep asks for
// the same triples over and over, so the answers are kept per group, layout and device.
using WeightKey = std::tuple<const FusionCategory*, std::vector<Sector>, std::vector<Sector>,
                             Sector, std::vector<Sector>, std::vector<Sector>, Sector,
                             std::vector<Sector>, std::vector<Sector>, Sector,
                             std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>,
                             std::vector<int8_t>, std::vector<int8_t>, std::vector<int8_t>,
                             std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>,
                             std::vector<int64_t>, int64_t>;

std::map<WeightKey, std::optional<std::complex<double>>>& weight_cache() {
  static std::map<WeightKey, std::optional<std::complex<double>>> cache;
  return cache;
}

// The number of weights one process keeps before the table is dropped.
constexpr std::size_t kWeightCacheLimit = 1U << 20U;

std::optional<std::complex<double>> compute_diagram_weight(
    const std::shared_ptr<FusionCategory>& group, const std::vector<Sector>& left_legs,
    const std::vector<Sector>& left_inner, const Sector& left_coupled,
    const std::vector<Sector>& right_legs, const std::vector<Sector>& right_inner,
    const Sector& right_coupled, const std::vector<Sector>& out_legs,
    const std::vector<Sector>& out_inner, const Sector& out_coupled,
    const std::vector<int64_t>& left_copies, const std::vector<int64_t>& right_copies,
    const std::vector<int64_t>& out_copies, const std::vector<int8_t>& left_dual,
    const std::vector<int8_t>& right_dual, const std::vector<int8_t>& out_dual,
    const std::vector<int64_t>& shared_left, const std::vector<int64_t>& shared_right,
    const std::vector<int64_t>& out_left_axis, const std::vector<int64_t>& out_right_axis,
    const c10::Device& device) {
  g_weight_decline = 0;
  // The weight below is exact only while the duality acts as the identity on every sector the
  // diagram touches: a non-self-dual sector carries the phase of the conjugate basis into the
  // recoupling, which this form does not model yet, so the caller keeps its Clebsch-Gordan route.
  const auto identity_duality = [&group](const Sector& sector) {
    return group->dual_sector(sector) == sector;
  };
  for (const std::vector<Sector>* tree : {&left_legs, &left_inner, &right_legs, &right_inner,
                                         &out_legs, &out_inner}) {
    for (const Sector& sector : *tree) {
      if (!identity_duality(sector)) {
        g_weight_decline = 10;
        return std::nullopt;
      }
    }
  }
  if (!identity_duality(left_coupled) || !identity_duality(right_coupled) ||
      !identity_duality(out_coupled)) {
    g_weight_decline = 10;
    return std::nullopt;
  }
  // The two trees of a glued pair are compared after a bend has moved the leg into the domain, so
  // they line up only when the pair carries the dual on exactly one side.  A pair that carries it
  // on both or on neither needs the metric of the identified leg, which this form does not model.
  const int64_t shared = static_cast<int64_t>(shared_left.size());
  for (int64_t index = 0; index < shared; ++index) {
    const int8_t left_flag = left_dual[shared_left[index]];
    const int8_t right_flag = right_dual[shared_right[index]];
    if (left_flag == right_flag) {
      g_weight_decline = 11;
      return std::nullopt;
    }
  }
  // The free legs of both operands, in the order the output lists them.
  const auto ordered = [](const std::vector<int64_t>& position) {
    std::vector<int64_t> free;
    for (int64_t axis = 0; axis < static_cast<int64_t>(position.size()); ++axis) {
      if (position[axis] >= 0) free.push_back(axis);
    }
    std::stable_sort(free.begin(), free.end(), [&position](int64_t one, int64_t two) {
      return position[one] < position[two];
    });
    return free;
  };
  const std::vector<int64_t> free_left = ordered(out_left_axis);
  const std::vector<int64_t> free_right = ordered(out_right_axis);
  const int64_t alpha = static_cast<int64_t>(free_left.size());
  const int64_t beta = static_cast<int64_t>(free_right.size());
  // The output names every free leg of both operands exactly once: a leg the output keeps and both
  // operands share would be counted twice, which this routine does not model.
  if (static_cast<int64_t>(out_legs.size()) != alpha + beta ||
      alpha + shared != static_cast<int64_t>(left_legs.size()) ||
      beta + shared != static_cast<int64_t>(right_legs.size())) {
    g_weight_decline = 1;
    return std::nullopt;
  }
  // A bend takes the last leg of the codomain, so the legs that leave it are put at its end: the
  // left keeps its free legs and bends its shared ones, the right keeps its shared ones and bends
  // its free ones, and the output keeps its left free legs and bends its right ones.
  std::vector<int64_t> left_order(free_left);
  for (auto it = shared_left.rbegin(); it != shared_left.rend(); ++it) left_order.push_back(*it);
  std::vector<int64_t> right_order(shared_right);
  for (auto it = free_right.rbegin(); it != free_right.rend(); ++it) right_order.push_back(*it);
  std::vector<int64_t> joined;
  for (int64_t axis : free_left) joined.push_back(out_left_axis[axis]);
  for (int64_t axis : free_right) joined.push_back(out_right_axis[axis]);
  std::vector<int64_t> out_order(joined);
  std::reverse(out_order.begin() + alpha, out_order.end());

  const PairTree left_tree{left_legs, left_dual, left_inner, left_copies, left_coupled};
  const PairTree right_tree{right_legs, right_dual, right_inner, right_copies, right_coupled};
  const PairTree out_tree{out_legs, out_dual, out_inner, out_copies, out_coupled};
  std::vector<TreeTerm> left_terms;
  std::vector<TreeTerm> right_terms;
  std::vector<TreeTerm> out_terms;
  const auto collect = [&](const PairTree& tree, const std::vector<int64_t>& order, int64_t keep,
                           std::vector<TreeTerm>& terms) {
    for (const auto& [moved, value] : permute_tree(group, tree, order, device)) {
      for (const auto& [pair, coefficient] :
           repartition(group, moved, empty_tree(group), keep, device)) {
        terms.push_back(TreeTerm{pair.first, pair.second, value * coefficient});
      }
    }
  };
  collect(left_tree, left_order, alpha, left_terms);
  collect(right_tree, right_order, shared, right_terms);
  collect(out_tree, out_order, alpha, out_terms);
  if (left_terms.empty() || right_terms.empty() || out_terms.empty()) {
    g_weight_decline = 2;
    return std::nullopt;
  }
  // The two operands meet in the shared tree, whose coupled sector is a channel of each of them, so
  // only its shape is compared; the free parts of both are the same basis elements as the output's,
  // coupled sector included.  The shared block closes with the quantum dimension of its channel, and
  // a shared coupling is read through the conjugate basis, so each one carries the category's
  // duality scalar.
  std::complex<double> total = 0.0;
  for (const TreeTerm& one : left_terms) {
    for (const TreeTerm& two : right_terms) {
      if (!same_shape(one.other, two.kept)) continue;
      for (const TreeTerm& three : out_terms) {
        if (!same_tree(one.kept, three.kept) || !same_tree(two.other, three.other)) continue;
        // The shared block closes with the quantum dimension of the channel it couples to.
        std::complex<double> factor = group->quantum_dim(two.kept.coupled);
        total += one.value * two.value * three.value * factor;
      }
    }
  }
  return total;
}
}  // namespace

void clear_weight_cache() {
  weight_cache().clear();
}

bool diagram_weight(const std::shared_ptr<FusionCategory>& group,
                    const std::vector<Sector>& left_legs, const std::vector<Sector>& left_inner,
                    const Sector& left_coupled, const std::vector<Sector>& right_legs,
                    const std::vector<Sector>& right_inner, const Sector& right_coupled,
                    const std::vector<Sector>& out_legs, const std::vector<Sector>& out_inner,
                    const Sector& out_coupled, const std::vector<int64_t>& left_copies,
                    const std::vector<int64_t>& right_copies,
                    const std::vector<int64_t>& out_copies,
                    const std::vector<int8_t>& left_dual, const std::vector<int8_t>& right_dual,
                    const std::vector<int8_t>& out_dual,
                    const std::vector<int64_t>& shared_left,
                    const std::vector<int64_t>& shared_right,
                    const std::vector<int64_t>& out_left_axis,
                    const std::vector<int64_t>& out_right_axis, const c10::Device& device,
                    std::complex<double>& weight) {
  const WeightKey key{group.get(),   left_legs,      left_inner,     left_coupled,
                      right_legs,    right_inner,    right_coupled,  out_legs,
                      out_inner,     out_coupled,    left_copies,    right_copies,
                      out_copies,    left_dual,      right_dual,     out_dual,
                      shared_left,   shared_right,   out_left_axis,  out_right_axis,
                      weight_device_code(device)};
  auto& cache = weight_cache();
  const auto found = cache.find(key);
  if (found != cache.end()) {
    if (!found->second.has_value()) return false;
    weight = *found->second;
    return true;
  }
  const std::optional<std::complex<double>> result =
      compute_diagram_weight(group, left_legs, left_inner, left_coupled, right_legs, right_inner,
                             right_coupled, out_legs, out_inner, out_coupled, left_copies,
                             right_copies, out_copies, left_dual, right_dual, out_dual,
                             shared_left, shared_right, out_left_axis, out_right_axis, device);
  if (cache.size() >= kWeightCacheLimit) cache.clear();
  cache.emplace(key, result);
  if (!result.has_value()) return false;
  weight = *result;
  return true;
}

}  // namespace sym
