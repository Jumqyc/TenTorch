#include "diagram.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace sym {
namespace {

// The guard that declined the last weight this translation unit computed, 0 when none did.
int g_decline = 0;

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

// A move that rewrote a pair of trees, and the coefficient of the move.
struct BendTerm {
  PairTree kept;
  PairTree other;
  std::complex<double> value;
};

// A move that rewrote one tree, and the coefficient of the move.
struct TreeValue {
  PairTree tree;
  std::complex<double> value;
};

// The two operands of one block combination, bent into the domain, with the leg order the output
// is brought into.  Building it once weighs every output tree of the combination against it,
// instead of redoing the operand half for each of them.
struct Diagram {
  std::shared_ptr<FusionCategory> group;
  std::vector<BendTerm> left;
  std::vector<BendTerm> right;
  std::vector<int64_t> out_order;
  int64_t alpha;
  int64_t beta;
};

// The nonzero entries of a symbol, as the zero-based label of every axis.  A symbol carries four
// axes at most; the entry of an impossible move is empty.
CopyEntry copy_entry(const at::Tensor& symbol) {
  if (symbol.dim() > 4) {
    throw not_implemented_error("A diagram weight reads the copy labels of four axes at most.");
  }
  // One host copy and one pass over its values, so a symbol costs its caller a single sync.
  const at::Tensor host = symbol.to(at::kCPU, at::kComplexDouble).contiguous();
  CopyEntry entry;
  if (host.numel() == 0) return entry;
  const c10::complex<double>* values = host.data_ptr<c10::complex<double>>();
  std::array<int64_t, 4> labels{0, 0, 0, 0};
  for (int64_t flat = 0; flat < host.numel(); ++flat) {
    // The last axis moves fastest, so a row-major walk decodes the label of every axis.
    int64_t rest = flat;
    for (int64_t axis = host.dim() - 1; axis >= 0; --axis) {
      labels[axis] = rest % host.size(axis);
      rest /= host.size(axis);
    }
    if (values[flat] != c10::complex<double>(0.0, 0.0)) {
      entry[labels] = std::complex<double>(values[flat].real(), values[flat].imag());
    }
  }
  return entry;
}

// The two associators of a swap and the braiding of the exchanged charge, contracted into the
// amplitudes of the four vertices the swap touches.
at::Tensor contract_swap(const at::Tensor& first, const at::Tensor& exchange,
                         const at::Tensor& second) {
  return at::einsum("abcd,cr,pqrd->abpq", {first, exchange, second});
}

// The recoupling matrix of swapping the legs (position, position + 1) of a chain: for every
// coupling ``old`` the swap may read and every ``fresh`` coupling it may write, the copy entries
// of the swapped and of the following vertex.
SwapMatrix move_matrix(const std::vector<Sector>& legs, const std::vector<Sector>& inner,
                       const Sector& coupled, int64_t position,
                       const std::shared_ptr<FusionCategory>& group, const c10::Device& device) {
  SwapMatrix out;
  const Sector left = legs[position];
  const Sector right = legs[position + 1];
  if (position == 0) {
    // Swapping the first two legs only relabels the copy of their vertex; the label of the vertex
    // after it is untouched, which the minus-one entries say.  The braiding's second axis is the
    // label the swapped vertex takes.
    const Sector old = inner.empty() ? coupled : inner[0];
    for (const auto& [labels, value] : copy_entry(group->braiding(left, right, old, device))) {
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
        for (const auto& [labels, value] : copy_entry(contract_swap(first, exchange, second))) {
          out[old][fresh][labels] += value;
        }
      }
    }
  }
  return out;
}

// Bring a chain into the given leg order, one adjacent swap at a time.  Empty when the order is
// not a permutation of the legs.
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
  std::vector<ChainState> states{{inner, initial, 1.0}};
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
        const SwapMatrix matrix = move_matrix(current, state.inner, coupled, position, group, device);
        // The first swap of a two-leg chain couples the two legs, so the moved state gains that
        // coupling instead of rewriting an existing one.
        const Sector old = slot < state.inner.size() ? state.inner[slot] : coupled;
        const int64_t old_copy = state.copies.at(slot);
        const int64_t after_copy = slot + 1 < state.copies.size() ? state.copies.at(slot + 1) : 1;
        const auto row = matrix.find(old);
        if (row == matrix.end()) continue;
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

// Whether every sector a tree touches is its own dual, which is what the recoupling below models.
bool self_dual(const std::shared_ptr<FusionCategory>& group, const PairTree& tree) {
  const auto identity = [&group](const Sector& sector) {
    return group->dual_sector(sector) == sector;
  };
  for (const Sector& sector : tree.uncoupled) {
    if (!identity(sector)) return false;
  }
  for (const Sector& sector : tree.inner) {
    if (!identity(sector)) return false;
  }
  return identity(tree.coupled);
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
std::vector<BendTerm> bend_right(const std::shared_ptr<FusionCategory>& group, const PairTree& first,
                                 const PairTree& second, const c10::Device& device) {
  std::vector<BendTerm> out;
  const int64_t n1 = static_cast<int64_t>(first.uncoupled.size());
  const int64_t n2 = static_cast<int64_t>(second.uncoupled.size());
  if (n1 == 0) return out;
  const Sector& c = first.coupled;
  const Sector a =
      n1 == 1 ? group->trivial_sector() : (n1 == 2 ? first.uncoupled[0] : first.inner.back());
  const Sector& b = first.uncoupled.back();
  double scale = std::sqrt(group->quantum_dim(b));
  if (first.isdual.back()) scale *= frobenius_schur_phase(group, group->dual_sector(b), device);
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
    grown.isdual.push_back(!first.isdual.back());
    if (n2 > 1) {
      grown.inner = second.inner;
      grown.inner.push_back(c);
    }
    if (n2 > 0) {
      grown.vertices = second.vertices;
      grown.vertices.push_back(labels[1]);
    }
    grown.coupled = a;
    out.push_back(BendTerm{head, grown, scale * value});
  }
  return out;
}

// Every pair a tree reaches when ``keep`` of its codomain legs are kept and the rest are bent off
// its end, one bend at a time, with the coefficient of the whole move.
std::vector<BendTerm> repartition(const std::shared_ptr<FusionCategory>& group, const PairTree& tree,
                                  int64_t keep, const c10::Device& device) {
  std::vector<BendTerm> states{
      {tree, PairTree{.coupled = group->trivial_sector()}, 1.0}};
  const int64_t steps = static_cast<int64_t>(tree.uncoupled.size()) - keep;
  for (int64_t step = 0; step < steps && !states.empty(); ++step) {
    std::vector<BendTerm> grown;
    for (const BendTerm& state : states) {
      for (const BendTerm& moved : bend_right(group, state.kept, state.other, device)) {
        grown.push_back(BendTerm{moved.kept, moved.other, state.value * moved.value});
      }
    }
    states = std::move(grown);
  }
  return states;
}

// Bring a tree into the leg order ``target`` by the braiding of every adjacent swap.
std::vector<TreeValue> permute_tree(const std::shared_ptr<FusionCategory>& group,
                                    const PairTree& tree, const std::vector<int64_t>& target,
                                    const c10::Device& device) {
  std::vector<TreeValue> out;
  if (tree.uncoupled.empty()) {
    out.push_back(TreeValue{tree, 1.0});
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
    out.push_back(TreeValue{moved, state.value});
  }
  return out;
}

// The leg order every tree of the diagram is brought into before its bends, and the number of free
// legs of each operand.  A bend takes the last leg of a tree, so the legs that stay go in front of
// the ones that leave: the left keeps its free legs and bends its shared ones, the right keeps its
// shared ones and bends its free ones, and the output keeps its left free legs and bends its right.
struct Orders {
  std::vector<int64_t> left;
  std::vector<int64_t> right;
  std::vector<int64_t> out;
  int64_t alpha;
  int64_t beta;
};

Orders diagram_orders(const Gluing& glue) {
  // The free legs of one operand, in the order the output lists them.
  const auto free = [](const std::vector<int64_t>& axis) {
    std::vector<int64_t> order;
    for (int64_t index = 0; index < static_cast<int64_t>(axis.size()); ++index) {
      if (axis[index] >= 0) order.push_back(index);
    }
    std::stable_sort(order.begin(), order.end(),
                     [&axis](int64_t one, int64_t two) { return axis[one] < axis[two]; });
    return order;
  };
  const std::vector<int64_t> free_left = free(glue.out_left);
  const std::vector<int64_t> free_right = free(glue.out_right);
  Orders orders;
  orders.alpha = static_cast<int64_t>(free_left.size());
  orders.beta = static_cast<int64_t>(free_right.size());
  orders.left = free_left;
  for (auto it = glue.left.rbegin(); it != glue.left.rend(); ++it) orders.left.push_back(*it);
  orders.right = glue.right;
  for (auto it = free_right.rbegin(); it != free_right.rend(); ++it) orders.right.push_back(*it);
  for (int64_t axis : free_left) orders.out.push_back(glue.out_left[axis]);
  for (int64_t axis : free_right) orders.out.push_back(glue.out_right[axis]);
  std::reverse(orders.out.begin() + orders.alpha, orders.out.end());
  return orders;
}

// The terms one operand reaches once it stands in ``order`` and keeps ``keep`` of its legs: every
// permutation of its free legs, then the bends of the rest.
std::vector<BendTerm> operand_terms(const std::shared_ptr<FusionCategory>& group,
                                    const PairTree& tree, const std::vector<int64_t>& order,
                                    int64_t keep, const c10::Device& device) {
  std::vector<BendTerm> terms;
  for (const TreeValue& moved : permute_tree(group, tree, order, device)) {
    for (const BendTerm& bend : repartition(group, moved.tree, keep, device)) {
      terms.push_back(BendTerm{bend.kept, bend.other, moved.value * bend.value});
    }
  }
  return terms;
}

// The guard on the operand half of a diagram, 0 when it covers it:
//   1  the output does not name every free leg of both operands exactly once
//   10 a sector of an operand is not its own dual
//   11 a glued pair carries the dual on both sides or on neither
int operand_guard(const std::shared_ptr<FusionCategory>& group, const PairTree& left,
                  const PairTree& right, const Gluing& glue) {
  if (!self_dual(group, left) || !self_dual(group, right)) return 10;
  const std::size_t shared = glue.left.size();
  for (std::size_t index = 0; index < shared; ++index) {
    if (left.isdual[glue.left[index]] == right.isdual[glue.right[index]]) return 11;
  }
  const auto free_count = [](const std::vector<int64_t>& axis) {
    return static_cast<std::size_t>(
        std::count_if(axis.begin(), axis.end(), [](int64_t one) { return one >= 0; }));
  };
  if (free_count(glue.out_left) + shared != left.uncoupled.size() ||
      free_count(glue.out_right) + shared != right.uncoupled.size()) {
    return 1;
  }
  return 0;
}

// Build the operand half of one block combination, or nothing when a guard declines the shape.
std::optional<Diagram> make_diagram(const std::shared_ptr<FusionCategory>& group,
                                    const PairTree& left, const PairTree& right,
                                    const Gluing& glue, const c10::Device& device) {
  g_decline = operand_guard(group, left, right, glue);
  if (g_decline != 0) return std::nullopt;
  const Orders orders = diagram_orders(glue);
  std::vector<BendTerm> left_terms = operand_terms(group, left, orders.left, orders.alpha, device);
  std::vector<BendTerm> right_terms =
      operand_terms(group, right, orders.right, static_cast<int64_t>(glue.left.size()), device);
  if (left_terms.empty() || right_terms.empty()) {
    g_decline = 2;
    return std::nullopt;
  }
  return Diagram{group, std::move(left_terms), std::move(right_terms), orders.out, orders.alpha,
                 orders.beta};
}

// The weight of one output tree against a built diagram, or nothing when a guard declines it.
std::optional<std::complex<double>> weigh(const Diagram& diagram, const PairTree& out,
                                          const c10::Device& device) {
  if (!self_dual(diagram.group, out)) {
    g_decline = 10;
    return std::nullopt;
  }
  if (static_cast<int64_t>(out.uncoupled.size()) != diagram.alpha + diagram.beta) {
    g_decline = 1;
    return std::nullopt;
  }
  const std::vector<BendTerm> terms =
      operand_terms(diagram.group, out, diagram.out_order, diagram.alpha, device);
  if (terms.empty()) {
    g_decline = 2;
    return std::nullopt;
  }
  // The two operands meet in the shared tree, whose coupled sector is a channel of each of them, so
  // only its shape is compared; the free parts of both are the same basis elements as the output's,
  // coupled sector included.  The shared block closes with the quantum dimension of its channel.
  std::complex<double> total = 0.0;
  for (const BendTerm& one : diagram.left) {
    for (const BendTerm& two : diagram.right) {
      if (!same_shape(one.other, two.kept)) continue;
      for (const BendTerm& three : terms) {
        if (!same_tree(one.kept, three.kept) || !same_tree(two.other, three.other)) continue;
        total += one.value * two.value * three.value * diagram.group->quantum_dim(two.kept.coupled);
      }
    }
  }
  return total;
}

// The operand half of a diagram, and the guard that declined it when it has none.
struct DiagramAnswer {
  std::optional<Diagram> diagram;
  int decline = 0;
};

// The weight of one output tree, and the guard that declined it.
struct WeightAnswer {
  std::complex<double> weight = 0.0;
  int decline = 0;
};

// A weight does not depend on the blocks that carry it, and a sweep asks for the same diagrams over
// and over, so the operand half and the weight are each kept per group, layout and device.
using DiagramKey = std::tuple<const FusionCategory*, PairTree, PairTree, Gluing, int64_t>;
using WeightKey = std::tuple<const FusionCategory*, PairTree, PairTree, Gluing, int64_t, PairTree>;

std::map<DiagramKey, DiagramAnswer>& diagram_cache() {
  static std::map<DiagramKey, DiagramAnswer> cache;
  return cache;
}

std::map<WeightKey, WeightAnswer>& weight_cache() {
  static std::map<WeightKey, WeightAnswer> cache;
  return cache;
}

// The number of weights one process keeps before the table is dropped.
constexpr std::size_t kWeightCacheLimit = 1U << 20U;

// The number of terms the diagram table keeps before it is dropped.  An entry of that table holds
// the operand terms themselves, which dwarf its key, and host memory is the scarce resource of this
// box, so the table is bounded by what it holds rather than by how many keys it has.
constexpr std::size_t kDiagramTermBudget = 1U << 16U;

// The terms the diagram table holds right now.
std::size_t g_diagram_terms = 0;

// The operand half of a diagram, from the table or built once.
const DiagramAnswer& cached_diagram(const std::shared_ptr<FusionCategory>& group,
                                    const PairTree& left, const PairTree& right, const Gluing& glue,
                                    const c10::Device& device) {
  const DiagramKey key{group.get(), left, right, glue, device_code(device)};
  auto& cache = diagram_cache();
  const auto found = cache.find(key);
  if (found != cache.end()) return found->second;
  std::optional<Diagram> diagram = make_diagram(group, left, right, glue, device);
  const int decline = g_decline;
  const std::size_t terms = diagram.has_value() ?
                                diagram->left.size() + diagram->right.size() :
                                0;
  if (g_diagram_terms + terms > kDiagramTermBudget) {
    cache.clear();
    g_diagram_terms = 0;
  }
  g_diagram_terms += terms;
  return cache.emplace(key, DiagramAnswer{std::move(diagram), decline}).first->second;
}

}  // namespace

PairTree pair_tree(const std::vector<Leg>& legs, const FusionTree& tree) {
  // A tree on no legs is the scalar block: the engine spells it as one sector holding the charge it
  // closes to, and that sector is not a leg, so only the coupled charge survives.
  if (legs.empty()) return PairTree{.coupled = tree.coupled()};
  std::vector<bool> isdual;
  isdual.reserve(legs.size());
  for (const Leg& leg : legs) isdual.push_back(leg.qconj() == -1);
  return PairTree{tree.sectors(), isdual, tree.inner(), tree.copies(), tree.coupled()};
}

Gluing glue_legs(const std::vector<int64_t>& left_labels, const std::vector<int64_t>& right_labels,
                 const std::vector<int64_t>& out_labels) {
  Gluing glue;
  glue.out_left.assign(left_labels.size(), -1);
  glue.out_right.assign(right_labels.size(), -1);
  // The axis each operand's leg occupies in the output, so the operands' free legs can be listed in
  // the output's order; a leg the output keeps and both operands share has an axis here too, and is
  // excluded from the glued legs below.
  for (std::size_t axis = 0; axis < left_labels.size(); ++axis) {
    const auto in_out = std::find(out_labels.begin(), out_labels.end(), left_labels[axis]);
    if (in_out != out_labels.end()) glue.out_left[axis] = in_out - out_labels.begin();
  }
  for (std::size_t axis = 0; axis < right_labels.size(); ++axis) {
    const auto in_out = std::find(out_labels.begin(), out_labels.end(), right_labels[axis]);
    if (in_out != out_labels.end()) glue.out_right[axis] = in_out - out_labels.begin();
  }
  for (std::size_t axis = 0; axis < left_labels.size(); ++axis) {
    if (glue.out_left[axis] >= 0) continue;
    const auto in_right = std::find(right_labels.begin(), right_labels.end(), left_labels[axis]);
    if (in_right == right_labels.end()) continue;
    glue.left.push_back(static_cast<int64_t>(axis));
    glue.right.push_back(in_right - right_labels.begin());
  }
  return glue;
}

void clear_weight_cache() {
  weight_cache().clear();
  diagram_cache().clear();
  g_diagram_terms = 0;
}

int diagram_weight(const std::shared_ptr<FusionCategory>& group, const PairTree& left,
                   const PairTree& right, const PairTree& out, const Gluing& glue,
                   const c10::Device& device, std::complex<double>& weight) {
  const WeightKey key{group.get(), left, right, glue, device_code(device), out};
  auto& cache = weight_cache();
  const auto found = cache.find(key);
  if (found != cache.end()) {
    if (found->second.decline != 0) return found->second.decline;
    weight = found->second.weight;
    return 0;
  }
  const DiagramAnswer& half = cached_diagram(group, left, right, glue, device);
  WeightAnswer answer;
  if (half.diagram.has_value()) {
    const std::optional<std::complex<double>> value = weigh(*half.diagram, out, device);
    if (value.has_value()) {
      answer.weight = *value;
    } else {
      answer.decline = g_decline;
    }
  } else {
    answer.decline = half.decline;
  }
  if (cache.size() >= kWeightCacheLimit) cache.clear();
  cache.emplace(key, answer);
  if (answer.decline != 0) return answer.decline;
  weight = answer.weight;
  return 0;
}

}  // namespace sym
