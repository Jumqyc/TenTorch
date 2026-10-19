#include "diagram.h"

#include <algorithm>
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

// The recoupling matrix of one adjacent swap: rows over the old coupling, columns over the new.
using SwapMatrix = std::map<Sector, std::map<Sector, std::complex<double>>>;

// A chain during the permutation: its inner couplings, the coupled charge excluded, and its
// coefficient.
struct ChainState {
  std::vector<Sector> inner;
  std::complex<double> value;
};

// The single entry of a one-element symbol tensor, real or complex.  False when the category
// pairs the symbol into a matrix over fusion copies, which this routine does not evaluate.
bool symbol_scalar(const at::Tensor& symbol, std::complex<double>& out) {
  if (symbol.numel() != 1) return false;
  const c10::complex<double> value = symbol.item().toComplexDouble();
  out = std::complex<double>(value.real(), value.imag());
  return true;
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
    const Sector old = inner.empty() ? coupled : inner[0];
    std::complex<double> exchange;
    if (!symbol_scalar(group->braiding(left, right, old, device), exchange)) return std::nullopt;
    out[old][old] = exchange;
    return out;
  }
  const Sector before = position == 1 ? legs[0] : inner[position - 2];
  const Sector after = position < static_cast<int64_t>(inner.size()) ? inner[position] : coupled;
  for (const Sector& f : group->fusion_outcomes(left, right)) {
    const std::vector<Sector> tail = group->fusion_outcomes(before, f);
    if (std::find(tail.begin(), tail.end(), after) == tail.end()) continue;
    std::complex<double> exchange;
    if (!symbol_scalar(group->braiding(left, right, f, device), exchange)) return std::nullopt;
    for (const Sector& old : group->fusion_outcomes(before, left)) {
      const std::vector<Sector> old_tail = group->fusion_outcomes(old, right);
      if (std::find(old_tail.begin(), old_tail.end(), after) == old_tail.end()) continue;
      std::complex<double> first;
      if (!symbol_scalar(group->associator(before, left, right, after, old, f, device), first)) {
        return std::nullopt;
      }
      for (const Sector& fresh : group->fusion_outcomes(before, right)) {
        const std::vector<Sector> fresh_tail = group->fusion_outcomes(fresh, left);
        if (std::find(fresh_tail.begin(), fresh_tail.end(), after) == fresh_tail.end()) continue;
        std::complex<double> second;
        if (!symbol_scalar(group->associator(before, right, left, after, fresh, f, device),
                           second)) {
          return std::nullopt;
        }
        out[old][fresh] += first * exchange * second;
      }
    }
  }
  return out;
}

// Bring a chain into the given leg order, one adjacent swap at a time.  Empty when the order is
// not a permutation of the legs or a symbol is not a scalar.
std::optional<std::vector<ChainState>> permute_chain(
    const std::vector<Sector>& legs, const std::vector<Sector>& inner, const Sector& coupled,
    const std::vector<int64_t>& order, const std::shared_ptr<FusionCategory>& group,
    const c10::Device& device) {
  if (order.size() != legs.size()) return std::nullopt;
  std::vector<int64_t> arrangement(legs.size());
  for (size_t axis = 0; axis < arrangement.size(); ++axis) {
    arrangement[axis] = static_cast<int64_t>(axis);
  }
  std::vector<ChainState> states{{inner, std::complex<double>(1.0, 0.0)}};
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
        const auto row = matrix->find(old);
        if (row == matrix->end()) continue;
        for (const auto& [fresh, factor] : row->second) {
          ChainState moved = state;
          // A two-leg chain has no inner vertex, so its swap leaves the coupling list empty.
          if (slot < moved.inner.size()) moved.inner[slot] = fresh;
          moved.value *= factor;
          next.push_back(std::move(moved));
        }
      }
      states = std::move(next);
      std::swap(arrangement[at], arrangement[at - 1]);
      --at;
    }
  }
  return states;
}

// The operand's legs in the order its chain is moved to: the shared legs first, then the free
// legs sorted by the axis each one occupies in the output.
std::vector<int64_t> moved_order(const std::vector<int64_t>& shared,
                                 const std::vector<int64_t>& out_axis) {
  std::vector<int64_t> order = shared;
  std::vector<int64_t> free_axes;
  for (int64_t axis = 0; axis < static_cast<int64_t>(out_axis.size()); ++axis) {
    if (out_axis[axis] >= 0) free_axes.push_back(axis);
  }
  std::stable_sort(free_axes.begin(), free_axes.end(),
                   [&out_axis](int64_t one, int64_t two) { return out_axis[one] < out_axis[two]; });
  order.insert(order.end(), free_axes.begin(), free_axes.end());
  return order;
}

// The couplings of the left free part that let its chain fuse to the shared block's own charge.
std::vector<std::vector<Sector>> free_frames(const std::vector<Sector>& legs, const Sector& block,
                                             const std::shared_ptr<FusionCategory>& group) {
  if (legs.size() < 2) return {{}};
  std::vector<std::vector<Sector>> frames;
  std::vector<Sector> prefix;
  std::function<void(int64_t, const Sector&)> extend = [&](int64_t axis, const Sector& reached) {
    if (axis >= static_cast<int64_t>(legs.size()) - 1) {
      const std::vector<Sector> fused = group->fusion_outcomes(reached, legs.back());
      if (std::find(fused.begin(), fused.end(), block) != fused.end()) frames.push_back(prefix);
      return;
    }
    const std::vector<Sector> outcomes = axis == 1 ? group->fusion_outcomes(legs[0], legs[1])
                                                   : group->fusion_outcomes(reached, legs[axis]);
    for (const Sector& outcome : outcomes) {
      prefix.push_back(outcome);
      extend(axis + 1, outcome);
      prefix.pop_back();
    }
  };
  extend(1, legs[0]);
  return frames;
}

// The couplings a chain carries once one of its legs stops coming from an operand of its own:
// fusing the running total with each following leg in turn, and ending on the output's total.
std::vector<std::vector<Sector>> concat_frames(const std::vector<Sector>& legs,
                                               const Sector& start, const Sector& total,
                                               const std::shared_ptr<FusionCategory>& group) {
  std::vector<std::vector<Sector>> frames;
  std::vector<Sector> prefix;
  std::function<void(std::size_t, const Sector&)> extend = [&](std::size_t index,
                                                              const Sector& current) {
    for (const Sector& fused : group->fusion_outcomes(current, legs[index])) {
      if (index + 1 == legs.size()) {
        if (fused == total) frames.push_back(prefix);
      } else {
        prefix.push_back(fused);
        extend(index + 1, fused);
        prefix.pop_back();
      }
    }
  };
  if (!legs.empty()) extend(0, start);
  return frames;
}

// The coupling an operand's chain carries at one position of its shared prefix.  A prefix that
// runs to the operand's last leg ends on that leg's total charge, which the tree keeps outside
// its inner list.
std::optional<Sector> prefix_coupling(const std::vector<Sector>& inner, const Sector& coupled,
                                      int64_t index) {
  if (index < static_cast<int64_t>(inner.size())) return inner[index];
  if (index == static_cast<int64_t>(inner.size())) return coupled;
  return std::nullopt;
}

// The couplings of a chain at one index, with the last leg's total charge standing in for the
// entry the inner list does not carry.
const Sector& chain_coupling(const std::vector<Sector>& inner, const Sector& coupled, int64_t index) {
  return index < static_cast<int64_t>(inner.size()) ? inner[index] : coupled;
}

// The scalar the junction carries: the loop weight of the shared block, and the recoupling that
// moves the block through the left free chain one vertex at a time, each step one associator.
// Empty when a step's associator is not a scalar.
std::optional<std::complex<double>> junction_scale(
    const Sector& block, const std::vector<Sector>& inner, const Sector& coupled,
    const std::vector<Sector>& frame, const std::vector<Sector>& free_legs, int64_t shared,
    const std::shared_ptr<FusionCategory>& group, const c10::Device& device) {
  // The block that closes the loop also fixes the local frame, and a loop whose pair carries the
  // duality on the left does not fuse back to the trivial sector without the self-duality sign.
  const double loop = 1.0 / std::sqrt(group->quantum_dim(block));
  std::complex<double> product = 1.0;
  Sector previous = free_legs.empty() ? block : free_legs[0];
  for (std::size_t step = 0; step < frame.size() && step + 1 < free_legs.size(); ++step) {
    std::complex<double> one;
    const Sector& after = chain_coupling(inner, coupled, shared + static_cast<int64_t>(step));
    const Sector& before = chain_coupling(inner, coupled, shared - 1 + static_cast<int64_t>(step));
    const at::Tensor symbol =
        group->associator(block, previous, free_legs[step + 1], after, before, frame[step], device);
    // An empty symbol says this fusion channel does not exist, so the frame contributes nothing:
    // the recoupling runs over the frames whose whole chain of associators exists.
    if (symbol.numel() == 0) return std::complex<double>(0.0, 0.0);
    if (!symbol_scalar(symbol, one)) return std::nullopt;
    product *= one;
    previous = frame[step];
  }
  return product * loop;
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
                             std::vector<int64_t>, std::vector<int64_t>, std::vector<int8_t>,
                             std::vector<int64_t>, std::vector<int64_t>, int64_t>;

std::map<WeightKey, std::optional<std::complex<double>>>& weight_cache() {
  static std::map<WeightKey, std::optional<std::complex<double>>> cache;
  return cache;
}

// The number of weights one process keeps before the table is dropped.
constexpr std::size_t kWeightCacheLimit = 1U << 20U;

std::optional<std::complex<double>> compute_diagram_weight(
    const std::shared_ptr<FusionCategory>& group, const std::vector<Sector>& left_legs, const std::vector<Sector>& left_inner,
                    const Sector& left_coupled, const std::vector<Sector>& right_legs,
                    const std::vector<Sector>& right_inner, const Sector& right_coupled,
                    const std::vector<Sector>& out_legs, const std::vector<Sector>& out_inner,
                    const Sector& out_coupled, const std::vector<int64_t>& left_copies,
                    const std::vector<int64_t>& right_copies,
                    const std::vector<int64_t>& out_copies,
                    const std::vector<int64_t>& shared_left,
                    const std::vector<int64_t>& shared_right,
                    const std::vector<int8_t>& shared_dual,
                    const std::vector<int64_t>& out_left_axis,
                    const std::vector<int64_t>& out_right_axis, const c10::Device& device) {
  const int64_t shared = static_cast<int64_t>(shared_left.size());
  g_weight_decline = 0;
  if (!group->categorical_chain_rule()) {
    g_weight_decline = 6;
    return std::nullopt;
  }
  const int64_t left_free = static_cast<int64_t>(left_legs.size()) - shared;
  const int64_t right_free = static_cast<int64_t>(right_legs.size()) - shared;
  // A scalar output reaches this routine either as no legs at all or as a single trivial one,
  // depending on how the caller enumerated its trees; neither carries a coupling tuple.
  const bool scalar_output =
      out_legs.empty() || (out_legs.size() == 1 && out_legs[0] == group->trivial_sector());
  // A leg both operands keep in the output batches instead of gluing, which this routine does
  // not model: its free legs would be counted twice.
  if (scalar_output ? left_free + right_free != 0
                    : static_cast<int64_t>(out_legs.size()) != left_free + right_free) {
    g_weight_decline = 1;
    return std::nullopt;
  }
  const std::vector<int64_t> order_left = moved_order(shared_left, out_left_axis);
  const std::vector<int64_t> order_right = moved_order(shared_right, out_right_axis);
  const auto left_states =
      permute_chain(left_legs, left_inner, left_coupled, order_left, group, device);
  const auto right_states =
      permute_chain(right_legs, right_inner, right_coupled, order_right, group, device);
  if (!left_states.has_value() || !right_states.has_value()) {
    g_weight_decline = 2;
    return std::nullopt;
  }
  std::vector<Sector> moved_left;
  for (int64_t axis : order_left) moved_left.push_back(left_legs[axis]);
  const std::vector<Sector> free_legs(moved_left.begin() + shared, moved_left.end());
  // The joined chain carries the left free legs before the right's while the output lists them
  // in its own order, and a coupling tuple only names a block in one and the same leg order, so
  // every state is moved into the output's order before it is matched.  Each operand reports the
  // output axis of each of its free legs, which places the joined legs.
  std::vector<Sector> joined_legs(free_legs);
  std::vector<int64_t> joined_axis;
  for (int64_t index = shared; index < static_cast<int64_t>(order_left.size()); ++index) {
    joined_axis.push_back(out_left_axis[order_left[index]]);
  }
  for (int64_t index = shared; index < static_cast<int64_t>(order_right.size()); ++index) {
    joined_legs.push_back(right_legs[order_right[index]]);
    joined_axis.push_back(out_right_axis[order_right[index]]);
  }
  std::vector<int64_t> to_output(joined_legs.size(), 0);
  for (size_t index = 0; index < joined_axis.size(); ++index) {
    to_output[joined_axis[index]] = static_cast<int64_t>(index);
  }
  // A glued pair carries the duality matrix on one of its two legs, and each dual leg on the left
  // contributes the self-duality sign of its own charge to the closed loop.
  double duality = 1.0;
  for (size_t index = 0; index < shared_dual.size() && index < shared_left.size(); ++index) {
    if (shared_dual[index] != 0) duality *= group->dual_sign(left_legs[shared_left[index]]);
  }
  std::complex<double> total = 0.0;
  // Nothing is glued, so the contraction is an outer product: its chain is the two operands'
  // chains concatenated through the left operand's own total charge, and the weight is the
  // recoupling that moves that chain into the output's leg order.
  if (shared == 0) {
    std::vector<Sector> right_joined;
    for (int64_t index = shared; index < static_cast<int64_t>(order_right.size()); ++index) {
      right_joined.push_back(right_legs[order_right[index]]);
    }
    std::complex<double> outer = 0.0;
    for (const ChainState& left : *left_states) {
      for (const ChainState& right : *right_states) {
        std::vector<std::vector<Sector>> frames;
        if (right_free == 0) {
          frames.push_back({});
        } else {
          frames = concat_frames(right_joined, left_coupled, out_coupled, group);
        }
        for (const std::vector<Sector>& frame : frames) {
          std::vector<Sector> inner(left.inner);
          if (right_free > 0) inner.push_back(left_coupled);
          inner.insert(inner.end(), frame.begin(), frame.end());
          const auto moved =
              permute_chain(joined_legs, inner, out_coupled, to_output, group, device);
          if (!moved.has_value()) return std::nullopt;
          for (const ChainState& state : *moved) {
            if (state.inner == out_inner) outer += left.value * right.value * state.value;
          }
        }
      }
    }
    return duality * outer;
  }
  for (const ChainState& left : *left_states) {
    const int64_t left_count = static_cast<int64_t>(left.inner.size());
    for (const ChainState& right : *right_states) {
      const int64_t right_count = static_cast<int64_t>(right.inner.size());
      // Every coupling inside the shared block has to agree.
      bool agrees = true;
      for (int64_t index = 0; index < shared - 1 && agrees; ++index) {
        const auto one = prefix_coupling(left.inner, left_coupled, index);
        const auto two = prefix_coupling(right.inner, right_coupled, index);
        agrees = one.has_value() && two.has_value() && *one == *two;
      }
      if (!agrees) continue;
      const Sector& block = shared == 1               ? moved_left[0]
                            : left_free == 0          ? left_coupled
                            : shared - 2 < left_count ? left.inner[shared - 2]
                                                      : left_coupled;
      const int64_t from = std::min(shared - 1, right_count);
      const std::vector<Sector> tail(right.inner.begin() + from, right.inner.end());
      for (const std::vector<Sector>& frame : free_frames(free_legs, block, group)) {
        const auto scale =
            junction_scale(block, left.inner, left_coupled, frame, free_legs, shared,
                           group, device);
        if (!scale.has_value()) {
          g_weight_decline = 3;
          return std::nullopt;
        }
        // A scalar output carries no coupling tuple, so nothing else pins the term down: it is
        // kept exactly when the two operands' totals can fuse to the output's total charge.
        if (scalar_output) {
          for (const Sector& fused : group->fusion_outcomes(block, right_coupled)) {
            if (fused == out_coupled) total += left.value * right.value * *scale;
          }
          continue;
        }
        // The junction coupling sits between the two free parts; a right operand with no free leg
        // leaves the joined chain inside the left's free part, with no room for it.
        const size_t drop = free_legs.size() < 2 ? 2 - free_legs.size() : 0;
        std::vector<Sector> joined(frame);
        if (right_free > 0) joined.push_back(block);
        joined.insert(joined.end(), tail.begin(), tail.end());
        joined.erase(joined.begin(), joined.begin() + std::min(drop, joined.size()));
        // A chain over count legs carries count - 2 couplings; anything else is a shape this
        // routine does not model, so the caller keeps its dense fallback.
        if (joined.size() + 2 != joined_legs.size()) {
          g_weight_decline = 4;
          return std::nullopt;
        }
        const auto moved =
            permute_chain(joined_legs, joined, out_coupled, to_output, group, device);
        if (!moved.has_value()) {
          g_weight_decline = 5;
          return std::nullopt;
        }
        for (const ChainState& state : *moved) {
          if (state.inner == out_inner) total += left.value * right.value * *scale * state.value;
        }
      }
    }
  }
  return duality * total;
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
                    const std::vector<int64_t>& shared_left,
                    const std::vector<int64_t>& shared_right,
                    const std::vector<int8_t>& shared_dual,
                    const std::vector<int64_t>& out_left_axis,
                    const std::vector<int64_t>& out_right_axis, const c10::Device& device,
                    std::complex<double>& weight) {
  const WeightKey key{group.get(),   left_legs,      left_inner,     left_coupled,
                      right_legs,    right_inner,    right_coupled,  out_legs,
                      out_inner,     out_coupled,    left_copies,    right_copies,
                      out_copies,    shared_left,    shared_right,   shared_dual,
                      out_left_axis, out_right_axis,  weight_device_code(device)};
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
                             right_copies, out_copies, shared_left, shared_right, shared_dual,
                             out_left_axis, out_right_axis, device);
  if (cache.size() >= kWeightCacheLimit) cache.clear();
  cache.emplace(key, result);
  if (!result.has_value()) return false;
  weight = *result;
  return true;
}

}  // namespace sym
