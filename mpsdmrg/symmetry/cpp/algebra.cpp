// The symmetry groups of the engine: sector labels, duality, fusion and quantum dimension.
#include "algebra.h"

#include <algorithm>
#include <limits>
#include <map>
#include <utility>


#include "groups/abelian.h"
#include "groups/su2.h"

#include <algorithm>
#include <cstddef>
#include <map>
#include <numeric>
#include <unordered_map>

namespace mpsdmrg {

namespace {

// The sector sizes and values in one plain vector, so a sequence of sectors can be a cache
// key without ambiguity between two of them.
std::vector<int64_t> flatten(const std::vector<Sector>& sectors) {
  std::vector<int64_t> out;
  for (const Sector& sector : sectors) {
    out.push_back(static_cast<int64_t>(sector.size()));
    out.insert(out.end(), sector.begin(), sector.end());
  }
  return out;
}

// Device of a cache key: the type distinguishes CPU from CUDA, the index the device.
int64_t device_code(const c10::Device& device) {
  return static_cast<int64_t>(device.type()) * 1000 + (device.has_index() ? device.index() : -1);
}

struct DenseKey {
  std::string group;
  int64_t device = 0;
  std::vector<int64_t> code;

  bool operator==(const DenseKey& other) const {
    return group == other.group && device == other.device && code == other.code;
  }
};

struct DenseKeyHash {
  std::size_t operator()(const DenseKey& key) const {
    std::size_t h = std::hash<std::string>()(key.group) ^ std::hash<int64_t>()(key.device);
    for (int64_t value : key.code) {
      h = h * 1000003 + std::hash<int64_t>()(value);
    }
    return h;
  }
};

// The expansions are read-only and recur in every sweep, so they are kept by tree and legs,
// and dropped wholesale when the table is full.
std::unordered_map<DenseKey, at::Tensor, DenseKeyHash> stored_dense_cache;
constexpr std::size_t stored_dense_limit = 8192;

// The order of the blocks of a leg by sector: stable, so the layout of equal legs matches.
std::vector<std::size_t> sector_order(const std::vector<Sector>& sectors) {
  std::vector<std::size_t> order(sectors.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&sectors](std::size_t a, std::size_t b) { return sectors[a] < sectors[b]; });
  return order;
}

std::size_t hash_sector(std::size_t seed, const Sector& sector) {
  for (int64_t value : sector) {
    seed = seed * 1000003 + std::hash<int64_t>()(value);
  }
  return seed;
}

}  // namespace

[[noreturn]] void assertion_failed(const std::string& message) {
  PyErr_SetString(PyExc_AssertionError, message.c_str());
  throw py::error_already_set();
}

void clear_stored_dense() { stored_dense_cache.clear(); }

std::string sector_repr(const Sector& sector) {
  std::string out = "(";
  for (std::size_t i = 0; i < sector.size(); ++i) {
    if (i != 0) out += ", ";
    out += std::to_string(sector[i]);
  }
  if (sector.size() == 1) out += ",";
  return out + ")";
}

std::string sectors_repr(const std::vector<Sector>& sectors) {
  std::string out = "(";
  for (std::size_t i = 0; i < sectors.size(); ++i) {
    if (i != 0) out += ", ";
    out += sector_repr(sectors[i]);
  }
  if (sectors.size() == 1) out += ",";
  return out + ")";
}

Leg::Leg(std::shared_ptr<Group> group, std::vector<Sector> sectors, std::vector<int64_t> slices,
         int64_t qconj)
    : group_(std::move(group)), qconj_(qconj) {
  if (!group_) {
    assertion_failed("A leg needs a group.");
  }
  if (qconj_ != 1 && qconj_ != -1) {
    assertion_failed("The charge direction of a leg is +1 or -1, not " + std::to_string(qconj_) +
                     ".");
  }
  if (slices.size() != sectors.size() + 1 || slices.front() != 0) {
    assertion_failed("The block boundaries of a leg start at zero and there is one more of them "
                     "than there are sectors.");
  }
  for (std::size_t i = 1; i < slices.size(); ++i) {
    if (slices[i] <= slices[i - 1]) {
      assertion_failed("The block boundaries of a leg increase strictly.");
    }
  }
  for (const Sector& sector : sectors) {
    if (!group_->is_valid_sector(sector)) {
      assertion_failed("A leg carries the invalid sector " + sector_repr(sector) + ".");
    }
  }
  const std::vector<std::size_t> order = sector_order(sectors);
  for (std::size_t i = 1; i < order.size(); ++i) {
    if (sectors[order[i - 1]] == sectors[order[i]]) {
      assertion_failed("The sector " + sector_repr(sectors[order[i]]) +
                       " appears on more than one block of a leg.");
    }
  }
  sectors_.reserve(order.size());
  slices_.assign(1, 0);
  for (std::size_t position : order) {
    sectors_.push_back(sectors[position]);
    slices_.push_back(slices_.back() + slices[position + 1] - slices[position]);
  }
}

Leg Leg::trivial(int64_t dim, std::shared_ptr<Group> group, int64_t qconj) {
  if (!group) {
    group = no_symmetry();
  }
  const Sector trivial = group->trivial_sector();
  return Leg(std::move(group), {trivial}, {0, dim}, qconj);
}

std::pair<Leg, std::vector<int64_t>> Leg::from_blocks(
    std::shared_ptr<Group> group, const std::vector<std::pair<Sector, int64_t>>& entries,
    int64_t qconj) {
  std::vector<Sector> given;
  given.reserve(entries.size());
  for (const auto& [sector, dim] : entries) {
    given.push_back(sector);
  }
  const std::vector<std::size_t> order = sector_order(given);
  std::vector<int64_t> rank(order.size());
  for (std::size_t position = 0; position < order.size(); ++position) {
    rank[order[position]] = static_cast<int64_t>(position);
  }
  std::vector<Sector> sectors;
  std::vector<int64_t> slices{0};
  sectors.reserve(order.size());
  for (std::size_t position : order) {
    sectors.push_back(entries[position].first);
    slices.push_back(slices.back() + entries[position].second);
  }
  return {Leg(std::move(group), std::move(sectors), std::move(slices), qconj), std::move(rank)};
}

int64_t Leg::block_dim(int64_t qi) const {
  const std::size_t index = static_cast<std::size_t>(qi);
  return slices_.at(index + 1) - slices_.at(index);
}

int64_t Leg::multiplicity(int64_t qi) const {
  return block_dim(qi) / group_->sector_dim(sectors_.at(static_cast<std::size_t>(qi)));
}

Sector Leg::sector(int64_t qi, bool dual) const {
  const Sector& stored = sectors_.at(static_cast<std::size_t>(qi));
  return dual && qconj_ == -1 ? group_->dual_sector(stored) : stored;
}

Leg Leg::fuse(const Leg& other) const {
  if (!same_group(*group_, *other.group_)) {
    assertion_failed("Cannot fuse legs of groups " + group_->group_name() + " and " +
                     other.group_->group_name() + ".");
  }
  std::map<Sector, int64_t> counts;
  for (std::size_t i = 0; i < sectors_.size(); ++i) {
    const int64_t left = multiplicity(static_cast<int64_t>(i));
    for (std::size_t j = 0; j < other.sectors_.size(); ++j) {
      const int64_t right = other.multiplicity(static_cast<int64_t>(j));
      for (const Sector& outcome : group_->fusion_outcomes(sectors_[i], other.sectors_[j])) {
        counts[outcome] += left * right * group_->n_symbol(sectors_[i], other.sectors_[j], outcome);
      }
    }
  }
  std::vector<std::pair<Sector, int64_t>> entries;
  entries.reserve(counts.size());
  for (const auto& [outcome, copies] : counts) {
    entries.emplace_back(outcome, group_->sector_dim(outcome) * copies);
  }
  return from_blocks(group_, entries, 1).first;
}

Leg Leg::conj() const { return Leg(group_, sectors_, slices_, -qconj_); }

std::string Leg::repr() const {
  return "Leg(dim=" + std::to_string(dim()) + ", qconj=" + std::to_string(qconj_) +
         ", blocks=" + std::to_string(nblocks()) + ", sectors=" + sectors_repr(sectors_) + ")";
}

bool Leg::operator==(const Leg& other) const {
  return same_group(*group_, *other.group_) && sectors_ == other.sectors_ &&
         slices_ == other.slices_ && qconj_ == other.qconj_;
}

std::size_t Leg::hash() const {
  std::size_t seed = std::hash<std::string>()(group_->group_name()) ^
                     std::hash<std::string>()(group_->type_name());
  for (const Sector& sector : sectors_) {
    seed = hash_sector(seed, sector);
  }
  for (int64_t value : slices_) {
    seed = seed * 1000003 + std::hash<int64_t>()(value);
  }
  return seed * 1000003 + std::hash<int64_t>()(qconj_);
}

FusionTree::FusionTree(std::shared_ptr<Group> group, std::vector<Sector> sectors,
                       std::vector<Sector> inner, Sector coupled)
    : group_(std::move(group)), sectors_(std::move(sectors)), inner_(std::move(inner)),
      coupled_(std::move(coupled)) {
  if (!group_) {
    assertion_failed("A fusion tree needs a group.");
  }
  if (sectors_.empty()) {
    assertion_failed("A fusion tree needs at least one leg.");
  }
  if (inner_.size() != (sectors_.size() > 2 ? sectors_.size() - 2 : 0)) {
    assertion_failed(std::to_string(inner_.size()) + " inner couplings cannot fuse " +
                     std::to_string(sectors_.size()) + " legs.");
  }
  const std::vector<Sector> couplings = chain();
  for (std::size_t k = 1; k < sectors_.size(); ++k) {
    if (!group_->can_fuse_to(couplings[k - 1], sectors_[k], couplings[k])) {
      assertion_failed("Leg " + std::to_string(k) + " of " + sectors_repr(sectors_) +
                       " does not fuse into the coupling " + sector_repr(couplings[k]) + ".");
    }
  }
}

FusionTree FusionTree::from_blocks(const std::shared_ptr<Group>& group,
                                   const std::vector<Leg>& legs, const BlockKey& key,
                                   const Sector& qtotal) {
  if (legs.empty()) {
    return FusionTree(group, {qtotal}, {}, qtotal);
  }
  std::vector<Sector> sectors;
  sectors.reserve(legs.size());
  for (std::size_t i = 0; i < legs.size(); ++i) {
    sectors.push_back(legs[i].sector(key.legs.at(i), true));
  }
  return FusionTree(group, std::move(sectors), key.inner, qtotal);
}

std::vector<Sector> FusionTree::chain() const {
  if (sectors_.size() == 1) {
    return {coupled_};
  }
  std::vector<Sector> out;
  out.reserve(sectors_.size() + 1);
  out.push_back(sectors_.front());
  out.insert(out.end(), inner_.begin(), inner_.end());
  out.push_back(coupled_);
  return out;
}

at::Tensor FusionTree::to_dense_block(const c10::Device& device) const {
  const auto options = at::TensorOptions().dtype(group_->fusion_dtype()).device(device);
  const std::vector<Sector> couplings = chain();
  at::Tensor out = at::eye(group_->sector_dim(sectors_.front()), options);
  // the expansion is the contraction of the vertex of every visited coupling, its last axis
  // the running magnetic index of the coupling chain
  for (std::size_t i = 1; i < sectors_.size(); ++i) {
    const at::Tensor table = group_->fusion_tensor(couplings[i - 1], sectors_[i], couplings[i],
                                                   device);
    const int64_t m = out.size(-1);
    const int64_t a = table.size(1);
    const int64_t b = table.size(2);
    std::vector<int64_t> shape(out.sizes().begin(), out.sizes().end() - 1);
    shape.push_back(a);
    shape.push_back(b);
    out = at::matmul(out.reshape({-1, m}), table.reshape({m, a * b})).reshape(shape);
  }
  return out;
}

std::string FusionTree::repr() const {
  return "FusionTree(sectors=" + sectors_repr(sectors_) + ", inner=" + sectors_repr(inner_) +
         ", coupled=" + sector_repr(coupled_) + ")";
}

bool FusionTree::operator==(const FusionTree& other) const {
  return same_group(*group_, *other.group_) && sectors_ == other.sectors_ &&
         inner_ == other.inner_ && coupled_ == other.coupled_;
}

std::size_t FusionTree::hash() const {
  std::size_t seed = std::hash<std::string>()(group_->group_name()) ^
                     std::hash<std::string>()(group_->type_name());
  for (const Sector& sector : sectors_) {
    seed = hash_sector(seed, sector);
  }
  for (const Sector& sector : inner_) {
    seed = hash_sector(seed, sector);
  }
  return hash_sector(seed, coupled_);
}

std::vector<FusionTree> fusion_trees(const std::shared_ptr<Group>& group,
                                     const std::vector<Sector>& sectors, const Sector& coupled) {
  if (sectors.empty()) {
    return coupled == group->trivial_sector()
               ? std::vector<FusionTree>{FusionTree(group, {group->trivial_sector()}, {}, coupled)}
               : std::vector<FusionTree>{};
  }
  if (sectors.size() == 1) {
    return sectors.front() == coupled
               ? std::vector<FusionTree>{FusionTree(group, sectors, {}, coupled)}
               : std::vector<FusionTree>{};
  }
  std::vector<std::pair<Sector, std::vector<Sector>>> states{{sectors.front(), {}}};
  for (std::size_t k = 1; k < sectors.size(); ++k) {
    const bool last = k + 1 == sectors.size();
    std::vector<std::pair<Sector, std::vector<Sector>>> following;
    for (const auto& [current, inner] : states) {
      for (const Sector& outcome : group->fusion_outcomes(current, sectors[k])) {
        if (last) {
          if (outcome == coupled) {
            following.emplace_back(outcome, inner);
          }
        } else {
          std::vector<Sector> grown(inner);
          grown.push_back(outcome);
          following.emplace_back(outcome, std::move(grown));
        }
      }
    }
    states = std::move(following);
  }
  std::vector<FusionTree> out;
  out.reserve(states.size());
  for (const auto& [fused, inner] : states) {
    out.emplace_back(group, sectors, inner, coupled);
  }
  return out;
}

double dual_sign(const Group& group, const Sector& sector) {
  return group.dual_sector(sector) == sector && (group.sector_dim(sector) - 1) % 2 != 0 ? -1.0
                                                                                        : 1.0;
}

at::Tensor stored_dense(const std::vector<Leg>& legs, const FusionTree& tree,
                        const c10::Device& device) {
  DenseKey key;
  key.group = tree.group()->type_name() + "|" + tree.group()->group_name();
  key.device = device_code(device);
  for (const Leg& leg : legs) {
    key.code.push_back(leg.qconj());
  }
  key.code.push_back(-1);
  const std::vector<int64_t> sector_code = flatten(tree.sectors());
  key.code.insert(key.code.end(), sector_code.begin(), sector_code.end());
  key.code.push_back(-1);
  const std::vector<int64_t> inner_code = flatten(tree.inner());
  key.code.insert(key.code.end(), inner_code.begin(), inner_code.end());
  key.code.push_back(-1);
  const std::vector<int64_t> coupled_code = flatten({tree.coupled()});
  key.code.insert(key.code.end(), coupled_code.begin(), coupled_code.end());

  const auto found = stored_dense_cache.find(key);
  if (found != stored_dense_cache.end()) {
    return found->second;
  }
  at::Tensor out = tree.to_dense_block(device);
  // a leg that carries duals stores the magnetic index flipped and carries the phase (-1)^m
  for (std::size_t i = 0; i < legs.size(); ++i) {
    if (legs[i].qconj() != -1) {
      continue;
    }
    const int64_t axis = static_cast<int64_t>(i);
    const at::Tensor matrix = legs[i].group()->duality_matrix(tree.sectors()[i], device);
    out = at::tensordot(out, matrix, {axis}, {1}).movedim(-1, axis);
  }
  if (stored_dense_cache.size() >= stored_dense_limit) {
    stored_dense_cache.clear();
  }
  stored_dense_cache.emplace(std::move(key), out);
  return out;
}

}  // namespace mpsdmrg

