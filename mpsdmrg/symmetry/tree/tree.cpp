// The charge-adapted legs and fusion trees of the engine: the block structure of a leg, the
// tree that fixes how its sectors fuse, and the dense Clebsch-Gordan expansion of a tree.
#include "tree.h"

#include <algorithm>
#include <cstddef>
#include <map>
#include <numeric>
#include <unordered_map>
#include <utility>

namespace sym {

namespace {

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

Leg::Leg(std::shared_ptr<FusionCategory> group, std::vector<Sector> sectors, std::vector<int64_t> slices,
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

Leg Leg::trivial(int64_t dim, std::shared_ptr<FusionCategory> group, int64_t qconj) {
  if (!group) {
    group = no_symmetry();
  }
  const Sector trivial = group->trivial_sector();
  return Leg(std::move(group), {trivial}, {0, dim}, qconj);
}

std::pair<Leg, std::vector<int64_t>> Leg::from_blocks(
    std::shared_ptr<FusionCategory> group, const std::vector<std::pair<Sector, int64_t>>& entries,
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

int64_t Leg::multiplets() const {
  // Every multiplet contributes one copy; its magnetic index is carried by the vertex.
  int64_t total = 0;
  for (int64_t qi = 0; qi < nblocks(); ++qi) total += multiplicity(qi);
  return total;
}

Sector Leg::sector(int64_t qi, bool dual) const {
  const Sector& stored = sectors_.at(static_cast<std::size_t>(qi));
  return dual && qconj_ == -1 ? group_->dual_sector(stored) : stored;
}

Leg Leg::fuse(const Leg& other) const {
  if (!same_group(*group_, *other.group_)) {
    assertion_failed("Cannot fuse legs of groups " + group_->name() + " and " +
                     other.group_->name() + ".");
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
  std::size_t seed = std::hash<std::string>()(group_->name()) ^
                     std::hash<std::string>()(group_->type_name());
  for (const Sector& sector : sectors_) {
    seed = hash_sector(seed, sector);
  }
  for (int64_t value : slices_) {
    seed = seed * 1000003 + std::hash<int64_t>()(value);
  }
  return seed * 1000003 + std::hash<int64_t>()(qconj_);
}

FusionTree::FusionTree(std::shared_ptr<FusionCategory> group, std::vector<Sector> sectors,
                       std::vector<Sector> inner, Sector coupled, std::vector<int64_t> copies)
    : group_(std::move(group)), sectors_(std::move(sectors)), inner_(std::move(inner)),
      coupled_(std::move(coupled)), copies_(std::move(copies)) {
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
  // every vertex of the tree realizes one of its couplings, the first one by default
  if (copies_.empty()) {
    copies_.assign(sectors_.size() - 1, 0);
  }
  if (copies_.size() != sectors_.size() - 1) {
    assertion_failed(std::to_string(copies_.size()) + " copies cannot label " +
                     std::to_string(sectors_.size()) + " legs.");
  }
  for (std::size_t k = 1; k < sectors_.size(); ++k) {
    const int64_t multiplicity = group_->n_symbol(couplings[k - 1], sectors_[k], couplings[k]);
    if (copies_[k - 1] < 0 || copies_[k - 1] >= multiplicity) {
      assertion_failed("The fusion of " + sector_repr(couplings[k - 1]) + " with " +
                       sector_repr(sectors_[k]) + " to " + sector_repr(couplings[k]) + " has " +
                       std::to_string(multiplicity) + " copies, not " +
                       std::to_string(copies_[k - 1]) + ".");
    }
  }
}

FusionTree FusionTree::from_blocks(const std::shared_ptr<FusionCategory>& group,
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
  return FusionTree(group, std::move(sectors), key.inner, qtotal, key.copies);
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

std::string FusionTree::repr() const {
  const std::string base = "FusionTree(sectors=" + sectors_repr(sectors_) + ", inner=" +
                           sectors_repr(inner_) + ", coupled=" + sector_repr(coupled_);
  return group_->carries_multiplicity() ? base + ", copies=" + std::to_string(copies_.size()) + ")"
                                        : base + ")";
}

bool FusionTree::operator==(const FusionTree& other) const {
  return same_group(*group_, *other.group_) && sectors_ == other.sectors_ &&
         inner_ == other.inner_ && coupled_ == other.coupled_ && copies_ == other.copies_;
}

std::size_t FusionTree::hash() const {
  std::size_t seed = std::hash<std::string>()(group_->name()) ^
                     std::hash<std::string>()(group_->type_name());
  for (const Sector& sector : sectors_) {
    seed = hash_sector(seed, sector);
  }
  for (const Sector& sector : inner_) {
    seed = hash_sector(seed, sector);
  }
  for (int64_t copy : copies_) {
    seed = seed * 1000003 + std::hash<int64_t>()(copy);
  }
  return hash_sector(seed, coupled_);
}

std::vector<FusionTree> fusion_trees(const std::shared_ptr<FusionCategory>& group,
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
  // one path per combination of the couplings and of the copy every vertex realizes
  struct State {
    Sector current;
    std::vector<Sector> inner;
    std::vector<int64_t> copies;
  };
  std::vector<State> states{{sectors.front(), {}, {}}};
  for (std::size_t k = 1; k < sectors.size(); ++k) {
    const bool last = k + 1 == sectors.size();
    std::vector<State> following;
    for (const State& state : states) {
      for (const Sector& outcome : group->fusion_outcomes(state.current, sectors[k])) {
        if (last && outcome != coupled) {
          continue;
        }
        const int64_t multiplicity = group->n_symbol(state.current, sectors[k], outcome);
        for (int64_t copy = 0; copy < multiplicity; ++copy) {
          State grown;
          grown.current = outcome;
          grown.inner = state.inner;
          if (!last) {
            grown.inner.push_back(outcome);
          }
          grown.copies = state.copies;
          grown.copies.push_back(copy);
          following.push_back(std::move(grown));
        }
      }
    }
    states = std::move(following);
  }
  std::vector<FusionTree> out;
  out.reserve(states.size());
  for (const State& state : states) {
    out.emplace_back(group, sectors, state.inner, coupled, state.copies);
  }
  return out;
}

Leg site_leg(const std::shared_ptr<FusionCategory>& group, int64_t dim, int64_t qconj) {
  // an abelian group has one charge, so a leg of dimension d is d copies of it; a non-abelian
  // group names a multiplet, which its dimension alone does not pick out
  if (!group->is_abelian()) {
    throw value_error("A non-abelian group does not name a leg by a dimension; name the multiplet "
                      "instead, e.g. multiplet_leg(group, (1, 0)).");
  }
  return Leg::trivial(dim, group, qconj);
}

Leg multiplet_leg(const std::shared_ptr<FusionCategory>& group, const Sector& sector, int64_t qconj) {
  // one copy of one multiplet, whose magnetic indices are carried by the vertex
  return Leg(group, {sector}, {0, group->sector_dim(sector)}, qconj);
}

Leg bond_leg(const std::shared_ptr<FusionCategory>& group, std::vector<std::pair<Sector, int64_t>> entries,
             int64_t qconj) {
  return Leg::from_blocks(group, std::move(entries), qconj).first;
}

Leg fuse_legs(const Leg& first, const Leg& second) { return first.fuse(second); }

Leg capped_leg(const Leg& leg, int64_t max_multiplets) {
  if (leg.multiplets() <= max_multiplets) return leg;
  const std::shared_ptr<FusionCategory>& group = leg.group();
  std::vector<std::pair<Sector, int64_t>> entries;
  int64_t total = 0;
  for (int64_t qi = 0; qi < leg.nblocks(); ++qi) {
    const Sector& sector = leg.sectors().at(qi);
    const int64_t keep = std::min(leg.multiplicity(qi), max_multiplets - total);
    if (keep > 0) {
      entries.emplace_back(sector, keep * group->sector_dim(sector));
      total += keep;
    }
  }
  return Leg::from_blocks(group, std::move(entries), leg.qconj()).first;
}

void bind_tree(py::module_& m) {
    m.def("site_leg", &sym::site_leg, py::arg("group"), py::arg("dim"), py::arg("qconj") = 1);
    m.def("multiplet_leg", &sym::multiplet_leg, py::arg("group"), py::arg("sector"),
          py::arg("qconj") = 1);
    m.def("bond_leg", &sym::bond_leg, py::arg("group"), py::arg("entries"), py::arg("qconj") = 1);
    m.def("fuse_legs", &sym::fuse_legs, py::arg("first"), py::arg("second"));
    m.def("capped_leg", &sym::capped_leg, py::arg("leg"), py::arg("max_multiplets"));
    py::class_<sym::Leg, std::shared_ptr<sym::Leg>>(m, "Leg")
        .def(py::init<std::shared_ptr<sym::FusionCategory>, std::vector<sym::Sector>,
                      std::vector<int64_t>, int64_t>(),
             py::arg("group"), py::arg("sectors"), py::arg("slices"), py::arg("qconj") = 1)
        .def_static("trivial", &sym::Leg::trivial, py::arg("dim"), py::arg("group") = nullptr,
                    py::arg("qconj") = 1)
        .def_static(
            "from_blocks",
            [](std::shared_ptr<sym::FusionCategory> group,
               const std::vector<std::pair<sym::Sector, int64_t>>& entries, int64_t qconj) {
              auto [leg, rank] = sym::Leg::from_blocks(std::move(group), entries, qconj);
              return py::make_tuple(py::cast(leg), py::tuple(py::cast(rank)));
            },
            py::arg("group"), py::arg("entries"), py::arg("qconj") = 1)
        .def_property_readonly("group", &sym::Leg::group)
        .def_property_readonly("sectors", [](const sym::Leg& leg) {
          return sym::to_python(leg.sectors());
        })
        .def_property_readonly("slices", [](const sym::Leg& leg) {
          return py::tuple(py::cast(leg.slices()));
        })
        .def_property_readonly("qconj", &sym::Leg::qconj)
        .def_property_readonly("dim", &sym::Leg::dim)
        .def_property_readonly("nblocks", &sym::Leg::nblocks)
        .def("sector",
             [](const sym::Leg& leg, int64_t qi, bool dual) {
               return sym::to_python(leg.sector(qi, dual));
             },
             py::arg("qi"), py::arg("dual") = false)
        .def("multiplicity", &sym::Leg::multiplicity, py::arg("qi"))
        .def_property_readonly("multiplets", &sym::Leg::multiplets)
        .def("fuse", &sym::Leg::fuse, py::arg("other"))
        .def("conj", &sym::Leg::conj)
        .def("__eq__",
             [](const sym::Leg& leg, py::object other) -> py::object {
               if (!py::isinstance<sym::Leg>(other)) {
                 return py::reinterpret_borrow<py::object>(Py_NotImplemented);
               }
               return py::cast(leg == other.cast<const sym::Leg&>());
             })
        .def("__hash__", &sym::Leg::hash)
        .def("__repr__", &sym::Leg::repr)
        .def("__reduce__", [](const sym::Leg& self) {
          return py::make_tuple(py::type::of<sym::Leg>(),
                                py::make_tuple(self.group(), sym::to_python(self.sectors()),
                                               py::cast(self.slices()), self.qconj()));
        }, "the constructor arguments, so that a leg survives a pickle round trip");

}

}  // namespace sym
