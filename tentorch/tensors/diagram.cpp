// The tree moves of the engine: bend, fold, cycle, braid and the pair permutation.

#include "../symmetry/cat/symbol.h"
#include "diagram.h"

#include <pybind11/complex.h>
#include <pybind11/stl.h>

#include <pybind11/complex.h>
#include <torch/extension.h>


#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sym::engine {
namespace {

double sqrtdim(const FusionCategory& group, const Sector& sector) {
  return std::sqrt(group.quantum_dim(sector));
}

// The sector a leg points at: a leg that carries duals stores the conjugate of the sector its line
// fuses with, and the symbols of the moves are written in the fused sectors.
Sector pointing(const FusionCategory& group, const FusionTree& tree, int64_t axis) {
  return tree.isdual.at(axis) ? group.dual_sector(tree.sectors.at(axis)) : tree.sectors.at(axis);
}

std::complex<double> entry(const at::Tensor& matrix, int64_t row, int64_t column) {
  const c10::complex<double> value =
      matrix.to(at::kCPU).select(0, row).select(0, column).item<c10::complex<double>>();
  return {value.real(), value.imag()};
}

}  // namespace

std::vector<PairTerm> bend_right(const FusionCategory& group, const FusionTreePair& pair,
                                 const Sector& qtotal, const c10::Device& device) {
  const FusionTree& out = pair.codomain;
  const FusionTree& in = pair.domain;
  const int64_t n1 = out.size();
  if (n1 < 1) throw std::invalid_argument("bend_right: the codomain side has no leg to bend");
  const Sector& c = out.coupled;
  const Sector& stored = out.sectors.back();
  const Sector b = pointing(group, out, n1 - 1);
  // The new root of the shortened side.
  // A side of one leg carries no vertex, so the root of the shortened side is the unit and the
  // symbol's first axis is a single entry.
  const Sector a = n1 == 1 ? group.trivial_sector()
                           : (n1 == 2 ? pointing(group, out, 0) : out.inner.back());
  const int64_t row = n1 >= 2 ? out.vertices.back() : 0;

  // The codomain side loses its last leg, so its tree is the prefix that used to fuse with `b` and
  // couples to `a`; the vertex the domain side gains carries the copy label of the B symbol.
  FusionTree new_out = out;
  if (n1 >= 1) new_out.sectors.pop_back();
  if (n1 >= 1) new_out.isdual.pop_back();
  if (n1 >= 3) new_out.inner.pop_back();
  if (n1 >= 2) new_out.vertices.pop_back();
  new_out.coupled = a;

  // The domain side gains that leg as a bra, so it keeps the sector it stores and points the other
  // way; its own old coupling becomes the new inner coupling, and its root is the sector that
  // closes `qtotal` with the root the codomain side kept.
  FusionTree new_in = in;
  new_in.sectors.push_back(stored);
  new_in.isdual.push_back(!out.isdual.back());
  if (in.size() > 1) new_in.inner.push_back(in.coupled);
  if (in.size() > 0) new_in.vertices.push_back(row);
  new_in.coupled = partner_root(group, a, qtotal);

  std::complex<double> coefficient = sqrtdim(group, c) / sqrtdim(group, a);
  if (out.isdual.back()) {
    coefficient *= std::conj(frobenius_schur_phase(group, group.dual_sector(stored), device));
  }
  const at::Tensor symbol = b_symbol(group, a, b, c, device);
  const int64_t copies = symbol.size(1);
  std::vector<PairTerm> terms;
  for (int64_t column = 0; column < copies; ++column) {
    // The copy label of the new vertex belongs to the vertex the domain side gains, not to the one
    // the codomain side loses.
    FusionTree branch = new_in;
    if (in.size() > 0) branch.vertices.back() = column;
    terms.push_back(PairTerm{FusionTreePair{new_out, branch}, coefficient * entry(symbol, row, column)});
  }
  return terms;
}

std::vector<PairTerm> bend_left(const FusionCategory& group, const FusionTreePair& pair,
                                const Sector& qtotal, const c10::Device& device) {
  const FusionTree& out = pair.codomain;
  const FusionTree& in = pair.domain;
  const int64_t n2 = in.size();
  if (n2 < 1) throw std::invalid_argument("bend_left: the domain side has no leg to bend");
  const Sector& c = in.coupled;
  const Sector& stored = in.sectors.back();
  const Sector b = pointing(group, in, n2 - 1);
  // A one-leg side carries no vertex, so the root of the shortened side is the unit: the symbol
  // then forces the old coupling to be that leg's sector.
  const Sector a = n2 == 1 ? group.trivial_sector()
                           : (n2 == 2 ? pointing(group, in, 0) : in.inner.back());
  const int64_t row = n2 >= 2 ? in.vertices.back() : 0;

  FusionTree new_in = in;
  new_in.sectors.pop_back();
  new_in.isdual.pop_back();
  if (n2 >= 3) new_in.inner.pop_back();
  if (n2 >= 2) new_in.vertices.pop_back();
  new_in.coupled = a;

  // The codomain side gains that leg as a ket, so it keeps the sector it stores and points the
  // other way; its own old coupling becomes the new inner coupling, and its root is the sector that
  // closes `qtotal` with the root the domain side kept.
  FusionTree new_out = out;
  new_out.sectors.push_back(stored);
  new_out.isdual.push_back(!in.isdual.back());
  if (out.size() > 1) new_out.inner.push_back(out.coupled);
  if (out.size() > 0) new_out.vertices.push_back(row);
  new_out.coupled = partner_root(group, a, qtotal);

  std::complex<double> coefficient = std::conj(sqrtdim(group, c) / sqrtdim(group, a));
  if (in.isdual.back()) {
    coefficient *= frobenius_schur_phase(group, group.dual_sector(b), device);
  }
  const at::Tensor symbol = a_symbol(group, a, b, c, device);
  const int64_t copies = symbol.size(1);
  std::vector<PairTerm> terms;
  for (int64_t column = 0; column < copies; ++column) {
    FusionTree branch = new_out;
    if (out.size() > 0) branch.vertices.back() = column;
    terms.push_back(PairTerm{FusionTreePair{branch, new_in}, coefficient * entry(symbol, row, column)});
  }
  return terms;
}

// The sector a tree couples to on the prefix ending at leg `k`, and the copy it realises there.
// `k` counts legs from one, and has to be at least two, so that the prefix has two legs to fuse.
namespace {

struct Vertex {
  Sector left;
  Sector right;
  int64_t copy;
};

Vertex vertex_at(const FusionTree& tree, int64_t k) {
  const int64_t n = tree.size();
  if (k < 2 || k > n) throw std::out_of_range("vertex_at: no vertex fuses leg " + std::to_string(k));
  return Vertex{k == 2 ? tree.sectors.at(0) : tree.inner.at(k - 3),
                k == n ? tree.coupled : tree.inner.at(k - 2), tree.vertices.at(k - 2)};
}

// The single entry of a one-dimensional symbol, which is all a move needs where every fusion is
// unique.
std::complex<double> scalar_of(const at::Tensor& symbol, const char* where) {
  if (symbol.numel() == 0) return {0.0, 0.0};
  if (symbol.numel() != 1) {
    throw std::invalid_argument(std::string(where) +
                                ": the move needs one copy at every vertex, but this category "
                                "couples the sectors in more than one way");
  }
  const c10::complex<double> value = symbol.reshape({-1}).select(0, 0).item<c10::complex<double>>();
  return {value.real(), value.imag()};
}

std::vector<LegCharge> charges_of(const FusionTree& tree) {
  std::vector<LegCharge> charges;
  charges.reserve(tree.sectors.size());
  for (size_t axis = 0; axis < tree.sectors.size(); ++axis)
    charges.push_back(LegCharge{tree.sectors[axis], tree.isdual[axis]});
  return charges;
}

}  // namespace

std::complex<double> multi_associator(const FusionCategory& group, const FusionTree& longer,
                                      const FusionTree& shorter, const c10::Device& device) {
  const int64_t n = longer.size();
  if (shorter.size() != n - 1) return 0.0;
  for (int64_t axis = 0; axis + 1 < n; ++axis) {
    if (longer.sectors.at(axis + 1) != shorter.sectors.at(axis)) return 0.0;
    if (longer.isdual.at(axis + 1) != shorter.isdual.at(axis)) return 0.0;
  }
  const Sector a = longer.sectors.at(0);
  std::complex<double> coefficient = 1.0;
  for (int64_t k = 2; k < n; ++k) {
    const Vertex longer_vertex = vertex_at(longer, k + 1);
    const Vertex shorter_vertex = vertex_at(shorter, k);
    // The associator of the pair fuses the third leg of the longer tree, one further right than the
    // leg the inverse move fuses.
    const at::Tensor symbol = group.associator(a, shorter_vertex.left, longer.sectors.at(k),
                                               longer_vertex.right, longer_vertex.left,
                                               shorter_vertex.right, device);
    coefficient *= scalar_of(symbol, "multi_associator");
    if (coefficient == std::complex<double>(0.0, 0.0)) return coefficient;
  }
  return coefficient;
}

std::vector<TreeTerm> multi_fmove(const FusionCategory& group, const FusionTree& tree,
                                  const c10::Device& device) {
  const int64_t n = tree.size();
  if (n < 1) throw std::invalid_argument("multi_fmove: the tree has no leg to split off");
  std::vector<TreeTerm> terms;
  if (n == 1) {
    FusionTree target;
    target.coupled = group.trivial_sector();
    terms.push_back(TreeTerm{target, 1.0});
    return terms;
  }
  FusionTree tail;
  tail.sectors.assign(tree.sectors.begin() + 1, tree.sectors.end());
  tail.isdual.assign(tree.isdual.begin() + 1, tree.isdual.end());
  tail.coupled = tree.sectors.at(1);
  const Sector a = tree.sectors.at(0);
  const Sector c = tree.coupled;
  const std::map<Sector, std::vector<FusionTree>> candidates =
      trees_by_charge(group, charges_of(tail));
  for (const auto& [charge, trees] : candidates) {
    const std::vector<Sector> outcomes = group.fusion_outcomes(a, charge);
    if (std::find(outcomes.begin(), outcomes.end(), c) == outcomes.end()) continue;
    for (const FusionTree& candidate : trees) {
      const std::complex<double> coefficient = multi_associator(group, tree, candidate, device);
      if (coefficient == std::complex<double>(0.0, 0.0)) continue;
      terms.push_back(TreeTerm{candidate, coefficient});
    }
  }
  return terms;
}

std::vector<TreeTerm> multi_fmove_inv(const FusionCategory& group, const Sector& a, const Sector& c,
                                      const FusionTree& tree, bool isduala,
                                      const c10::Device& device) {
  const Sector b = tree.coupled;
  const std::vector<Sector> outcomes = group.fusion_outcomes(a, b);
  if (std::find(outcomes.begin(), outcomes.end(), c) == outcomes.end()) return {};
  FusionTree head;
  head.sectors.reserve(tree.size() + 1);
  head.sectors.push_back(a);
  head.sectors.insert(head.sectors.end(), tree.sectors.begin(), tree.sectors.end());
  head.isdual.reserve(tree.size() + 1);
  head.isdual.push_back(isduala);
  head.isdual.insert(head.isdual.end(), tree.isdual.begin(), tree.isdual.end());
  head.coupled = c;
  std::vector<TreeTerm> terms;
  if (tree.size() == 0) {
    terms.push_back(TreeTerm{head, 1.0});
    return terms;
  }
  const std::map<Sector, std::vector<FusionTree>> candidates =
      trees_by_charge(group, charges_of(head));
  const auto found = candidates.find(c);
  if (found == candidates.end()) return {};
  for (const FusionTree& candidate : found->second) {
    {
      std::complex<double> coefficient = 1.0;
      const int64_t n = tree.size();
      for (int64_t k = 2; k <= n; ++k) {
        const Vertex source = vertex_at(tree, k);
        const Vertex target = vertex_at(candidate, k + 1);
        const at::Tensor symbol = group.associator(a, source.left, tree.sectors.at(k - 1),
                                                   target.right, target.left, source.right, device);
        coefficient *= std::conj(scalar_of(symbol, "multi_fmove_inv"));
        if (coefficient == std::complex<double>(0.0, 0.0)) break;
      }
      if (coefficient == std::complex<double>(0.0, 0.0)) continue;
      terms.push_back(TreeTerm{candidate, coefficient});
    }
  }
  return terms;
}

std::vector<PairTerm> repartition(const FusionCategory& group, const FusionTreePair& pair,
                                  int64_t keep, const Sector& qtotal, const c10::Device& device) {
  const int64_t total = pair.codomain.size() + pair.domain.size();
  if (keep < 0 || keep > total) {
    throw std::out_of_range("repartition: a tree pair of " + std::to_string(total) +
                            " legs cannot carry " + std::to_string(keep) + " of them on one side");
  }
  std::vector<PairTerm> current{PairTerm{pair, 1.0}};
  // A bend that reaches no tree has nowhere to go; reporting it keeps a silent empty product from
  // turning into a crash further up.
  const auto require_terms = [](const std::vector<PairTerm>& terms, const char* side) {
    if (terms.empty()) {
      throw std::invalid_argument(std::string("repartition: no tree carries the ") + side +
                                  " legs after bending one leg off");
    }
  };
  while (static_cast<int64_t>(current.front().pair.codomain.size()) > keep) {
    std::vector<PairTerm> grown;
    for (const PairTerm& term : current)
      for (const PairTerm& next : bend_right(group, term.pair, qtotal, device))
        grown.push_back(PairTerm{next.pair, term.coefficient * next.coefficient});
    require_terms(grown, "codomain");
    current = std::move(grown);
  }
  while (static_cast<int64_t>(current.front().pair.domain.size()) > total - keep) {
    std::vector<PairTerm> grown;
    for (const PairTerm& term : current)
      for (const PairTerm& next : bend_left(group, term.pair, qtotal, device))
        grown.push_back(PairTerm{next.pair, term.coefficient * next.coefficient});
    require_terms(grown, "domain");
    current = std::move(grown);
  }
  return current;
}

std::vector<PairTerm> fold_right(const FusionCategory& group, const FusionTreePair& pair,
                                 const c10::Device& device) {
  const FusionTree& out = pair.codomain;
  if (out.size() < 1) throw std::invalid_argument("fold_right: the codomain side has no leg to fold");
  const Sector a = out.sectors.front();
  const bool isduala = out.isdual.front();
  const Sector c = out.coupled;
  const std::complex<double> kappa = frobenius_schur_phase(group, a, device);
  std::vector<PairTerm> terms;
  for (const TreeTerm& left : multi_fmove(group, out, device)) {
    const Sector b = left.tree.coupled;
    const std::complex<double> head = std::sqrt(group.quantum_dim(c) / group.quantum_dim(b)) *
                                     scalar_of(a_symbol(group, a, b, c, device), "fold_right");
    for (const TreeTerm& right : multi_fmove_inv(group, group.dual_sector(a), b, pair.domain,
                                                 !isduala, device)) {
      std::complex<double> coefficient = head * left.coefficient * right.coefficient;
      if (isduala) coefficient *= kappa;
      terms.push_back(PairTerm{FusionTreePair{left.tree, right.tree}, coefficient});
    }
  }
  return terms;
}

std::vector<PairTerm> fold_left(const FusionCategory& group, const FusionTreePair& pair,
                                const c10::Device& device) {
  if (pair.domain.size() < 1) throw std::invalid_argument("fold_left: the domain side has no leg to fold");
  std::vector<PairTerm> terms;
  for (const PairTerm& term : fold_right(group, FusionTreePair{pair.domain, pair.codomain}, device)) {
    terms.push_back(PairTerm{FusionTreePair{term.pair.domain, term.pair.codomain},
                             std::conj(term.coefficient)});
  }
  return terms;
}

std::vector<PairTerm> cycle_clockwise(const FusionCategory& group, const FusionTreePair& pair,
                                      const Sector& qtotal, const c10::Device& device) {
  const bool folds_first = pair.codomain.size() > 0;
  const std::vector<PairTerm> firsts =
      folds_first ? fold_right(group, pair, device) : bend_left(group, pair, qtotal, device);
  std::vector<PairTerm> terms;
  for (const PairTerm& first : firsts) {
    const std::vector<PairTerm> nexts =
        folds_first ? bend_left(group, first.pair, qtotal, device) : fold_right(group, first.pair, device);
    for (const PairTerm& next : nexts) {
      const std::complex<double> coefficient = next.coefficient * first.coefficient;
      if (coefficient == std::complex<double>(0.0, 0.0)) continue;
      terms.push_back(PairTerm{next.pair, coefficient});
    }
  }
  return terms;
}

std::vector<PairTerm> cycle_anticlockwise(const FusionCategory& group, const FusionTreePair& pair,
                                          const Sector& qtotal, const c10::Device& device) {
  const bool folds_first = pair.domain.size() > 0;
  const std::vector<PairTerm> firsts =
      folds_first ? fold_left(group, pair, device) : bend_right(group, pair, qtotal, device);
  std::vector<PairTerm> terms;
  for (const PairTerm& first : firsts) {
    const std::vector<PairTerm> nexts =
        folds_first ? bend_right(group, first.pair, qtotal, device) : fold_left(group, first.pair, device);
    for (const PairTerm& next : nexts) {
      const std::complex<double> coefficient = next.coefficient * first.coefficient;
      if (coefficient == std::complex<double>(0.0, 0.0)) continue;
      terms.push_back(PairTerm{next.pair, coefficient});
    }
  }
  return terms;
}

at::Tensor pair_tensor(const FusionCategory& group, const FusionTreePair& pair,
                       const c10::Device& device, bool dress) {
  if (pair.codomain.coupled != pair.domain.coupled) {
    throw std::invalid_argument("pair_tensor: the two sides of a pair couple to different sectors");
  }
  const at::Tensor out = tree_tensor(group, pair.codomain, device, dress);
  const at::Tensor in = tree_tensor(group, pair.domain, device, dress);
  return at::tensordot(out, in, {-1}, {-1});
}

}  // namespace sym::engine

namespace sym::engine {
namespace {

// A tree as Python hands it in: its sectors, duality flags, inner couplings and vertex copies.
FusionTree tree_of(const py::dict& data) {
  FusionTree tree;
  tree.sectors = data["sectors"].cast<std::vector<Sector>>();
  tree.isdual = data["isdual"].cast<std::vector<bool>>();
  tree.inner = data["inner"].cast<std::vector<Sector>>();
  tree.vertices = data["vertices"].cast<std::vector<int64_t>>();
  tree.coupled = data["coupled"].cast<Sector>();
  return tree;
}

py::dict tree_as(const FusionTree& tree) {
  py::dict data;
  data["sectors"] = tree.sectors;
  data["isdual"] = tree.isdual;
  data["inner"] = tree.inner;
  data["vertices"] = tree.vertices;
  data["coupled"] = py::tuple(py::cast(tree.coupled));
  return data;
}

FusionTreePair pair_of(const py::dict& data) {
  return FusionTreePair{tree_of(data["codomain"].cast<py::dict>()),
                        tree_of(data["domain"].cast<py::dict>())};
}

py::dict pair_as(const FusionTreePair& pair) {
  py::dict data;
  data["codomain"] = tree_as(pair.codomain);
  data["domain"] = tree_as(pair.domain);
  return data;
}

py::list terms_as(const std::vector<PairTerm>& terms) {
  py::list out;
  for (const PairTerm& term : terms) out.append(py::make_tuple(pair_as(term.pair), term.coefficient));
  return out;
}

}  // namespace

void bind_duality(py::module_& m) {
  m.def("engine_bend_right", [](const std::shared_ptr<FusionCategory> group, const py::dict& pair,
                                const Sector& qtotal, const py::object& device) {
    return terms_as(bend_right(*group, pair_of(pair), qtotal, device_of(device)));
  });
  m.def("engine_bend_left", [](const std::shared_ptr<FusionCategory> group, const py::dict& pair,
                               const Sector& qtotal, const py::object& device) {
    return terms_as(bend_left(*group, pair_of(pair), qtotal, device_of(device)));
  });
  m.def("engine_repartition", [](const std::shared_ptr<FusionCategory> group, const py::dict& pair,
                                 int64_t keep, const Sector& qtotal, const py::object& device) {
    return terms_as(repartition(*group, pair_of(pair), keep, qtotal, device_of(device)));
  });
  m.def("engine_fold_right", [](const std::shared_ptr<FusionCategory> group, const py::dict& pair,
                                const py::object& device) {
    return terms_as(fold_right(*group, pair_of(pair), device_of(device)));
  });
  m.def("engine_fold_left", [](const std::shared_ptr<FusionCategory> group, const py::dict& pair,
                               const py::object& device) {
    return terms_as(fold_left(*group, pair_of(pair), device_of(device)));
  });
  m.def("engine_cycle_clockwise", [](const std::shared_ptr<FusionCategory> group, const py::dict& pair,
                                     const Sector& qtotal, const py::object& device) {
    return terms_as(cycle_clockwise(*group, pair_of(pair), qtotal, device_of(device)));
  });
  m.def("engine_cycle_anticlockwise", [](const std::shared_ptr<FusionCategory> group,
                                         const py::dict& pair, const Sector& qtotal,
                                         const py::object& device) {
    return terms_as(cycle_anticlockwise(*group, pair_of(pair), qtotal, device_of(device)));
  });
  m.def("engine_multi_fmove", [](const std::shared_ptr<FusionCategory> group, const py::dict& tree,
                                 const py::object& device) {
    py::list out;
    for (const TreeTerm& term : multi_fmove(*group, tree_of(tree), device_of(device)))
      out.append(py::make_tuple(tree_as(term.tree), term.coefficient));
    return out;
  });
  m.def("engine_pair_tensor", [](const std::shared_ptr<FusionCategory> group, const py::dict& pair,
                                 const py::object& device, bool dress) {
    return pair_tensor(*group, pair_of(pair), device_of(device), dress);
  }, py::arg("group"), py::arg("pair"), py::arg("device") = py::none(), py::arg("dress") = false);
  m.def("engine_tree_describe", [](const std::shared_ptr<FusionCategory> group, const py::list& legs,
                                   const py::tuple& charge, int64_t index) {
    std::vector<LegCharge> charges;
    for (const py::handle& leg : legs) {
      const py::tuple data = leg.cast<py::tuple>();
      const auto sectors = data[0].cast<std::vector<std::pair<Sector, int64_t>>>();
      const int64_t qconj = data[1].cast<int64_t>();
      for (const auto& [sector, multiplicity] : sectors)
        for (int64_t copy = 0; copy < multiplicity; ++copy) charges.push_back(LegCharge{sector, qconj == -1});
    }
    const std::map<Sector, std::vector<FusionTree>> trees = trees_by_charge(*group, charges);
    const auto found = trees.find(charge.cast<Sector>());
    const std::vector<FusionTree>& of_charge = found == trees.end() ? std::vector<FusionTree>{} : found->second;
    if (index < 0 || index >= static_cast<int64_t>(of_charge.size())) {
      throw std::out_of_range("engine_tree_describe: no tree at index " + std::to_string(index));
    }
    return py::make_tuple(py::tuple(py::cast(of_charge.at(index).coupled)), tree_as(of_charge.at(index)));
  });
}

}  // namespace sym::engine

// The exchange of two neighbouring legs of a tree, and the permutation of a tree pair built from it.


#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

namespace sym::engine {
namespace {

// The sector a tree couples to on the prefix ending at leg `k`, counting legs from one.
Sector inner_extended(const FusionCategory& group, const FusionTree& tree, int64_t k) {
  const int64_t n = tree.size();
  if (k == 1) return pointing(group, tree, 0);
  if (k == n) return tree.coupled;
  return tree.inner.at(k - 2);
}

// The entries of a symbol, read once into host memory.
std::vector<std::complex<double>> entries_of(const at::Tensor& symbol) {
  const at::Tensor host = symbol.to(at::kCPU, at::kComplexDouble).contiguous();
  const int64_t count = host.numel();
  const c10::complex<double>* data = host.data_ptr<c10::complex<double>>();
  std::vector<std::complex<double>> entries(static_cast<size_t>(count));
  for (int64_t index = 0; index < count; ++index) {
    const c10::complex<double> value = data[index];
    entries[static_cast<size_t>(index)] = std::complex<double>(value.real(), value.imag());
  }
  return entries;
}

}  // namespace

std::vector<TreeTerm> swap_adjacent(const FusionCategory& group, const FusionTree& tree, int64_t k,
                                    bool inverse, const c10::Device& device) {
  const int64_t n = tree.size();
  if (k < 0 || k + 1 >= n) {
    throw std::out_of_range("swap_adjacent: no leg to exchange with the leg at " + std::to_string(k));
  }
  // The symbols of the moves read the sectors the legs point at, not the ones they store.
  const Sector a = pointing(group, tree, k);
  const Sector b = pointing(group, tree, k + 1);
  FusionTree swapped = tree;
  std::swap(swapped.sectors.at(k), swapped.sectors.at(k + 1));
  const bool dual_left = swapped.isdual.at(k);
  const bool dual_right = swapped.isdual.at(k + 1);
  swapped.isdual.at(k) = dual_right;
  swapped.isdual.at(k + 1) = dual_left;
  std::vector<TreeTerm> terms;
  if (k == 0) {
    // The exchange happens at the top vertex, so it costs the braiding of the two legs alone.
    const Sector c = n > 2 ? tree.inner.at(0) : tree.coupled;
    const at::Tensor symbol = group.braiding(a, b, c, device);
    if (symbol.numel() == 1) {
      const std::complex<double> value = scalar_of(symbol, "swap_adjacent");
      terms.push_back(TreeTerm{swapped, inverse ? std::conj(value) : value});
      return terms;
    }
    const std::vector<std::complex<double>> values = entries_of(symbol);
    const int64_t columns = symbol.size(1);
    const int64_t row = tree.vertices.at(0);
    for (int64_t column = 0; column < columns; ++column) {
      FusionTree branch = swapped;
      branch.vertices.at(0) = column;
      terms.push_back(TreeTerm{branch, inverse ? std::conj(values.at(row * columns + column))
                                               : values.at(row * columns + column)});
    }
    return terms;
  }
  // Below the top the exchange also recouples the two vertices it sits between.
  const Sector inner_a = inner_extended(group, tree, k);
  const Sector inner_c = inner_extended(group, tree, k + 1);
  const Sector inner_e = inner_extended(group, tree, k + 2);
  const std::vector<Sector> left = group.fusion_outcomes(inner_a, b);
  const std::vector<Sector> right = group.fusion_outcomes(inner_e, group.dual_sector(a));
  for (const Sector& fresh : left) {
    if (std::find(right.begin(), right.end(), fresh) == right.end()) continue;
    const at::Tensor braid_first = group.braiding(inner_c, b, inner_e, device);
    const at::Tensor braid_second = group.braiding(inner_a, b, fresh, device);
    const at::Tensor recouple =
        group.associator(b, inner_a, pointing(group, tree, k), inner_e, fresh, inner_c, device);
    FusionTree branch = swapped;
    branch.inner.at(k - 1) = fresh;
    if (braid_first.numel() == 1 && braid_second.numel() == 1 && recouple.numel() == 1) {
      const std::complex<double> first = scalar_of(braid_first, "swap_adjacent");
      const std::complex<double> second = scalar_of(braid_second, "swap_adjacent");
      const std::complex<double> middle = scalar_of(recouple, "swap_adjacent");
      const std::complex<double> coefficient =
          inverse ? std::conj(first * middle) * second : first * std::conj(middle * second);
      terms.push_back(TreeTerm{branch, coefficient});
      continue;
    }
    // A vertex that couples more than one way turns the exchange into a matrix over its labels,
    // whose entries couple the copies of the two vertices the exchange sits between.
    const std::vector<std::complex<double>> first = entries_of(braid_first);
    const std::vector<std::complex<double>> second = entries_of(braid_second);
    const std::vector<std::complex<double>> middle = entries_of(recouple);
    const int64_t rows_second = braid_second.size(0);
    const int64_t rows_first = braid_first.size(0);
    const int64_t columns_first = braid_first.size(1);
    const int64_t left_copies = recouple.size(0);
    const int64_t middle_copies = recouple.size(1);
    const int64_t inner_copies = recouple.size(2);
    const int64_t right_copies = recouple.size(3);
    // The exchange sits between the vertex that produces ``inner_c`` and the one that produces
    // ``inner_e``; their copies label the rows and columns of the recoupling matrix.
    const int64_t row = tree.vertices.at(k - 1);
    const int64_t column_copy = tree.vertices.at(k);
    (void)rows_first;
    (void)inner_copies;
    for (int64_t fresh_copy = 0; fresh_copy < rows_second; ++fresh_copy) {
      for (int64_t next_copy = 0; next_copy < middle_copies; ++next_copy) {
        std::complex<double> coefficient{0.0, 0.0};
        for (int64_t column = 0; column < columns_first; ++column) {
          const std::complex<double> one_first =
              first[static_cast<size_t>(column_copy * columns_first + column)];
          for (int64_t back = 0; back < left_copies; ++back) {
            const std::complex<double> one_middle = middle[static_cast<size_t>(
                ((back * middle_copies + next_copy) * inner_copies + row) * right_copies + column)];
            const std::complex<double> one_second =
                second[static_cast<size_t>(fresh_copy * left_copies + back)];
            if (inverse) {
              coefficient += std::conj(one_first) * std::conj(one_middle) * one_second;
            } else {
              coefficient += one_first * std::conj(one_middle) * std::conj(one_second);
            }
          }
        }
        if (std::abs(coefficient) < 1e-14) continue;
        FusionTree copy = branch;
        copy.vertices.at(k - 1) = fresh_copy;
        copy.vertices.at(k) = next_copy;
        terms.push_back(TreeTerm{copy, coefficient});
      }
    }
  }
  return terms;
}

std::vector<TreeTerm> braid_tree(const FusionCategory& group, const FusionTree& tree,
                                 const std::vector<int64_t>& order, const c10::Device& device) {
  const int64_t n = tree.size();
  if (static_cast<int64_t>(order.size()) != n) {
    throw std::invalid_argument("braid_tree: the order names " + std::to_string(order.size()) +
                                " legs of a tree with " + std::to_string(n));
  }
  std::vector<bool> seen(static_cast<size_t>(n), false);
  for (const int64_t position : order) {
    if (position < 1 || position > n || seen.at(static_cast<size_t>(position - 1))) {
      throw std::invalid_argument("braid_tree: the order is not a permutation of one to the leg count");
    }
    seen.at(static_cast<size_t>(position - 1)) = true;
  }
  // The exchanges are found on a copy of the order and applied to the identity arrangement, so the
  // sequence turns the identity into the requested order.
  std::vector<int64_t> current = order;
  std::vector<int64_t> swaps;
  for (int64_t target = 0; target < n; ++target) {
    const auto found =
        std::find(current.begin() + target, current.end(), target + 1) - current.begin();
    for (int64_t at = found; at > target; --at) {
      swaps.push_back(at - 1);
      std::swap(current.at(static_cast<size_t>(at - 1)), current.at(static_cast<size_t>(at)));
    }
  }
  // The sequence was found by sorting the order back to the identity, so applying it backwards
  // takes the identity arrangement to the order.
  std::reverse(swaps.begin(), swaps.end());
  std::vector<int64_t> levels(static_cast<size_t>(n));
  std::iota(levels.begin(), levels.end(), 0);
  std::vector<TreeTerm> current_terms{TreeTerm{tree, 1.0}};
  for (const int64_t position : swaps) {
    const bool inverse = levels.at(static_cast<size_t>(position)) >
                         levels.at(static_cast<size_t>(position + 1));
    std::vector<TreeTerm> grown;
    for (const TreeTerm& term : current_terms) {
      for (const TreeTerm& next : swap_adjacent(group, term.tree, position, inverse, device)) {
        grown.push_back(TreeTerm{next.tree, term.coefficient * next.coefficient});
      }
    }
    current_terms = std::move(grown);
    std::swap(levels.at(static_cast<size_t>(position)),
              levels.at(static_cast<size_t>(position + 1)));
  }
  return current_terms;
}

std::vector<PairTerm> permute_pair(const FusionCategory& group, const FusionTreePair& pair,
                                   int64_t keep, const std::vector<int64_t>& order,
                                   const c10::Device& device) {
  const int64_t codomain = static_cast<int64_t>(pair.codomain.size());
  const int64_t total = codomain + static_cast<int64_t>(pair.domain.size());
  // A permutation keeps the two sides apart: the order names the source legs of the codomain first
  // and those of the domain after, so neither tree changes its root.
  if (keep != codomain) {
    throw std::invalid_argument("permute_pair: a permutation keeps the codomain legs of a pair");
  }
  if (static_cast<int64_t>(order.size()) != total) {
    throw std::invalid_argument("permute_pair: the order names " + std::to_string(order.size()) +
                                " legs of a pair with " + std::to_string(total));
  }
  std::vector<int64_t> codomain_order;
  std::vector<int64_t> domain_order;
  for (int64_t at = 0; at < total; ++at) {
    const int64_t leg = order.at(static_cast<size_t>(at));
    if (leg < 1 || leg > total) {
      throw std::invalid_argument("permute_pair: the order names a leg the pair does not carry");
    }
    if ((leg <= codomain) != (at < keep)) {
      throw std::invalid_argument("permute_pair: a permutation cannot move a leg to the other side");
    }
    (leg <= codomain ? codomain_order : domain_order).push_back(leg <= codomain ? leg : leg - codomain);
  }
  std::vector<PairTerm> terms;
  for (const TreeTerm& left : braid_tree(group, pair.codomain, codomain_order, device)) {
    for (const TreeTerm& right : braid_tree(group, pair.domain, domain_order, device)) {
      terms.push_back(PairTerm{FusionTreePair{left.tree, right.tree},
                               left.coefficient * right.coefficient});
    }
  }
  return terms;
}

}  // namespace sym::engine

namespace sym::engine {
namespace {

// The device a Python argument names, defaulting to the one ATen is set to.
c10::Device braid_device_of(const py::object& value) {
  if (value.is_none()) return at::Device(at::kCPU);
  return value.cast<c10::Device>();
}

FusionTree braid_tree_of(const py::dict& data) {
  FusionTree tree;
  tree.sectors = data["sectors"].cast<std::vector<Sector>>();
  tree.isdual = data["isdual"].cast<std::vector<bool>>();
  tree.inner = data["inner"].cast<std::vector<Sector>>();
  tree.vertices = data["vertices"].cast<std::vector<int64_t>>();
  tree.coupled = data["coupled"].cast<Sector>();
  return tree;
}

py::dict braid_tree_as(const FusionTree& tree) {
  py::dict data;
  data["sectors"] = tree.sectors;
  data["isdual"] = tree.isdual;
  data["inner"] = tree.inner;
  data["vertices"] = tree.vertices;
  data["coupled"] = py::tuple(py::cast(tree.coupled));
  return data;
}

FusionTreePair braid_pair_of(const py::dict& data) {
  return FusionTreePair{braid_tree_of(data["codomain"].cast<py::dict>()),
                        braid_tree_of(data["domain"].cast<py::dict>())};
}

py::dict braid_pair_as(const FusionTreePair& pair) {
  py::dict data;
  data["codomain"] = braid_tree_as(pair.codomain);
  data["domain"] = braid_tree_as(pair.domain);
  return data;
}

py::list braid_terms_as(const std::vector<PairTerm>& terms) {
  py::list out;
  for (const PairTerm& term : terms) out.append(py::make_tuple(braid_pair_as(term.pair), term.coefficient));
  return out;
}

}  // namespace

void bind_braid(py::module_& m) {
  m.def("engine_permute_pair", [](const std::shared_ptr<FusionCategory> group, const py::dict& pair,
                                  int64_t keep, const std::vector<int64_t>& order,
                                  const py::object& device) {
    return braid_terms_as(permute_pair(*group, braid_pair_of(pair), keep, order, braid_device_of(device)));
  });
  m.def("engine_braid_tree", [](const std::shared_ptr<FusionCategory> group, const py::dict& tree,
                                const std::vector<int64_t>& order, const py::object& device) {
    py::list out;
    for (const TreeTerm& term : braid_tree(*group, braid_tree_of(tree), order, braid_device_of(device))) {
      out.append(py::make_tuple(braid_tree_as(term.tree), term.coefficient));
    }
    return out;
  });
}

}  // namespace sym::engine
