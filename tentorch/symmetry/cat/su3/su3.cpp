#include "su3.h"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sym::su3 {
namespace {

constexpr double kTolerance = 1e-10;
constexpr double kInverseRoot3 = 0.57735026918962576451;  // 1 / sqrt(3)
constexpr double kMixing = 0.61803398874989484820;       // splits the degenerate weights

// The eight hermitian generators of the fundamental representation.
const std::vector<at::Tensor>& gell_mann() {
  static const std::vector<at::Tensor> matrices = [] {
    const auto options = at::kComplexDouble;
    const c10::complex<double> imaginary(0.0, 1.0);
    std::vector<at::Tensor> out;
    for (int64_t a = 0; a < 8; ++a) {
      at::Tensor m = at::zeros({3, 3}, options);
      switch (a) {
        case 0: m.index_put_({0, 1}, 1.0); m.index_put_({1, 0}, 1.0); break;
        case 1: m.index_put_({0, 1}, -imaginary); m.index_put_({1, 0}, imaginary); break;
        case 2: m.index_put_({0, 0}, 1.0); m.index_put_({1, 1}, -1.0); break;
        case 3: m.index_put_({0, 2}, 1.0); m.index_put_({2, 0}, 1.0); break;
        case 4: m.index_put_({0, 2}, -imaginary); m.index_put_({2, 0}, imaginary); break;
        case 5: m.index_put_({1, 2}, 1.0); m.index_put_({2, 1}, 1.0); break;
        case 6: m.index_put_({1, 2}, -imaginary); m.index_put_({2, 1}, imaginary); break;
        default:
          m.index_put_({0, 0}, kInverseRoot3);
          m.index_put_({1, 1}, kInverseRoot3);
          m.index_put_({2, 2}, -2.0 * kInverseRoot3);
          break;
      }
      out.push_back(m);
    }
    return out;
  }();
  return matrices;
}

int64_t power_of_three(int64_t exponent) {
  int64_t result = 1;
  for (int64_t k = 0; k < exponent; ++k) result *= 3;
  return result;
}

// The generator acting on the space with `upper` upper and `lower` lower indices.
at::Tensor action_matrix(const at::Tensor& generator, int64_t upper, int64_t lower) {
  const int64_t total = upper + lower;
  at::Tensor result = at::zeros({power_of_three(total), power_of_three(total)}, at::kComplexDouble);
  const at::Tensor identity = at::eye(3, at::kComplexDouble);
  for (int64_t k = 0; k < total; ++k) {
    const bool is_upper = k < upper;
    // a transposed generator is not contiguous, and at::kron views its operands
    const at::Tensor slot = is_upper ? generator : (-generator.t()).contiguous();
    // a one-by-one identity: the kron below builds the total matrix from the slots
    at::Tensor term = at::eye(1, at::kComplexDouble);
    for (int64_t j = 0; j < total; ++j) term = at::kron(term, j == k ? slot : identity);
    result = result + term;
  }
  return result;
}

// An orthonormal basis of the tensors symmetric among the uppers and among the lowers.
at::Tensor symmetric_basis(int64_t upper, int64_t lower) {
  const int64_t total = upper + lower;
  const int64_t size = power_of_three(total);
  at::Tensor projector = at::zeros({size, size}, at::kComplexDouble);
  std::vector<int64_t> ups(upper);
  std::iota(ups.begin(), ups.end(), 0);
  std::vector<int64_t> downs(lower);
  std::iota(downs.begin(), downs.end(), 0);
  int64_t count = 0;
  std::vector<int64_t> shape(total, 3);
  shape.push_back(size);
  do {
    do {
      std::vector<int64_t> axes = ups;
      for (int64_t k = 0; k < lower; ++k) axes.push_back(upper + downs[k]);
      axes.push_back(total);
      at::Tensor identity = at::eye(size, at::kComplexDouble).reshape(shape);
      projector = projector + identity.permute(axes).reshape({size, size});
      ++count;
    } while (std::next_permutation(downs.begin(), downs.end()));
  } while (std::next_permutation(ups.begin(), ups.end()));
  projector = projector / static_cast<double>(count);
  auto [vectors, values, _] = at::linalg_svd(projector);
  return vectors.index_select(1, (values > kTolerance).nonzero().squeeze(-1));
}

// The trace map contracting the first upper with the first lower index of the symmetric basis.
at::Tensor trace_map(int64_t upper, int64_t lower) {
  const int64_t total = upper + lower;
  const at::Tensor source = symmetric_basis(upper, lower);
  const at::Tensor target = symmetric_basis(upper - 1, lower - 1);
  std::string labels;
  for (int64_t k = 0; k < total; ++k) labels.push_back(static_cast<char>('a' + k));
  std::string contracted = labels;
  contracted[upper] = labels[0];
  std::string kept;
  for (int64_t k = 0; k < total; ++k) {
    if (k != 0 && k != upper) kept.push_back(labels[k]);
  }
  const std::string equation = contracted + "->" + kept;
  at::Tensor rows = at::zeros({target.size(1), source.size(1)}, at::kComplexDouble);
  for (int64_t k = 0; k < source.size(1); ++k) {
    at::Tensor traced = at::einsum(equation, {source.select(1, k).reshape(std::vector<int64_t>(total, 3))});
    rows.select(1, k).copy_(target.conj().t().matmul(traced.reshape({-1})));
  }
  return rows;
}

struct Irrep {
  at::Tensor generators;  // (8, d, d)
  at::Tensor weights;     // (d, 2) of (T3, T8)
};

// The memo of `irrep`, so that the cache can be dropped.
std::map<Sector, Irrep>& irrep_cache() {
  static std::map<Sector, Irrep> cache;
  return cache;
}

// The generators and weights of the sector in its weight basis, memoised.
const Irrep& irrep(const Sector& sector) {
  std::map<Sector, Irrep>& cache = irrep_cache();
  const auto found = cache.find(sector);
  if (found != cache.end()) return found->second;
  const int64_t p = sector.at(0);
  const int64_t q = sector.at(1);
  at::Tensor basis = symmetric_basis(p, q);
  if (p > 0 && q > 0) {
    // the kernel of the trace map: the rows of Vh past the non-zero singular values, so the
    // full Vh is needed (the reduced one keeps only as many rows as the map has columns)
    auto [_, values, vectors] = at::linalg_svd(trace_map(p, q), true);
    const int64_t rank = (values > kTolerance).sum().item<int64_t>();
    basis = basis.matmul(vectors.narrow(0, rank, vectors.size(0) - rank).conj().t());
  }
  const auto& gell = gell_mann();
  const auto project = [&](const at::Tensor& current) {
    std::vector<at::Tensor> out;
    for (const at::Tensor& generator : gell) {
      out.push_back(current.conj().t().matmul(action_matrix(generator, p, q)).matmul(current));
    }
    return at::stack(out, 0);
  };
  at::Tensor generators = project(basis);
  auto [values, vectors] = at::linalg_eigh(generators[2] + kMixing * generators[7]);
  basis = basis.matmul(vectors);
  generators = project(basis);
  at::Tensor weights = at::stack({at::real(at::diagonal(generators[2])),
                                  at::real(at::diagonal(generators[7]))}, -1);
  const at::Tensor order = at::argsort(weights.select(1, 0) * 4.0 + weights.select(1, 1), 0, true);
  Irrep result;
  // ``order`` sorts the weights, so it indexes the two matrix axes, not the eight generators
  result.generators = generators.index_select(1, order).index_select(2, order);
  result.weights = weights.index_select(0, order);
  return cache.emplace(sector, result).first->second;
}

std::array<int64_t, 3> weight_key(double t3, double t8) {
  const double third = t8 * kInverseRoot3;
  const auto scaled = [](double value) { return std::llround(value * 6.0); };
  return {scaled((t3 + third) * 0.5), scaled((third - t3) * 0.5), scaled(-third)};
}

std::array<int64_t, 3> weight_key(const at::Tensor& weights, int64_t index) {
  return weight_key(weights[index][0].item<double>(), weights[index][1].item<double>());
}

// The six permutations of three indices with their parity.
const std::vector<std::pair<std::array<int64_t, 3>, int64_t>>& permutations() {
  static const std::vector<std::pair<std::array<int64_t, 3>, int64_t>> table = [] {
    std::vector<std::pair<std::array<int64_t, 3>, int64_t>> out;
    std::array<int64_t, 3> permutation{0, 1, 2};
    do {
      const int64_t parity =
          ((permutation[0] > permutation[1] ? 1 : 0) + (permutation[0] > permutation[2] ? 1 : 0)
           + (permutation[1] > permutation[2] ? 1 : 0)) % 2;
      out.push_back({permutation, parity});
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    return out;
  }();
  return table;
}

// The integer weight lattice: the simple roots and the Weyl vector.  The Dynkin labels (p, q)
// have highest weight $(2p + q, -p + q, -p - 2q)$, and the Weyl group permutes the three
// coordinates, so every weight of (p, q) is $h - i \alpha_1 - j \alpha_2$ with $i, j \ge 0$.
constexpr std::array<int64_t, 3> kAlpha1{3, -3, 0};
constexpr std::array<int64_t, 3> kAlpha2{0, 3, -3};
constexpr std::array<int64_t, 3> kWeylVector{3, 0, -3};

std::array<int64_t, 3> highest_weight(int64_t p, int64_t q) {
  return {2 * p + q, -p + q, -p - 2 * q};
}

std::array<int64_t, 3> permute(const std::array<int64_t, 3>& vector,
                               const std::array<int64_t, 3>& permutation) {
  return {vector[permutation[0]], vector[permutation[1]], vector[permutation[2]]};
}

int64_t dot(const std::array<int64_t, 3>& left, const std::array<int64_t, 3>& right) {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

// The number of ways to write $x \alpha_1 + y \alpha_2$ as a non-negative sum of positive roots,
// which is $\min(x, y) + 1$ on the positive root cone and zero outside it.
int64_t partition_count(const std::array<int64_t, 3>& weight) {
  if (weight[0] % 3 != 0 || weight[2] % 3 != 0) return 0;
  const int64_t x = weight[0] / 3;
  const int64_t y = -weight[2] / 3;
  if (x < 0 || y < 0) return 0;
  return (x < y ? x : y) + 1;
}

// Kostant's formula: $m(\mu) = \sum_{w \in W} \epsilon(w) P(w(\lambda + \rho) - \rho - \mu)$.
int64_t weight_multiplicity(const std::array<int64_t, 3>& highest,
                            const std::array<int64_t, 3>& weight) {
  const std::array<int64_t, 3> shifted{
      highest[0] + kWeylVector[0], highest[1] + kWeylVector[1], highest[2] + kWeylVector[2]};
  int64_t multiplicity = 0;
  for (const auto& [permutation, parity] : permutations()) {
    const std::array<int64_t, 3> image = permute(shifted, permutation);
    const std::array<int64_t, 3> argument{
        image[0] - kWeylVector[0] - weight[0], image[1] - kWeylVector[1] - weight[1],
        image[2] - kWeylVector[2] - weight[2]};
    multiplicity += (parity ? -1 : 1) * partition_count(argument);
  }
  return multiplicity;
}

struct WeightTerm {
  std::array<int64_t, 3> key;
  int64_t multiplicity;
};

// The weights of a sector as (key, multiplicity) pairs: the lattice points of the Weyl polytope
// $h - Q_+$ that are also inside $\operatorname{conv}(W h)$, counted by Kostant's formula.
std::map<Sector, std::vector<WeightTerm>>& weight_terms_cache() {
  static std::map<Sector, std::vector<WeightTerm>> cache;
  return cache;
}

const std::vector<WeightTerm>& weight_terms(const Sector& sector) {
  std::map<Sector, std::vector<WeightTerm>>& cache = weight_terms_cache();
  const auto found = cache.find(sector);
  if (found != cache.end()) return found->second;
  const int64_t p = sector.at(0);
  const int64_t q = sector.at(1);
  const std::array<int64_t, 3> highest = highest_weight(p, q);
  std::vector<std::array<int64_t, 3>> orbit;
  for (const auto& entry : permutations()) orbit.push_back(permute(highest, entry.first));
  std::sort(orbit.begin(), orbit.end());
  orbit.erase(std::unique(orbit.begin(), orbit.end()), orbit.end());
  const int64_t norm = dot(highest, highest);
  const int64_t span = p + q;
  std::vector<WeightTerm> terms;
  for (int64_t i = 0; i <= span; ++i) {
    for (int64_t j = 0; j <= span; ++j) {
      // h - i alpha_1 - j alpha_2
      const std::array<int64_t, 3> weight{
          highest[0] - i * kAlpha1[0] - j * kAlpha2[0],
          highest[1] - i * kAlpha1[1] - j * kAlpha2[1],
          highest[2] - i * kAlpha1[2] - j * kAlpha2[2]};
      const bool inside = std::all_of(orbit.begin(), orbit.end(), [&](const auto& vertex) {
        return dot(vertex, weight) <= norm;
      });
      if (!inside) continue;
      const int64_t multiplicity = weight_multiplicity(highest, weight);
      if (multiplicity > 0) terms.push_back({weight, multiplicity});
    }
  }
  return cache.emplace(sector, std::move(terms)).first->second;
}

}  // namespace

int64_t dimension(const Sector& sector) {
  const int64_t p = sector.at(0);
  const int64_t q = sector.at(1);
  return (p + 1) * (q + 1) * (p + q + 2) / 2;
}


// The memo of `weight_keys`, so that the cache can be dropped.
std::map<Sector, std::vector<std::array<int64_t, 3>>>& weight_keys_cache() {
  static std::map<Sector, std::vector<std::array<int64_t, 3>>> cache;
  return cache;
}

const std::vector<std::array<int64_t, 3>>& weight_keys(const Sector& sector) {
  std::map<Sector, std::vector<std::array<int64_t, 3>>>& cache = weight_keys_cache();
  const auto found = cache.find(sector);
  if (found != cache.end()) return found->second;
  const Irrep& data = irrep(sector);
  std::vector<std::array<int64_t, 3>> keys;
  for (int64_t k = 0; k < data.weights.size(0); ++k) keys.push_back(weight_key(data.weights, k));
  return cache.emplace(sector, keys).first->second;
}

std::map<Sector, int64_t> fusion(const Sector& first, const Sector& second) {
  const std::vector<WeightTerm>& left = weight_terms(first);
  const std::vector<WeightTerm>& right = weight_terms(second);
  // The multiplicity of a weight of the product is a sum over the smaller weight list, with the
  // other one as a lookup table.
  const bool left_is_smaller = left.size() <= right.size();
  const std::vector<WeightTerm>& scan = left_is_smaller ? left : right;
  const std::vector<WeightTerm>& table = left_is_smaller ? right : left;
  std::map<std::array<int64_t, 3>, int64_t> index;
  for (const auto& term : table) index.emplace(term.key, term.multiplicity);
  const auto product_multiplicity = [&](const std::array<int64_t, 3>& total) {
    int64_t multiplicity = 0;
    for (const auto& term : scan) {
      const std::array<int64_t, 3> remainder{
          total[0] - term.key[0], total[1] - term.key[1], total[2] - term.key[2]};
      const auto hit = index.find(remainder);
      if (hit != index.end()) multiplicity += term.multiplicity * hit->second;
    }
    return multiplicity;
  };
  const std::array<int64_t, 3> first_highest = highest_weight(first.at(0), first.at(1));
  const std::array<int64_t, 3> second_highest = highest_weight(second.at(0), second.at(1));
  const std::array<int64_t, 3> top{first_highest[0] + second_highest[0],
                                   first_highest[1] + second_highest[1],
                                   first_highest[2] + second_highest[2]};
  // Every weight of the product is top - i alpha_1 - j alpha_2, and the largest i and j are the
  // sums of the two spans.
  const int64_t span = first.at(0) + first.at(1) + second.at(0) + second.at(1);
  std::map<Sector, int64_t> found;
  for (int64_t i = 0; i <= span; ++i) {
    for (int64_t j = 0; j <= span; ++j) {
      const std::array<int64_t, 3> key{top[0] - i * kAlpha1[0] - j * kAlpha2[0],
                                       top[1] - i * kAlpha1[1] - j * kAlpha2[1],
                                       top[2] - i * kAlpha1[2] - j * kAlpha2[2]};
      if (key[0] < key[1] || key[1] < key[2]) continue;
      // Racah-Speiser: $N^c_{ab} = \sum_{w \in W} \epsilon(w) m_{a \otimes b}(c + \rho - w\rho)$.
      int64_t multiplicity = 0;
      for (const auto& [permutation, parity] : permutations()) {
        const std::array<int64_t, 3> shifted{
            key[0] + kWeylVector[0] - kWeylVector[permutation[0]],
            key[1] + kWeylVector[1] - kWeylVector[permutation[1]],
            key[2] + kWeylVector[2] - kWeylVector[permutation[2]]};
        multiplicity += (parity ? -1 : 1) * product_multiplicity(shifted);
      }
      if (multiplicity != 0)
        found[{(key[0] - key[1]) / 3, (key[1] - key[2]) / 3}] = multiplicity;
    }
  }
  return found;
}

std::map<Sector, int64_t> fusion_from_weights(const Sector& first, const Sector& second) {
  std::map<std::array<int64_t, 3>, int64_t> product;
  for (const auto& left : weight_keys(first)) {
    for (const auto& right : weight_keys(second)) {
      std::array<int64_t, 3> sum{left[0] + right[0], left[1] + right[1], left[2] + right[2]};
      ++product[sum];
    }
  }
  const std::array<int64_t, 3> rho{6, 0, -6};
  std::map<Sector, int64_t> found;
  for (const auto& [key, count] : product) {
    if (key[0] < key[1] || key[1] < key[2]) continue;
    int64_t total = 0;
    for (const auto& [permutation, parity] : permutations()) {
      const std::array<int64_t, 3> shifted{key[0] + rho[0] - rho[permutation[0]],
                                          key[1] + rho[1] - rho[permutation[1]],
                                          key[2] + rho[2] - rho[permutation[2]]};
      const auto hit = product.find(shifted);
      total += (parity ? -1 : 1) * (hit == product.end() ? 0 : hit->second);
    }
    if (total != 0) found[{(key[0] - key[1]) / 6, (key[1] - key[2]) / 6}] = total;
  }
  return found;
}

at::Tensor clebsch_copies(const Sector& first, const Sector& second, const Sector& third,
                          const c10::Device& device) {
  const std::map<Sector, int64_t> fused = fusion(first, second);
  TORCH_CHECK(fused.count(third) > 0, "The sectors do not fuse to the requested one.");
  const Irrep& left = irrep(first);
  const Irrep& right = irrep(second);
  const Irrep& conjugate = irrep(Sector{third.at(1), third.at(0)});
  const int64_t da = left.generators.size(2);
  const int64_t db = right.generators.size(2);
  const int64_t dc = conjugate.generators.size(2);
  std::vector<int64_t> zero;
  for (int64_t i = 0; i < da; ++i) {
    const std::array<int64_t, 3> first_key = weight_key(left.weights, i);
    for (int64_t j = 0; j < db; ++j) {
      const std::array<int64_t, 3> second_key = weight_key(right.weights, j);
      for (int64_t k = 0; k < dc; ++k) {
        const std::array<int64_t, 3> third_key = weight_key(conjugate.weights, k);
        if (first_key[0] + second_key[0] + third_key[0] == 0
            && first_key[1] + second_key[1] + third_key[1] == 0
            && first_key[2] + second_key[2] + third_key[2] == 0) {
          zero.push_back((i * db + j) * dc + k);
        }
      }
    }
  }
  const int64_t size = da * db * dc;
  const int64_t kept = static_cast<int64_t>(zero.size());
  TORCH_CHECK(kept > 0, "The coupling has no invariant vertex.");
  const at::Tensor index = at::tensor(zero, at::kLong);
  // The Casimir acts inside the subspace whose weights add up to zero, so it is applied to a basis
  // of that subspace instead of being written out on the full triple product, which is (da db dc)^2
  // entries and dominates both the time and the memory of a vertex.
  const auto action = [&](const at::Tensor& vector, const at::Tensor& generator, int64_t axis) {
    std::vector<int64_t> order{0, 1, 2, 3};
    order.erase(order.begin() + axis);
    order.insert(order.begin(), axis);
    std::vector<int64_t> back(order.size());
    for (size_t at = 0; at < order.size(); ++at) back.at(order.at(at)) = static_cast<int64_t>(at);
    const at::Tensor turned = vector.reshape({da, db, dc, kept}).permute(order);
    const int64_t width = turned.numel() / turned.size(0);
    return generator.matmul(turned.reshape({turned.size(0), width})).reshape(turned.sizes()).permute(back);
  };
  // The basis vectors of the weight-zero subspace as the columns of a (da db dc) x kept matrix.
  at::Tensor basis = at::zeros({size, kept}, at::kComplexDouble);
  basis.index_copy_(0, index, at::eye(kept, at::kComplexDouble));
  at::Tensor restricted = at::zeros({kept, kept}, at::kComplexDouble);
  for (int64_t a = 0; a < 8; ++a) {
    const at::Tensor total = action(basis, left.generators[a], 0)
        + action(basis, right.generators[a], 1)
        + action(basis, conjugate.generators[a], 2);
    const at::Tensor applied = action(total, left.generators[a], 0)
        + action(total, right.generators[a], 1) + action(total, conjugate.generators[a], 2);
    restricted = restricted + applied.reshape({size, kept}).index_select(0, index);
  }
  // A vertex is an eigenvector of the Casimir with eigenvalue zero; the others are its eigenvalues.
  auto [values, vectors] = at::linalg_eigh(restricted);
  const double largest = values.abs().max().item<double>();
  const at::Tensor kernel = (values.abs() < 1e-6 * std::max(1.0, largest)).nonzero().squeeze(-1);
  std::vector<at::Tensor> copies;
  for (int64_t copy = 0; copy < kernel.numel(); ++copy) {
    at::Tensor vector = at::zeros({size}, at::kComplexDouble);
    vector.index_copy_(0, index, vectors.select(1, kernel.select(0, copy).item<int64_t>()));
    at::Tensor tensor = vector.reshape({da, db, dc});
    tensor = tensor / tensor.norm();
    // the vertex keeps the normalization of the engine: the squared norm is the dimension of the
    // coupled sector
    tensor = tensor * std::sqrt(static_cast<double>(dimension(third)));
    const at::Tensor flat = tensor.reshape({-1});
    const int64_t head = (flat.abs() > kTolerance).nonzero().squeeze(-1).select(0, 0).item<int64_t>();
    // the representation basis is built on the host; the vertex goes to the caller's device
    copies.push_back((tensor * (flat[head].conj() / flat[head].abs())).to(device));
  }
  return at::stack(copies, 0);
}

at::Tensor clebsch(const Sector& first, const Sector& second, const Sector& third,
                   const c10::Device& device) {
  // a coupling the engine carries as a single vertex appears exactly once
  const at::Tensor copies = clebsch_copies(first, second, third, device);
  TORCH_CHECK(copies.size(0) == 1, "The coupling of these sectors couples ",
              copies.size(0), " times, not once.");
  return copies.select(0, 0);
}

at::Tensor metric(const Sector& sector, const c10::Device& device) {
  const at::Tensor form = clebsch(sector, Sector{sector.at(1), sector.at(0)}, Sector{0, 0}, device);
  TORCH_CHECK(form.defined(), "The sector has no invariant form.");
  const double scale = std::sqrt(static_cast<double>(dimension(sector)));
  return form.reshape({form.size(0), form.size(1)}) * scale;
}

// Drop the memo tables of the representation data.
void clear_caches() {
  irrep_cache().clear();
  weight_keys_cache().clear();
  weight_terms_cache().clear();
}

}  // namespace sym::su3

namespace sym {

SU3Symmetry::SU3Symmetry(c10::Device device) : Group("SU(3)", device) {}

std::vector<Outcome> SU3Symmetry::compute_outcomes(const Sector& a, const Sector& b) const {
  std::vector<Outcome> fused;
  for (const auto& [sector, multiplicity] : su3::fusion(a, b)) {
    fused.push_back({sector, static_cast<int>(multiplicity)});
  }
  return fused;
}

Sector SU3Symmetry::dual(Sector a) const { return {a.at(1), a.at(0)}; }

Sector SU3Symmetry::trivial() const { return {0, 0}; }

double SU3Symmetry::q_dim(Sector a) const { return static_cast<double>(su3::dimension(a)); }

// Every self-dual SU(3) sector has a symmetric invariant form in this gauge.
double SU3Symmetry::frobenius_schur_sign(const Sector&) const { return 1.0; }

c10::ScalarType SU3Symmetry::dtype() const { return at::kComplexDouble; }

at::Tensor SU3Symmetry::compute_clebsch_gordan(const Sector& a, const Sector& b, const Sector& c,
                                       int64_t copy) const {
  // The third factor of the invariant is the conjugate irrep in its own basis, so the metric of the
  // fused sector converts that axis into the basis of the leg it is contracted with.
  return at::einsum(
      "pq,abq->abp",
      {su3::metric(c, device()), su3::clebsch_copies(a, b, c, device()).select(0, copy)});
}

const std::shared_ptr<Group>& su3_symmetry() {
  static const std::shared_ptr<Group> instance = std::make_shared<SU3Symmetry>();
  return instance;
}

}  // namespace sym
