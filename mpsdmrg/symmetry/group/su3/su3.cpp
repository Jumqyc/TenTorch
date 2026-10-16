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

// The generators and weights of the sector in its weight basis, memoised.
const Irrep& irrep(const Sector& sector) {
  static std::map<Sector, Irrep> cache;
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

}  // namespace

int64_t dimension(const Sector& sector) {
  const int64_t p = sector.at(0);
  const int64_t q = sector.at(1);
  return (p + 1) * (q + 1) * (p + q + 2) / 2;
}

double casimir(const Sector& sector) {
  const double p = static_cast<double>(sector.at(0));
  const double q = static_cast<double>(sector.at(1));
  return 4.0 / 3.0 * (p * p + q * q + p * q + 3.0 * p + 3.0 * q);
}

const std::vector<std::array<int64_t, 3>>& weight_keys(const Sector& sector) {
  static std::map<Sector, std::vector<std::array<int64_t, 3>>> cache;
  const auto found = cache.find(sector);
  if (found != cache.end()) return found->second;
  const Irrep& data = irrep(sector);
  std::vector<std::array<int64_t, 3>> keys;
  for (int64_t k = 0; k < data.weights.size(0); ++k) keys.push_back(weight_key(data.weights, k));
  return cache.emplace(sector, keys).first->second;
}

std::map<Sector, int64_t> fusion(const Sector& first, const Sector& second) {
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
  const auto outcome = fusion(first, second).find(third);
  if (outcome == fusion(first, second).end()) {
    throw value_error("The sectors do not fuse to the requested one.");
  }
  const Irrep& left = irrep(first);
  const Irrep& right = irrep(second);
  const Irrep& conjugate = irrep(Sector{third.at(1), third.at(0)});
  const int64_t da = left.generators.size(2);
  const int64_t db = right.generators.size(2);
  const int64_t dc = conjugate.generators.size(2);
  const at::Tensor ia = at::eye(da, at::kComplexDouble);
  const at::Tensor ib = at::eye(db, at::kComplexDouble);
  const at::Tensor ic = at::eye(dc, at::kComplexDouble);
  const int64_t size = da * db * dc;
  at::Tensor casimir = at::zeros({size, size}, at::kComplexDouble);
  for (int64_t a = 0; a < 8; ++a) {
    const at::Tensor total = at::kron(at::kron(left.generators[a], ib), ic)
        + at::kron(at::kron(ia, right.generators[a]), ic)
        + at::kron(at::kron(ia, ib), conjugate.generators[a]);
    casimir = casimir + total.matmul(total);
  }
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
  const at::Tensor index = at::tensor(zero, at::kLong);
  const at::Tensor restricted = casimir.index_select(0, index).index_select(1, index);
  auto [values, vectors] = at::linalg_eigh(restricted);
  const at::Tensor kernel = (values.abs() < 1e-6).nonzero().squeeze(-1);
  if (kernel.numel() == 0) throw value_error("The coupling has no invariant vertex.");
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
  if (copies.size(0) != 1) {
    throw not_implemented_error("The coupling of these sectors couples more than once.");
  }
  return copies.select(0, 0);
}

at::Tensor metric(const Sector& sector, const c10::Device& device) {
  const at::Tensor form = clebsch(sector, Sector{sector.at(1), sector.at(0)}, Sector{0, 0}, device);
  if (!form.defined()) throw not_implemented_error("The sector has no invariant form.");
  const double scale = std::sqrt(static_cast<double>(dimension(sector)));
  return form.reshape({form.size(0), form.size(1)}) * scale;
}

}  // namespace sym::su3

namespace sym {

namespace {

// The Dynkin labels of a sector, the way an error message prints them.
std::string labels(const Sector& a) {
  return "(" + std::to_string(a.at(0)) + ", " + std::to_string(a.at(1)) + ")";
}

}  // namespace

SU3Symmetry::SU3Symmetry()
    : Group("SU3Symmetry", FusionStyle::general, BraidingStyle::bosonic, {0, 0}, "SU(3)",
            std::numeric_limits<double>::infinity()) {}

bool SU3Symmetry::is_valid_sector(const Sector& a) const {
  return a.size() == 2 && a[0] >= 0 && a[1] >= 0;
}

Sector SU3Symmetry::dual_sector(const Sector& a) const { return {a.at(1), a.at(0)}; }

std::vector<Sector> SU3Symmetry::fusion_outcomes(const Sector& a, const Sector& b) const {
  std::vector<Sector> outcomes;
  for (const auto& [sector, multiplicity] : su3::fusion(a, b)) {
    outcomes.push_back(sector);
  }
  return outcomes;
}

int64_t SU3Symmetry::sector_dim(const Sector& a) const { return su3::dimension(a); }

at::Tensor SU3Symmetry::fusion_tensor(const Sector& a, const Sector& b, const Sector& c,
                                      const c10::Device& device, int64_t copy) const {
  const std::map<Sector, int64_t> outcomes = su3::fusion(a, b);
  const auto found = outcomes.find(c);
  if (found == outcomes.end()) {
    throw value_error("Sectors " + labels(a) + " and " + labels(b) + " do not fuse to " + labels(c) + ".");
  }
  if (copy < 0 || copy >= found->second) {
    throw value_error("The fusion of " + labels(a) + " and " + labels(b) + " to " + labels(c) +
                      " couples " + std::to_string(found->second) + " times, not " +
                      std::to_string(copy + 1) + ".");
  }
  // the third factor of the invariant is the conjugate irrep in its own basis, so the metric of
  // the fused sector converts that axis into the basis of the leg it is contracted with
  return at::einsum("pq,abq->abp",
                    {su3::metric(c, device), su3::clebsch_copies(a, b, c, device).select(0, copy)});
}

int64_t SU3Symmetry::reference_index(const Sector&) const {
  return 0;
}

c10::ScalarType SU3Symmetry::fusion_dtype() const {
  return at::kComplexDouble;
}

int64_t SU3Symmetry::n_symbol(const Sector& a, const Sector& b, const Sector& c) const {
  const std::map<Sector, int64_t> outcomes = su3::fusion(a, b);
  const auto found = outcomes.find(c);
  return found == outcomes.end() ? 0 : found->second;
}

at::Tensor SU3Symmetry::duality_matrix(const Sector& a, const c10::Device& device) const {
  // the weight basis of an SU(3) irrep runs from the highest weight, so the basis of the dual
  // leg is the reversed one, and the invariant form of an irrep with an odd label sum is the
  // negative of the element-wise conjugate: one matrix carries both
  const double sign = (a.at(0) + a.at(1)) % 2 == 0 ? 1.0 : -1.0;
  return su3::metric(a, device) * sign;
}

at::Tensor SU3Symmetry::invariant_form(const Sector& a, const c10::Device& device) const {
  return su3::metric(a, device);
}

double SU3Symmetry::dual_sign(const Sector&) const {
  // the sign of the conjugation lives in duality_matrix, so no extra factor enters here
  return 1.0;
}

double SU3Symmetry::dual_phase(const Sector& sector) const {
  // the phase the conjugate component carries: (-1)^(p + q)
  return (sector.at(0) + sector.at(1)) % 2 == 0 ? 1.0 : -1.0;
}

Sector SU3Symmetry::sector_of_dimension(int64_t dim) const {
  for (int64_t q = 0; q <= dim; ++q) {
    for (int64_t p = 0; p <= dim; ++p) {
      if (su3::dimension({p, q}) == dim) {
        return {p, q};
      }
    }
  }
  throw value_error("No SU(3) sector has dimension " + std::to_string(dim) + ".");
}

bool SU3Symmetry::is_same_symmetry(const Group& other) const {
  return other.type_name() == "SU3Symmetry";
}

const std::shared_ptr<Group>& su3_symmetry() {
  static const std::shared_ptr<Group> instance = std::make_shared<SU3Symmetry>();
  return instance;
}

void bind_su3(py::module_& m) {
  py::class_<sym::SU3Symmetry, sym::Group, std::shared_ptr<sym::SU3Symmetry>>(
      m, "SU3Symmetry")
      .def(py::init<>());
  m.attr("su3_symmetry") = py::cast(sym::su3_symmetry());
}

}  // namespace sym
