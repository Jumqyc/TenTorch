// The SU(2) group, with its sectors labelled by twice the spin.
#include "su2.h"

#include <map>
#include <string>
#include <vector>

namespace sym {

namespace {

std::string half_label(int64_t jj) {
  return std::to_string(jj / 2) + (jj % 2 == 0 ? ".0" : ".5");
}

}  // namespace

SU2Symmetry::SU2Symmetry()
    : Group("SU2Symmetry", FusionStyle::multiple_unique, BraidingStyle::bosonic, {0}, "SU(2)") {}

bool SU2Symmetry::is_valid_sector(const Sector& a) const {
  return a.size() == 1 && a[0] >= 0;
}

Sector SU2Symmetry::dual_sector(const Sector& a) const {
  return a;
}

std::vector<Sector> SU2Symmetry::fusion_outcomes(const Sector& a, const Sector& b) const {
  std::vector<Sector> outcomes;
  for (int64_t jj = std::abs(a.at(0) - b.at(0)); jj <= a.at(0) + b.at(0); jj += 2) {
    outcomes.push_back({jj});
  }
  return outcomes;
}

int64_t SU2Symmetry::sector_dim(const Sector& a) const {
  return a.at(0) + 1;
}

at::Tensor SU2Symmetry::fusion_tensor(const Sector& a, const Sector& b, const Sector& c,
                                      const c10::Device& device, int64_t) const {
  const std::vector<Sector> outcomes = fusion_outcomes(a, b);
  if (std::find(outcomes.begin(), outcomes.end(), c) == outcomes.end()) {
    throw value_error("Spins " + half_label(a.at(0)) + " and " + half_label(b.at(0)) +
                      " do not fuse to " + half_label(c.at(0)) + ".");
  }
  return su2::fusion_tensor(a.at(0), b.at(0), c.at(0), device);
}

bool SU2Symmetry::is_same_symmetry(const FusionCategory& other) const {
  return other.type_name() == "SU2Symmetry";
}

double SU2Symmetry::dual_sign(const Sector& sector) const {
  return dual_sector(sector) == sector && (sector_dim(sector) - 1) % 2 != 0 ? -1.0 : 1.0;
}

double SU2Symmetry::dual_phase(const Sector& sector) const {
  // the phase the conjugate component carries: (-1)^spin
  return (sector.at(0) / 2) % 2 == 0 ? 1.0 : -1.0;
}

const std::shared_ptr<Group>& su2_symmetry() {
  static const std::shared_ptr<Group> instance = std::make_shared<SU2Symmetry>();
  return instance;
}


}  // namespace sym

namespace sym {

int64_t SU2Symmetry::reference_index(const Sector& a) const {
  return a.at(0) / 2;
}

c10::ScalarType SU2Symmetry::fusion_dtype() const {
  return at::kDouble;
}

void SU2Symmetry::clear_caches() const {
  su2::clear_tables();
}

at::Tensor SU2Symmetry::invariant_form(const Sector& a, const c10::Device& device) const {
  // the invariant pairing of two multiplets is the same flip and phase as the dual leg
  return duality_matrix(a, device);
}

at::Tensor SU2Symmetry::duality_matrix(const Sector& a, const c10::Device& device) const {
  const int64_t spin = a.at(0);
  const int64_t dim = spin + 1;
  const auto options = at::TensorOptions().dtype(at::kDouble).device(device);
  at::Tensor phase = at::ones({dim}, options);
  phase.slice(0, 1, dim, 2).fill_(-1.0);
  const double sign = spin % 2 == 0 ? 1.0 : -1.0;
  return (phase * sign).unsqueeze(1) * at::flip(at::eye(dim, options), {0});
}

void bind_su2(py::module_& m) {
  py::class_<sym::SU2Symmetry, sym::Group, std::shared_ptr<sym::SU2Symmetry>>(
      m, "SU2Symmetry")
      .def(py::init<>());
  m.attr("su2_symmetry") = py::cast(sym::su2_symmetry());
}

}  // namespace sym

// The Clebsch-Gordan core of SU(2); cg.h states the construction and the API.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

namespace sym {
namespace su2 {

namespace {

using Levels = std::map<int64_t, std::vector<double>>;

// Unnormalised level-MM vector, from the level MM + 2 components of the same total spin.
std::vector<double> lower(const std::vector<double>& vec, int64_t j1, int64_t j2, int64_t MM) {
  const int64_t n1 = j1 + 1;
  std::vector<double> out(n1, 0.0);
  for (int64_t k1 = 0; k1 < n1; ++k1) {
    const int64_t m1 = 2 * k1 - j1;
    const int64_t m2 = MM + 2 - m1;
    if (std::abs(m2) > j2) continue;
    if (std::abs(m2 - 2) <= j2) {
      out[k1] += vec[k1] * std::sqrt((j2 + m2) * (j2 - m2 + 2)) / 2;
    }
    if (k1 > 0 && std::abs(m1 - 2) <= j1) {
      out[k1 - 1] += vec[k1] * std::sqrt((j1 + m1) * (j1 - m1 + 2)) / 2;
    }
  }
  return out;
}

}  // namespace

int64_t entries(int64_t j1, int64_t j2, int64_t j3) {
  return (j1 + 1) * (j2 + 1) * (j3 + 1);
}

int64_t at(int64_t j2, int64_t j3, int64_t k1, int64_t k2, int64_t k3) {
  return (k1 * (j2 + 1) + k2) * (j3 + 1) + k3;
}

std::map<int64_t, std::vector<double>> coefficients(int64_t j1, int64_t j2) {
  const int64_t lo = std::abs(j1 - j2), hi = j1 + j2, n1 = j1 + 1;
  std::map<int64_t, std::vector<double>> out;
  out[hi] = std::vector<double>(entries(j1, j2, hi), 0.0);
  out[hi][at(j2, hi, j1, j2, hi)] = 1.0;
  Levels previous;
  previous[hi] = std::vector<double>(n1, 0.0);
  previous[hi][j1] = 1.0;
  for (int64_t MM = hi - 2; MM >= -hi; MM -= 2) {
    Levels level;
    for (auto it = previous.rbegin(); it != previous.rend(); ++it) {
      if (it->first < std::abs(MM)) continue;
      const double scale = std::sqrt((it->first + MM + 2) * (it->first - MM)) / 2;
      std::vector<double> vec = lower(it->second, j1, j2, MM);
      for (double& value : vec) value /= scale;
      level[it->first] = vec;
    }
    if (MM >= lo && MM >= 0) {
      // The state missing at this level is the part of the highest-weight state orthogonal
      // to the states already there, with a positive m1 = j1 component.
      std::vector<double> vec(n1, 0.0);
      vec[j1] = 1.0;
      for (auto it = level.rbegin(); it != level.rend(); ++it) {
        double dot = 0.0;
        for (int64_t k1 = 0; k1 < n1; ++k1) dot += it->second[k1] * vec[k1];
        for (int64_t k1 = 0; k1 < n1; ++k1) vec[k1] -= dot * it->second[k1];
      }
      double norm = 0.0;
      for (double value : vec) norm += value * value;
      norm = std::sqrt(norm);
      TORCH_CHECK(norm > 1e-12, "Degenerate orthogonalization for j1=", j1, ", j2=", j2,
                  ", J=", MM, ".");
      for (double& value : vec) value /= norm;
      level[MM] = vec;
    }
    previous = level;
    for (const auto& [JJ, vec] : level) {
      if (out.find(JJ) == out.end()) out[JJ] = std::vector<double>(entries(j1, j2, JJ), 0.0);
      std::vector<double>& tensor = out[JJ];
      for (int64_t k1 = 0; k1 < n1; ++k1) {
        const int64_t m1 = 2 * k1 - j1;
        const int64_t m2 = MM - m1;
        if (std::abs(m2) <= j2 && vec[k1] != 0.0) {
          tensor[at(j2, JJ, k1, (m2 + j2) / 2, (MM + JJ) / 2)] += vec[k1];
        }
      }
    }
  }
  return out;
}

std::map<int64_t, at::Tensor> table(int64_t j1, int64_t j2) {
  std::map<int64_t, at::Tensor> out;
  for (const auto& [j3, values] : coefficients(j1, j2)) {
    at::Tensor tensor = torch::empty({j1 + 1, j2 + 1, j3 + 1}, torch::kFloat64);
    std::memcpy(tensor.data_ptr<double>(), values.data(), values.size() * sizeof(double));
    out[j3] = tensor;
  }
  return out;
}

namespace {

// The tables are read-only and recur in every contraction, so they are kept per device.
std::map<std::tuple<int64_t, int64_t, std::string>, std::map<int64_t, at::Tensor>> tables;

}  // namespace

std::map<int64_t, at::Tensor> fusion_table(int64_t j1, int64_t j2, const torch::Device& device) {
  const auto key = std::make_tuple(j1, j2, device.str());
  auto kept = tables.find(key);
  if (kept == tables.end()) {
    std::map<int64_t, at::Tensor> moved;
    for (auto& [j3, tensor] : table(j1, j2)) moved.emplace(j3, tensor.to(device));
    kept = tables.emplace(key, std::move(moved)).first;
  }
  return kept->second;
}

at::Tensor fusion_tensor(int64_t j1, int64_t j2, int64_t j3, const torch::Device& device) {
  const std::map<int64_t, at::Tensor>& kept = fusion_table(j1, j2, device);
  const auto found = kept.find(j3);
  TORCH_CHECK(found != kept.end(), "Spins ", j1, " and ", j2, " do not fuse to ", j3, ".");
  return found->second;
}

void clear_tables() {
  tables.clear();
}

}  // namespace su2
}  // namespace sym
