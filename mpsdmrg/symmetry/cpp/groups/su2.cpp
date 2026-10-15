// The SU(2) group, with its sectors labelled by twice the spin.
#include "su2.h"

#include <map>
#include <string>
#include <vector>

namespace mpsdmrg {

namespace {

std::string half_label(int64_t jj) {
  return std::to_string(jj / 2) + (jj % 2 == 0 ? ".0" : ".5");
}

}  // namespace

SU2Symmetry::SU2Symmetry()
    : Group("SU2Symmetry", FusionStyle::multiple_unique, BraidingStyle::bosonic, {0}, "SU(2)",
            std::numeric_limits<double>::infinity()) {}

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
                                      const c10::Device& device) const {
  const std::vector<Sector> outcomes = fusion_outcomes(a, b);
  if (std::find(outcomes.begin(), outcomes.end(), c) == outcomes.end()) {
    throw value_error("Spins " + half_label(a.at(0)) + " and " + half_label(b.at(0)) +
                      " do not fuse to " + half_label(c.at(0)) + ".");
  }
  return su2::fusion_tensor(a.at(0), b.at(0), c.at(0), device);
}

bool SU2Symmetry::is_same_symmetry(const Group& other) const {
  return other.type_name() == "SU2Symmetry";
}

Sector SU2Symmetry::sector_of_dimension(int64_t dim) const {
  if (dim < 1) {
    throw value_error("A spin multiplet has a positive dimension, not " + std::to_string(dim) +
                      ".");
  }
  // the spin-j multiplet of dimension 2j + 1 is stored as jj = 2j
  return Sector{dim - 1};
}
const std::shared_ptr<Group>& su2_symmetry() {
  static const std::shared_ptr<Group> instance = std::make_shared<SU2Symmetry>();
  return instance;
}


}  // namespace mpsdmrg

namespace mpsdmrg {

int64_t SU2Symmetry::reference_index(const Sector& a) const {
  return a.at(0) / 2;
}

c10::ScalarType SU2Symmetry::fusion_dtype() const {
  return at::kDouble;
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
  m.def("fusion_tensor", &mpsdmrg::su2::fusion_tensor, "Clebsch-Gordan tensor of one SU(2) vertex",
        py::arg("j1"), py::arg("j2"), py::arg("j3"), py::arg("device"));
  m.def("fusion_table", &mpsdmrg::su2::fusion_table, "Clebsch-Gordan tensors of one SU(2) pair",
        py::arg("j1"), py::arg("j2"), py::arg("device"));
  m.def("recoupling", &mpsdmrg::su2::recoupling,
        "the overlap of the two coupling schemes of three SU(2) spins",
        py::arg("j1"), py::arg("j2"), py::arg("j3"), py::arg("j4"));
  py::class_<mpsdmrg::SU2Symmetry, mpsdmrg::Group, std::shared_ptr<mpsdmrg::SU2Symmetry>>(
      m, "SU2Symmetry")
      .def(py::init<>());
  m.attr("su2_symmetry") = py::cast(mpsdmrg::su2_symmetry());
}

}  // namespace mpsdmrg

// The Clebsch-Gordan core of SU(2); cg.h states the construction and the API.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

namespace mpsdmrg {
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

at::Tensor highest(const std::map<int64_t, at::Tensor>& table, int64_t j1, int64_t j2,
                      int64_t j4) {
  return table.at(j4).slice(2, j4, j4 + 1).squeeze(2);
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

std::tuple<std::vector<int64_t>, std::vector<int64_t>, at::Tensor> recoupling(
    int64_t j1, int64_t j2, int64_t j3, int64_t j4) {
  const std::map<int64_t, at::Tensor> pair12 = table(j1, j2);
  const std::map<int64_t, at::Tensor> pair23 = table(j2, j3);
  const std::map<int64_t, at::Tensor> fused34 = table(j3, j4);
  const std::map<int64_t, at::Tensor> fused14 = table(j1, j4);
  std::vector<int64_t> rows, cols;
  for (const auto& [j12, unused] : pair12) if (fused34.count(j12)) rows.push_back(j12);
  for (const auto& [j23, unused] : pair23) if (fused14.count(j23)) cols.push_back(j23);
  // Both states carry the total spin j4, so the overlap is the same at every total
  // component; the highest one keeps the sum short.
  std::vector<at::Tensor> right_totals;
  for (int64_t j23 : cols) right_totals.push_back(highest(table(j1, j23), j1, j23, j4));
  std::vector<double> values;
  values.reserve(rows.size() * cols.size());
  for (int64_t j12 : rows) {
    const at::Tensor left_total = highest(table(j12, j3), j12, j3, j4);
    for (size_t k = 0; k < cols.size(); ++k) {
      values.push_back(torch::einsum("abe,ec,bcf,af->",
                                     {pair12.at(j12), left_total, pair23.at(cols[k]),
                                      right_totals[k]}).item<double>());
    }
  }
  at::Tensor f = torch::from_blob(values.data(),
                                     {static_cast<int64_t>(rows.size()),
                                      static_cast<int64_t>(cols.size())},
                                     torch::kFloat64).clone();
  return {rows, cols, f};
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
}  // namespace mpsdmrg
