// Clebsch-Gordan coefficients of SU(2), labelled by twice the spin: the compiled core of
// mpsdmrg/symmetry/data/su2data.py.
//
// A spin pair (j1, j2) is built level by level, from M = j1 + j2 down to -(j1 + j2) in
// steps of two.  The states already at a level are lowered by J_-, and the one state
// missing there is the part of the highest-weight state orthogonal to them.  The level-M
// state of total spin J is the vector over m1 of <j1 m1; j2 (M - m1) | J M>, so
// X[k1, k2, k3] = <j1 m1; j2 m2 | j3 m3> with m = 2 k - j.  The overlap of the two
// coupling schemes of three spins contracts those tensors on the host.
//
// Build and load through mpsdmrg/symmetry/data/su2data.py (torch.utils.cpp_extension.load).
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <tuple>
#include <vector>

namespace {

// X of one pair (j1, j2) at fixed j3, in row-major (k1, k2, k3), and its flat index.
int64_t entries(int64_t j1, int64_t j2, int64_t j3) {
  return (j1 + 1) * (j2 + 1) * (j3 + 1);
}

int64_t at(int64_t j2, int64_t j3, int64_t k1, int64_t k2, int64_t k3) {
  return (k1 * (j2 + 1) + k2) * (j3 + 1) + k3;
}

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

// {j3: X} of one pair (j1, j2), for every allowed j3.
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

// Single-component slice of X at its highest m3, in shape (j1 + 1, j2 + 1).
torch::Tensor highest(const std::map<int64_t, torch::Tensor>& table, int64_t j1, int64_t j2,
                      int64_t j4) {
  return table.at(j4).slice(2, j4, j4 + 1).squeeze(2);
}

std::map<int64_t, torch::Tensor> table(int64_t j1, int64_t j2) {
  std::map<int64_t, torch::Tensor> out;
  for (const auto& [j3, values] : coefficients(j1, j2)) {
    torch::Tensor tensor = torch::empty({j1 + 1, j2 + 1, j3 + 1}, torch::kFloat64);
    std::memcpy(tensor.data_ptr<double>(), values.data(), values.size() * sizeof(double));
    out[j3] = tensor;
  }
  return out;
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>, torch::Tensor> recoupling(
    int64_t j1, int64_t j2, int64_t j3, int64_t j4) {
  const std::map<int64_t, torch::Tensor> pair12 = table(j1, j2);
  const std::map<int64_t, torch::Tensor> pair23 = table(j2, j3);
  const std::map<int64_t, torch::Tensor> fused34 = table(j3, j4);
  const std::map<int64_t, torch::Tensor> fused14 = table(j1, j4);
  std::vector<int64_t> rows, cols;
  for (const auto& [j12, unused] : pair12) if (fused34.count(j12)) rows.push_back(j12);
  for (const auto& [j23, unused] : pair23) if (fused14.count(j23)) cols.push_back(j23);
  // Both states carry the total spin j4, so the overlap is the same at every total
  // component; the highest one keeps the sum short.
  std::vector<torch::Tensor> right_totals;
  for (int64_t j23 : cols) right_totals.push_back(highest(table(j1, j23), j1, j23, j4));
  std::vector<double> values;
  values.reserve(rows.size() * cols.size());
  for (int64_t j12 : rows) {
    const torch::Tensor left_total = highest(table(j12, j3), j12, j3, j4);
    for (size_t k = 0; k < cols.size(); ++k) {
      values.push_back(torch::einsum("abe,ec,bcf,af->",
                                     {pair12.at(j12), left_total, pair23.at(cols[k]),
                                      right_totals[k]}).item<double>());
    }
  }
  torch::Tensor f = torch::from_blob(values.data(),
                                     {static_cast<int64_t>(rows.size()),
                                      static_cast<int64_t>(cols.size())},
                                     torch::kFloat64).clone();
  return {rows, cols, f};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("table", &table, "Clebsch-Gordan tensors of one spin pair, by fused spin");
  m.def("recoupling", &recoupling, "overlap of the two coupling schemes of three spins");
}
