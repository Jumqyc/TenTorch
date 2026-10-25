#pragma once

// The Clebsch-Gordan core of SU(2), labelled by twice the spin: the tables `SU2Symmetry`
// reads to build its fusion tensor, kept out of the symmetry class itself.
//
// A spin pair (j1, j2) is built level by level, from M = j1 + j2 down to -(j1 + j2) in
// steps of two.  The states already at a level are lowered by J_-, and the one state
// missing there is the part of the highest-weight state orthogonal to them.  The level-M
// state of total spin J is the vector over m1 of <j1 m1; j2 (M - m1) | J M>, so
// X[k1, k2, k3] = <j1 m1; j2 m2 | j3 m3> with m = 2 k - j.  The overlap of the two
// coupling schemes of three spins contracts those tensors on the host.

#include <ATen/ATen.h>
#include <cstdint>
#include <map>
#include <tuple>
#include <vector>

namespace sym {
namespace su2 {

// X of one pair (j1, j2) at fixed j3, in row-major (k1, k2, k3), and its flat index.
int64_t entries(int64_t j1, int64_t j2, int64_t j3);
int64_t at(int64_t j2, int64_t j3, int64_t k1, int64_t k2, int64_t k3);

// {j3: X} of one pair (j1, j2), for every allowed j3.
std::map<int64_t, std::vector<double>> coefficients(int64_t j1, int64_t j2);

// {j3: X} of one pair (j1, j2) as float64 host tensors of shape (j1 + 1, j2 + 1, j3 + 1).
std::map<int64_t, at::Tensor> table(int64_t j1, int64_t j2);

// X of one vertex, kept per (j1, j2, device) so the host tables are built once.
// Args:
//   j1, j2, j3: twice the fused spins, j3 an outcome of fusing j1 and j2.
//   device: device of the returned tensor.
at::Tensor fusion_tensor(int64_t j1, int64_t j2, int64_t j3, const c10::Device& device);

// {j3: X} of one pair, moved to the device and kept with it; read-only.
std::map<int64_t, at::Tensor> fusion_table(int64_t j1, int64_t j2, const c10::Device& device);

// Drop every kept table.
void clear_tables();

}  // namespace su2
}  // namespace sym
