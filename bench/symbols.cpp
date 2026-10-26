// What the category layer costs with its caches warm: the associator, the braiding, a vertex, and
// the SU(3) fusion every block key of a tensor goes through.
#include <cstdio>
#include <memory>

#include "../tentorch/symmetry/cat/su2/su2.h"
#include "../tentorch/symmetry/cat/su3/su3.h"
#include "timing.h"

using namespace sym;

int main()
{
  const auto cpu = at::Device(at::DeviceType::CPU);
  const auto su2 = std::make_shared<SU2Symmetry>(cpu);
  const auto su3 = std::make_shared<SU3Symmetry>(cpu);
  const Sector half{1};
  const Sector one{2};
  const Sector octet{1, 1};

  std::printf("SU(2)\n");
  bench::repeat("F(half, half, half -> 0)", 4000, [&] { (void) su2->F_symbol(half, half, half, su2->trivial()); });
  bench::repeat("F(half, half, half -> one)", 4000, [&] { (void) su2->F_symbol(half, half, half, one); });
  bench::repeat("R(half, half)", 4000, [&] { (void) su2->R_symbol(half, half); });
  bench::repeat("CG(half, half -> 0, copy 0)", 4000,
                [&] { (void) su2->clebsch_gordan(half, half, su2->trivial(), 0); });

  std::printf("SU(3)\n");
  bench::repeat("F(8, 8, 8 -> 1)", 400, [&] { (void) su3->F_symbol(octet, octet, octet, Sector{1, 1}); });
  bench::repeat("F(8, 8, 8 -> 8)", 200, [&] { (void) su3->F_symbol(octet, octet, octet, octet); });
  bench::repeat("R(8, 8)", 400, [&] { (void) su3->R_symbol(octet, octet); });
  bench::repeat("CG(8, 8 -> 0, copy 0)", 200,
                [&] { (void) su3->clebsch_gordan(octet, octet, su3->trivial(), 0); });

  // A tensor's block key asks for N(c, dual(c'), q), so the fusion of two large labels is on the
  // hot path of every construction, not only of a contraction.
  std::printf("SU(3) fusion\n");
  bench::repeat("n_symbol((4, 4), (1, 1), (0, 0))", 200,
                [&] { (void) su3->n_symbol(Sector{4, 4}, octet, su3->trivial()); });
  bench::repeat("n_symbol((2, 5), (1, 1), (0, 0))", 200,
                [&] { (void) su3->n_symbol(Sector{2, 5}, octet, su3->trivial()); });
  bench::repeat("n_symbol((6, 6), (1, 1), (0, 0))", 50,
                [&] { (void) su3->n_symbol(Sector{6, 6}, octet, su3->trivial()); });
  bench::repeat("outcomes((4, 4), (1, 1))", 200, [&] { (void) su3->outcomes(Sector{4, 4}, octet); });
  bench::repeat("outcomes((6, 6), (6, 6))", 50, [&] { (void) su3->outcomes(Sector{6, 6}, Sector{6, 6}); });
  return 0;
}
