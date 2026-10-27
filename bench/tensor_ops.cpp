// What a leg move costs: the exchange a permutation is built from, the permutation, and the bend
// that follows it, on a tensor with one block and on one with several copies per leg.
#include <cstdio>
#include <vector>

#include "../tentorch/symmetry/cat/su2/su2.h"
#include "../tentorch/tensors/tensor.h"
#include "timing.h"

using namespace sym;

int main()
{
  const auto cpu = at::Device(at::DeviceType::CPU);
  const auto su2 = std::make_shared<SU2Symmetry>(cpu);
  const Leg half = Leg::from_blocks(su2, {{{1}, 1}}).first;
  const Leg two_copies = Leg::from_blocks(su2, {{{1}, 2}}).first;

  const std::vector<Leg> six(6, half);
  const Tensor tensor(su2, six, {}, su2->trivial());
  std::printf("six half legs: %zu blocks, %lld rows\n", tensor.blocks().size(),
              static_cast<long long>(tensor.rows_of(su2->trivial()).rows()));
  bench::repeat("place(1,0,2,3,4,5)", 500, [&] { (void) tensor.place({1, 0, 2, 3, 4, 5}); });
  bench::repeat("place(0,1,3,2,4,5)", 500, [&] { (void) tensor.place({0, 1, 3, 2, 4, 5}); });
  bench::repeat("place(1,2,3,4,5,0)", 300, [&] { (void) tensor.place({1, 2, 3, 4, 5, 0}); });
  bench::repeat("place(3,4,5,0,1,2)", 200, [&] { (void) tensor.place({3, 4, 5, 0, 1, 2}); });
  bench::repeat("place({1..5}, {0})", 200, [&] { (void) tensor.place({1, 2, 3, 4, 5}, {0}); });
  bench::repeat("place({0..4}, {5})", 300, [&] { (void) tensor.place({0, 1, 2, 3, 4}, {5}); });
  bench::repeat("place(0,1,2)", 200, [&] { (void) tensor.place({0, 1, 2}); });

  // Three doublets cannot fuse to the trivial charge, so the total charge is the half-integer one.
  const std::vector<Leg> copies(3, two_copies);
  const Tensor multi(su2, copies, {}, Sector{1});
  std::printf("three two-copy legs: %zu blocks\n", multi.blocks().size());
  bench::repeat("multi place(1,0,2)", 300, [&] { (void) multi.place({1, 0, 2}); });
  bench::repeat("multi place(2,1,0)", 200, [&] { (void) multi.place({2, 1, 0}); });
  return 0;
}
