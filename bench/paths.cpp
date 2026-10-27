// How long an equation with several operands takes: the order the operands are contracted in decides
// how large the intermediate tensors are, so these equations are the ones a plan can shorten.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include "tentorch/linalg/graded/einsum.h"
#include "tentorch/symmetry/cat/su2/su2.h"
#include "tentorch/tensors/tensor.h"
#include "bench/timing.h"

using namespace sym;

// A leg of `multiplets` copies of the doublet, the bond of a matrix product state.
static Leg bond(const std::shared_ptr<const FusionCategory> &cat, int64_t multiplets)
{
  return Leg::from_blocks(cat, {{{0}, multiplets / 2}, {{2}, multiplets}}).first;
}

// Fill every block with a different value, so the benchmark moves data and not only indices.
static void fill(Tensor &tensor, double phase)
{
  int64_t at = 0;
  for (const auto &entry : tensor.blocks())
    entry.second.fill_(c10::complex<double>(std::cos(phase + 0.3 * at), std::sin(phase + 0.7 * ++at)));
}

int main(int argc, char **argv)
{
  double seconds = 1.0;
  for (int64_t at = 1; at < argc; ++at)
    if (std::string(argv[at]) == "--seconds" && at + 1 < argc) seconds = std::atof(argv[at + 1]);
  const auto cat = std::make_shared<const SU2Symmetry>(c10::Device(c10::DeviceType::CPU));
  const Leg physical = Leg::from_blocks(cat, {{{2}, 1}}).first;
  const Leg small = bond(cat, 4);

  // Four two-leg tensors whose written order multiplies two of them without a shared label, which is
  // the outer product a plan avoids.
  Tensor first(cat, {small, small}, {}, cat->trivial());
  Tensor second(cat, {small, small}, {}, cat->trivial());
  Tensor third(cat, {small, small}, {}, cat->trivial());
  Tensor fourth(cat, {small, small}, {}, cat->trivial());
  fill(first, 0.3);
  fill(second, 1.1);
  fill(third, 2.0);
  fill(fourth, 2.7);
  bench::run_for("einsum(\"AB,CD,AC,BD->\")", seconds, [&]
                 { (void)einsum("AB,CD,AC,BD->", {first, second, third, fourth}, cat->trivial()); });

  // A chain of four tensors that share a bond with the next one and carry a physical leg each.
  Tensor first_link(cat, {small, physical}, {small}, cat->trivial());
  Tensor second_link(cat, {small, physical}, {small}, cat->trivial());
  fill(first_link, 0.5);
  fill(second_link, 1.7);
  bench::run_for("einsum(\"ASb,BTc,CUd,DVe->astuve\")", seconds, [&] {
    (void)einsum("ASb,BTc,CUd,DVe->astuve",
                 {first_link, second_link, first_link, second_link}, cat->trivial());
  });

  // Three tensors whose labels form a closed loop, so the order decides how many legs stay open.
  bench::run_for("einsum(\"AB,BC,CA->\")", seconds, [&] {
    (void)einsum("AB,BC,CA->", {first, second, first}, cat->trivial());
  });
  return 0;
}
