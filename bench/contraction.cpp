// The contractions a sweep issues, at three bond dimensions: a bond contraction, an environment
// update of three operands, and the elementary contraction of two blocks over two bond legs.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "../tentorch/linalg/graded/contract.h"
#include "../tentorch/linalg/graded/einsum.h"
#include "../tentorch/symmetry/cat/su2/su2.h"
#include "../tentorch/tensors/tensor.h"
#include "timing.h"

using namespace sym;

namespace
{

  /// A bond leg of `multiplets` triplet copies and half as many trivial ones.
  Leg bond(const std::shared_ptr<const FusionCategory> &cat, int64_t multiplets)
  {
    return Leg::from_blocks(cat, {{{0}, multiplets / 2}, {{2}, multiplets}}).first;
  }

  /// A spin one leg, one copy, which keeps every total charge of a tensor even.
  Leg spin(const std::shared_ptr<const FusionCategory> &cat)
  {
    return Leg::from_blocks(cat, {{{2}, 1}}).first;
  }

  /// Fills every block with a reproducible constant, so a product is not a product of zeros.
  void fill(Tensor &tensor, double phase)
  {
    int64_t at = 0;
    for (const BlockKey &key : tensor.keys())
      tensor.block(key.codomain, key.domain, key.copy)
          .fill_(c10::complex<double>(std::cos(phase + static_cast<double>(at)),
                                      std::sin(phase + 3.0 * static_cast<double>(at++))));
  }

  /// Prints the shape of a tensor the way `render` does, without the tree details.
  void describe(const char *name, const Tensor &tensor)
  {
    std::printf("  %-22s %lld cod, %lld dom, %zu blocks\n", name,
                static_cast<long long>(tensor.codomain().size()),
                static_cast<long long>(tensor.domain().size()), tensor.blocks().size());
  }

} // namespace

int main(int argc, char **argv)
{
  double seconds = 1.0;
  for (int at = 1; at < argc; ++at)
    if (std::string(argv[at]) == "--seconds" && at + 1 < argc)
      seconds = std::atof(argv[++at]);

  const auto cpu = at::Device(at::DeviceType::CPU);
  const auto su2 = std::make_shared<SU2Symmetry>(cpu);
  const Sector neutral = su2->trivial();
  const Leg half = spin(su2);

  for (const int64_t multiplets : {2, 4, 8})
  {
    const Leg leg = bond(su2, multiplets);
    std::printf("bond multiplets %lld\n", static_cast<long long>(multiplets));

    // A bond contraction: (left bond, spin) x (spin, right bond), summed over the middle bond.
    {
      Tensor left(su2, {leg, half}, {leg}, neutral);
      Tensor right(su2, {leg, leg}, {leg}, neutral);
      fill(left, 0.3);
      fill(right, 1.1);
      describe("left", left);
      describe("right", right);
      const Tensor product = einsum("lsr,ruv->lsuv", {left, right}, neutral);
      describe("product", product);
      bench::run_for("einsum(\"lsr,ruv->lsuv\")", seconds,
                     [&] { (void) einsum("lsr,ruv->lsuv", {left, right}, neutral); });
    }

    // An environment update: a state block, an MPO block and the next state block.
    {
      Tensor state(su2, {leg, half}, {leg}, neutral);
      Tensor mpo(su2, {leg}, {leg}, neutral);
      Tensor next(su2, {leg}, {leg, half}, neutral);
      fill(state, 0.7);
      fill(mpo, 1.9);
      fill(next, 2.6);
      describe("state", state);
      describe("mpo", mpo);
      describe("next", next);
      const Tensor updated = einsum("asb,bc,cds->ad", {state, mpo, next}, neutral);
      describe("updated", updated);
      bench::run_for("einsum(\"asb,bc,cds->ad\")", seconds,
                     [&] { (void) einsum("asb,bc,cds->ad", {state, mpo, next}, neutral); });
    }

    // The elementary contraction: two bond legs of one block against two of the other.
    {
      Tensor left(su2, {leg, half}, {leg, leg}, neutral);
      Tensor right(su2, {leg, leg}, {leg}, neutral);
      fill(left, 0.5);
      fill(right, 1.3);
      describe("left", left);
      describe("right", right);
      const Tensor product = contract(left, right, {0, 1}, {0, 1}, neutral);
      describe("product", product);
      bench::run_for("contract(two bond legs)", seconds,
                     [&] { (void) contract(left, right, {0, 1}, {0, 1}, neutral); });
    }
  }
  return 0;
}
