// Checks the combinatorial SU(3) fusion against the weight-based reference and times both paths.
// The heavy before numbers are behind --old-heavy because they take several minutes.
#include <chrono>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../tentorch/symmetry/cat/cat.h"
#include "../tentorch/symmetry/cat/su3/su3.h"
#include "../tentorch/tensors/tensor.h"

using namespace sym;
using Clock = std::chrono::steady_clock;

namespace
{

  // A category whose fusion is the weight-based reference, used only for the comparison.
  class OldSU3 : public SU3Symmetry
  {
  protected:
    std::vector<Outcome> compute_outcomes(const Sector &a, const Sector &b) const override
    {
      std::vector<Outcome> fused;
      for (const auto &[sector, multiplicity] : su3::fusion_from_weights(a, b))
        fused.push_back({sector, static_cast<int>(multiplicity)});
      return fused;
    }
  };

  template <typename Call>
  double seconds(Call &&call)
  {
    const auto start = Clock::now();
    call();
    return std::chrono::duration<double>(Clock::now() - start).count();
  }

  Leg octet_leg(const std::shared_ptr<const FusionCategory> &cat)
  {
    return Leg::from_blocks(cat, {{Sector{1, 1}, 1}}).first;
  }

  bool same_fusion(const std::map<Sector, int64_t> &left, const std::map<Sector, int64_t> &right)
  {
    return left == right;
  }

  int64_t dimension_sum(const std::map<Sector, int64_t> &fused)
  {
    int64_t total = 0;
    for (const auto &[sector, multiplicity] : fused) total += multiplicity * su3::dimension(sector);
    return total;
  }

  // The N-ality is $p + 2q \bmod 3$ and is additive over a fusion.
  int64_t nality(const Sector &sector) { return (sector.at(0) + 2 * sector.at(1)) % 3; }

  bool conserves_nality(const Sector &a, const Sector &b, const std::map<Sector, int64_t> &fused)
  {
    const int64_t wanted = (nality(a) + nality(b)) % 3;
    for (const auto &entry : fused)
      if (nality(entry.first) != wanted) return false;
    return true;
  }

} // namespace

int main(int argc, char **argv)
{
  const bool heavy = argc > 1 && std::string(argv[1]) == "--old-heavy";

  // Every sector with p + q <= 4, in both operand orders.
  std::vector<Sector> sectors;
  for (int64_t p = 0; p <= 4; ++p)
    for (int64_t q = 0; p + q <= 4; ++q) sectors.emplace_back(Sector{p, q});

  std::printf("== new fusion against the weight-based reference ==\n");
  std::printf("  sectors with p + q <= 4: %zu\n", sectors.size());
  int64_t compared = 0;
  int64_t compared_outcomes = 0;
  int64_t differ = 0;
  for (const Sector &a : sectors)
  {
    for (const Sector &b : sectors)
    {
      const std::map<Sector, int64_t> got = su3::fusion(a, b);
      const std::map<Sector, int64_t> want = su3::fusion_from_weights(a, b);
      ++compared;
      compared_outcomes += static_cast<int64_t>(got.size());
      if (!same_fusion(got, want))
      {
        ++differ;
        std::printf("  differ: (%lld,%lld) x (%lld,%lld)\n", static_cast<long long>(a.at(0)),
                    static_cast<long long>(a.at(1)), static_cast<long long>(b.at(0)),
                    static_cast<long long>(b.at(1)));
      }
    }
  }
  std::printf("  compared %lld sector pairs (%lld outcome triples), %lld differ\n",
              static_cast<long long>(compared), static_cast<long long>(compared_outcomes),
              static_cast<long long>(differ));

  // The vector itself, because the order of `outcomes` is part of the interface contract.
  {
    const auto old_cat = std::make_shared<OldSU3>();
    const auto new_cat = std::make_shared<SU3Symmetry>();
    int64_t order_differ = 0;
    for (const Sector &a : sectors)
    {
      for (const Sector &b : sectors)
      {
        const std::vector<Outcome> &left = old_cat->outcomes(a, b);
        const std::vector<Outcome> &right = new_cat->outcomes(a, b);
        bool same = left.size() == right.size();
        for (size_t k = 0; same && k < left.size(); ++k)
          same = left[k].sector == right[k].sector && left[k].multiplicity == right[k].multiplicity;
        if (!same) ++order_differ;
      }
    }
    std::printf("  compute_outcomes order and multiplicity: %lld differ\n",
                static_cast<long long>(order_differ));
  }

  const std::vector<std::pair<Sector, Sector>> spots{
      {{2, 2}, {2, 2}}, {{3, 0}, {0, 3}}, {{4, 0}, {0, 4}}, {{1, 2}, {2, 1}},
      {{3, 3}, {3, 3}}, {{4, 2}, {2, 4}}, {{5, 1}, {1, 5}}, {{6, 0}, {0, 6}}};
  for (const auto &[a, b] : spots)
  {
    const std::map<Sector, int64_t> got = su3::fusion(a, b);
    const std::map<Sector, int64_t> want = su3::fusion_from_weights(a, b);
    std::printf("  spot (%lld,%lld) + (%lld,%lld): %zu outcomes, %s\n",
                static_cast<long long>(a.at(0)), static_cast<long long>(a.at(1)),
                static_cast<long long>(b.at(0)), static_cast<long long>(b.at(1)), got.size(),
                same_fusion(got, want) ? "agree" : "DIFFER");
  }

  // Large labels have no affordable weight-based reference: the dimension sum and the N-ality
  // are two independent properties every fusion must satisfy.
  std::printf("== large labels ==\n");
  const std::vector<std::pair<Sector, Sector>> large{
      {{12, 0}, {1, 1}}, {{6, 6}, {1, 1}}, {{0, 15}, {1, 1}}, {{12, 0}, {12, 0}},
      {{6, 6}, {6, 6}}, {{0, 15}, {12, 0}}};
  for (const auto &[a, b] : large)
  {
    su3::clear_caches();
    std::map<Sector, int64_t> fused;
    const double elapsed = seconds([&] { fused = su3::fusion(a, b); });
    const int64_t want = su3::dimension(a) * su3::dimension(b);
    std::printf("  (%lld,%lld) x (%lld,%lld): %zu outcomes, sum m d = %lld (want %lld), %s, "
                "%.1f us\n",
                static_cast<long long>(a.at(0)), static_cast<long long>(a.at(1)),
                static_cast<long long>(b.at(0)), static_cast<long long>(b.at(1)), fused.size(),
                static_cast<long long>(dimension_sum(fused)), static_cast<long long>(want),
                conserves_nality(a, b, fused) ? "N-ality ok" : "N-ality BAD", elapsed * 1e6);
  }

  // The category path: n_symbol on a cold cache, for the same sectors the problem measured.
  const std::vector<Sector> heavy_sectors{{4, 4}, {2, 5}, {6, 6}};
  std::printf("== n_symbol on a cold cache ==\n");
  {
    const auto warm = std::make_shared<SU3Symmetry>();
    (void)warm->n_symbol(Sector{1, 1}, Sector{1, 1}, Sector{0, 0});
  }
  for (const Sector &a : heavy_sectors)
  {
    {
      const auto fresh = std::make_shared<SU3Symmetry>();
      fresh->clear_cache();
      su3::clear_caches();
      int64_t value = 0;
      const double elapsed = seconds([&] { value = fresh->n_symbol(a, Sector{0, 0}, Sector{0, 0}); });
      std::printf("  new n_symbol((%lld,%lld), 1, 1) = %lld: %.1f us\n",
                  static_cast<long long>(a.at(0)), static_cast<long long>(a.at(1)),
                  static_cast<long long>(value), elapsed * 1e6);
    }
    {
      const auto fresh = std::make_shared<SU3Symmetry>();
      fresh->clear_cache();
      su3::clear_caches();
      int64_t value = 0;
      const double elapsed = seconds([&] {
        value = fresh->n_symbol(a, Sector{1, 1}, Sector{1, 1});
      });
      std::printf("  new n_symbol((%lld,%lld), (1,1), (1,1)) = %lld: %.1f us\n",
                  static_cast<long long>(a.at(0)), static_cast<long long>(a.at(1)),
                  static_cast<long long>(value), elapsed * 1e6);
    }
  }

  // The first torch allocation pays the context initialisation, so warm it up once.
  {
    const auto warm = std::make_shared<SU3Symmetry>();
    const Leg leg = octet_leg(warm);
    const Tensor warmup(warm, {leg, leg, leg, leg}, {}, warm->trivial());
    (void)warmup;
  }

  // The 4-octet tensor reaches (4,4) on its codomain, so the old path pays the dense irrep there.
  const auto fresh = std::make_shared<SU3Symmetry>();
  fresh->clear_cache();
  su3::clear_caches();
  const Leg octet = octet_leg(fresh);
  size_t blocks = 0;
  const double tensor_time = seconds([&] {
    const Tensor tensor(fresh, {octet, octet, octet, octet}, {}, fresh->trivial());
    blocks = tensor.blocks().size();
  });
  std::printf("  new Tensor(4 octets, trivial): %zu blocks, %.1f us\n", blocks,
              tensor_time * 1e6);

  if (!heavy)
  {
    std::printf("== old heavy timings skipped: pass --old-heavy ==\n");
    return 0;
  }

  std::printf("== old weight-based timings ==\n");
  for (const Sector &a : std::vector<Sector>{{4, 4}, {2, 5}})
  {
    const auto old = std::make_shared<OldSU3>();
    old->clear_cache();
    su3::clear_caches();
    int64_t value = 0;
    const double elapsed = seconds([&] { value = old->n_symbol(a, Sector{0, 0}, Sector{0, 0}); });
    std::printf("  old n_symbol((%lld,%lld), 1, 1) = %lld: %.1f s\n",
                static_cast<long long>(a.at(0)), static_cast<long long>(a.at(1)),
                static_cast<long long>(value), elapsed);
  }
  std::printf("  old n_symbol((6,6), 1, 1): not measurable, the dense projector is 3^12 by 3^12\n");

  const auto old = std::make_shared<OldSU3>();
  const Leg old_octet = octet_leg(old);
  old->clear_cache();
  su3::clear_caches();
  size_t old_blocks = 0;
  const double old_tensor_time = seconds([&] {
    const Tensor tensor(old, {old_octet, old_octet, old_octet, old_octet}, {}, old->trivial());
    old_blocks = tensor.blocks().size();
  });
  std::printf("  old Tensor(4 octets, trivial): %zu blocks, %.1f s\n", old_blocks,
              old_tensor_time);
  return 0;
}
