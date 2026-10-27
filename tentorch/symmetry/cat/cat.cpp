// The categorical interface: the fusion every category shares, and the two questions that follow
// from it so that no category answers them twice.
#include "cat.h"

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace sym
{

  c10::Device default_device()
  {
    // A small system profiles better on the host: a block is a few hundred entries, which a kernel
    // launch costs more than to multiply.
    if (const char *wanted = std::getenv("TENTORCH_DEVICE"); wanted != nullptr && std::string(wanted) == "cpu")
      return c10::Device(c10::kCPU);
    return at::globalContext().hasCUDA() ? c10::Device(c10::kCUDA) : c10::Device(c10::kCPU);
  }

  FusionCategory::FusionCategory(std::string name, c10::Device device)
      : name_(std::move(name)), device_(device) {}

  const std::string &FusionCategory::name() const { return name_; }

  c10::Device FusionCategory::device() const { return device_; }

  const std::vector<Outcome> &FusionCategory::outcomes(const Sector &a, const Sector &b) const
  {
    const std::pair<Sector, Sector> asked{a, b};
    auto found = outcomes_cache_.find(asked);
    if (found == outcomes_cache_.end())
    {
      found = outcomes_cache_.emplace(asked, compute_outcomes(a, b)).first;
    }
    return found->second;
  }

  void FusionCategory::clear_cache() const { outcomes_cache_.clear(); }

  // The multiplicity of `a (x) b -> c`: how many distinct vertices couple the pair into `c`, and
  // zero when they do not couple to it at all.
  int64_t FusionCategory::n_symbol(const Sector &a, const Sector &b, const Sector &c) const
  {
    for (const Outcome &outcome : outcomes(a, b))
    {
      if (outcome.sector == c)
      {
        return outcome.multiplicity;
      }
    }
    return 0;
  }

  // Whether `a (x) b` couples to `c` at all.
  bool FusionCategory::can_fuse_to(const Sector &a, const Sector &b, const Sector &c) const
  {
    return n_symbol(a, b, c) > 0;
  }

  std::vector<FIndex> FusionCategory::f_rows(const Sector &a, const Sector &b, const Sector &c,
                                             const Sector &d) const
  {
    std::vector<FIndex> rows;
    for (const Outcome &e : outcomes(a, b))
    {
      const int64_t copies = n_symbol(e.sector, c, d);
      for (int64_t inner = 0; inner < e.multiplicity; ++inner)
      {
        for (int64_t outer = 0; outer < copies; ++outer)
        {
          rows.push_back(FIndex{e.sector, inner, outer});
        }
      }
    }
    return rows;
  }

  std::vector<FIndex> FusionCategory::f_columns(const Sector &a, const Sector &b, const Sector &c,
                                                const Sector &d) const
  {
    std::vector<FIndex> columns;
    for (const Outcome &f : outcomes(b, c))
    {
      const int64_t copies = n_symbol(a, f.sector, d);
      for (int64_t inner = 0; inner < f.multiplicity; ++inner)
      {
        for (int64_t outer = 0; outer < copies; ++outer)
        {
          columns.push_back(FIndex{f.sector, inner, outer});
        }
      }
    }
    return columns;
  }

  std::vector<RIndex> FusionCategory::r_rows(const Sector &b, const Sector &c) const
  {
    std::vector<RIndex> rows;
    for (const Outcome &e : outcomes(b, c))
    {
      for (int64_t copy = 0; copy < e.multiplicity; ++copy)
      {
        rows.push_back(RIndex{e.sector, copy});
      }
    }
    return rows;
  }

} // namespace sym
