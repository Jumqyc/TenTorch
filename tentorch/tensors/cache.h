// The state a regrouping reads and writes on its way: the plans already built, the matrices they
// multiply with, the buffers a chain fills, the layouts of the sides, and the names plans are
// found by.
#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "layout.h"
#include "regroup.h"

namespace sym
{

  /**
   * @brief The name of a regrouping in the plan cache.
   * @details Two plans are the same when they come from the same category, the same sides, the same
   * block, the same total charge, and the same step: `kind` is the step (an exchange or a bend),
   * `arm` the position an exchange picks or the side a bend takes from.
   *
   * `sides` points at a vector the cache holds, so comparing two names compares pointers.
   */
  struct PlanQuery
  {
    std::shared_ptr<const FusionCategory> cat;
    const std::vector<int64_t> *sides;
    BlockKey key;
    Sector qtotal;
    int64_t kind;
    int64_t arm;

    bool operator==(const PlanQuery &other) const
    {
      return std::tie(cat, sides, key, qtotal, kind, arm) ==
             std::tie(other.cat, other.sides, other.key, other.qtotal, other.kind, other.arm);
    }
  };

  /**
   * @brief The hash of a plan name.
   * @details A mix of its fields: `cat` and `sides` by address, the block by its two sectors and its
   * copy, and the rest by value.
   */
  struct PlanQueryHash
  {
    size_t operator()(const PlanQuery &query) const;
  };

  /**
   * @brief Everything a regrouping remembers between calls.
   * @details The plans, the matrices a plan multiplies with, the buffers a chain fills, and the
   * layouts and the interns the names are made of.
   *
   * A plan keeps its address while it is cached: the matrix cache is keyed by it, so a plan is
   * dropped together with its matrices and never leaves a stale key behind.
   *
   * The budget of each cache is a cap, not a policy: when the next entry would take a cache past
   * its cap the whole cache is dropped.  A DMRG sweep asks for the same plans every step, so a
   * cache that clears when it is full costs one step of rebuilding and pays for itself.
   */
  class TensorCaches final : public LayoutSource
  {
  public:
    /// The cache the tensor layer uses.
    static TensorCaches &instance();

    const SideLayout &side(const FusionCategory &category,
                           const std::vector<Leg> &legs,
                           const Sector &root) const override;

    /// The rows a side at one root spans, without keeping the layout.
    int64_t rows(const std::shared_ptr<const FusionCategory> &category,
                 const std::vector<Leg> &legs,
                 const Sector &root) const;

    /// The interned legs of both sides of a tensor: a plan name holds this pointer.
    const std::vector<int64_t> *intern(const std::vector<Leg> &codomain,
                                       const std::vector<Leg> &domain);

    /// The cached plan of `query`, or `nullptr`.
    const Regrouping *find(const PlanQuery &query) const;

    /// Keeps `plan` under `query` and returns the copy the cache holds.
    const Regrouping &keep(const PlanQuery &query, Regrouping plan);

    /// Adds the block `target` of `source` into `into` through `plan`.
    void apply(const Regrouping &plan,
               const BlockKey &target,
               const at::Tensor &source,
               at::Tensor &into);

    /// A dense buffer of `rows x columns`, taken from this thread's pool when one is free.
    at::Tensor borrow(const at::TensorOptions &options, int64_t rows, int64_t columns);

    /// Returns a buffer `borrow` handed out.
    void give_back(at::Tensor buffer);

  private:
    // A matrix of one plan on one block, kept when it fits the budget.
    at::Tensor keep_matrix(const std::tuple<const Regrouping *, int64_t, BlockKey> &name,
                           const at::Tensor &matrix,
                           int64_t bytes);

    // The matrix a shift from `target` needs on `device`: dense, or sparse above the size one dense
    // block is worth.
    at::Tensor matrix_of(const Regrouping &plan,
                         const BlockKey &target,
                         int64_t source_entries,
                         int64_t target_entries,
                         const at::TensorOptions &options);

    // The matrices of the cached plans, keyed by the plan, the device, and the block.  Every matrix
    // carries a row and a column per entry of a block, so the set grows with the shapes a sweep walks
    // through and not with the tensor it serves; past the budget the whole cache goes, which keeps
    // the pointer a matrix is keyed by from outliving the plan it names.
    std::map<std::tuple<const Regrouping *, int64_t, BlockKey>, at::Tensor> matrices_;
    int64_t matrix_bytes_ = 0;

    // The plans, keyed by their name.
    std::unordered_map<PlanQuery, Regrouping, PlanQueryHash> plans_;
    int64_t plan_bytes_ = 0;

    // The layouts of the sides, keyed by the category, the interned legs, and the root.
    mutable std::map<std::tuple<const void *, const std::vector<int64_t> *, Sector>, SideLayout> layouts_;

    // The rows of a side, keyed by the category and the interned legs: a tensor reads this once per
    // root when it is built, and the layouts themselves are only built when a step needs one.
    mutable std::map<std::pair<std::shared_ptr<const FusionCategory>, std::vector<int64_t>>, int64_t> rows_;

    // The interned legs: the serialization of a side, or of both sides, that a name points into.
    mutable std::set<std::vector<int64_t>> interned_;
  };

} // namespace sym
