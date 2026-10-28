// The caches the tensor layer keeps: plans, their matrices, buffers, layouts, and interns.
#include "cache.h"

#include <algorithm>
#include <tuple>
#include <utility>

namespace sym
{

  namespace
  {
    // A plan costs one walk over the fusion trees and a matrix one walk over its plan, so both are
    // kept until the shapes a sweep walks through would cost more than rebuilding them.
    constexpr int64_t kMatrixBudget = 1536LL * 1024 * 1024;
    // The plans a run needs, keyed by leg structure, block and step, and the bytes their piece lists
    // hold.  The set grows while the bond is still ramping and then stops — a 32 site chain at bond 64
    // settles at about 800 MB after four sweeps and grows no further — so the budget is a ceiling for
    // the case where it does keep growing, not a target.  Past it the whole cache goes and the plans
    // are rebuilt on demand: at bond 64 a ceiling of 256 MB costs about a sixth of a sweep and saves
    // 600 MB, while one much below a sweep's working set rebuilds every plan of every sweep.
    constexpr int64_t kPlanBudget = 256LL * 1024 * 1024;
    constexpr int64_t kBufferBudget = 64LL * 1024 * 1024;

    // One side of a step, written out: the legs and the blocks they carry.
    void write_legs(std::vector<int64_t> &out, const std::vector<Leg> &legs)
    {
      out.push_back(static_cast<int64_t>(legs.size()));
      for (const Leg &leg : legs)
      {
        out.push_back(leg.nblocks());
        for (int64_t block = 0; block < leg.nblocks(); ++block)
        {
          const Sector &sector = leg.sector(block);
          out.push_back(static_cast<int64_t>(sector.size()));
          out.insert(out.end(), sector.begin(), sector.end());
          out.push_back(leg.multiplicity(block));
        }
      }
    }

    // A shape of a side, written so that two sides with the same legs and root share one key.
    std::vector<int64_t> shape_key(const std::vector<Leg> &legs, const Sector &root)
    {
      std::vector<int64_t> key;
      key.push_back(static_cast<int64_t>(root.size()));
      key.insert(key.end(), root.begin(), root.end());
      for (const Leg &leg : legs)
      {
        key.push_back(leg.nblocks());
        for (int64_t block = 0; block < leg.nblocks(); ++block)
        {
          const Sector &sector = leg.sector(block);
          key.push_back(static_cast<int64_t>(sector.size()));
          key.insert(key.end(), sector.begin(), sector.end());
          key.push_back(leg.multiplicity(block));
        }
      }
      return key;
    }

    // Adds the runs of a shift to the block it names.  A block of the engine is a contiguous host
    // matrix of double or complex double, which the walk reads through pointers; anything else — a
    // block on another device, of another dtype, or a view of the storage — carries the runs as index
    // lists and lets one scatter add them.
    void add_pieces(const std::vector<Piece> &pieces, const at::Tensor &source, at::Tensor &target)
    {
      if (pieces.empty()) return;
      const bool host = source.is_contiguous() && target.is_contiguous() && source.device().is_cpu() &&
                        target.device().is_cpu() &&
                        (target.scalar_type() == at::kComplexDouble || target.scalar_type() == at::kDouble);
      if (host && target.scalar_type() == at::kComplexDouble)
      {
        const c10::complex<double> *from = source.data_ptr<c10::complex<double>>();
        c10::complex<double> *to = target.data_ptr<c10::complex<double>>();
        for (const Piece &piece : pieces)
        {
          const c10::complex<double> factor(piece.coefficient.real(), piece.coefficient.imag());
          for (int64_t at = 0; at < piece.run; ++at)
            to[piece.last + at * piece.to_step] += factor * from[piece.first + at * piece.from_step];
        }
        return;
      }
      if (host)
      {
        const double *from = source.data_ptr<double>();
        double *to = target.data_ptr<double>();
        for (const Piece &piece : pieces)
          for (int64_t at = 0; at < piece.run; ++at)
            to[piece.last + at * piece.to_step] +=
              piece.coefficient.real() * from[piece.first + at * piece.from_step];
        return;
      }
      const at::Tensor from = source.contiguous().reshape({-1});
      const bool plain = target.is_contiguous();
      const at::Tensor to = plain ? target.reshape({-1}) : target.contiguous().reshape({-1});
      int64_t total = 0;
      for (const Piece &piece : pieces) total += piece.run;
      const at::TensorOptions index_options = at::TensorOptions().dtype(at::kLong).device(target.device());
      std::vector<int64_t> reads(static_cast<size_t>(total));
      std::vector<int64_t> writes(static_cast<size_t>(total));
      at::Tensor scale = at::empty({total}, at::TensorOptions().dtype(at::kComplexDouble));
      c10::complex<double> *scale_at = scale.data_ptr<c10::complex<double>>();
      int64_t at = 0;
      for (const Piece &piece : pieces)
        for (int64_t step = 0; step < piece.run; ++step, ++at)
        {
          reads.at(static_cast<size_t>(at)) = piece.first + step * piece.from_step;
          writes.at(static_cast<size_t>(at)) = piece.last + step * piece.to_step;
          scale_at[at] = c10::complex<double>(piece.coefficient.real(), piece.coefficient.imag());
        }
      const at::Tensor values = from.index_select(0, at::tensor(reads, index_options)) *
                                scale.to(target.device()).to(target.scalar_type());
      to.index_put_({at::tensor(writes, index_options)}, values, true);
      if (!plain) target.copy_(to.reshape(target.sizes()));
    }

    // The buffers a chain fills and drops again. Every elementary move writes a fresh block, so
    // without this each step asks the allocator for one and the allocator costs more than the move.
    struct BufferPool
    {
      std::map<std::pair<int64_t, int64_t>, std::vector<at::Tensor>> held;
      int64_t bytes = 0;
    };

    BufferPool &buffer_pool()
    {
      thread_local BufferPool pool;
      return pool;
    }

    // A probe beats a walk down a tree: a plan is found by hashing its name, and the name compares
    // by address for the two sides, so a hit costs about as much as reading the block key.
    void mix(size_t &state, size_t value)
    {
      state ^= value + 0x9e3779b97f4a7c15ULL + (state << 6) + (state >> 2);
    }
  } // namespace

  size_t PlanQueryHash::operator()(const PlanQuery &query) const
  {
    size_t state = std::hash<const void *>()(query.cat.get());
    mix(state, std::hash<const void *>()(query.sides));
    for (const Sector *charge : {&query.key.codomain, &query.key.domain, &query.qtotal})
      for (const int64_t value : *charge) mix(state, std::hash<int64_t>()(value));
    mix(state, std::hash<int64_t>()(query.key.copy));
    mix(state, std::hash<int64_t>()(query.kind));
    mix(state, std::hash<int64_t>()(query.arm));
    return state;
  }

  TensorCaches &TensorCaches::instance()
  {
    static TensorCaches caches;
    return caches;
  }

  // The interned sides of a step: equal legs give the same vector, and the vector stays where it is
  // for as long as the program runs, so a plan can be named by where it sits.
  const std::vector<int64_t> *TensorCaches::intern(const std::vector<Leg> &codomain,
                                                   const std::vector<Leg> &domain)
  {
    std::vector<int64_t> key;
    write_legs(key, codomain);
    write_legs(key, domain);
    return &*interned_.insert(std::move(key)).first;
  }

  // The layout of one side at one root.  A plan needs the layout of every side it starts from and
  // ends at, and those repeat for every block of a tensor and every sweep, so the walk over the
  // fusion trees is done once per (side, root) instead of once per block.
  const SideLayout &TensorCaches::side(const FusionCategory &category, const std::vector<Leg> &legs,
                                       const Sector &root) const
  {
    std::vector<int64_t> bytes;
    write_legs(bytes, legs);
    const auto asked = std::make_tuple(static_cast<const void *>(&category),
                                       &*interned_.insert(std::move(bytes)).first, root);
    const auto found = layouts_.find(asked);
    if (found != layouts_.end()) return found->second;
    return layouts_.emplace(asked, layout_of(category, legs, root)).first->second;
  }

  // The rows a root gives on a side, built once per shape: a layout is a search over the fusion
  // trees, which is dearer than the walk it sizes, and a chain asks for the same shapes again.
  int64_t TensorCaches::rows(const std::shared_ptr<const FusionCategory> &category,
                             const std::vector<Leg> &legs, const Sector &root) const
  {
    const auto asked = std::make_pair(category, shape_key(legs, root));
    const auto found = rows_.find(asked);
    if (found != rows_.end()) return found->second;
    return rows_.emplace(asked, layout_of(*category, legs, root).rows()).first->second;
  }

  const Regrouping *TensorCaches::find(const PlanQuery &query) const
  {
    const auto found = plans_.find(query);
    return found == plans_.end() ? nullptr : &found->second;
  }

  const Regrouping &TensorCaches::keep(const PlanQuery &query, Regrouping plan)
  {
    int64_t added = 0;
    for (const Shift &shift : plan.shifts)
      added += static_cast<int64_t>(shift.pieces.size()) * static_cast<int64_t>(sizeof(Piece));
    if (kPlanBudget < plan_bytes_ + added)
    {
      plans_.clear();
      plan_bytes_ = 0;
      // The matrices are keyed by the address of the plan they came from, so they go with the plans.
      matrices_.clear();
      matrix_bytes_ = 0;
    }
    plan_bytes_ += added;
    return plans_.emplace(query, std::move(plan)).first->second;
  }

  // Keeps a matrix unless it would push the cache past its budget, in which case the cache is
  // dropped whole first: a matrix costs one walk over its plan to build, a cache that never shrinks
  // costs the device memory of every shape a sweep has walked through.
  at::Tensor TensorCaches::keep_matrix(const std::tuple<const Regrouping *, int64_t, BlockKey> &name,
                                       const at::Tensor &matrix, int64_t bytes)
  {
    if (bytes > kMatrixBudget) return matrix;
    if (matrix_bytes_ + bytes > kMatrixBudget)
    {
      matrices_.clear();
      matrix_bytes_ = 0;
    }
    matrix_bytes_ += bytes;
    return matrices_.emplace(name, matrix).first->second;
  }

  // The plan's matrix for one target block: one row and one column per entry of the blocks, built as
  // a dense matrix while it is small enough to pay for itself and as a sparse one beyond that, since
  // a sparse product costs a fixed setup on the device that a small block cannot repay.
  at::Tensor TensorCaches::matrix_of(const Regrouping &plan, const BlockKey &target,
                                     int64_t source_entries, int64_t target_entries,
                                     const at::TensorOptions &options)
  {
    const std::tuple<const Regrouping *, int64_t, BlockKey> name(
      &plan, static_cast<int64_t>(options.device().index()), target);
    const auto found = matrices_.find(name);
    if (found != matrices_.end()) return found->second;
    std::vector<std::pair<int64_t, std::pair<int64_t, std::complex<double>>>> entries;
    for (const Shift &shift : plan.shifts)
    {
      if (shift.to != target) continue;
      for (const Piece &piece : shift.pieces)
        for (int64_t at = 0; at < piece.run; ++at)
          entries.emplace_back(piece.last + at * piece.to_step,
                               std::make_pair(piece.first + at * piece.from_step, piece.coefficient));
    }
    std::sort(entries.begin(), entries.end(),
              [](const auto &left, const auto &right) { return left.first < right.first; });
    const auto device = options.device();
    const at::TensorOptions dense_options = at::TensorOptions().dtype(at::kComplexDouble).device(device);
    if (source_entries * target_entries <= 65536)
    {
      std::vector<c10::complex<double>> table(static_cast<size_t>(source_entries * target_entries),
                                              c10::complex<double>(0.0, 0.0));
      for (const auto &entry : entries)
        table.at(static_cast<size_t>(entry.first) * source_entries + entry.second.first) +=
          c10::complex<double>(entry.second.second.real(), entry.second.second.imag());
      const at::Tensor dense = at::from_blob(table.data(), {target_entries, source_entries}, at::kComplexDouble)
                                 .to(device)
                                 .clone();
      return keep_matrix(name, dense, 16 * dense.numel());
    }
    std::vector<int64_t> offsets(static_cast<size_t>(target_entries) + 1, 0), columns;
    std::vector<c10::complex<double>> values;
    for (const auto &entry : entries) ++offsets.at(static_cast<size_t>(entry.first) + 1);
    for (int64_t row = 0; row < target_entries; ++row) offsets.at(row + 1) += offsets.at(row);
    columns.reserve(entries.size());
    values.reserve(entries.size());
    for (const auto &entry : entries)
    {
      columns.push_back(entry.second.first);
      values.push_back(c10::complex<double>(entry.second.second.real(), entry.second.second.imag()));
    }
    // The indices are built dense and sorted here, so the checked constructor has nothing left to
    // verify and would only warn that the format is still beta.
    const auto sparse = at::_sparse_csr_tensor_unsafe(
      at::from_blob(offsets.data(), {static_cast<int64_t>(offsets.size())}, at::kLong).to(device).clone(),
      at::from_blob(columns.data(), {static_cast<int64_t>(columns.size())}, at::kLong).to(device).clone(),
      at::from_blob(values.data(), {static_cast<int64_t>(values.size())}, at::kComplexDouble).to(device).clone(),
      {target_entries, source_entries}, dense_options);
    return keep_matrix(name, sparse, 16 * sparse.values().numel() + 8 * sparse.col_indices().numel() +
                                              8 * sparse.crow_indices().numel());
  }

  // Adds what one source block sends into one target block: through the pointer walk when both are
  // host blocks, and through one sparse product otherwise, so a move on another device costs one
  // kernel per block and not one per run.
  void TensorCaches::apply(const Regrouping &plan, const BlockKey &target, const at::Tensor &source,
                           at::Tensor &into)
  {
    const bool host = source.is_contiguous() && into.is_contiguous() && source.device().is_cpu() &&
                      into.device().is_cpu() &&
                      (into.scalar_type() == at::kComplexDouble || into.scalar_type() == at::kDouble);
    if (host)
    {
      for (const Shift &shift : plan.shifts)
        if (shift.to == target) add_pieces(shift.pieces, source, into);
      return;
    }
    if (into.numel() == 0) return;
    at::Tensor matrix;
    {
      matrix = matrix_of(plan, target, source.numel(), into.numel(), into.options());
    }
    const at::Tensor added =
      at::matmul(matrix, source.contiguous().reshape({source.numel(), 1})).reshape({into.numel()});
    if (into.is_contiguous()) into.reshape({into.numel()}).add_(added.to(into.scalar_type()));
    else into.copy_((into.contiguous().reshape({into.numel()}) + added.to(into.scalar_type())).reshape(into.sizes()));
  }

  // A zeroed buffer of the shape asked for, reused from the pool when one is held.
  at::Tensor TensorCaches::borrow(const at::TensorOptions &options, int64_t rows, int64_t columns)
  {
    BufferPool &pool = buffer_pool();
    const auto found = pool.held.find({rows, columns});
    if (found == pool.held.end() || found->second.empty())
    {
      return at::zeros({rows, columns}, options);
    }
    at::Tensor taken = std::move(found->second.back());
    found->second.pop_back();
    pool.bytes -= taken.numel() * taken.element_size();
    return taken.zero_();
  }

  // Hands a buffer back, unless the pool already holds as much as it will.
  void TensorCaches::give_back(at::Tensor buffer)
  {
    BufferPool &pool = buffer_pool();
    const int64_t size = buffer.numel() * buffer.element_size();
    if (pool.bytes + size > kBufferBudget) return;
    pool.bytes += size;
    pool.held[{buffer.size(0), buffer.size(1)}].push_back(std::move(buffer));
  }

} // namespace sym
