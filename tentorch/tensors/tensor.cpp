#include "tensor.h"

#include <algorithm>
#include <complex>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace sym
{

  namespace
  {
    // The rows the legs of one side span at a root, before the layouts of a tensor are built.
    int64_t side_rows(const std::shared_ptr<const FusionCategory> &category, const std::vector<Leg> &legs,
                      const Sector &root);


    // The block of a leg that carries `sector`, which the leg holds sorted.
    int64_t block_of(const Leg &leg, const Sector &sector)
    {
      const std::vector<Sector> &sectors = leg.sectors();
      const auto found = std::lower_bound(sectors.begin(), sectors.end(), sector);
      if (found == sectors.end() || *found != sector)
        throw std::invalid_argument("a leg of " + std::to_string(leg.nblocks()) +
                                    " blocks carries none in sector " + render(sector));
      return found - sectors.begin();
    }

    // The charge the legs up to `last` couple to, the trivial charge before the first.
    Sector chain_at(const FusionCategory &category, const FusionTree &tree, int64_t last)
    {
      if (last < 0) return category.trivial();
      if (last + 1 == tree.size()) return tree.coupled;
      if (last == 0) return tree.sectors.at(0);
      return tree.inner.at(static_cast<size_t>(last - 1));
    }

  } // namespace

  SideLayout layout_of(const FusionCategory &category, const std::vector<Leg> &legs, const Sector &root)
  {
    SideLayout layout;
    layout.trees = fusion_trees(category, legs, root);
    layout.offsets.push_back(0);
    for (const FusionTree &tree : layout.trees)
    {
      // One row per combination of the multiplicities the tree's legs carry.
      int64_t rows = 1;
      for (size_t axis = 0; axis < legs.size(); ++axis)
        rows *= legs.at(axis).multiplicity(block_of(legs.at(axis), tree.sectors.at(axis)));
      layout.offsets.push_back(layout.offsets.back() + rows);
    }
    return layout;
  }

  std::vector<BlockKey> block_keys(const FusionCategory &category,
                                   const std::vector<Leg> &codomain,
                                   const std::vector<Leg> &domain,
                                   const Sector &qtotal)
  {
    const std::vector<Sector> out_roots = reachable_charges(category, codomain);
    const std::vector<Sector> in_roots = reachable_charges(category, domain);
    std::vector<BlockKey> keys;
    for (const Sector &r : out_roots)
      for (const Sector &s : in_roots)
        for (int64_t copy = 0; copy < category.n_symbol(r, category.dual(s), qtotal); ++copy)
          keys.push_back(BlockKey{r, s, copy});
    return keys;
  }

  Tensor::Tensor(std::shared_ptr<const FusionCategory> cat,
                 std::vector<Leg> codomain,
                 std::vector<Leg> domain,
                 Sector qtotal,
                 std::optional<c10::ScalarType> dtype,
                 std::optional<c10::Device> device)
      : cat_(std::move(cat)),
        codomain_(std::move(codomain)),
        domain_(std::move(domain)),
        qtotal_(std::move(qtotal))
  {
    if (!cat_) throw std::invalid_argument("a tensor needs a category");
    dtype_ = dtype.value_or(cat_->dtype());
    device_ = device.value_or(cat_->device());
    for (const Leg &leg : codomain_)
      if (!same_group(*leg.cat(), *cat_))
        throw std::invalid_argument("a codomain leg is over a different category");
    for (const Leg &leg : domain_)
      if (!same_group(*leg.cat(), *cat_))
        throw std::invalid_argument("a domain leg is over a different category");
    const at::TensorOptions options = at::TensorOptions().dtype(dtype_).device(device_);
    // One layout per root and not one per key: the keys of a side share it, and the layout is cached
    // on the tensor so that a later access of the same root costs a lookup.
    const std::vector<BlockKey> keys = block_keys(*cat_, codomain_, domain_, qtotal_);
    // The keys of a side share a root, and a side of a root is the same rows or columns every time,
    // so each root is looked up once.
    std::map<Sector, int64_t> rows_of_root;
    std::map<Sector, int64_t> columns_of_root;
    for (const BlockKey &key : keys)
    {
      auto rows = rows_of_root.find(key.codomain);
      if (rows == rows_of_root.end())
        rows = rows_of_root.emplace(key.codomain, side_rows(cat_, codomain_, key.codomain)).first;
      auto columns = columns_of_root.find(key.domain);
      if (columns == columns_of_root.end())
        columns = columns_of_root.emplace(key.domain, side_rows(cat_, domain_, key.domain)).first;
      blocks_.emplace(key, at::zeros({rows->second, columns->second}, options));
    }
  }

  std::vector<Leg> Tensor::legs() const
  {
    std::vector<Leg> out = codomain_;
    out.insert(out.end(), domain_.begin(), domain_.end());
    return out;
  }

  const at::Tensor &Tensor::block(const Sector &codomain_root, const Sector &domain_root, int64_t copy) const
  {
    const auto found = blocks_.find(BlockKey{codomain_root, domain_root, copy});
    if (found == blocks_.end())
      throw std::out_of_range("the tensor has no block at " + render(codomain_root) + " x " + render(domain_root) +
                              " of copy " + std::to_string(copy));
    return found->second;
  }

  at::Tensor &Tensor::block(const Sector &codomain_root, const Sector &domain_root, int64_t copy)
  {
    const auto found = blocks_.find(BlockKey{codomain_root, domain_root, copy});
    if (found == blocks_.end())
      throw std::out_of_range("the tensor has no block at " + render(codomain_root) + " x " + render(domain_root) +
                              " of copy " + std::to_string(copy));
    return found->second;
  }

  const SideLayout &Tensor::rows_of(const Sector &codomain_root) const
  {
    const auto found = codomain_layouts_.find(codomain_root);
    if (found != codomain_layouts_.end()) return found->second;
    return codomain_layouts_.emplace(codomain_root, layout_of(*cat_, codomain_, codomain_root)).first->second;
  }

  const SideLayout &Tensor::columns_of(const Sector &domain_root) const
  {
    const auto found = domain_layouts_.find(domain_root);
    if (found != domain_layouts_.end()) return found->second;
    return domain_layouts_.emplace(domain_root, layout_of(*cat_, domain_, domain_root)).first->second;
  }

  std::vector<BlockKey> Tensor::keys() const
  {
    std::vector<BlockKey> out;
    for (const auto &[key, matrix] : blocks_) out.push_back(key);
    return out;
  }

  namespace
  {

    // One run of entries of a regrouping: the entry `first + at * from_step` of the block a shift
    // reads is added to the entry `last + at * to_step` of the block it writes, scaled by
    // `coefficient`.  A block is a contiguous matrix, so a run along the columns has step one on both
    // sides, while a run along the rows carries the row stride.
    struct Piece
    {
      int64_t first = 0;
      int64_t last = 0;
      int64_t from_step = 1;
      int64_t to_step = 1;
      int64_t run = 1;
      std::complex<double> coefficient{0.0, 0.0};
    };

    // Where the entries of one block go: the runs that land in one block of the result, and the shape
    // of that block, so that a step need not look the layout up again for every block it fills.
    struct Shift
    {
      BlockKey from;
      BlockKey to;
      int64_t rows = 0;
      int64_t columns = 0;
      std::vector<Piece> pieces;
    };

    struct Regrouping
    {
      std::vector<Shift> shifts;
    };

    // The matrices the plans multiply device blocks with, one per plan, device and target block, and the
    // bytes they hold.  Every matrix carries a row and a column per entry of a block, so a cache that
    // keeps them all grows with the number of shapes a sweep walks through and not with the tensor it
    // serves; past the budget the cache is dropped whole and the matrices are built again on demand.
    struct MatrixCache
    {
      std::map<std::tuple<const Regrouping *, int64_t, BlockKey>, at::Tensor> held;
      int64_t bytes = 0;
    };

    constexpr int64_t kMatrixBudget = 1536LL * 1024 * 1024;

    MatrixCache &matrix_cache()
    {
      static MatrixCache cache;
      return cache;
    }

    // Keeps a matrix unless it would push the cache past its budget, in which case the cache is dropped
    // whole first: a matrix costs one walk over its plan to build, a cache that never shrinks costs the
    // device memory of every shape a sweep has walked through.
    at::Tensor keep_matrix(MatrixCache &cache, const std::tuple<const Regrouping *, int64_t, BlockKey> &name,
                           const at::Tensor &matrix, int64_t bytes)
    {
      if (bytes > kMatrixBudget) return matrix;
      if (cache.bytes + bytes > kMatrixBudget)
      {
        cache.held.clear();
        cache.bytes = 0;
      }
      cache.bytes += bytes;
      return cache.held.emplace(name, matrix).first->second;
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

    // The plan's matrix for one target block: one row and one column per entry of the blocks, built as
    // a dense matrix while it is small enough to pay for itself and as a sparse one beyond that, since
    // a sparse product costs a fixed setup on the device that a small block cannot repay.
    at::Tensor matrix_of(const Regrouping &plan, const BlockKey &target, int64_t source_entries,
                         int64_t target_entries, const at::TensorOptions &options, MatrixCache &cache)
    {
      const std::tuple<const Regrouping *, int64_t, BlockKey> name(
        &plan, static_cast<int64_t>(options.device().index()), target);
      const auto found = cache.held.find(name);
      if (found != cache.held.end()) return found->second;
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
        return keep_matrix(cache, name, dense, 16 * dense.numel());
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
      return keep_matrix(cache, name, sparse, 16 * sparse.values().numel() + 8 * sparse.col_indices().numel() +
                                                8 * sparse.crow_indices().numel());
    }

    // Adds what one source block sends into one target block: through the pointer walk when both are
    // host blocks, and through one sparse product otherwise, so a move on another device costs one
    // kernel per block and not one per run.
    void add_into(const Regrouping &plan, const BlockKey &target, const at::Tensor &source, at::Tensor &into,
                  MatrixCache &cache)
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
        matrix = matrix_of(plan, target, source.numel(), into.numel(), into.options(), cache);
      }
      const at::Tensor added =
        at::matmul(matrix, source.contiguous().reshape({source.numel(), 1})).reshape({into.numel()});
      if (into.is_contiguous()) into.reshape({into.numel()}).add_(added.to(into.scalar_type()));
      else into.copy_((into.contiguous().reshape({into.numel()}) + added.to(into.scalar_type())).reshape(into.sizes()));
    }

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

    // The interned sides of a step: equal legs give the same vector, and the vector stays where it is
    // for as long as the program runs, so a plan can be named by where it sits.
    const std::vector<int64_t> *interned_sides(const std::vector<Leg> &codomain, const std::vector<Leg> &domain)
    {
      static std::set<std::vector<int64_t>> held;
      std::vector<int64_t> key;
      write_legs(key, codomain);
      write_legs(key, domain);
      return &*held.insert(std::move(key)).first;
    }

    // The interned legs of one side, for the same reason.
    const std::vector<int64_t> *interned_legs(const std::vector<Leg> &legs)
    {
      static std::set<std::vector<int64_t>> held;
      std::vector<int64_t> key;
      write_legs(key, legs);
      return &*held.insert(std::move(key)).first;
    }

    // The layout of one side at one root.  A plan needs the layout of every side it starts from and
    // ends at, and those repeat for every block of a tensor and every sweep, so the walk over the
    // fusion trees is done once per (side, root) instead of once per block.
    const SideLayout &cached_layout(const FusionCategory &cat,
                                    const std::vector<Leg> &legs,
                                    const Sector &root)
    {
      static std::map<std::tuple<const void *, const std::vector<int64_t> *, Sector>, SideLayout> held;
      const auto asked = std::make_tuple(static_cast<const void *>(&cat), interned_legs(legs), root);
      const auto found = held.find(asked);
      if (found != held.end()) return found->second;
      return held.emplace(asked, layout_of(cat, legs, root)).first->second;
    }

    // The name of a plan: the two sides of the step, the block it acts on, and which of the
    // elementary moves it is.  The sides compare by address, which is what the interning buys.
    struct PlanName
    {
      std::shared_ptr<const FusionCategory> cat;
      const std::vector<int64_t> *sides = nullptr;
      BlockKey key;
      Sector qtotal;
      // Which plan of the two builders this is, and where in the step: the two builders share the cache,
      // so the kind is part of the name rather than implied by the one that is asking.
      int64_t kind = 0;
      int64_t arm = 0;

      bool operator==(const PlanName &other) const
      {
        return std::tie(cat, sides, key, qtotal, kind, arm) ==
               std::tie(other.cat, other.sides, other.key, other.qtotal, other.kind, other.arm);
      }
    };

    // A probe beats a walk down a tree: a plan is found by hashing the name, and the name compares by
    // address for the two sides, so a hit costs about as much as reading the block key.
    struct PlanNameHash
    {
      static void mix(size_t &state, size_t value)
      {
        state ^= value + 0x9e3779b97f4a7c15ULL + (state << 6) + (state >> 2);
      }

      size_t operator()(const PlanName &name) const
      {
        size_t state = std::hash<const void *>()(name.cat.get());
        mix(state, std::hash<const void *>()(name.sides));
        for (const Sector *charge : {&name.key.codomain, &name.key.domain, &name.qtotal})
          for (const int64_t value : *charge) mix(state, std::hash<int64_t>()(value));
        mix(state, std::hash<int64_t>()(name.key.copy));
        mix(state, std::hash<int64_t>()(name.kind));
        mix(state, std::hash<int64_t>()(name.arm));
        return state;
      }
    };

    // The plans a run needs, keyed by leg structure, block and step, and the bytes their piece lists
    // hold.  The set grows while the bond is still ramping and then stops — a 32 site chain at bond 64
    // settles at about 800 MB after four sweeps and grows no further — so the budget is a ceiling for
    // the case where it does keep growing, not a target.  Past it the whole cache goes and the plans
    // are rebuilt on demand: at bond 64 a ceiling of 256 MB costs about a sixth of a sweep and saves
    // 600 MB, while one much below a sweep's working set rebuilds every plan of every sweep.
    constexpr int64_t kPlanBudget = 256LL * 1024 * 1024;

    struct PlanCache
    {
      std::unordered_map<PlanName, Regrouping, PlanNameHash> plans;
      int64_t bytes = 0;
    };

    PlanCache &plan_cache()
    {
      static PlanCache cache;
      return cache;
    }


    // The regrouping of exchanging the codomain legs `p` and `p + 1` of one block: the trees that
    // agree outside the pair recouple by F and R, and the width the two legs carry is read at the
    // exchanged position on the other side.
    const Regrouping &exchange_regrouping(const std::shared_ptr<const FusionCategory> &cat,
                                          const std::vector<int64_t> *sides,
                                          const std::vector<Leg> &codomain,
                                          const std::vector<Leg> &domain,
                                          const BlockKey &key,
                                          const Sector &qtotal,
                                          int64_t p)
    {
      PlanCache &cache = plan_cache();
      const PlanName name{cat, sides, key, qtotal, 0, p};
      const auto held = cache.plans.find(name);
      if (held != cache.plans.end())
      {
        return held->second;
      }
      const int64_t count = static_cast<int64_t>(codomain.size());
      std::vector<Leg> exchanged = codomain;
      std::swap(exchanged.at(static_cast<size_t>(p)), exchanged.at(static_cast<size_t>(p + 1)));
      const SideLayout &source = cached_layout(*cat, codomain, key.codomain);
      const SideLayout &target = cached_layout(*cat, exchanged, key.codomain);
      const int64_t stride = cached_layout(*cat, domain, key.domain).rows();
      Regrouping plan;
      // An exchange moves entries about inside a block and leaves the block the same size, so the
      // shape is the one the step already has.
      plan.shifts.push_back(Shift{key, key, 0, 0, {}});

      // Everything about a tree that the exchange leaves alone.  A tree of the exchanged side is read
      // in the order before the exchange, so that the two sides meet on the same key.
      const auto signature = [&](const FusionTree &tree, bool exchanged_tree)
      {
        std::vector<int64_t> out;
        for (int64_t axis = 0; axis < count; ++axis)
        {
          const int64_t from = exchanged_tree && (axis == p || axis == p + 1) ? (axis == p ? p + 1 : p) : axis;
          const Sector &sector = tree.sectors.at(static_cast<size_t>(from));
          out.insert(out.end(), sector.begin(), sector.end());
        }
        for (int64_t at = 0; at < static_cast<int64_t>(tree.inner.size()); ++at)
        {
          if (at == p - 1) continue;
          const Sector &charge = tree.inner.at(static_cast<size_t>(at));
          out.insert(out.end(), charge.begin(), charge.end());
        }
        for (int64_t at = 0; at < static_cast<int64_t>(tree.copies.size()); ++at)
        {
          if (at == p - 1 || at == p) continue;
          out.push_back(tree.copies.at(static_cast<size_t>(at)));
        }
        return out;
      };

      // Where the axis of a side ends up in the other order.
      const auto moved = [p](int64_t axis) { return axis == p ? p + 1 : axis == p + 1 ? p : axis; };

      // The multiplicities the legs of a tree carry, one per axis of `legs`.
      const auto shape_of = [&](const FusionTree &tree, const std::vector<Leg> &legs)
      {
        std::vector<int64_t> out;
        for (int64_t axis = 0; axis < count; ++axis)
        {
          const Leg &leg = legs.at(static_cast<size_t>(axis));
          out.push_back(leg.multiplicity(block_of(leg, tree.sectors.at(static_cast<size_t>(axis)))));
        }
        return out;
      };

      // The trees the exchange recouples: they agree on everything outside the pair.
      std::map<std::vector<int64_t>, std::vector<int64_t>> groups;
      for (int64_t at = 0; at < static_cast<int64_t>(source.trees.size()); ++at)
        groups[signature(source.trees.at(static_cast<size_t>(at)), false)].push_back(at);
      std::map<std::vector<int64_t>, std::vector<int64_t>> partners;
      for (int64_t at = 0; at < static_cast<int64_t>(target.trees.size()); ++at)
        partners[signature(target.trees.at(static_cast<size_t>(at)), true)].push_back(at);

      // The coefficient of every pair of trees the exchange couples: the braiding between the two
      // intermediate bases is applied to the copies of one outcome only, which is the middle map.
      std::map<std::pair<int64_t, int64_t>, std::complex<double>> factors;
      for (const auto &[marks, olds] : groups)
      {
        const auto found = partners.find(marks);
        if (found == partners.end())
          throw std::logic_error("the exchange has no image for a tree of the codomain");
        const std::vector<int64_t> &news = found->second;
        const FusionTree &sample = source.trees.at(static_cast<size_t>(olds.front()));
        const Sector y = chain_at(*cat, sample, p - 1);
        const Sector d = chain_at(*cat, sample, p + 1);
        const Sector &b = sample.sectors.at(static_cast<size_t>(p));
        const Sector &c = sample.sectors.at(static_cast<size_t>(p + 1));
        const at::Tensor straight = cat->F_symbol(y, b, c, d).to(at::kComplexDouble);
        const at::Tensor braided = cat->F_symbol(y, c, b, d).to(at::kComplexDouble);
        TORCH_CHECK(straight.size(0) == static_cast<int64_t>(olds.size()) &&
                      braided.size(0) == static_cast<int64_t>(news.size()),
                    "the exchange does not match the trees of the block");
        // The outcomes are a handful of sectors, so their offsets are read from a flat list: a map
        // here would allocate on every group for no reason.
        std::vector<std::pair<Sector, int64_t>> offset;
        int64_t at = 0;
        for (const Outcome &outcome : cat->outcomes(b, c))
        {
          offset.emplace_back(outcome.sector, at);
          at += outcome.multiplicity;
        }
        const std::vector<FIndex> from = cat->f_columns(y, b, c, d);
        const std::vector<FIndex> to = cat->f_columns(y, c, b, d);
        const at::Tensor braid = cat->R_symbol(b, c).to(at::kComplexDouble);
        // Where the braiding is read: a row is a tree of the right bracketing and a column one of the
        // left, addressed by the outcome and the copy, so a bracketing the braiding does not see has
        // no entry.
        const auto braid_at = [&](const FIndex &index)
        {
          for (const auto &[sector, start] : offset)
            if (sector == index.sector) return start + index.inner;
          return static_cast<int64_t>(-1);
        };
        std::vector<int64_t> rows;
        for (const FIndex &index : to) rows.push_back(braid_at(index));
        std::vector<int64_t> columns;
        for (const FIndex &index : from) columns.push_back(braid_at(index));
        // The coupled coefficient is `braided . middle . straight^dagger`, where `middle` is the
        // braiding scattered onto the pairs of bracketings that agree on the intermediate sector and
        // on the vertex above it.  The matrices are a few entries wide, so the product is summed here
        // instead of assembled out of tensor operations.
        const int64_t to_size = static_cast<int64_t>(to.size());
        const int64_t from_size = static_cast<int64_t>(from.size());
        const int64_t braid_columns = braid.size(1);
        const c10::complex<double> *braided_entry = braided.data_ptr<c10::complex<double>>();
        const c10::complex<double> *straight_entry = straight.data_ptr<c10::complex<double>>();
        const c10::complex<double> *braid_entry = braid.data_ptr<c10::complex<double>>();
        // The entries the product visits, read once: which row of the braided matrix, which entry of
        // the braiding, and which column of the straight matrix.  Only the two tree indices are left
        // to the loop below, so its body is arithmetic and no longer a scan of matching sectors.
        struct Term
        {
          int64_t braided_at;
          int64_t braid_at;
          int64_t straight_at;
        };
        std::vector<Term> terms;
        for (int64_t i = 0; i < to_size; ++i)
        {
          if (rows.at(static_cast<size_t>(i)) < 0) continue;
          for (int64_t j = 0; j < from_size; ++j)
          {
            if (columns.at(static_cast<size_t>(j)) < 0) continue;
            if (to.at(static_cast<size_t>(i)).sector != from.at(static_cast<size_t>(j)).sector) continue;
            if (to.at(static_cast<size_t>(i)).outer != from.at(static_cast<size_t>(j)).outer) continue;
            terms.push_back(Term{i,
                                 rows.at(static_cast<size_t>(i)) * braid_columns +
                                   columns.at(static_cast<size_t>(j)),
                                 j});
          }
        }
        for (size_t one = 0; one < news.size(); ++one)
          for (size_t other = 0; other < olds.size(); ++other)
          {
            const c10::complex<double> *braided_row =
              braided_entry + static_cast<int64_t>(one) * to_size;
            const c10::complex<double> *straight_row =
              straight_entry + static_cast<int64_t>(other) * from_size;
            c10::complex<double> value{0.0, 0.0};
            for (const Term &term : terms)
              value += braided_row[term.braided_at] * braid_entry[term.braid_at] *
                       std::conj(straight_row[term.straight_at]);
            if (value.real() == 0.0 && value.imag() == 0.0) continue;
            factors.emplace(std::make_pair(news.at(one), olds.at(other)),
                            std::complex<double>(value.real(), value.imag()));
          }
      }

      // Trees whose legs carry the same multiplicities move in one piece: the width index is a mixed
      // radix over those multiplicities, and the exchange only moves its digits, which keeps the size
      // of a class the same on both sides even when the two legs carry different numbers of copies.
      std::map<std::vector<int64_t>, std::vector<int64_t>> before_classes;
      std::map<std::vector<int64_t>, std::vector<int64_t>> after_classes;
      std::vector<std::vector<int64_t>> after_shapes(target.trees.size());
      for (int64_t at = 0; at < static_cast<int64_t>(source.trees.size()); ++at)
        before_classes[shape_of(source.trees.at(static_cast<size_t>(at)), codomain)].push_back(at);
      for (int64_t at = 0; at < static_cast<int64_t>(target.trees.size()); ++at)
      {
        // A partner carries the same sector on the same leg, so its shape is read back in the order
        // the legs had before the exchange.
        const std::vector<int64_t> shape = shape_of(target.trees.at(static_cast<size_t>(at)), exchanged);
        std::vector<int64_t> back(shape.size(), 0);
        for (int64_t axis = 0; axis < count; ++axis)
          back.at(static_cast<size_t>(moved(axis))) = shape.at(static_cast<size_t>(axis));
        after_shapes.at(static_cast<size_t>(at)) = shape;
        after_classes[back].push_back(at);
      }
      std::vector<Piece> &pieces = plan.shifts.back().pieces;
      for (const auto &[shape, olds] : before_classes)
      {
        const auto partner = after_classes.find(shape);
        if (partner == after_classes.end())
          throw std::logic_error("the exchange has no image for a block of trees");
        const std::vector<int64_t> &news = partner->second;
        int64_t width = 1;
        for (const int64_t factor : shape) width *= factor;
        // The width of a class reads the copies of a leg in the order of the legs, so an index moves
        // its digits to the positions the exchanged legs put them at, under that side's radices.
        const std::vector<int64_t> &target_shape = after_shapes.at(static_cast<size_t>(news.front()));
        std::vector<int64_t> width_of(static_cast<size_t>(width));
        for (int64_t at = 0; at < width; ++at)
        {
          std::vector<int64_t> digits(shape.size(), 0);
          std::vector<int64_t> placed(shape.size(), 0);
          int64_t left = at;
          for (size_t axis = shape.size(); axis-- > 0;)
          {
            digits.at(axis) = left % shape.at(axis);
            left /= shape.at(axis);
          }
          for (int64_t axis = 0; axis < count; ++axis)
            placed.at(static_cast<size_t>(moved(axis))) = digits.at(static_cast<size_t>(axis));
          int64_t index = 0;
          for (int64_t axis = 0; axis < count; ++axis)
            index = index * target_shape.at(static_cast<size_t>(axis)) + placed.at(static_cast<size_t>(axis));
          width_of.at(static_cast<size_t>(at)) = index;
        }
        for (const int64_t one : news)
          for (const int64_t other : olds)
          {
            const auto factor = factors.find(std::make_pair(one, other));
            if (factor == factors.end()) continue;
            // Digits the exchange leaves in place stay adjacent on both sides, so a stretch of width
            // indices that moves without a gap is read and written as one run.
            for (int64_t at = 0; at < width;)
            {
              int64_t run = 1;
              while (at + run < width &&
                     width_of.at(static_cast<size_t>(at + run)) == width_of.at(static_cast<size_t>(at)) + run)
                ++run;
              pieces.push_back(Piece{(source.offsets.at(static_cast<size_t>(other)) + at) * stride,
                                     (target.offsets.at(static_cast<size_t>(one)) +
                                      width_of.at(static_cast<size_t>(at))) * stride,
                                     1, 1, run * stride, factor->second});
              at += run;
            }
          }
      }
      int64_t added = 0;
      for (const Shift &shift : plan.shifts)
        added += static_cast<int64_t>(shift.pieces.size()) * static_cast<int64_t>(sizeof(Piece));
      if (kPlanBudget < cache.bytes + added)
      {
        cache.plans.clear();
        cache.bytes = 0;
        // The matrices are keyed by the address of the plan they came from, so they go with the plans.
        matrix_cache().held.clear();
        matrix_cache().bytes = 0;
      }
      cache.bytes += added;
      return cache.plans.emplace(name, std::move(plan)).first->second;
    }

    // The regrouping of bending the last leg of one side onto the other, for one block.  The giving
    // tree keeps everything but its last vertex, the taking tree gains one, and the recoupling of the
    // two vertices carries the copies of the moved leg into the columns of the taking side.
    const Regrouping &bend_regrouping(const std::shared_ptr<const FusionCategory> &cat,
                                      const std::vector<int64_t> *sides,
                                      const std::vector<Leg> &codomain,
                                      const std::vector<Leg> &domain,
                                      const BlockKey &key,
                                      const Sector &qtotal,
                                      bool from_codomain)
    {
      PlanCache &cache = plan_cache();
      const PlanName name{cat, sides, key, qtotal, 1, from_codomain ? 1 : 0};
      const auto held = cache.plans.find(name);
      if (held != cache.plans.end())
      {
        return held->second;
      }
      const std::vector<Leg> &giving = from_codomain ? codomain : domain;
      const std::vector<Leg> &taking = from_codomain ? domain : codomain;
      const Leg moved_leg = giving.back();
      const std::vector<Leg> kept_giving(giving.begin(), giving.end() - 1);
      std::vector<Leg> kept_taking = taking;
      kept_taking.push_back(moved_leg);
      const std::vector<Leg> out_codomain = from_codomain ? kept_giving : kept_taking;
      const std::vector<Leg> out_domain = from_codomain ? kept_taking : kept_giving;
      const SideLayout &moved_side = cached_layout(*cat, from_codomain ? codomain : domain,
                                                   from_codomain ? key.codomain : key.domain);
      const SideLayout &gains_side = cached_layout(*cat, from_codomain ? domain : codomain,
                                                   from_codomain ? key.domain : key.codomain);
      const Sector moved_root = from_codomain ? key.codomain : key.domain;
      const Sector gains_root = from_codomain ? key.domain : key.codomain;
      const int64_t source_stride = (from_codomain ? gains_side : moved_side).rows();

      // The layouts of the two sides of one output root pair, with the trees indexed by fusion so that
      // a recoupling finds them without a scan.
      struct OutputLayout
      {
        const SideLayout *rows = nullptr;
        const SideLayout *columns = nullptr;
        std::map<FusionTree, int64_t> row_of;
        std::map<FusionTree, int64_t> column_of;
      };

      Regrouping plan;
      std::map<BlockKey, size_t> at_of;
      const auto index_of = [&](const BlockKey &to, int64_t rows, int64_t columns) -> size_t
      {
        const auto found = at_of.find(to);
        if (found != at_of.end()) return found->second;
        const size_t index = plan.shifts.size();
        at_of.emplace(to, index);
        plan.shifts.push_back(Shift{key, to, rows, columns, {}});
        return index;
      };
      std::map<std::pair<Sector, Sector>, OutputLayout> out_layouts;

      for (size_t gains_at = 0; gains_at < gains_side.trees.size(); ++gains_at)
      {
        const FusionTree &gains_tree = gains_side.trees.at(gains_at);
        const int64_t gains = gains_side.offsets.at(gains_at + 1) - gains_side.offsets.at(gains_at);
        for (size_t moved_at = 0; moved_at < moved_side.trees.size(); ++moved_at)
        {
          const FusionTree &moved_tree = moved_side.trees.at(moved_at);
          const int64_t leg_block = block_of(moved_leg, moved_tree.sectors.back());
          const Sector charge = moved_leg.sector(leg_block);
          const int64_t copies = moved_leg.multiplicity(leg_block);
          const int64_t span = moved_side.offsets.at(moved_at + 1) - moved_side.offsets.at(moved_at);
          const int64_t inside = span / copies;
          // The tree the giving side keeps: its last vertex is gone.
          const FusionTree kept_tree = trimmed(*cat, moved_tree);
          const Sector prefix = kept_tree.coupled;
          const int64_t copy = moved_tree.size() >= 2 ? moved_tree.copies.back() : 0;
          // The vertex the leg bends through: on the codomain it is (prefix, leg) and the other root
          // closes last, on the domain the roles swap, which is the same recoupling read from the
          // other side of the tensor.
          const Sector first = from_codomain ? prefix : gains_root;
          const Sector middle = from_codomain ? charge : cat->dual(charge);
          const Sector last = from_codomain ? cat->dual(gains_root) : cat->dual(prefix);
          const at::Tensor coefficients = cat->F_symbol(first, middle, last, qtotal).to(at::kComplexDouble);
          const c10::complex<double> *entry = coefficients.data_ptr<c10::complex<double>>();
          const int64_t width = coefficients.size(1);
          const std::vector<FIndex> across = cat->f_rows(first, middle, last, qtotal);
          const std::vector<FIndex> along = cat->f_columns(first, middle, last, qtotal);
          // The index the moved tree already names, and the one it leaves free.
          const std::vector<FIndex> &named = from_codomain ? across : along;
          const std::vector<FIndex> &free = from_codomain ? along : across;
          const Sector named_root = from_codomain ? moved_root : cat->dual(moved_root);
          std::vector<int64_t> taken;
          for (size_t at = 0; at < named.size(); ++at)
            if (named.at(at).sector == named_root && named.at(at).inner == copy &&
                named.at(at).outer == key.copy)
              taken.push_back(static_cast<int64_t>(at));
          if (taken.empty()) continue;
          const int64_t source_row = moved_side.offsets.at(moved_at);
          const int64_t source_column = gains_side.offsets.at(gains_at);
          for (const int64_t index : taken)
            for (size_t column = 0; column < free.size(); ++column)
            {
              const FIndex &taken_side = free.at(column);
              const c10::complex<double> value = entry[index * width + static_cast<int64_t>(column)];
              if (value.real() == 0.0 && value.imag() == 0.0) continue;
              // The tree the taking side gains: the bent leg is its last one, and its root is the
              // outcome of the recoupling read from that side.
              const Sector extended_root = from_codomain ? cat->dual(taken_side.sector) : taken_side.sector;
              FusionTree extended = gains_tree;
              // The two sides label the same space by dual charges, so the leg carries the dual of the
              // charge it was read with once it lands on the other side.
              extended.sectors.push_back(cat->dual(charge));
              // The old root becomes an inner vertex only once the tree has one of its own: a side with
              // fewer than two legs carries no inner charge.
              if (gains_tree.size() >= 2) extended.inner.push_back(gains_root);
              // A tree of n legs carries one copy per vertex, so a tree of one leg carries none.
              if (gains_tree.size() >= 1) extended.copies.push_back(taken_side.inner);
              extended.coupled = extended_root;
              const Sector out_codomain_root = from_codomain ? prefix : extended_root;
              const Sector out_domain_root = from_codomain ? extended_root : prefix;
              const std::pair<Sector, Sector> roots{out_codomain_root, out_domain_root};
              auto found = out_layouts.find(roots);
              if (found == out_layouts.end())
              {
                OutputLayout fresh;
                fresh.rows = &cached_layout(*cat, out_codomain, out_codomain_root);
                fresh.columns = &cached_layout(*cat, out_domain, out_domain_root);
                for (size_t at = 0; at < fresh.rows->trees.size(); ++at)
                  fresh.row_of.emplace(fresh.rows->trees.at(at), static_cast<int64_t>(at));
                for (size_t at = 0; at < fresh.columns->trees.size(); ++at)
                  fresh.column_of.emplace(fresh.columns->trees.at(at), static_cast<int64_t>(at));
                found = out_layouts.emplace(roots, std::move(fresh)).first;
              }
              const OutputLayout &layout = found->second;
              const auto row_at = layout.row_of.find(from_codomain ? kept_tree : extended);
              const auto column_at = layout.column_of.find(from_codomain ? extended : kept_tree);
              if (row_at == layout.row_of.end() || column_at == layout.column_of.end()) continue;
              const int64_t row = layout.rows->offsets.at(static_cast<size_t>(row_at->second));
              const int64_t column_offset = layout.columns->offsets.at(static_cast<size_t>(column_at->second));
              const int64_t block_rows = layout.rows->rows();
              const int64_t block_columns = layout.columns->rows();
              const int64_t target_stride = block_columns;
              std::vector<Piece> &pieces =
                plan.shifts
                  .at(index_of(BlockKey{out_codomain_root, out_domain_root, taken_side.outer}, block_rows,
                               block_columns))
                  .pieces;
              const std::complex<double> factor(value.real(), value.imag());
              if (from_codomain)
              {
                for (int64_t shifted = 0; shifted < inside; ++shifted)
                  for (int64_t gain = 0; gain < gains; ++gain)
                    pieces.push_back(
                      Piece{(source_row + shifted * copies) * source_stride + source_column + gain,
                            (row + shifted) * target_stride + column_offset + gain * copies,
                            source_stride, 1, copies, factor});
              }
              else
              {
                for (int64_t gain = 0; gain < gains; ++gain)
                  for (int64_t copy_at = 0; copy_at < copies; ++copy_at)
                    pieces.push_back(
                      Piece{(source_column + gain) * source_stride + source_row + copy_at,
                            (row + gain * copies + copy_at) * target_stride + column_offset,
                            copies, 1, inside, factor});
              }
            }
        }
      }
      int64_t added = 0;
      for (const Shift &shift : plan.shifts)
        added += static_cast<int64_t>(shift.pieces.size()) * static_cast<int64_t>(sizeof(Piece));
      if (kPlanBudget < cache.bytes + added)
      {
        cache.plans.clear();
        cache.bytes = 0;
        // The matrices are keyed by the address of the plan they came from, so they go with the plans.
        matrix_cache().held.clear();
        matrix_cache().bytes = 0;
      }
      cache.bytes += added;
      return cache.plans.emplace(name, std::move(plan)).first->second;
    }

  } // namespace

  namespace
  {

    // One step of a chain: the legs the tensor has after it and the kind of regrouping that takes the
    // tensor there.
    struct Rebase
    {
      enum class Kind
      {
        exchange,
        to_codomain,
        to_domain,
      };

      Kind kind = Kind::exchange;
      int64_t position = 0;
      std::vector<Leg> codomain;
      std::vector<Leg> domain;
      const std::vector<int64_t> *sides = nullptr;
    };

    // A chain of steps together with the legs the tensor has after each of them, built by the moves
    // that compose to it.
    struct Chain
    {
      std::vector<Rebase> steps;
      std::vector<Leg> codomain;
      std::vector<Leg> domain;

      Chain(std::vector<Leg> codomain, std::vector<Leg> domain)
          : codomain(std::move(codomain)), domain(std::move(domain))
      {
      }

      // Two adjacent codomain legs trade places.
      void exchange(int64_t position)
      {
        std::swap(codomain.at(static_cast<size_t>(position)), codomain.at(static_cast<size_t>(position + 1)));
        add(Rebase::Kind::exchange, position);
      }

      // The last leg of one side is bent onto the end of the other side.
      void fold(bool from_codomain)
      {
        if (from_codomain)
        {
          domain.push_back(codomain.back());
          codomain.pop_back();
        }
        else
        {
          codomain.push_back(domain.back());
          domain.pop_back();
        }
        add(from_codomain ? Rebase::Kind::to_domain : Rebase::Kind::to_codomain, 0);
      }

      // The exchanges a bubble sort takes to reach `order`, which leaves the codomain legs in that
      // order.  A number in `order` is a position in the codomain as it stands now, not a leg
      // number: the two agree only while no leg has been lifted off the domain, so a caller that
      // has already folded legs up has to map its leg numbers through the layout it built.
      void sort(const std::vector<int64_t> &order)
      {
        std::vector<int64_t> which(codomain.size());
        std::iota(which.begin(), which.end(), 0);
        for (int64_t at = 0; at < static_cast<int64_t>(order.size()); ++at)
        {
          int64_t from = at;
          while (which.at(static_cast<size_t>(from)) != order.at(static_cast<size_t>(at))) ++from;
          while (from > at)
          {
            exchange(from - 1);
            std::swap(which.at(static_cast<size_t>(from - 1)), which.at(static_cast<size_t>(from)));
            --from;
          }
        }
      }

    private:
      void add(Rebase::Kind kind, int64_t position)
      {
        Rebase step;
        step.kind = kind;
        step.position = position;
        step.codomain = codomain;
        step.domain = domain;
        step.sides = interned_sides(codomain, domain);
        steps.push_back(std::move(step));
      }
    };

    // The regrouping of one block through one step.
    const Regrouping &regrouping_of(const Rebase &step, const std::shared_ptr<const FusionCategory> &category,
                                    const std::vector<Leg> &codomain, const std::vector<Leg> &domain,
                                    const BlockKey &key, const Sector &qtotal)
    {
      if (step.kind == Rebase::Kind::exchange)
        return exchange_regrouping(category, step.sides, codomain, domain, key, qtotal, step.position);
      return bend_regrouping(category, step.sides, codomain, domain, key, qtotal,
                             step.kind == Rebase::Kind::to_domain);
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

    // The rows a root gives on a side, built once per shape: a layout is a search over the fusion
    // trees, which is dearer than the walk it sizes, and a chain asks for the same shapes again.
    int64_t side_rows(const std::shared_ptr<const FusionCategory> &category, const std::vector<Leg> &legs,
                      const Sector &root)
    {
      static std::map<std::pair<std::shared_ptr<const FusionCategory>, std::vector<int64_t>>, int64_t> cache;
      std::vector<int64_t> key = shape_key(legs, root);
      const auto found = cache.find(std::make_pair(category, key));
      if (found != cache.end()) return found->second;
      return cache.emplace(std::make_pair(category, std::move(key)), layout_of(*category, legs, root).rows())
        .first->second;
    }

    // The buffers a chain fills and drops again. Every elementary move writes a fresh block, so
    // without this each step asks the allocator for one and the allocator costs more than the move.
    struct BufferPool
    {
      std::map<std::pair<int64_t, int64_t>, std::vector<at::Tensor>> held;
      int64_t bytes = 0;
    };

    constexpr int64_t kBufferBudget = 64LL * 1024 * 1024;

    BufferPool &buffer_pool()
    {
      thread_local BufferPool pool;
      return pool;
    }

    // A zeroed buffer of the shape asked for, reused from the pool when one is held.
    at::Tensor borrow_buffer(const at::TensorOptions &options, int64_t rows, int64_t columns)
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
    void return_buffer(at::Tensor buffer)
    {
      BufferPool &pool = buffer_pool();
      const int64_t size = buffer.numel() * buffer.element_size();
      if (pool.bytes + size > kBufferBudget) return;
      pool.bytes += size;
      pool.held[{buffer.size(0), buffer.size(1)}].push_back(std::move(buffer));
    }

    // Runs a chain of steps over every block of a tensor: the entries of a block are carried in
    // buffers from step to step, so the chain builds one tensor and not one per step.
    Tensor carried(const Tensor &tensor, const std::vector<Rebase> &steps)
    {
      if (steps.empty()) return tensor;
      const at::TensorOptions options = at::TensorOptions().dtype(tensor.dtype()).device(tensor.device());
      Tensor out(tensor.cat(), steps.back().codomain, steps.back().domain, tensor.qtotal(), tensor.dtype(),
                 tensor.device());
      MatrixCache &matrices = matrix_cache();
      for (const auto &[start, block] : tensor.blocks())
      {
        // The entries of this block as they travel: the blocks of the legs the chain has reached, and
        // the buffers it borrows for the step it is on.
        std::map<BlockKey, at::Tensor> live;
        live.emplace(start, block.contiguous());
        // The legs of the step, which the chain owns: the entries only need to read them.
        const std::vector<Leg> *codomain = &tensor.codomain();
        const std::vector<Leg> *domain = &tensor.domain();
        for (size_t at = 0; at < steps.size(); ++at)
        {
          const Rebase &step = steps.at(at);
          const bool last = at + 1 == steps.size();
          std::map<BlockKey, at::Tensor> next;
          // The block a shift lands in: the result on the last step, a wiped buffer before it.
          const auto target_of = [&](const BlockKey &key, int64_t rows, int64_t columns) -> at::Tensor & {
            if (last) return out.block(key.codomain, key.domain, key.copy);
            const auto found = next.find(key);
            if (found != next.end()) return found->second;
            return next.emplace(key, borrow_buffer(options, rows, columns)).first->second;
          };
          for (const auto &[key, buffer] : live)
          {
            const Regrouping &plan =
              regrouping_of(step, tensor.cat(), *codomain, *domain, key, tensor.qtotal());
            // Every target block of the plan, with the shape the plan recorded for it; an exchange
            // leaves a block the size it already has.
            std::vector<std::pair<BlockKey, std::pair<int64_t, int64_t>>> targets;
            for (const Shift &shift : plan.shifts)
              if (std::none_of(targets.begin(), targets.end(),
                               [&](const auto &seen) { return seen.first == shift.to; }))
                targets.emplace_back(shift.to, std::make_pair(shift.rows == 0 ? buffer.size(0) : shift.rows,
                                                              shift.columns == 0 ? buffer.size(1) : shift.columns));
            for (const auto &[target, shape] : targets)
              add_into(plan, target, buffer, target_of(target, shape.first, shape.second), matrices);
          }
          codomain = &step.codomain;
          domain = &step.domain;
          // The buffer of the first step is the block of the tensor itself or a view of it, so it is
          // the one buffer the chain did not borrow.
          if (at > 0)
            for (auto &entry : live) return_buffer(std::move(entry.second));
          live = std::move(next);
        }
      }
      return out;
    }

  } // namespace

  Tensor Tensor::place(const std::vector<int64_t> &above) const
  {
    const int64_t total = nlegs();
    std::vector<bool> named(static_cast<size_t>(total), false);
    for (const int64_t leg : above)
    {
      if (leg < 0 || leg >= total)
        throw std::invalid_argument("the leg " + std::to_string(leg) + " does not exist, the tensor has " +
                                    std::to_string(total) + " legs");
      if (named.at(static_cast<size_t>(leg)))
        throw std::invalid_argument("the leg " + std::to_string(leg) + " is named twice");
      named.at(static_cast<size_t>(leg)) = true;
    }
    std::vector<int64_t> below;
    below.reserve(static_cast<size_t>(total - static_cast<int64_t>(above.size())));
    for (int64_t leg = 0; leg < total; ++leg)
      if (!named.at(static_cast<size_t>(leg))) below.push_back(leg);
    return place(above, below);
  }

  Tensor Tensor::place(const std::vector<int64_t> &above, const std::vector<int64_t> &below) const
  {
    const int64_t height = static_cast<int64_t>(codomain_.size());
    const int64_t depth = static_cast<int64_t>(domain_.size());
    const int64_t total = height + depth;
    // A layout names every leg once: a number that is out of range or repeated, or a leg that is
    // named on neither side, leaves the result without a meaning.
    std::vector<bool> named(static_cast<size_t>(total), false);
    for (const std::vector<int64_t> *side : {&above, &below})
      for (const int64_t leg : *side)
      {
        if (leg < 0 || leg >= total)
          throw std::invalid_argument("the leg " + std::to_string(leg) + " does not exist, the tensor has " +
                                      std::to_string(total) + " legs");
        if (named.at(static_cast<size_t>(leg)))
          throw std::invalid_argument("the leg " + std::to_string(leg) + " is named twice");
        named.at(static_cast<size_t>(leg)) = true;
      }
    if (static_cast<int64_t>(above.size() + below.size()) != total)
    {
      int64_t left_out = 0;
      while (named.at(static_cast<size_t>(left_out))) ++left_out;
      throw std::invalid_argument("the leg " + std::to_string(left_out) + " is named on neither side");
    }
    Chain chain(codomain_, domain_);
    // The front of the domain can be left alone when nothing has to come down and the legs named
    // below start with the legs that are already there: then only the back of the domain is lifted,
    // and every leg in front of it is neither lifted nor folded back.
    bool descends = false;
    for (size_t at = 0; !descends && at < below.size(); ++at) descends = below.at(at) < height;
    int64_t kept = 0;
    while (!descends && kept < depth && kept < static_cast<int64_t>(below.size()) &&
           below.at(static_cast<size_t>(kept)) == height + kept)
      ++kept;
    if (kept > 0)
    {
      // The legs behind the ones that stay are lifted off the end of the domain, one fold each; the
      // folds take the end of the codomain first, so the legs that come back down are named last
      // first, and they land on the domain in the order the caller named them in.
      for (int64_t at = kept; at < depth; ++at) chain.fold(false);
      const auto position_of = [height, depth](const int64_t leg) {
        return leg < height ? leg : height + (depth - 1 - (leg - height));
      };
      std::vector<int64_t> order;
      order.reserve(static_cast<size_t>(total - kept));
      for (const int64_t leg : above) order.push_back(position_of(leg));
      for (auto leg = below.rbegin(); leg != below.rend() - kept; ++leg) order.push_back(position_of(*leg));
      chain.sort(order);
      for (int64_t at = kept; at < static_cast<int64_t>(below.size()); ++at) chain.fold(true);
      return carried(*this, chain.steps);
    }
    // Otherwise every leg is bent onto the codomain, the only side that can be ordered; a leg taken
    // off its end goes back last named first, so the folds below land the domain in the named order.
    while (!chain.domain.empty()) chain.fold(false);
    const auto position_of = [height, depth](const int64_t leg) {
      return leg < height ? leg : height + (depth - 1 - (leg - height));
    };
    std::vector<int64_t> order;
    order.reserve(static_cast<size_t>(total));
    for (const int64_t leg : above) order.push_back(position_of(leg));
    for (auto leg = below.rbegin(); leg != below.rend(); ++leg) order.push_back(position_of(*leg));
    chain.sort(order);
    for (int64_t at = 0; at < static_cast<int64_t>(below.size()); ++at) chain.fold(true);
    return carried(*this, chain.steps);
  }

  Tensor Tensor::conj() const
  {
    // The rows of the conjugate are the columns of the tensor: the two sides lay their rows out
    // through the same tree order, so the block turns over as it conjugates.
    Tensor out(cat_, domain_, codomain_, cat_->dual(qtotal_), dtype_, device_);
    for (const auto &[key, block] : blocks_)
      out.block(key.domain, key.codomain, key.copy).copy_(block.conj().transpose(0, 1));
    return out;
  }

  std::vector<int64_t> codomain_legs(const Tensor &tensor)
  {
    std::vector<int64_t> legs(static_cast<size_t>(tensor.codomain().size()));
    for (int64_t at = 0; at < static_cast<int64_t>(legs.size()); ++at) legs.at(static_cast<size_t>(at)) = at;
    return legs;
  }

  std::vector<int64_t> domain_legs(const Tensor &tensor)
  {
    const int64_t above = static_cast<int64_t>(tensor.codomain().size());
    std::vector<int64_t> legs(static_cast<size_t>(tensor.domain().size()));
    for (int64_t at = 0; at < static_cast<int64_t>(legs.size()); ++at)
      legs.at(static_cast<size_t>(at)) = above + at;
    return legs;
  }

  Tensor Tensor::clone() const
  {
    Tensor out(*this);
    for (auto &[key, block] : out.blocks_) block = block.clone();
    return out;
  }

  Tensor &Tensor::operator+=(const Tensor &other)
  {
    if (!same_group(*cat_, *other.cat_) || !(qtotal_ == other.qtotal_) ||
        !(codomain_ == other.codomain_) || !(domain_ == other.domain_))
      throw std::invalid_argument("a sum needs both tensors with the same legs and the same charge");
    for (const auto &[key, block] : other.blocks_)
      blocks_.at(key).add_(block);
    return *this;
  }

  Tensor &Tensor::operator-=(const Tensor &other)
  {
    if (!same_group(*cat_, *other.cat_) || !(qtotal_ == other.qtotal_) ||
        !(codomain_ == other.codomain_) || !(domain_ == other.domain_))
      throw std::invalid_argument("a difference needs both tensors with the same legs and the same "
                                  "charge");
    for (const auto &[key, block] : other.blocks_)
      blocks_.at(key).sub_(block);
    return *this;
  }

  Tensor &Tensor::operator*=(double scalar)
  {
    for (auto &[key, block] : blocks_) block.mul_(scalar);
    return *this;
  }

  Tensor &Tensor::operator*=(std::complex<double> scalar)
  {
    if (scalar.imag() == 0.0) return *this *= scalar.real();
    for (auto &[key, block] : blocks_)
    {
      if (!block.is_complex())
        throw std::invalid_argument("a block of real entries cannot take the complex factor " +
                                    std::to_string(scalar.real()) + "+" +
                                    std::to_string(scalar.imag()) + "i");
      block.mul_(c10::complex<double>(scalar.real(), scalar.imag()));
    }
    return *this;
  }

  Tensor &Tensor::operator/=(double scalar)
  {
    if (scalar == 0.0) throw std::invalid_argument("a tensor cannot be divided by zero");
    for (auto &[key, block] : blocks_) block.div_(scalar);
    return *this;
  }

} // namespace sym
