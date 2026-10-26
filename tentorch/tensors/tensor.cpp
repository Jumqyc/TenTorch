#include "tensor.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace sym
{

  namespace
  {

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
    for (const BlockKey &key : keys)
      blocks_.emplace(key, at::zeros({rows_of(key.codomain).rows(), columns_of(key.domain).rows()}, options));
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

    // The width index a tree of a swapped block reads its rows at: the exchange swaps the two width
    // digits that sit at the pair, and the mixed radix of the widths has the last leg fastest.
    int64_t exchanged_width(const std::vector<int64_t> &shape, int64_t p, int64_t at)
    {
      std::vector<int64_t> digits(shape.size(), 0);
      for (size_t axis = shape.size(); axis-- > 0;)
      {
        digits.at(axis) = at % shape.at(axis);
        at /= shape.at(axis);
      }
      std::swap(digits.at(static_cast<size_t>(p)), digits.at(static_cast<size_t>(p + 1)));
      int64_t out = 0;
      for (size_t axis = 0; axis < shape.size(); ++axis) out = out * shape.at(axis) + digits.at(axis);
      return out;
    }

    // One class of trees of an exchange: every old tree of the class lands on a new tree of the same
    // class, whose recoupling row contracts the old rows.  `tree_old` and `tree_new` are the indices
    // of the trees in the recoupling matrix, `row_old` and `row_new` where those trees start in the
    // two blocks.
    struct SwapClass
    {
      const std::vector<int64_t> &tree_old;
      const std::vector<int64_t> &tree_new;
      const std::vector<int64_t> &row_old;
      const std::vector<int64_t> &row_new;
      const std::vector<int64_t> &width_of;
      const at::Tensor &recoupling;
      int64_t before;
      int64_t width;
      int64_t columns;
    };

    template <typename scalar>
    void add_class(const SwapClass &step, const scalar *source, scalar *target, int64_t source_stride,
                   int64_t target_stride)
    {
      const scalar *recoupling = step.recoupling.data_ptr<scalar>();
      for (size_t at = 0; at < step.tree_new.size(); ++at)
      {
        const scalar *row = recoupling + step.tree_new.at(at) * step.before;
        for (int64_t width = 0; width < step.width; ++width)
          for (size_t old = 0; old < step.tree_old.size(); ++old)
          {
            const scalar factor = row[step.tree_old.at(old)];
            if (factor == scalar(0)) continue;
            const scalar *from = source + (step.row_old.at(old) + width) * source_stride;
            scalar *to = target + (step.row_new.at(at) + step.width_of.at(static_cast<size_t>(width))) * target_stride;
            for (int64_t entry = 0; entry < step.columns; ++entry) to[entry] += factor * from[entry];
          }
      }
    }

    // A class of trees of an exchange: the trees of a block that share every multiplicity outside
    // the pair, the rows they start at, and how their width digits are read after the exchange.
    struct ExchangeClass
    {
      std::vector<int64_t> old_trees;
      std::vector<int64_t> new_trees;
      std::vector<int64_t> old_rows;
      std::vector<int64_t> new_rows;
      std::vector<int64_t> width_of;
      std::vector<int64_t> shape;
      int64_t width = 1;
    };

    // Everything an exchange does to a block: the recoupling over its trees and the width classes.
    struct ExchangePlan
    {
      int64_t before = 0;
      int64_t after = 0;
      int64_t rows_before = 0;
      int64_t rows_after = 0;
      int64_t columns = 0;
      at::Tensor recoupling;
      std::vector<ExchangeClass> classes;
    };

    // The key of a plan: the position and the leg structures decide the trees and the coefficients,
    // the block and the tensor options decide the shape and where the matrix is carried.
    std::vector<int64_t> plan_key(const std::vector<Leg> &codomain, const std::vector<Leg> &domain,
                                  const BlockKey &key, int64_t p, const at::Tensor &matrix)
    {
      std::vector<int64_t> fingerprint{p, static_cast<int64_t>(matrix.device().index()),
                                       static_cast<int64_t>(matrix.scalar_type())};
      for (const std::vector<Leg> *legs : {&codomain, &domain})
      {
        fingerprint.push_back(static_cast<int64_t>(legs->size()));
        for (const Leg &leg : *legs)
        {
          fingerprint.push_back(leg.nblocks());
          for (int64_t block = 0; block < leg.nblocks(); ++block)
          {
            const Sector &sector = leg.sector(block);
            fingerprint.push_back(static_cast<int64_t>(sector.size()));
            fingerprint.insert(fingerprint.end(), sector.begin(), sector.end());
            fingerprint.push_back(leg.multiplicity(block));
          }
        }
      }
      for (const Sector *charge : {&key.codomain, &key.domain})
      {
        fingerprint.push_back(static_cast<int64_t>(charge->size()));
        fingerprint.insert(fingerprint.end(), charge->begin(), charge->end());
      }
      fingerprint.push_back(key.copy);
      return fingerprint;
    }

    // The plan of exchanging the codomain legs `p` and `p + 1` of a block, built once per structure:
    // the coefficients are category data, so a later exchange of the same shape reads them.
    const ExchangePlan &exchange_plan(const std::shared_ptr<const FusionCategory> &cat,
                                      const std::vector<Leg> &codomain, const std::vector<Leg> &domain,
                                      const BlockKey &key, int64_t p, const at::Tensor &matrix)
    {
      using PlanKey = std::pair<std::shared_ptr<const FusionCategory>, std::vector<int64_t>>;
      static std::map<PlanKey, ExchangePlan> plans;
      const PlanKey name{cat, plan_key(codomain, domain, key, p, matrix)};
      const auto held = plans.find(name);
      if (held != plans.end()) return held->second;

      const int64_t count = static_cast<int64_t>(codomain.size());
      std::vector<Leg> exchanged = codomain;
      std::swap(exchanged.at(static_cast<size_t>(p)), exchanged.at(static_cast<size_t>(p + 1)));
      const SideLayout source = layout_of(*cat, codomain, key.codomain);
      const SideLayout target = layout_of(*cat, exchanged, key.codomain);
      ExchangePlan plan;
      plan.before = static_cast<int64_t>(source.trees.size());
      plan.after = static_cast<int64_t>(target.trees.size());
      plan.rows_before = source.rows();
      plan.rows_after = target.rows();
      plan.columns = matrix.size(1);
      const at::TensorOptions index_options = at::TensorOptions().dtype(at::kLong).device(matrix.device());

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
      const auto moved = [p](int64_t axis)
      { return axis == p ? p + 1 : axis == p + 1 ? p : axis; };

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
      for (int64_t at = 0; at < plan.before; ++at)
        groups[signature(source.trees.at(static_cast<size_t>(at)), false)].push_back(at);
      std::map<std::vector<int64_t>, std::vector<int64_t>> partners;
      for (int64_t at = 0; at < plan.after; ++at)
        partners[signature(target.trees.at(static_cast<size_t>(at)), true)].push_back(at);

      // One recoupling per group, packed into the tree space of the block.
      std::vector<at::Tensor> pieces;
      std::vector<int64_t> row_order;
      std::vector<int64_t> column_order;
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

        const at::Tensor straight = cat->F_symbol(y, b, c, d).to(matrix.device());
        const at::Tensor braided = cat->F_symbol(y, c, b, d).to(matrix.device());
        TORCH_CHECK(straight.size(0) == static_cast<int64_t>(olds.size()) &&
                        braided.size(0) == static_cast<int64_t>(news.size()),
                    "the exchange does not match the trees of the block");

        // The braiding between the two intermediate bases, on the copies of one outcome only.
        std::map<Sector, int64_t> offset;
        int64_t at = 0;
        for (const Outcome &outcome : cat->outcomes(b, c))
        {
          offset.emplace(outcome.sector, at);
          at += outcome.multiplicity;
        }
        const std::vector<FIndex> from = cat->f_columns(y, b, c, d);
        const std::vector<FIndex> to = cat->f_columns(y, c, b, d);
        const at::Tensor braid = cat->R_symbol(b, c).to(matrix.device());
        std::vector<int64_t> places;
        std::vector<int64_t> sources;
        for (int64_t i = 0; i < static_cast<int64_t>(to.size()); ++i)
        {
          for (int64_t j = 0; j < static_cast<int64_t>(from.size()); ++j)
          {
            if (to.at(static_cast<size_t>(i)).sector != from.at(static_cast<size_t>(j)).sector) continue;
            if (to.at(static_cast<size_t>(i)).outer != from.at(static_cast<size_t>(j)).outer) continue;
            places.push_back(i * static_cast<int64_t>(from.size()) + j);
            sources.push_back((offset.at(to.at(static_cast<size_t>(i)).sector) +
                               to.at(static_cast<size_t>(i)).inner) *
                                braid.size(1) +
                              offset.at(from.at(static_cast<size_t>(j)).sector) + from.at(static_cast<size_t>(j)).inner);
          }
        }
        at::Tensor middle = at::zeros({static_cast<int64_t>(to.size()) * static_cast<int64_t>(from.size())},
                                      braid.options());
        middle.index_put_({at::tensor(places, index_options)},
                          braid.flatten().index_select(0, at::tensor(sources, index_options)));
        middle = middle.reshape({static_cast<int64_t>(to.size()), static_cast<int64_t>(from.size())});
        pieces.push_back(at::matmul(braided, at::matmul(middle, straight.conj().transpose(0, 1))));
        row_order.insert(row_order.end(), news.begin(), news.end());
        column_order.insert(column_order.end(), olds.begin(), olds.end());
      }

      // Scatter the pieces onto the trees they belong to: the groups partition the trees, so both
      // orders are permutations and each piece lands in one region.
      at::Tensor packed = at::zeros({plan.after, plan.before}, matrix.options());
      int64_t row_at = 0;
      int64_t column_at = 0;
      for (const at::Tensor &piece : pieces)
      {
        const int64_t rows = piece.size(0);
        const int64_t width = piece.size(1);
        packed.slice(0, row_at, row_at + rows).slice(1, column_at, column_at + width).copy_(piece);
        row_at += rows;
        column_at += width;
      }
      at::Tensor stacked = at::zeros({plan.after, plan.before}, matrix.options());
      stacked.index_copy_(0, at::tensor(row_order, index_options), packed);
      plan.recoupling = at::zeros({plan.after, plan.before}, matrix.options());
      plan.recoupling.index_copy_(1, at::tensor(column_order, index_options), stacked);

      // Trees whose legs carry the same multiplicities move in one piece: the width index is a mixed
      // radix over those multiplicities, and the exchange only moves its digits, which keeps the size
      // of a class the same on both sides even when the two legs carry different numbers of copies.
      std::map<std::vector<int64_t>, std::vector<int64_t>> before_classes;
      std::map<std::vector<int64_t>, std::vector<int64_t>> after_classes;
      std::vector<std::vector<int64_t>> after_shapes(static_cast<size_t>(plan.after));
      for (int64_t at = 0; at < plan.before; ++at)
        before_classes[shape_of(source.trees.at(static_cast<size_t>(at)), codomain)].push_back(at);
      for (int64_t at = 0; at < plan.after; ++at)
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
      for (const auto &[shape, olds] : before_classes)
      {
        const auto partner = after_classes.find(shape);
        if (partner == after_classes.end())
          throw std::logic_error("the exchange has no image for a block of trees");
        ExchangeClass klass;
        klass.old_trees = olds;
        klass.new_trees = partner->second;
        klass.shape = shape;
        for (const int64_t tree : klass.old_trees) klass.old_rows.push_back(source.offsets.at(static_cast<size_t>(tree)));
        for (const int64_t tree : klass.new_trees) klass.new_rows.push_back(target.offsets.at(static_cast<size_t>(tree)));
        for (const int64_t factor : shape) klass.width *= factor;
        // The width of a class reads the copies of a leg in the order of the legs, so an index moves
        // its digits to the positions the exchanged legs put them at, under that side's radices.
        const std::vector<int64_t> &target_shape = after_shapes.at(static_cast<size_t>(klass.new_trees.front()));
        klass.width_of.resize(static_cast<size_t>(klass.width));
        for (int64_t at = 0; at < klass.width; ++at)
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
          klass.width_of.at(static_cast<size_t>(at)) = index;
        }
        plan.classes.push_back(std::move(klass));
      }
      return plans.emplace(name, std::move(plan)).first->second;
    }

    // The exchange of two adjacent codomain legs as a square matrix over the rows of a block: the
    // entry (new row, old row) is the coefficient the exchange copies that pair with.
    at::Tensor exchange_matrix(const ExchangePlan &plan, const at::TensorOptions &options)
    {
      const at::TensorOptions index_options = at::TensorOptions().dtype(at::kLong).device(options.device());
      at::Tensor out = at::zeros({plan.rows_after, plan.rows_before}, options);
      for (const ExchangeClass &klass : plan.classes)
      {
        const at::Tensor piece =
          plan.recoupling
            .index_select(0, at::tensor(klass.new_trees, index_options))
            .index_select(1, at::tensor(klass.old_trees, index_options));
        std::vector<int64_t> rows_at, columns_at;
        for (int64_t width = 0; width < klass.width; ++width)
          for (size_t at = 0; at < klass.new_trees.size(); ++at)
            for (size_t old = 0; old < klass.old_trees.size(); ++old)
            {
              rows_at.push_back(klass.new_rows.at(at) + klass.width_of.at(static_cast<size_t>(width)));
              columns_at.push_back(klass.old_rows.at(old) + width);
            }
        out.index_put_({at::tensor(rows_at, index_options), at::tensor(columns_at, index_options)},
                       piece.flatten().repeat({klass.width}));
      }
      return out;
    }

    // The key of a permutation: the leg structures and the block decide the trees and the
    // coefficients, the order decides where each tree lands.
    // One exchange of a block, walked over its entries: each class of trees is a small recoupling
    // applied to the rows of the trees it joins, so the cost is the size of the block and not the
    // square of its rows.
    template <typename scalar>
    void exchange_walk(const ExchangePlan &plan, const at::Tensor &source, at::Tensor &target)
    {
      const int64_t columns = source.size(1);
      const int64_t before = plan.recoupling.size(0);
      const scalar *from = source.data_ptr<scalar>();
      scalar *to = target.data_ptr<scalar>();
      for (const ExchangeClass &klass : plan.classes)
      {
        const SwapClass step{klass.old_trees, klass.new_trees, klass.old_rows, klass.new_rows,
                             klass.width_of, plan.recoupling, before, klass.width, columns};
        add_class(step, from, to, source.stride(0), target.stride(0));
      }
    }

    // One exchange of a block, written into `target`, which is cleared first.
    void exchanged(const ExchangePlan &plan, const at::Tensor &matrix, at::Tensor &target)
    {
      target.zero_();
      if (matrix.scalar_type() == at::kComplexDouble) exchange_walk<c10::complex<double>>(plan, matrix, target);
      else exchange_walk<double>(plan, matrix, target);
    }

  } // namespace

  Tensor Tensor::swap(int64_t position) const
  {
    const int64_t count = static_cast<int64_t>(codomain_.size());
    if (position < 0 || position + 1 >= count)
    {
      throw std::invalid_argument("a swap needs two adjacent codomain legs, got position " +
                                  std::to_string(position));
    }
    const int64_t p = position;
    std::vector<Leg> codomain = codomain_;
    std::swap(codomain.at(static_cast<size_t>(p)), codomain.at(static_cast<size_t>(p + 1)));
    Tensor out(cat_, codomain, domain_, qtotal_, dtype(), device());
    for (const auto &[key, matrix] : blocks_)
    {
      const ExchangePlan &plan = exchange_plan(cat_, codomain_, domain_, key, p, matrix);
      if (plan.classes.empty()) continue;
      at::Tensor &result = out.block(key.codomain, key.domain, key.copy);
      const at::TensorOptions index_options = at::TensorOptions().dtype(at::kLong).device(matrix.device());
      // A block of the engine is a contiguous host matrix, so a class is a walk over its rows
      // instead of a chain of tensor expressions; anything else keeps the tensor algebra.
      const bool walk = matrix.is_contiguous() && result.is_contiguous() && matrix.device().is_cpu() &&
                        result.device().is_cpu() &&
                        (matrix.scalar_type() == at::kComplexDouble || matrix.scalar_type() == at::kDouble);
      for (const ExchangeClass &klass : plan.classes)
      {
        if (walk)
        {
          const SwapClass step{klass.old_trees, klass.new_trees, klass.old_rows, klass.new_rows,
                               klass.width_of, plan.recoupling, plan.before, klass.width, plan.columns};
          if (matrix.scalar_type() == at::kComplexDouble)
            add_class(step, matrix.data_ptr<c10::complex<double>>(), result.data_ptr<c10::complex<double>>(),
                      matrix.stride(0), result.stride(0));
          else
            add_class(step, matrix.data_ptr<double>(), result.data_ptr<double>(), matrix.stride(0),
                      result.stride(0));
          continue;
        }
        // The rows of a class land on other rows of the same class: gather the old rows in the
        // order the widths are walked, recouple every width at once, and scatter the new rows.
        std::vector<int64_t> gather;
        std::vector<int64_t> scatter;
        const int64_t trees_old = static_cast<int64_t>(klass.old_trees.size());
        const int64_t trees_new = static_cast<int64_t>(klass.new_trees.size());
        for (int64_t width = 0; width < klass.width; ++width)
        {
          for (const int64_t row : klass.old_rows) gather.push_back(row + width);
          for (size_t at = 0; at < klass.new_rows.size(); ++at)
            scatter.push_back(klass.new_rows.at(at) + klass.width_of.at(static_cast<size_t>(width)));
        }
        const at::Tensor piece = matrix.index_select(0, at::tensor(gather, index_options))
                                   .reshape({klass.width, trees_old, plan.columns});
        const at::Tensor selected = plan.recoupling.index_select(0, at::tensor(klass.new_trees, index_options))
                                      .index_select(1, at::tensor(klass.old_trees, index_options));
        const at::Tensor moved = at::matmul(selected.unsqueeze(0).expand({klass.width, -1, -1}), piece)
                                   .reshape({klass.width * trees_new, plan.columns});
        result.index_copy_(0, at::tensor(scatter, index_options), moved);
      }
    }
    return out;
  }

  namespace
  {

    // The charge the codomain legs before the last one fuse to: the root the codomain keeps when
    // its last leg is bent across.
    Sector before_last(const FusionCategory &category, const FusionTree &tree)
    {
      const int64_t count = tree.size();
      if (count <= 1) return category.trivial();
      if (count == 2) return tree.sectors.at(0);
      return tree.inner.at(static_cast<size_t>(count - 3));
    }

    // The layouts of the two sides of an output root, with the trees indexed by fusion so that a
    // recoupling finds them without a scan.
    struct OutputLayout
    {
      SideLayout rows;
      SideLayout columns;
      std::map<FusionTree, int64_t> row_of;
      std::map<FusionTree, int64_t> column_of;
    };

    // One contribution of a bend: the recoupling entry `factor` copies the region of the source
    // block that the moved tree and the gaining tree name into the region of the target block that
    // the two resulting trees name.  The counts `inside`, `gains` and `copies` are the mixed radix
    // both sides read the same entries with: on the giving side the copy digit sits between the
    // inside index and the gaining index, on the taking side the two are exchanged.
    struct BendStep
    {
      at::Tensor *target;
      const at::Tensor *source;
      int64_t row_at;
      int64_t column_at;
      int64_t source_row;
      int64_t source_column;
      int64_t inside;
      int64_t gains;
      int64_t copies;
      bool moved_rows;
      std::complex<double> factor;
    };

    template <typename scalar>
    void add_entries(const BendStep &step,
                     const scalar *source,
                     scalar *target,
                     int64_t source_stride,
                     int64_t target_stride,
                     scalar factor)
    {
      for (int64_t inside = 0; inside < step.inside; ++inside)
        for (int64_t gain = 0; gain < step.gains; ++gain)
          for (int64_t copy = 0; copy < step.copies; ++copy)
          {
            if (step.moved_rows)
              target[(step.row_at + inside) * target_stride + step.column_at + gain * step.copies + copy] +=
                factor * source[(step.source_row + inside * step.copies + copy) * source_stride + step.source_column + gain];
            else
              target[(step.row_at + gain * step.copies + copy) * target_stride + step.column_at + inside] +=
                factor * source[(step.source_column + gain) * source_stride + step.source_row + inside * step.copies + copy];
          }
    }

    // Adds one step to its target block.  A block of the engine is a contiguous matrix, so a step is
    // a walk over three indices instead of a chain of tensor expressions; a block that is not a
    // contiguous host matrix of double or complex double keeps the tensor algebra.
    void add_step(const BendStep &step)
    {
      at::Tensor &target = *step.target;
      const at::Tensor &source = *step.source;
      const int64_t rows = step.moved_rows ? step.inside : step.gains * step.copies;
      const int64_t columns = step.moved_rows ? step.gains * step.copies : step.inside;
      const bool entry = target.scalar_type() == at::kComplexDouble || target.scalar_type() == at::kDouble;
      if (!entry || !source.device().is_cpu() || !target.device().is_cpu() ||
          !source.is_contiguous() || !target.is_contiguous())
      {
        const at::Tensor piece =
          step.moved_rows
            ? source.narrow(0, step.source_row, step.inside * step.copies)
                .narrow(1, step.source_column, step.gains)
                .reshape({step.inside, step.copies, step.gains})
                .transpose(1, 2)
                .reshape({step.inside, step.gains * step.copies})
            : source.narrow(0, step.source_column, step.gains)
                .narrow(1, step.source_row, step.inside * step.copies)
                .reshape({step.gains, step.inside, step.copies})
                .transpose(1, 2)
                .reshape({step.gains * step.copies, step.inside});
        const at::Scalar factor =
          target.is_complex()
            ? at::Scalar(c10::complex<double>(step.factor.real(), step.factor.imag()))
            : at::Scalar(step.factor.real());
        target.narrow(0, step.row_at, rows).narrow(1, step.column_at, columns).add_(piece, factor);
        return;
      }
      if (target.scalar_type() == at::kComplexDouble)
        add_entries(step,
                    source.data_ptr<c10::complex<double>>(),
                    target.data_ptr<c10::complex<double>>(),
                    source.stride(0),
                    target.stride(0),
                    c10::complex<double>(step.factor.real(), step.factor.imag()));
      else
        add_entries(step, source.data_ptr<double>(), target.data_ptr<double>(), source.stride(0), target.stride(0),
                    step.factor.real());
    }

    // Bends the last leg of one side onto the end of the other side.
    // `from_codomain` names the side that gives the leg and nothing else: a tree is a tree, so the
    // two directions run the same recoupling with the roles of the two trees exchanged.
    Tensor bend(const Tensor &tensor, bool from_codomain)
    {
      const std::shared_ptr<const FusionCategory> cat = tensor.cat();
      const Sector qtotal = tensor.qtotal();
      const std::vector<Leg> &giving = from_codomain ? tensor.codomain() : tensor.domain();
      const std::vector<Leg> &taking = from_codomain ? tensor.domain() : tensor.codomain();
      const Leg held = giving.back();
      const std::vector<Leg> kept_giving(giving.begin(), giving.end() - 1);
      std::vector<Leg> kept_taking = taking;
      kept_taking.push_back(held);
      const std::vector<Leg> out_codomain = from_codomain ? kept_giving : kept_taking;
      const std::vector<Leg> out_domain = from_codomain ? kept_taking : kept_giving;
      Tensor out(cat, out_codomain, out_domain, qtotal, tensor.dtype(), tensor.device());

      // An output layout also carries the index of each of its trees so that a recoupling does not
      // search for them; the source layouts come from the cache on the tensor.
      std::map<std::pair<Sector, Sector>, OutputLayout> out_layouts;
      std::vector<BendStep> steps;

      for (const auto &[key, matrix] : tensor.blocks())
      {
        const SideLayout &rows = tensor.rows_of(key.codomain);
        const SideLayout &columns = tensor.columns_of(key.domain);
        // The tree that loses its last vertex, and the one that takes a vertex.
        const SideLayout &moved_layout = from_codomain ? rows : columns;
        const SideLayout &gains_layout = from_codomain ? columns : rows;
        const Sector moved_root = from_codomain ? key.codomain : key.domain;
        const Sector gains_root = from_codomain ? key.domain : key.codomain;
        for (size_t gains_at = 0; gains_at < gains_layout.trees.size(); ++gains_at)
        {
          const FusionTree &gains_tree = gains_layout.trees.at(gains_at);
          const int64_t gains_width =
            gains_layout.offsets.at(gains_at + 1) - gains_layout.offsets.at(gains_at);
          for (size_t moved_at = 0; moved_at < moved_layout.trees.size(); ++moved_at)
          {
            const FusionTree &moved_tree = moved_layout.trees.at(moved_at);
            const int64_t leg_block = block_of(held, moved_tree.sectors.back());
            const Sector charge = held.sector(leg_block);
            const int64_t copies = held.multiplicity(leg_block);
            const int64_t span = moved_layout.offsets.at(moved_at + 1) - moved_layout.offsets.at(moved_at);
            const int64_t inside = span / copies;
            const Sector prefix = before_last(*cat, moved_tree);
            const int64_t copy = moved_tree.size() >= 2 ? moved_tree.copies.back() : 0;
            // The vertex the leg bends through: on the codomain it is (prefix, leg) and the other
            // root closes last, on the domain the roles swap, which is the same recoupling read
            // from the other side of the tensor.
            const Sector first = from_codomain ? prefix : gains_root;
            const Sector middle = from_codomain ? charge : cat->dual(charge);
            const Sector last = from_codomain ? cat->dual(gains_root) : cat->dual(prefix);
            const at::Tensor recoupling = cat->F_symbol(first, middle, last, qtotal);
            const at::Tensor coefficients = recoupling.to(at::kCPU)
                                              .to(recoupling.is_complex() ? at::kComplexDouble : at::kDouble)
                                              .contiguous();
            const c10::complex<double> *complex_entry = nullptr;
            const double *real_entry = nullptr;
            if (recoupling.is_complex())
              complex_entry = coefficients.data_ptr<c10::complex<double>>();
            else
              real_entry = coefficients.data_ptr<double>();
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
            // The tree the giving side keeps: its last vertex is gone.
            FusionTree trimmed;
            trimmed.sectors = moved_tree.sectors;
            trimmed.sectors.pop_back();
            trimmed.inner = moved_tree.inner;
            if (!trimmed.inner.empty()) trimmed.inner.pop_back();
            trimmed.copies = moved_tree.copies;
            if (!trimmed.copies.empty()) trimmed.copies.pop_back();
            trimmed.coupled = prefix;
            for (const int64_t index : taken)
            {
              for (size_t column = 0; column < free.size(); ++column)
              {
                const FIndex &taken_side = free.at(column);
                // The tree the taking side gains: the bent leg is its last one, and its root is
                // the outcome of the recoupling read from that side.
                const Sector extended_root =
                  from_codomain ? cat->dual(taken_side.sector) : taken_side.sector;
                FusionTree extended = gains_tree;
                // The two sides label the same space by dual charges, so the leg carries the dual of
                // the charge it was read with once it lands on the other side.
                extended.sectors.push_back(cat->dual(charge));
                // The old root becomes an inner vertex only once the tree has one of its own: a
                // side with fewer than two legs carries no inner charge.
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
                  fresh.rows = out.rows_of(out_codomain_root);
                  fresh.columns = out.columns_of(out_domain_root);
                  for (size_t at = 0; at < fresh.rows.trees.size(); ++at)
                    fresh.row_of.emplace(fresh.rows.trees.at(at), static_cast<int64_t>(at));
                  for (size_t at = 0; at < fresh.columns.trees.size(); ++at)
                    fresh.column_of.emplace(fresh.columns.trees.at(at), static_cast<int64_t>(at));
                  found = out_layouts.emplace(roots, std::move(fresh)).first;
                }
                const OutputLayout &layout = found->second;
                const auto row_at = layout.row_of.find(from_codomain ? trimmed : extended);
                const auto column_at = layout.column_of.find(from_codomain ? extended : trimmed);
                if (row_at == layout.row_of.end() || column_at == layout.column_of.end()) continue;
                const std::complex<double> factor =
                  complex_entry != nullptr
                    ? std::complex<double>(complex_entry[index * width + static_cast<int64_t>(column)].real(),
                                           complex_entry[index * width + static_cast<int64_t>(column)].imag())
                    : std::complex<double>(real_entry[index * width + static_cast<int64_t>(column)], 0.0);
                if (factor.real() == 0.0 && factor.imag() == 0.0) continue;
                BendStep step;
                step.target = &out.block(out_codomain_root, out_domain_root, taken_side.outer);
                step.source = &matrix;
                step.row_at = layout.rows.offsets.at(static_cast<size_t>(row_at->second));
                step.column_at = layout.columns.offsets.at(static_cast<size_t>(column_at->second));
                step.source_row = moved_layout.offsets.at(moved_at);
                step.source_column = gains_layout.offsets.at(gains_at);
                step.inside = inside;
                step.gains = gains_width;
                step.copies = copies;
                step.moved_rows = from_codomain;
                step.factor = factor;
                steps.push_back(step);
              }
            }
          }
        }
      }
      for (const BendStep &step : steps) add_step(step);
      return out;
    }

  } // namespace

  Tensor Tensor::move_across(int64_t position, bool from_codomain) const
  {
    const int64_t count = static_cast<int64_t>((from_codomain ? codomain_ : domain_).size());
    if (position < 0 || position >= count)
      throw std::invalid_argument("there is no leg on that side at position " + std::to_string(position));
    if (position + 1 == count) return bend(*this, from_codomain);
    if (from_codomain)
    {
      // Bring the leg to the end of the codomain, so the bend always sees it there.
      std::vector<int64_t> order;
      for (int64_t at = 0; at < count; ++at)
        if (at != position) order.push_back(at);
      order.push_back(position);
      return bend(permute(order), true);
    }
    // A domain leg is always bent from the end of its side, so the legs behind it are parked on the
    // codomain one at a time and brought back afterwards, which keeps their order.
    Tensor walked = *this;
    int64_t parked = 0;
    while (static_cast<int64_t>(walked.domain().size()) > position + 1)
    {
      walked = bend(walked, false);
      ++parked;
    }
    walked = bend(walked, false);
    ++parked;
    // The bent leg was taken last, so it is the end of the tail; it goes back in front of the parked
    // ones, whose order is kept, so that bringing them back appends them in the order they had.
    const int64_t count_codomain = static_cast<int64_t>(walked.codomain().size());
    const int64_t kept = count_codomain - parked;
    std::vector<int64_t> order;
    for (int64_t at = 0; at < kept; ++at) order.push_back(at);
    order.push_back(count_codomain - 1);
    for (int64_t at = kept; at + 1 < count_codomain; ++at) order.push_back(at);
    walked = walked.permute(order);
    for (int64_t back = 1; back < parked; ++back) walked = bend(walked, true);
    return walked;
  }

  Tensor Tensor::permute(const std::vector<int64_t> &order) const
  {
    const int64_t count = static_cast<int64_t>(codomain_.size());
    if (static_cast<int64_t>(order.size()) != count)
    {
      throw std::invalid_argument("a permutation needs one entry per codomain leg, got " +
                                  std::to_string(order.size()));
    }
    std::vector<bool> seen(static_cast<size_t>(count), false);
    for (const int64_t leg : order)
    {
      if (leg < 0 || leg >= count || seen.at(static_cast<size_t>(leg)))
        throw std::invalid_argument("the entry " + std::to_string(leg) +
                                    " does not name a distinct codomain leg");
      seen.at(static_cast<size_t>(leg)) = true;
    }
    // The exchanges the permutation takes, read off a bubble sort: `steps` holds the position of each
    // exchange and `layouts` the codomain legs as they stand before it.
    std::vector<int64_t> which(static_cast<size_t>(count));
    std::iota(which.begin(), which.end(), 0);
    std::vector<std::vector<Leg>> layouts{codomain_};
    std::vector<int64_t> steps;
    for (int64_t position = 0; position < count; ++position)
    {
      int64_t at = position;
      while (which.at(static_cast<size_t>(at)) != order.at(static_cast<size_t>(position))) ++at;
      while (at > position)
      {
        std::vector<Leg> next = layouts.back();
        std::swap(next.at(static_cast<size_t>(at - 1)), next.at(static_cast<size_t>(at)));
        layouts.push_back(std::move(next));
        steps.push_back(at - 1);
        std::swap(which.at(static_cast<size_t>(at - 1)), which.at(static_cast<size_t>(at)));
        --at;
      }
    }
    std::vector<Leg> codomain;
    for (const int64_t leg : order) codomain.push_back(codomain_.at(static_cast<size_t>(leg)));
    if (steps.empty()) return *this;
    // A contiguous host block is carried through the exchanges in place, one walk per exchange;
    // anything else keeps them one at a time through `swap`.
    bool walked = true;
    for (const auto &[key, matrix] : blocks_)
      walked = walked && matrix.is_contiguous() && matrix.device().is_cpu() &&
               (matrix.scalar_type() == at::kComplexDouble || matrix.scalar_type() == at::kDouble);
    if (!walked)
    {
      Tensor carried = *this;
      for (const int64_t position : steps) carried = carried.swap(position);
      return carried;
    }
    Tensor out(cat_, codomain, domain_, qtotal_, dtype(), device());
    for (const auto &[key, matrix] : blocks_)
    {
      at::Tensor carried = matrix;
      at::Tensor scratch = at::empty({carried.size(0), carried.size(1)}, matrix.options());
      for (size_t at = 0; at < steps.size(); ++at)
      {
        exchanged(exchange_plan(cat_, layouts.at(at), domain_, key, steps.at(at), matrix), carried,
                  scratch);
        std::swap(carried, scratch);
      }
      out.block(key.codomain, key.domain, key.copy).copy_(carried);
    }
    return out;
  }

  Tensor Tensor::repartition(const std::vector<int64_t> &codomain) const
  {
    const int64_t given = static_cast<int64_t>(codomain_.size());
    const int64_t count = static_cast<int64_t>(domain_.size());
    const int64_t total = given + count;
    std::vector<bool> wanted(static_cast<size_t>(total), false);
    for (const int64_t leg : codomain)
    {
      if (leg < 0 || leg >= total)
        throw std::invalid_argument("the leg " + std::to_string(leg) + " does not exist, the tensor has " +
                                    std::to_string(total) + " legs");
      if (wanted.at(static_cast<size_t>(leg)))
        throw std::invalid_argument("the leg " + std::to_string(leg) + " is named twice");
      wanted.at(static_cast<size_t>(leg)) = true;
    }
    // A split that already holds costs nothing, which is the common case inside a sweep.
    bool kept = static_cast<int64_t>(codomain.size()) == given;
    for (int64_t at = 0; kept && at < given; ++at) kept = codomain.at(static_cast<size_t>(at)) == at;
    if (kept) return *this;
    // Every leg is bent onto the codomain, which is the side that can be permuted, so the legs can
    // be put in any order there.  The domain legs are taken off from its end, so they arrive in the
    // reverse of their original order.
    Tensor out = *this;
    while (!out.domain_.empty())
      out = out.move_across(static_cast<int64_t>(out.domain_.size()) - 1, false);
    auto position_of = [given, count](int64_t leg) {
      return leg < given ? leg : given + (count - 1 - (leg - given));
    };
    // The legs that stay on the codomain keep the order they were named in; the legs that go back
    // to the domain are put there last one first, so they land in their original order.
    std::vector<int64_t> order;
    order.reserve(static_cast<size_t>(total));
    for (const int64_t leg : codomain) order.push_back(position_of(leg));
    for (int64_t leg = total - 1; leg >= 0; --leg)
      if (!wanted.at(static_cast<size_t>(leg))) order.push_back(position_of(leg));
    out = out.permute(order);
    for (int64_t at = 0; at < total - static_cast<int64_t>(codomain.size()); ++at)
      out = out.move_across(static_cast<int64_t>(out.codomain_.size()) - 1, true);
    return out;
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

  Tensor &Tensor::operator/=(double scalar)
  {
    if (scalar == 0.0) throw std::invalid_argument("a tensor cannot be divided by zero");
    for (auto &[key, block] : blocks_) block.div_(scalar);
    return *this;
  }

} // namespace sym
