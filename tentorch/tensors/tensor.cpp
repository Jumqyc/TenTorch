#include "tensor.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

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
    // One layout per root and not one per key: the keys of a side share it.
    const std::vector<BlockKey> keys = block_keys(*cat_, codomain_, domain_, qtotal_);
    std::map<Sector, int64_t> rows;
    std::map<Sector, int64_t> columns;
    for (const BlockKey &key : keys)
    {
      if (rows.count(key.codomain) == 0) rows.emplace(key.codomain, layout_of(*cat_, codomain_, key.codomain).rows());
      if (columns.count(key.domain) == 0) columns.emplace(key.domain, layout_of(*cat_, domain_, key.domain).rows());
    }
    for (const BlockKey &key : keys)
      blocks_.emplace(key, at::zeros({rows.at(key.codomain), columns.at(key.domain)}, options));
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

  SideLayout Tensor::rows_of(const Sector &codomain_root) const
  {
    return layout_of(*cat_, codomain_, codomain_root);
  }

  SideLayout Tensor::columns_of(const Sector &domain_root) const
  {
    return layout_of(*cat_, domain_, domain_root);
  }

  std::vector<BlockKey> Tensor::keys() const
  {
    std::vector<BlockKey> out;
    for (const auto &[key, matrix] : blocks_) out.push_back(key);
    return out;
  }

  Tensor Tensor::swap(int64_t position) const
  {
    const int64_t count = static_cast<int64_t>(codomain_.size());
    if (position < 0 || position + 1 >= count)
    {
      throw std::invalid_argument("a swap needs two adjacent codomain legs, got position " +
                                  std::to_string(position));
    }
    const int64_t p = position;
    // The vertex that is the coupling of the pair, which `p - 1` is when the pair is not the first.
    const int64_t vertex = p - 1;

    std::vector<Leg> codomain = codomain_;
    std::swap(codomain.at(static_cast<size_t>(p)), codomain.at(static_cast<size_t>(p + 1)));
    Tensor out(cat_, codomain, domain_, qtotal_, dtype(), device());

    // Everything about a tree that the exchange leaves alone.  A tree of the exchanged side is read
    // in the order before the exchange, so that the two sides meet on the same key.
    auto signature = [&](const FusionTree &tree, bool exchanged)
    {
      std::vector<int64_t> key;
      for (int64_t axis = 0; axis < count; ++axis)
      {
        const int64_t from = exchanged && (axis == p || axis == p + 1) ? (axis == p ? p + 1 : p) : axis;
        const Sector &sector = tree.sectors.at(static_cast<size_t>(from));
        key.insert(key.end(), sector.begin(), sector.end());
      }
      for (int64_t at = 0; at < static_cast<int64_t>(tree.inner.size()); ++at)
      {
        if (at == vertex) continue;
        const Sector &charge = tree.inner.at(static_cast<size_t>(at));
        key.insert(key.end(), charge.begin(), charge.end());
      }
      for (int64_t at = 0; at < static_cast<int64_t>(tree.copies.size()); ++at)
      {
        if (at == p - 1 || at == p) continue;
        key.push_back(tree.copies.at(static_cast<size_t>(at)));
      }
      return key;
    };

    // The multiplicities of the legs of a tree, in the same reading order.
    auto widths = [&](const FusionTree &tree, const std::vector<Leg> &legs, bool exchanged)
    {
      std::vector<int64_t> out;
      for (int64_t axis = 0; axis < count; ++axis)
      {
        const int64_t from = exchanged && (axis == p || axis == p + 1) ? (axis == p ? p + 1 : p) : axis;
        const Leg &leg = legs.at(static_cast<size_t>(from));
        out.push_back(leg.multiplicity(block_of(leg, tree.sectors.at(static_cast<size_t>(from)))));
      }
      return out;
    };

    for (const auto &[key, matrix] : blocks_)
    {
      const Sector &root = key.codomain;
      const SideLayout source = layout_of(*cat_, codomain_, root);
      const SideLayout target = layout_of(*cat_, codomain, root);
      const int64_t before = static_cast<int64_t>(source.trees.size());
      const int64_t after = static_cast<int64_t>(target.trees.size());
      const int64_t columns = matrix.size(1);
      if (before == 0 && after == 0) continue;

      const at::TensorOptions index_options = at::TensorOptions().dtype(at::kLong).device(matrix.device());

      // The trees the exchange recouples: they agree on everything outside the pair.
      std::map<std::vector<int64_t>, std::vector<int64_t>> groups;
      for (int64_t at = 0; at < before; ++at)
        groups[signature(source.trees.at(static_cast<size_t>(at)), false)].push_back(at);
      std::map<std::vector<int64_t>, std::vector<int64_t>> partners;
      for (int64_t at = 0; at < after; ++at)
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
        const Sector y = chain_at(*cat_, sample, p - 1);
        const Sector d = chain_at(*cat_, sample, p + 1);
        const Sector &b = sample.sectors.at(static_cast<size_t>(p));
        const Sector &c = sample.sectors.at(static_cast<size_t>(p + 1));

        const at::Tensor straight = cat_->F_symbol(y, b, c, d).to(matrix.device());
        const at::Tensor exchanged = cat_->F_symbol(y, c, b, d).to(matrix.device());
        TORCH_CHECK(straight.size(0) == static_cast<int64_t>(olds.size()) &&
                        exchanged.size(0) == static_cast<int64_t>(news.size()),
                    "the exchange does not match the trees of the block");

        // The braiding between the two intermediate bases, on the copies of one outcome only.
        std::map<Sector, int64_t> offset;
        int64_t at = 0;
        for (const Outcome &outcome : cat_->outcomes(b, c))
        {
          offset.emplace(outcome.sector, at);
          at += outcome.multiplicity;
        }
        const std::vector<FIndex> from = cat_->f_columns(y, b, c, d);
        const std::vector<FIndex> to = cat_->f_columns(y, c, b, d);
        const at::Tensor braid = cat_->R_symbol(b, c).to(matrix.device());
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
        pieces.push_back(
          at::matmul(exchanged, at::matmul(middle, straight.conj().transpose(0, 1))));
        row_order.insert(row_order.end(), news.begin(), news.end());
        column_order.insert(column_order.end(), olds.begin(), olds.end());
      }

      // Scatter the pieces onto the trees they belong to: the groups partition the trees, so both
      // orders are permutations and each piece lands in one block.
      at::Tensor packed = at::zeros({after, before}, matrix.options());
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
      at::Tensor stacked = at::zeros({after, before}, matrix.options());
      stacked.index_copy_(0, at::tensor(row_order, index_options), packed);
      at::Tensor recoupling = at::zeros({after, before}, matrix.options());
      recoupling.index_copy_(1, at::tensor(column_order, index_options), stacked);

      // Trees whose legs carry the same multiplicities move in one piece: the width index is then a
      // mixed radix whose two digits the exchange swaps, which is one transpose of two axes.
      std::map<std::vector<int64_t>, std::vector<int64_t>> before_classes;
      std::map<std::vector<int64_t>, std::vector<int64_t>> after_classes;
      for (int64_t at = 0; at < before; ++at)
        before_classes[widths(source.trees.at(static_cast<size_t>(at)), codomain_, false)].push_back(at);
      for (int64_t at = 0; at < after; ++at)
        after_classes[widths(target.trees.at(static_cast<size_t>(at)), codomain, true)].push_back(at);

      at::Tensor &result = out.block(root, key.domain, key.copy);
      for (const auto &[shape, olds] : before_classes)
      {
        const auto found = after_classes.find(shape);
        if (found == after_classes.end())
          throw std::logic_error("the exchange has no image for a block of trees");
        const std::vector<int64_t> &news = found->second;
        int64_t width = 1;
        for (const int64_t factor : shape) width *= factor;

        std::vector<int64_t> gather;
        gather.reserve(olds.size() * static_cast<size_t>(width));
        for (const int64_t tree : olds)
          for (int64_t at = 0; at < width; ++at)
            gather.push_back(source.offsets.at(static_cast<size_t>(tree)) + at);
        const at::Tensor piece =
          matrix.index_select(0, at::tensor(gather, index_options))
            .reshape({static_cast<int64_t>(olds.size()), width * columns});
        const at::Tensor selected = recoupling.index_select(0, at::tensor(news, index_options))
                                      .index_select(1, at::tensor(olds, index_options));
        at::Tensor moved = at::matmul(selected, piece)
                             .reshape({static_cast<int64_t>(news.size()), width, columns});
        if (width > 1)
        {
          std::vector<int64_t> digits{static_cast<int64_t>(news.size())};
          digits.insert(digits.end(), shape.begin(), shape.end());
          digits.push_back(columns);
          moved = moved.view(digits)
                    .transpose(p + 1, p + 2)
                    .contiguous()
                    .reshape({static_cast<int64_t>(news.size()), width, columns});
        }
        for (int64_t at = 0; at < static_cast<int64_t>(news.size()); ++at)
          result.narrow(0, target.offsets.at(static_cast<size_t>(news.at(static_cast<size_t>(at)))), width)
            .copy_(moved.select(0, at));
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

    // Whether two trees fuse the same legs the same way. The coupled charge is the root of the
    // layout they are looked up in, so it is not compared: a layout already holds one root.
    bool same_fusion(const FusionTree &left, const FusionTree &right)
    {
      return left.sectors == right.sectors && left.inner == right.inner && left.copies == right.copies;
    }

    // The index of `tree` in `trees`, which the layouts of one root list in a fixed order.
    int64_t tree_index(const SideLayout &layout, const FusionTree &tree)
    {
      for (size_t at = 0; at < layout.trees.size(); ++at)
        if (same_fusion(layout.trees.at(at), tree)) return static_cast<int64_t>(at);
      return -1;
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
      for (const auto &[key, matrix] : tensor.blocks())
      {
        const SideLayout rows = layout_of(*cat, tensor.codomain(), key.codomain);
        const SideLayout columns = layout_of(*cat, tensor.domain(), key.domain);
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
            // The block rows of the moved tree with its copies split off the end, read as rows of
            // the new block (the giving side) or as its columns (the taking side).
            const at::Tensor piece =
              from_codomain
                ? matrix.narrow(0, moved_layout.offsets.at(moved_at), span)
                    .narrow(1, gains_layout.offsets.at(gains_at), gains_width)
                    .reshape({inside, copies, gains_width})
                    .transpose(1, 2)
                    .reshape({inside, gains_width * copies})
                : matrix.narrow(0, gains_layout.offsets.at(gains_at), gains_width)
                    .narrow(1, moved_layout.offsets.at(moved_at), span)
                    .reshape({gains_width, inside, copies})
                    .transpose(1, 2)
                    .reshape({gains_width * copies, inside});
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
                const Sector extended_root = from_codomain ? cat->dual(taken_side.sector) : taken_side.sector;
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
                const SideLayout out_rows = layout_of(*cat, out_codomain, out_codomain_root);
                const SideLayout out_columns = layout_of(*cat, out_domain, out_domain_root);
                const int64_t row_at = tree_index(out_rows, from_codomain ? trimmed : extended);
                const int64_t column_at = tree_index(out_columns, from_codomain ? extended : trimmed);
                if (row_at < 0 || column_at < 0) continue;
                at::Tensor factor = recoupling.index({index, static_cast<int64_t>(column)});
                factor = matrix.is_complex() ? factor.to(matrix.scalar_type()) : at::real(factor);
                if (factor.item<double>() == 0.0) continue;
                const int64_t row_width = from_codomain ? inside : gains_width * copies;
                const int64_t column_width = from_codomain ? gains_width * copies : inside;
                // The total charge closes on the taking side now, so its closing copy is the one
                // the free index names.
                out.block(out_codomain_root, out_domain_root, taken_side.outer)
                  .narrow(0, out_rows.offsets.at(static_cast<size_t>(row_at)), row_width)
                  .narrow(1, out_columns.offsets.at(static_cast<size_t>(column_at)), column_width)
                  .add_(piece * factor);
              }
            }
          }
        }
      }
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
    // Bring the wanted leg to each position in turn, one exchange at a time.
    std::vector<int64_t> which(static_cast<size_t>(count));
    std::iota(which.begin(), which.end(), 0);
    Tensor out = *this;
    for (int64_t position = 0; position < count; ++position)
    {
      int64_t at = position;
      while (which.at(static_cast<size_t>(at)) != order.at(static_cast<size_t>(position))) ++at;
      while (at > position)
      {
        out = out.swap(at - 1);
        std::swap(which.at(static_cast<size_t>(at - 1)), which.at(static_cast<size_t>(at)));
        --at;
      }
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
