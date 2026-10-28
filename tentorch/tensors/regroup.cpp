// The regrouping of one block along one step of a chain.
#include "regroup.h"

#include <algorithm>
#include <complex>
#include <map>
#include <tuple>
#include <utility>

namespace sym
{

  namespace
  {
    // The charge the legs up to `last` couple to, the trivial charge before the first.
    Sector chain_at(const FusionCategory &category, const FusionTree &tree, int64_t last)
    {
      if (last < 0) return category.trivial();
      if (last + 1 == tree.size()) return tree.coupled;
      if (last == 0) return tree.sectors.at(0);
      return tree.inner.at(static_cast<size_t>(last - 1));
    }
  } // namespace

  Regrouping exchange_regrouping(const std::shared_ptr<const FusionCategory> &cat,
                                 const std::vector<Leg> &codomain,
                                 const std::vector<Leg> &domain,
                                 const BlockKey &key,
                                 const Sector &qtotal,
                                 int64_t p,
                                 const LayoutSource &layouts)
  {
    const int64_t count = static_cast<int64_t>(codomain.size());
    std::vector<Leg> exchanged = codomain;
    std::swap(exchanged.at(static_cast<size_t>(p)), exchanged.at(static_cast<size_t>(p + 1)));
    const SideLayout &source = layouts.side(*cat, codomain, key.codomain);
    const SideLayout &target = layouts.side(*cat, exchanged, key.codomain);
    const int64_t stride = layouts.side(*cat, domain, key.domain).rows();
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
    return plan;
  }

  Regrouping bend_regrouping(const std::shared_ptr<const FusionCategory> &cat,
                             const std::vector<Leg> &codomain,
                             const std::vector<Leg> &domain,
                             const BlockKey &key,
                             const Sector &qtotal,
                             bool from_codomain,
                             const LayoutSource &layouts)
  {
    const std::vector<Leg> &giving = from_codomain ? codomain : domain;
    const std::vector<Leg> &taking = from_codomain ? domain : codomain;
    const Leg moved_leg = giving.back();
    const std::vector<Leg> kept_giving(giving.begin(), giving.end() - 1);
    std::vector<Leg> kept_taking = taking;
    kept_taking.push_back(moved_leg);
    const std::vector<Leg> out_codomain = from_codomain ? kept_giving : kept_taking;
    const std::vector<Leg> out_domain = from_codomain ? kept_taking : kept_giving;
    const SideLayout &moved_side = layouts.side(*cat, from_codomain ? codomain : domain,
                                                 from_codomain ? key.codomain : key.domain);
    const SideLayout &gains_side = layouts.side(*cat, from_codomain ? domain : codomain,
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
              fresh.rows = &layouts.side(*cat, out_codomain, out_codomain_root);
              fresh.columns = &layouts.side(*cat, out_domain, out_domain_root);
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
    return plan;
  }

} // namespace sym
