#include "tensor.h"

#include <algorithm>
#include <complex>
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

    // Where the entries of one block go: the runs that land in one block of the result.
    struct Shift
    {
      BlockKey from;
      BlockKey to;
      std::vector<Piece> pieces;
    };

    // One re-reading of a tensor: the result starts at zero and every shift adds to it.
    struct Regrouping
    {
      std::vector<Shift> shifts;
    };

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

    // The key of a regrouping: the legs and the block decide the trees and the coefficients.
    std::vector<int64_t> plan_key(const std::vector<Leg> &codomain,
                                  const std::vector<Leg> &domain,
                                  const BlockKey &key,
                                  const Sector &qtotal)
    {
      std::vector<int64_t> out;
      for (const std::vector<Leg> *legs : {&codomain, &domain})
      {
        out.push_back(static_cast<int64_t>(legs->size()));
        for (const Leg &leg : *legs)
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
      for (const Sector *charge : {&key.codomain, &key.domain, &qtotal})
      {
        out.push_back(static_cast<int64_t>(charge->size()));
        out.insert(out.end(), charge->begin(), charge->end());
      }
      out.push_back(key.copy);
      return out;
    }

    // The charge the codomain legs before the last one fuse to: the root the codomain keeps when its
    // last leg is bent across.
    Sector before_last(const FusionCategory &category, const FusionTree &tree)
    {
      const int64_t count = tree.size();
      if (count <= 1) return category.trivial();
      if (count == 2) return tree.sectors.at(0);
      return tree.inner.at(static_cast<size_t>(count - 3));
    }

    // The regrouping of exchanging the codomain legs `p` and `p + 1` of one block: the trees that
    // agree outside the pair recouple by F and R, and the width the two legs carry is read at the
    // exchanged position on the other side.
    const Regrouping &exchange_regrouping(const std::shared_ptr<const FusionCategory> &cat,
                                          const std::vector<Leg> &codomain,
                                          const std::vector<Leg> &domain,
                                          const BlockKey &key,
                                          const Sector &qtotal,
                                          int64_t p)
    {
      using PlanKey = std::pair<std::shared_ptr<const FusionCategory>, std::vector<int64_t>>;
      static std::map<PlanKey, Regrouping> plans;
      std::vector<int64_t> fingerprint = plan_key(codomain, domain, key, qtotal);
      fingerprint.insert(fingerprint.begin(), p);
      const PlanKey name{cat, fingerprint};
      const auto held = plans.find(name);
      if (held != plans.end()) return held->second;

      const int64_t count = static_cast<int64_t>(codomain.size());
      std::vector<Leg> exchanged = codomain;
      std::swap(exchanged.at(static_cast<size_t>(p)), exchanged.at(static_cast<size_t>(p + 1)));
      const SideLayout source = layout_of(*cat, codomain, key.codomain);
      const SideLayout target = layout_of(*cat, exchanged, key.codomain);
      const int64_t stride = layout_of(*cat, domain, key.domain).rows();
      Regrouping plan;
      plan.shifts.push_back(Shift{key, key, {}});

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
        const at::Tensor straight = cat->F_symbol(y, b, c, d);
        const at::Tensor braided = cat->F_symbol(y, c, b, d);
        TORCH_CHECK(straight.size(0) == static_cast<int64_t>(olds.size()) &&
                      braided.size(0) == static_cast<int64_t>(news.size()),
                    "the exchange does not match the trees of the block");
        std::map<Sector, int64_t> offset;
        int64_t at = 0;
        for (const Outcome &outcome : cat->outcomes(b, c))
        {
          offset.emplace(outcome.sector, at);
          at += outcome.multiplicity;
        }
        const std::vector<FIndex> from = cat->f_columns(y, b, c, d);
        const std::vector<FIndex> to = cat->f_columns(y, c, b, d);
        const at::Tensor braid = cat->R_symbol(b, c);
        std::vector<int64_t> places;
        std::vector<int64_t> sources;
        for (int64_t i = 0; i < static_cast<int64_t>(to.size()); ++i)
          for (int64_t j = 0; j < static_cast<int64_t>(from.size()); ++j)
          {
            if (to.at(static_cast<size_t>(i)).sector != from.at(static_cast<size_t>(j)).sector) continue;
            if (to.at(static_cast<size_t>(i)).outer != from.at(static_cast<size_t>(j)).outer) continue;
            places.push_back(i * static_cast<int64_t>(from.size()) + j);
            sources.push_back((offset.at(to.at(static_cast<size_t>(i)).sector) + to.at(static_cast<size_t>(i)).inner) *
                                braid.size(1) +
                              offset.at(from.at(static_cast<size_t>(j)).sector) + from.at(static_cast<size_t>(j)).inner);
          }
        const at::TensorOptions index_options = at::TensorOptions().dtype(at::kLong).device(braid.device());
        at::Tensor middle = at::zeros({static_cast<int64_t>(to.size()) * static_cast<int64_t>(from.size())},
                                      braid.options());
        middle.index_put_({at::tensor(places, index_options)},
                          braid.flatten().index_select(0, at::tensor(sources, index_options)));
        middle = middle.reshape({static_cast<int64_t>(to.size()), static_cast<int64_t>(from.size())});
        const at::Tensor coupled =
          at::matmul(braided, at::matmul(middle, straight.conj().transpose(0, 1)))
            .to(at::kCPU)
            .to(at::kComplexDouble)
            .contiguous();
        const c10::complex<double> *entry = coupled.data_ptr<c10::complex<double>>();
        for (size_t one = 0; one < news.size(); ++one)
          for (size_t other = 0; other < olds.size(); ++other)
          {
            const c10::complex<double> value = entry[one * olds.size() + other];
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
            for (int64_t at = 0; at < width; ++at)
              pieces.push_back(Piece{(source.offsets.at(static_cast<size_t>(other)) + at) * stride,
                                     (target.offsets.at(static_cast<size_t>(one)) +
                                      width_of.at(static_cast<size_t>(at))) * stride,
                                     1, 1, stride, factor->second});
          }
      }
      return plans.emplace(name, std::move(plan)).first->second;
    }

    // The regrouping of bending the last leg of one side onto the other, for one block.  The giving
    // tree keeps everything but its last vertex, the taking tree gains one, and the recoupling of the
    // two vertices carries the copies of the moved leg into the columns of the taking side.
    const Regrouping &bend_regrouping(const std::shared_ptr<const FusionCategory> &cat,
                                      const std::vector<Leg> &codomain,
                                      const std::vector<Leg> &domain,
                                      const BlockKey &key,
                                      const Sector &qtotal,
                                      bool from_codomain)
    {
      using PlanKey = std::pair<std::shared_ptr<const FusionCategory>, std::vector<int64_t>>;
      static std::map<PlanKey, Regrouping> plans;
      std::vector<int64_t> fingerprint = plan_key(codomain, domain, key, qtotal);
      fingerprint.insert(fingerprint.begin(), from_codomain ? 1 : 0);
      const PlanKey name{cat, fingerprint};
      const auto held = plans.find(name);
      if (held != plans.end()) return held->second;

      const std::vector<Leg> &giving = from_codomain ? codomain : domain;
      const std::vector<Leg> &taking = from_codomain ? domain : codomain;
      const Leg moved_leg = giving.back();
      const std::vector<Leg> kept_giving(giving.begin(), giving.end() - 1);
      std::vector<Leg> kept_taking = taking;
      kept_taking.push_back(moved_leg);
      const std::vector<Leg> out_codomain = from_codomain ? kept_giving : kept_taking;
      const std::vector<Leg> out_domain = from_codomain ? kept_taking : kept_giving;
      const SideLayout moved_side = from_codomain ? layout_of(*cat, codomain, key.codomain)
                                                  : layout_of(*cat, domain, key.domain);
      const SideLayout gains_side = from_codomain ? layout_of(*cat, domain, key.domain)
                                                  : layout_of(*cat, codomain, key.codomain);
      const Sector moved_root = from_codomain ? key.codomain : key.domain;
      const Sector gains_root = from_codomain ? key.domain : key.codomain;
      const int64_t source_stride = (from_codomain ? gains_side : moved_side).rows();

      // The layouts of the two sides of one output root pair, with the trees indexed by fusion so that
      // a recoupling finds them without a scan.
      struct OutputLayout
      {
        SideLayout rows;
        SideLayout columns;
        std::map<FusionTree, int64_t> row_of;
        std::map<FusionTree, int64_t> column_of;
      };

      Regrouping plan;
      std::map<BlockKey, size_t> at_of;
      const auto index_of = [&](const BlockKey &to) -> size_t
      {
        const auto found = at_of.find(to);
        if (found != at_of.end()) return found->second;
        const size_t index = plan.shifts.size();
        at_of.emplace(to, index);
        plan.shifts.push_back(Shift{key, to, {}});
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
          const Sector prefix = before_last(*cat, moved_tree);
          const int64_t copy = moved_tree.size() >= 2 ? moved_tree.copies.back() : 0;
          // The vertex the leg bends through: on the codomain it is (prefix, leg) and the other root
          // closes last, on the domain the roles swap, which is the same recoupling read from the
          // other side of the tensor.
          const Sector first = from_codomain ? prefix : gains_root;
          const Sector middle = from_codomain ? charge : cat->dual(charge);
          const Sector last = from_codomain ? cat->dual(gains_root) : cat->dual(prefix);
          const at::Tensor coefficients =
            cat->F_symbol(first, middle, last, qtotal).to(at::kCPU).to(at::kComplexDouble).contiguous();
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
          // The tree the giving side keeps: its last vertex is gone.
          FusionTree trimmed;
          trimmed.sectors = moved_tree.sectors;
          trimmed.sectors.pop_back();
          trimmed.inner = moved_tree.inner;
          if (!trimmed.inner.empty()) trimmed.inner.pop_back();
          trimmed.copies = moved_tree.copies;
          if (!trimmed.copies.empty()) trimmed.copies.pop_back();
          trimmed.coupled = prefix;
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
                fresh.rows = layout_of(*cat, out_codomain, out_codomain_root);
                fresh.columns = layout_of(*cat, out_domain, out_domain_root);
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
              const int64_t row = layout.rows.offsets.at(static_cast<size_t>(row_at->second));
              const int64_t column_offset = layout.columns.offsets.at(static_cast<size_t>(column_at->second));
              const int64_t target_stride = layout.columns.rows();
              std::vector<Piece> &pieces =
                plan.shifts
                  .at(index_of(BlockKey{out_codomain_root, out_domain_root, taken_side.outer}))
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
      return plans.emplace(name, std::move(plan)).first->second;
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
    std::vector<Leg> codomain = codomain_;
    std::swap(codomain.at(static_cast<size_t>(position)), codomain.at(static_cast<size_t>(position + 1)));
    Tensor out(cat_, codomain, domain_, qtotal_, dtype(), device());
    for (const auto &[key, matrix] : blocks_)
    {
      const Regrouping &plan = exchange_regrouping(cat_, codomain_, domain_, key, qtotal_, position);
      for (const Shift &shift : plan.shifts)
        add_pieces(shift.pieces, matrix, out.block(shift.to.codomain, shift.to.domain, shift.to.copy));
    }
    return out;
  }

  namespace
  {

    // Bends the last leg of one side onto the end of the other side.
    // `from_codomain` names the side that gives the leg and nothing else: a tree is a tree, so the
    // two directions run the same recoupling with the roles of the two trees exchanged.
    Tensor bend(const Tensor &tensor, bool from_codomain)
    {
      const std::vector<Leg> &giving = from_codomain ? tensor.codomain() : tensor.domain();
      const std::vector<Leg> &taking = from_codomain ? tensor.domain() : tensor.codomain();
      const std::vector<Leg> kept_giving(giving.begin(), giving.end() - 1);
      std::vector<Leg> kept_taking = taking;
      kept_taking.push_back(giving.back());
      const std::vector<Leg> out_codomain = from_codomain ? kept_giving : kept_taking;
      const std::vector<Leg> out_domain = from_codomain ? kept_taking : kept_giving;
      Tensor out(tensor.cat(), out_codomain, out_domain, tensor.qtotal(), tensor.dtype(), tensor.device());
      for (const auto &[key, matrix] : tensor.blocks())
      {
        const Regrouping &plan =
          bend_regrouping(tensor.cat(), tensor.codomain(), tensor.domain(), key, tensor.qtotal(), from_codomain);
        for (const Shift &shift : plan.shifts)
          add_pieces(shift.pieces, matrix, out.block(shift.to.codomain, shift.to.domain, shift.to.copy));
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
    // Every block is carried through the exchanges in one pair of buffers, so a permutation costs the
    // walks and no intermediate tensor.
    Tensor out(cat_, codomain, domain_, qtotal_, dtype(), device());
    for (const auto &[key, matrix] : blocks_)
    {
      at::Tensor carried = matrix.contiguous().clone();
      at::Tensor scratch = at::empty_like(carried);
      for (size_t at = 0; at < steps.size(); ++at)
      {
        const Regrouping &plan = exchange_regrouping(cat_, layouts.at(at), domain_, key, qtotal_, steps.at(at));
        scratch.zero_();
        for (const Shift &shift : plan.shifts) add_pieces(shift.pieces, carried, scratch);
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
