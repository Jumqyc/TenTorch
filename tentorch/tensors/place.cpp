// Running a chain of steps over a tensor: find the plan of a block, build it when it is not cached,
// keep it, and move the entries through it.
#include "tensor.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

#include "cache.h"
#include "regroup.h"

namespace sym
{

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
        step.sides = TensorCaches::instance().intern(codomain, domain);
        steps.push_back(std::move(step));
      }
    };

    // The regrouping of one block through one step: the cached plan, or the one this call builds.
    const Regrouping &plan_of(const Rebase &step, const std::shared_ptr<const FusionCategory> &category,
                              const std::vector<Leg> &codomain, const std::vector<Leg> &domain,
                              const BlockKey &key, const Sector &qtotal, TensorCaches &scratch)
    {
      const bool exchange = step.kind == Rebase::Kind::exchange;
      const PlanQuery query{category, step.sides, key, qtotal, exchange ? 0 : 1,
                            exchange ? step.position : (step.kind == Rebase::Kind::to_domain ? 1 : 0)};
      if (const Regrouping *held = scratch.find(query)) return *held;
      return scratch.keep(query, exchange
                                     ? exchange_regrouping(category, codomain, domain, key, qtotal,
                                                           step.position, scratch)
                                     : bend_regrouping(category, codomain, domain, key, qtotal,
                                                       step.kind == Rebase::Kind::to_domain, scratch));
    }

    // Runs a chain of steps over every block of a tensor: the entries of a block are carried in
    // buffers from step to step, so the chain builds one tensor and not one per step.
    Tensor carried(const Tensor &tensor, const std::vector<Rebase> &steps)
    {
      if (steps.empty()) return tensor;
      const at::TensorOptions options = at::TensorOptions().dtype(tensor.dtype()).device(tensor.device());
      Tensor out(tensor.cat(), steps.back().codomain, steps.back().domain, tensor.qtotal(), tensor.dtype(),
                 tensor.device());
      TensorCaches &scratch = TensorCaches::instance();
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
            return next.emplace(key, scratch.borrow(options, rows, columns)).first->second;
          };
          for (const auto &[key, buffer] : live)
          {
            const Regrouping &plan =
              plan_of(step, tensor.cat(), *codomain, *domain, key, tensor.qtotal(), scratch);
            // Every target block of the plan, with the shape the plan recorded for it; an exchange
            // leaves a block the size it already has.
            std::vector<std::pair<BlockKey, std::pair<int64_t, int64_t>>> targets;
            for (const Shift &shift : plan.shifts)
              if (std::none_of(targets.begin(), targets.end(),
                               [&](const auto &seen) { return seen.first == shift.to; }))
                targets.emplace_back(shift.to, std::make_pair(shift.rows == 0 ? buffer.size(0) : shift.rows,
                                                              shift.columns == 0 ? buffer.size(1) : shift.columns));
            for (const auto &[target, shape] : targets)
              scratch.apply(plan, target, buffer, target_of(target, shape.first, shape.second));
          }
          codomain = &step.codomain;
          domain = &step.domain;
          // The buffer of the first step is the block of the tensor itself or a view of it, so it is
          // the one buffer the chain did not borrow.
          if (at > 0)
            for (auto &entry : live) scratch.give_back(std::move(entry.second));
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

} // namespace sym
