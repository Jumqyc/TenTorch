#include "einsum.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "contract.h"
#include "../../symmetry/cat/cat.h"

namespace sym
{

  namespace
  {

    // The most charges one fold, or one combination of traces, may carry.
    constexpr int64_t kMaxCharges = 4096;

    // One group of the equation: its labels, lower cased so that the case reads as the side a leg has
    // to sit on rather than as part of its name, and the side each one asks for, `true` the codomain.
    struct Side
    {
      std::vector<std::string> labels;
      std::vector<bool> above;
    };

    Side side_of(const std::string &text)
    {
      Side out;
      for (const char letter : text)
      {
        if (letter == ' ') continue;
        out.labels.emplace_back(1, static_cast<char>(std::tolower(static_cast<unsigned char>(letter))));
        out.above.push_back(letter >= 'A' && letter <= 'Z');
      }
      return out;
    }

    int64_t position_of(const std::vector<std::string> &labels, const std::string &label)
    {
      const auto found = std::find(labels.begin(), labels.end(), label);
      if (found == labels.end())
        throw std::logic_error("the label '" + label + "' left the equation");
      return static_cast<int64_t>(std::distance(labels.begin(), found));
    }

    bool holds(const std::vector<std::string> &labels, const std::string &label)
    {
      return std::find(labels.begin(), labels.end(), label) != labels.end();
    }

    // The tensor with every leg the equation sums left where it sits and every leg it keeps on the
    // side its case declares, together with the labels in the order the legs then read them.  A
    // summed leg is paired with the leg it meets in the other operand, so which side it takes is the
    // contraction's to decide and the case only says where it would like it; a kept leg is free, so
    // its case is binding and the leg is bent when the tensor disagrees.
    std::pair<Tensor, std::vector<std::string>> laid_out(const Tensor &tensor, const Side &side,
                                                         const std::vector<std::string> &summed)
    {
      const int64_t codomain = static_cast<int64_t>(tensor.codomain().size());
      std::vector<int64_t> above;
      std::vector<int64_t> below;
      for (int64_t leg = 0; leg < tensor.nlegs(); ++leg)
      {
        const size_t at = static_cast<size_t>(leg);
        const bool stays = holds(summed, side.labels.at(at)) ? leg < codomain : side.above.at(at);
        (stays ? above : below).push_back(leg);
      }
      std::vector<std::string> labels;
      for (const int64_t leg : above) labels.push_back(side.labels.at(static_cast<size_t>(leg)));
      for (const int64_t leg : below) labels.push_back(side.labels.at(static_cast<size_t>(leg)));
      return {tensor.place(above, below), std::move(labels)};
    }

    // Where a tree sits in the layout of its own root; the layouts list the trees in a fixed order.
    int64_t tree_position(const SideLayout &layout, const FusionTree &tree)
    {
      for (size_t at = 0; at < layout.trees.size(); ++at)
        if (layout.trees.at(at) == tree) return static_cast<int64_t>(at);
      return -1;
    }

    // The partial trace over the last leg of the codomain and the last leg of the domain, which name
    // the same space: both last vertices are removed and the copies the two legs run over summed.
    // What the kept roots close to is the charge of the result, and a pair of kept roots can close to
    // more than one charge, so the sum can leave more than one tensor, one per charge it reaches.
    std::vector<std::pair<Sector, Tensor>> partial_trace(const Tensor &tensor)
    {
      const std::shared_ptr<const FusionCategory> cat = tensor.cat();
      const Sector qtotal = tensor.qtotal();
      const std::vector<Leg> out_codomain(tensor.codomain().begin(), tensor.codomain().end() - 1);
      const std::vector<Leg> out_domain(tensor.domain().begin(), tensor.domain().end() - 1);
      std::map<Sector, Tensor> out;
      for (const auto &[key, matrix] : tensor.blocks())
      {
        // A block whose own roots close in more than one way leaves the coefficient undetermined.
        require_single_closing(*cat, key.codomain, key.domain, qtotal, "the roots", "a trace");
        const SideLayout &rows = layouts().side(*cat, tensor.codomain(), key.codomain);
        const SideLayout &columns = layouts().side(*cat, tensor.domain(), key.domain);
        for (size_t row_at = 0; row_at < rows.trees.size(); ++row_at)
        {
          const FusionTree &above_tree = rows.trees.at(row_at);
          const int64_t row_span = rows.offsets.at(row_at + 1) - rows.offsets.at(row_at);
          const FusionTree kept_above = trimmed(*cat, above_tree);
          for (size_t column_at = 0; column_at < columns.trees.size(); ++column_at)
          {
            const FusionTree &under_tree = columns.trees.at(column_at);
            const int64_t column_span =
              columns.offsets.at(column_at + 1) - columns.offsets.at(column_at);
            // The two traced legs name one space, so both trees have to pick the same sector of it.
            if (!(above_tree.sectors.back() == under_tree.sectors.back())) continue;
            const FusionTree kept_under = trimmed(*cat, under_tree);
            const SideLayout &out_rows = layouts().side(*cat, out_codomain, kept_above.coupled);
            const SideLayout &out_columns = layouts().side(*cat, out_domain, kept_under.coupled);
            const int64_t new_row = tree_position(out_rows, kept_above);
            const int64_t new_column = tree_position(out_columns, kept_under);
            if (new_row < 0 || new_column < 0)
              throw std::logic_error("a kept tree of a trace is not in the layout it was trimmed "
                                     "from");
            // Dropping the traced leg drops its copy digit, the last and fastest radix of a tree.
            const int64_t inside_rows = out_rows.offsets.at(static_cast<size_t>(new_row) + 1) -
                                        out_rows.offsets.at(static_cast<size_t>(new_row));
            const int64_t inside_columns = out_columns.offsets.at(static_cast<size_t>(new_column) + 1) -
                                           out_columns.offsets.at(static_cast<size_t>(new_column));
            const int64_t copies = row_span / inside_rows;
            // Both legs carry the copy as their last radix, so the sum over the two copies is the
            // diagonal of that pair of axes.
            at::Tensor piece =
              matrix.narrow(0, rows.offsets.at(row_at), row_span)
                .narrow(1, columns.offsets.at(column_at), column_span)
                .reshape({inside_rows, copies, inside_columns, copies})
                .diagonal(0, 1, 3)
                .sum(-1);
            for (const Outcome &closing :
                 cat->outcomes(kept_above.coupled, cat->dual(kept_under.coupled)))
            {
              require_single_closing(*cat, kept_above.coupled, kept_under.coupled, closing.sector,
                                     "the kept roots", "a trace");
              const auto slot = out.try_emplace(closing.sector, cat, out_codomain, out_domain,
                                                closing.sector, tensor.dtype(), tensor.device());
              slot.first->second.block(kept_above.coupled, kept_under.coupled)
                .narrow(0, out_rows.offsets.at(static_cast<size_t>(new_row)), inside_rows)
                .narrow(1, out_columns.offsets.at(static_cast<size_t>(new_column)), inside_columns)
                .add_(piece);
            }
          }
        }
      }
      return std::vector<std::pair<Sector, Tensor>>(std::make_move_iterator(out.begin()),
                                                    std::make_move_iterator(out.end()));
    }

    // The number of components of a leg, which is the size its label contributes to a tensor.
    double leg_size(const FusionCategory &cat, const Leg &leg)
    {
      double out = 0;
      for (int64_t block = 0; block < leg.nblocks(); ++block)
        out += static_cast<double>(leg.multiplicity(block)) * cat.q_dim(leg.sector(block));
      return out;
    }

    // The size of a tensor over `labels`: the product of their dimensions.
    double tensor_size(const std::vector<std::string> &labels,
                       const std::map<std::string, double> &dims)
    {
      double out = 1;
      for (const std::string &label : labels) out *= dims.at(label);
      return out;
    }

    // Every label of either list, each once.
    std::vector<std::string> union_of(const std::vector<std::string> &left,
                                      const std::vector<std::string> &right)
    {
      std::vector<std::string> out = left;
      for (const std::string &label : right)
        if (!holds(out, label)) out.push_back(label);
      return out;
    }

    // The labels both lists name.
    std::vector<std::string> shared_of(const std::vector<std::string> &left,
                                       const std::vector<std::string> &right)
    {
      std::vector<std::string> out;
      for (const std::string &label : left)
        if (holds(right, label)) out.push_back(label);
      return out;
    }

    // The labels only one of the two lists names: the union without the ones both name, in its
    // order, so the left keeps its labels first and the right brings its own after them.
    std::vector<std::string> product_labels(const std::vector<std::string> &left,
                                            const std::vector<std::string> &right)
    {
      const std::vector<std::string> shared = shared_of(left, right);
      std::vector<std::string> out;
      for (const std::string &label : union_of(left, right))
        if (!holds(shared, label)) out.push_back(label);
      return out;
    }

    // The order the operands are folded in: the first operand goes into the accumulator and then each
    // of the others once, the order being the one that makes the products smallest.  Up to twelve
    // operands that order is found exactly, by a split over every subset of the others; beyond that
    // the next operand taken is the one whose product with the accumulator is smallest.  A tie goes
    // to the step that sums more labels at once and then to the order the equation is written in.
    std::vector<int64_t> fold_order(const std::vector<std::vector<std::string>> &inputs,
                                    const std::map<std::string, double> &dims)
    {
      const int64_t count = static_cast<int64_t>(inputs.size());
      std::vector<int64_t> order;
      if (count <= 12)
      {
        const int64_t masks = int64_t(1) << (count - 1);
        std::vector<std::vector<std::string>> labels(masks, inputs.front());
        for (int64_t mask = 1; mask < masks; ++mask)
        {
          const int64_t bit = __builtin_ctzll(static_cast<unsigned long long>(mask));
          labels.at(mask) = union_of(labels.at(mask ^ (int64_t(1) << bit)), inputs.at(bit + 1));
        }
        // A step that sums nothing is taken only when the accumulator shares no label with any
        // operand left: an outer product keeps every leg of both, so it is the last resort.
        for (int pass = 0; pass < 2; ++pass)
        {
          std::vector<double> cost(masks, std::numeric_limits<double>::infinity());
          std::vector<int64_t> summed(masks, 0);
          std::vector<int64_t> last(masks, 0);
          cost.at(0) = 0;
          for (int64_t mask = 1; mask < masks; ++mask)
            for (int64_t bit = 0; bit < count - 1; ++bit)
              if (mask & (int64_t(1) << bit))
              {
                const std::vector<std::string> shared =
                  shared_of(labels.at(mask ^ (int64_t(1) << bit)), inputs.at(bit + 1));
                if (pass == 0 && shared.empty()) continue;
                const double candidate = cost.at(mask ^ (int64_t(1) << bit)) +
                                         tensor_size(labels.at(mask), dims);
                // The smallest product first; a tie goes to the step that sums the most labels with
                // it, which keeps the fewest legs open in the accumulator.
                const bool better = candidate < cost.at(mask) ||
                                    (candidate == cost.at(mask) &&
                                     static_cast<int64_t>(shared.size()) > summed.at(mask)) ||
                                    (candidate == cost.at(mask) &&
                                     static_cast<int64_t>(shared.size()) == summed.at(mask) &&
                                     bit > last.at(mask));
                if (better)
                {
                  cost.at(mask) = candidate;
                  summed.at(mask) = static_cast<int64_t>(shared.size());
                  last.at(mask) = bit;
                }
              }
          if (cost.at(masks - 1) == std::numeric_limits<double>::infinity()) continue;
          for (int64_t mask = masks - 1; mask; mask ^= int64_t(1) << last.at(mask))
            order.push_back(last.at(mask) + 1);
          order.push_back(0);
          std::reverse(order.begin(), order.end());
          return order;
        }
      }

      // Too many operands for every subset: take the smallest product each time, and one that shares
      // no label with the accumulator only when nothing else is left, since a product without a sum
      // keeps every leg of both operands.
      std::vector<bool> taken(inputs.size(), false);
      std::vector<std::string> live = inputs.front();
      order.push_back(0);
      taken.at(0) = true;
      for (int64_t step = 1; step < count; ++step)
      {
        int64_t best = -1;
        double best_size = 0;
        bool found = false;
        for (int pass = 0; pass < 2 && !found; ++pass)
          for (int64_t at = 0; at < count; ++at)
          {
            if (taken.at(at)) continue;
            if (shared_of(live, inputs.at(at)).empty() != (pass == 1)) continue;
            const double size = tensor_size(union_of(live, inputs.at(at)), dims);
            if (!found || size < best_size)
            {
              found = true;
              best = at;
              best_size = size;
            }
          }
        order.push_back(best);
        taken.at(best) = true;
        live = union_of(live, inputs.at(best));
      }
      return order;
    }

    // The product of two slots: every label they share is summed with one contraction, and a pair
    // that shares none is an outer product.  A sum over a charge can leave more than one charge, so
    // a slot carries every partial result it reaches.
    std::vector<Tensor> combine(const std::vector<Tensor> &left_states,
                                const std::vector<std::string> &left_labels,
                                const std::vector<Tensor> &right_states,
                                const std::vector<std::string> &right_labels)
    {
      // The sectors the two charges can close to, each listed once.
      const auto closings_of = [](const FusionCategory &cat, const Sector &first, const Sector &second)
      {
        std::vector<Sector> out;
        for (const Outcome &outcome : cat.outcomes(first, second))
          if (std::find(out.begin(), out.end(), outcome.sector) == out.end())
            out.push_back(outcome.sector);
        return out;
      };

      // Every shared label is summed, so its leg of the left slot pairs with its leg of the right
      // one; a slot that shares nothing has no pair and the product is the outer one.
      const std::vector<std::string> shared = shared_of(left_labels, right_labels);
      std::vector<int64_t> left_legs;
      std::vector<int64_t> right_legs;
      for (const std::string &label : shared)
      {
        left_legs.push_back(position_of(left_labels, label));
        right_legs.push_back(position_of(right_labels, label));
      }

      std::vector<Tensor> folded;
      for (const Tensor &left : left_states)
        for (const Tensor &right : right_states)
        {
          // Which charge the pair closes to does not change the layout, so the two are laid out once
          // and every closing only redoes the product over the blocks.
          const auto [above, under] = align(left, right, left_legs, right_legs);
          const std::vector<int64_t> above_legs = domain_legs(above);
          const std::vector<int64_t> under_legs = codomain_legs(under);
          for (const Sector &closed : closings_of(*left.cat(), left.qtotal(), right.qtotal()))
            folded.push_back(contract(above, under, above_legs, under_legs, closed));
        }
      return folded;
    }

    // The operands folded in the order `order` gives, each step summing every label the accumulator
    // and the next operand share.  A step that sums a charge can leave more than one charge, so the
    // fold carries every partial result it reaches.  The legs the output keeps are laid out as its
    // two groups read them, the upper case ones on the codomain and the lower case ones under them.
    std::vector<Tensor> contract_path(const std::vector<std::vector<std::string>> &inputs,
                                      const Side &output,
                                      const std::vector<Tensor> &operands,
                                      const std::vector<int64_t> &order)
    {
      std::vector<std::string> labels = inputs.front();
      std::vector<Tensor> states{operands.front()};
      for (size_t step = 1; step < order.size(); ++step)
      {
        const int64_t at = order.at(step);
        std::vector<Tensor> folded = combine(states, labels, {operands.at(at)}, inputs.at(at));
        if (static_cast<int64_t>(folded.size()) > kMaxCharges)
          throw std::invalid_argument("the products of the equation leave more than " +
                                      std::to_string(kMaxCharges) + " charges to sum, which is "
                                      "more than this takes");
        labels = product_labels(labels, inputs.at(at));
        states = std::move(folded);
      }
      std::vector<int64_t> above;
      std::vector<int64_t> below;
      for (size_t at = 0; at < output.labels.size(); ++at)
        (output.above.at(at) ? above : below).push_back(position_of(labels, output.labels.at(at)));
      std::vector<Tensor> out;
      for (Tensor &state : states) out.push_back(state.place(above, below));
      return out;
    }

    // Every operand, once per charge a self-contracted label of it can leave, and the labels it keeps.
    struct Prepared
    {
      std::vector<std::vector<Tensor>> operands;
      std::vector<std::vector<std::string>> labels;
    };

    // Every operand with the labels it names twice summed away, and the labels each one keeps.
    Prepared self_contract(const std::vector<Tensor> &operands,
                           const std::vector<std::vector<std::string>> &inputs,
                           const std::string &spec)
    {
      Prepared prepared;
      for (size_t at = 0; at < operands.size(); ++at)
      {
        std::vector<Tensor> alternatives{operands.at(at)};
        std::vector<std::string> labels = inputs.at(at);
        while (true)
        {
          int64_t first = -1;
          int64_t second = -1;
          for (size_t one = 0; one < labels.size() && first < 0; ++one)
            for (size_t other = one + 1; other < labels.size(); ++other)
              if (labels.at(one) == labels.at(other))
              {
                first = static_cast<int64_t>(one);
                second = static_cast<int64_t>(other);
                break;
              }
          if (first < 0) break;
          const std::string &label = labels.at(static_cast<size_t>(first));
          const int64_t codomain = static_cast<int64_t>(alternatives.front().codomain().size());
          if (!(alternatives.front().legs().at(static_cast<size_t>(first)) ==
                alternatives.front().legs().at(static_cast<size_t>(second))))
            throw std::invalid_argument("the label '" + label + "' of '" + spec +
                                        "' names two different spaces in the operand " +
                                        std::to_string(at) + ", and a trace sums a space with "
                                        "itself");
          // A trace reads one leg of the label above and the other below, and the placement does the
          // recoupling when the two sit on one side: the leg already above is the upper one, so only
          // the order the two are trimmed from changes when it is the second.
          const int64_t above_leg = first < codomain ? first : second;
          const int64_t below_leg = first < codomain ? second : first;
          std::vector<int64_t> above;
          std::vector<int64_t> below;
          for (int64_t leg = 0; leg < alternatives.front().nlegs(); ++leg)
            if (leg != above_leg && leg != below_leg) (leg < codomain ? above : below).push_back(leg);
          above.push_back(above_leg);
          below.push_back(below_leg);
          std::vector<Tensor> traced;
          for (const Tensor &alternative : alternatives)
            for (auto &answer : partial_trace(alternative.place(above, below)))
              traced.push_back(std::move(answer.second));
          alternatives = std::move(traced);
          labels.erase(labels.begin() + second);
          labels.erase(labels.begin() + first);
        }
        prepared.operands.push_back(std::move(alternatives));
        prepared.labels.push_back(labels);
      }
      return prepared;
    }

  } // namespace


  Tensor einsum(const std::string &spec, const std::vector<Tensor> &operands, const Sector &charge)
  {
    if (operands.empty())
      throw std::invalid_argument("an einsum needs at least one operand");
    const std::string arrow = "->";
    const size_t at_arrow = spec.find(arrow);
    if (at_arrow == std::string::npos)
      throw std::invalid_argument("the equation '" + spec + "' does not say what it leaves");

    std::vector<Side> groups;
    std::string left = spec.substr(0, at_arrow);
    size_t start = 0;
    while (true)
    {
      const size_t comma = left.find(',', start);
      groups.push_back(side_of(left.substr(start, comma - start)));
      if (comma == std::string::npos) break;
      start = comma + 1;
    }
    const Side output = side_of(spec.substr(at_arrow + arrow.size()));

    if (groups.size() != operands.size())
      throw std::invalid_argument("the equation '" + spec + "' names " +
                                  std::to_string(groups.size()) + " operands but got " +
                                  std::to_string(operands.size()));
    const FusionCategory &cat = *operands.front().cat();
    for (size_t at = 0; at < operands.size(); ++at)
    {
      if (!same_group(*operands.at(at).cat(), cat))
        throw std::invalid_argument("an einsum needs every operand over one category");
      if (groups.at(at).labels.size() != static_cast<size_t>(operands.at(at).nlegs()))
        throw std::invalid_argument("the group " + std::to_string(at) + " of '" + spec +
                                    "' names " + std::to_string(groups.at(at).labels.size()) +
                                    " legs but the tensor has " +
                                    std::to_string(operands.at(at).nlegs()));
    }

    // A label is summed over at most two legs, and it is either summed away or kept, not both.
    std::vector<std::string> seen;
    for (const Side &group : groups)
      for (const std::string &label : group.labels)
      {
        if (std::count(seen.begin(), seen.end(), label) >= 2)
          throw std::invalid_argument("the label '" + label + "' of '" + spec +
                                      "' names more than two legs");
        seen.push_back(label);
      }
    for (const std::string &label : seen)
    {
      const auto count = std::count(seen.begin(), seen.end(), label);
      if (count == 1 && !holds(output.labels, label))
        throw std::invalid_argument("the label '" + label + "' of '" + spec +
                                    "' names one leg but is not kept in the output, which would "
                                    "leave it open");
      if (count == 2 && holds(output.labels, label))
        throw std::invalid_argument("the label '" + label + "' of '" + spec +
                                    "' is summed over and still kept");
    }
    for (const std::string &label : output.labels)
      if (std::count(seen.begin(), seen.end(), label) != 1)
        throw std::invalid_argument("the output label '" + label + "' of '" + spec +
                                    "' does not name a leg of the operands");

    // Every leg the equation keeps takes the side its case declares, and every leg it sums keeps the
    // side it already has: the two legs of a sum are paired through the contraction, so one of them
    // has to change side anyway and the case cannot say which.
    std::vector<std::string> summed;
    for (const std::string &label : seen)
      if (std::count(seen.begin(), seen.end(), label) == 2 && !holds(summed, label))
        summed.push_back(label);
    std::vector<Tensor> laid;
    std::vector<std::vector<std::string>> inputs;
    for (size_t at = 0; at < operands.size(); ++at)
    {
      auto [tensor, labels] = laid_out(operands.at(at), groups.at(at), summed);
      laid.push_back(std::move(tensor));
      inputs.push_back(std::move(labels));
    }

    const Prepared prepared = self_contract(laid, inputs, spec);

    // The legs the operands keep have to close to the charge the result carries: the roots the legs
    // the output puts above reach and the roots the ones it puts below reach have to pair to it.
    std::vector<Leg> above_legs;
    std::vector<Leg> below_legs;
    for (size_t one = 0; one < output.labels.size(); ++one)
      for (size_t at = 0; at < prepared.labels.size(); ++at)
        if (holds(prepared.labels.at(at), output.labels.at(one)))
        {
          const Leg leg = prepared.operands.at(at).front().legs().at(static_cast<size_t>(
            position_of(prepared.labels.at(at), output.labels.at(one))));
          (output.above.at(one) ? above_legs : below_legs).push_back(leg);
          break;
        }
    bool closes = output.labels.empty();
    for (const Sector &above : reachable_charges(cat, above_legs))
      for (const Sector &below : reachable_charges(cat, below_legs))
        if (cat.n_symbol(above, cat.dual(below), charge) > 0) closes = true;
    if (!closes)
      throw std::invalid_argument("the output legs of '" + spec + "' cannot close to " +
                                  render(charge));

    // A self-contracted leg leaves the charge its kept roots close to, so an operand can come with
    // more than one charge and the sum runs over every combination of them.
    size_t combinations = 1;
    for (const std::vector<Tensor> &alternatives : prepared.operands)
    {
      if (alternatives.empty())
        throw std::invalid_argument("the traces of '" + spec + "' leave nothing to sum");
      if (static_cast<int64_t>(alternatives.size()) >
          kMaxCharges / static_cast<int64_t>(combinations))
        throw std::invalid_argument("the traces of '" + spec + "' leave more than " +
                                    std::to_string(kMaxCharges) +
                                    " combinations of charges, which is more than this sums");
      combinations *= alternatives.size();
    }

    // The order the operands are contracted in.  A label keeps the space its leg names, so the size
    // of every partial product is read from the dimensions of the legs.
    std::map<std::string, double> dims;
    for (size_t at = 0; at < prepared.labels.size(); ++at)
    {
      const std::vector<Leg> legs = prepared.operands.at(at).front().legs();
      for (size_t leg = 0; leg < prepared.labels.at(at).size(); ++leg)
        dims[prepared.labels.at(at).at(leg)] = leg_size(cat, legs.at(leg));
    }
    const std::vector<int64_t> order = fold_order(prepared.labels, dims);

    std::optional<Tensor> total;
    std::vector<size_t> choice(prepared.operands.size(), 0);
    while (true)
    {
      std::vector<Tensor> operands;
      for (size_t at = 0; at < choice.size(); ++at)
        operands.push_back(prepared.operands.at(at).at(choice.at(at)));
      for (const Tensor &result : contract_path(prepared.labels, output, operands, order))
      {
        // The folds that land on another charge belong to a different sum.
        if (!(result.qtotal() == charge) || result.blocks().empty()) continue;
        if (!total) total = result;
        else *total += result;
      }
      size_t digit = 0;
      while (digit < choice.size() && ++choice.at(digit) == prepared.operands.at(digit).size())
      {
        choice.at(digit) = 0;
        ++digit;
      }
      if (digit == choice.size()) break;
    }
    if (!total)
      throw std::invalid_argument("no way for the charges of '" + spec + "' to fuse to " +
                                  render(charge));
    return *total;
  }

  Tensor einsum(const std::string &spec, const std::vector<Tensor> &operands)
  {
    if (operands.empty())
      throw std::invalid_argument("an einsum needs at least one operand");
    return einsum(spec, operands, operands.front().cat()->trivial());
  }

} // namespace sym
