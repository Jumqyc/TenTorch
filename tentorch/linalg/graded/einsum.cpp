#include "einsum.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
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

    std::vector<std::string> letters(const std::string &text)
    {
      std::vector<std::string> labels;
      for (const char letter : text)
        if (letter != ' ') labels.emplace_back(1, letter);
      return labels;
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

    // The tensor with every leg on the codomain: its codomain in place and then its domain, last one
    // first, which is where a bend puts a leg every time.
    Tensor all_up(const Tensor &tensor)
    {
      Tensor out = tensor;
      for (int64_t at = static_cast<int64_t>(out.domain().size()); at > 0; --at)
        out = out.move_across(at - 1, false);
      return out;
    }

    // Where a leg of `tensor` sits once every leg is on the codomain, numbered as `legs()` numbers.
    int64_t raised(const Tensor &tensor, int64_t leg)
    {
      const int64_t codomain = static_cast<int64_t>(tensor.codomain().size());
      return leg < codomain ? leg : codomain + (static_cast<int64_t>(tensor.domain().size()) - 1 -
                                                (leg - codomain));
    }

    // The tensor over the legs `above` on the codomain and `below` under them on the domain, each
    // list in the order it is to read, every leg of the tensor named by one of them.  This is the
    // general form of Tensor::repartition, which asks for the legs above in the order they already
    // have and leaves the rest under them; the order below has to be spelled out here because a
    // traced leg is put last on both sides.
    Tensor place(const Tensor &tensor, const std::vector<int64_t> &above,
                 const std::vector<int64_t> &below)
    {
      std::vector<int64_t> order;
      for (const int64_t leg : above) order.push_back(raised(tensor, leg));
      for (auto at = below.rbegin(); at != below.rend(); ++at) order.push_back(raised(tensor, *at));
      Tensor out = all_up(tensor).permute(order);
      for (size_t at = 0; at < below.size(); ++at)
        out = out.move_across(static_cast<int64_t>(out.codomain().size()) - 1, true);
      return out;
    }

    // The fusion tree with its last vertex removed: one leg fewer, the root the remaining legs
    // couple to.  The side with one leg leaves the unit tree.
    FusionTree trimmed(const FusionCategory &cat, const FusionTree &tree)
    {
      const int64_t count = tree.size();
      FusionTree out = tree;
      out.sectors.pop_back();
      if (!out.inner.empty()) out.inner.pop_back();
      if (!out.copies.empty()) out.copies.pop_back();
      if (count <= 1) out.coupled = cat.trivial();
      else if (count == 2) out.coupled = tree.sectors.at(0);
      else out.coupled = tree.inner.at(static_cast<size_t>(count - 3));
      return out;
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
        if (cat->n_symbol(key.codomain, cat->dual(key.domain), qtotal) > 1)
          throw std::invalid_argument("the roots " + render(key.codomain) + " and " +
                                      render(key.domain) + " of a trace close to " + render(qtotal) +
                                      " in more than one way, which a trace does not recouple yet");
        const SideLayout rows = layout_of(*cat, tensor.codomain(), key.codomain);
        const SideLayout columns = layout_of(*cat, tensor.domain(), key.domain);
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
            const SideLayout out_rows = layout_of(*cat, out_codomain, kept_above.coupled);
            const SideLayout out_columns = layout_of(*cat, out_domain, kept_under.coupled);
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
              if (closing.multiplicity > 1)
                throw std::invalid_argument(
                  "the kept roots " + render(kept_above.coupled) + " and " +
                  render(kept_under.coupled) + " of a trace close to " + render(closing.sector) +
                  " in more than one way, which a trace does not recouple yet");
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

    // The product of two operands that share no label: one keeps every leg above and the other
    // brings every leg below, so the block of the product is one block product with nothing summed.
    Tensor outer(const Tensor &above, const Tensor &under, const Sector &charge)
    {
      const FusionCategory &cat = *above.cat();
      if (!above.domain().empty() || !under.codomain().empty())
        throw std::logic_error("an outer product takes one operand with its legs above and one with "
                               "its legs below");
      const int64_t closings = cat.n_symbol(above.qtotal(), under.qtotal(), charge);
      if (closings > 1)
        throw std::invalid_argument("the charges " + render(above.qtotal()) + " and " +
                                    render(under.qtotal()) + " close to " + render(charge) +
                                    " in more than one way, which an outer product does not "
                                    "recouple yet");
      // A block of an operand is labelled by the roots of its own closing vertex, so a closing that
      // happens more than one way has no single copy to multiply and the product would drop it.
      const auto refuse_copies = [&](const Tensor &operand, const char *where) {
        for (const auto &[key, block] : operand.blocks())
          if (cat.n_symbol(key.codomain, cat.dual(key.domain), operand.qtotal()) > 1)
            throw std::invalid_argument("the roots " + render(key.codomain) + " and " +
                                        render(key.domain) + " of the operand " + where +
                                        " close to " + render(operand.qtotal()) +
                                        " in more than one way, which an outer product does not "
                                        "recouple yet");
      };
      refuse_copies(above, "above");
      refuse_copies(under, "below");
      Tensor out(above.cat(), above.codomain(), under.domain(), charge, above.dtype(),
                 above.device());
      for (const auto &[key, left] : above.blocks())
        for (const auto &[held, right] : under.blocks())
          out.block(key.codomain, held.domain).addmm_(left, right);
      return out;
    }

    // The operands folded left to right: each step sums every label the state and the operand share
    // by one contraction, so a label whose two ends meet is summed in that same move.  A glue sums
    // over the charge it lands on and can leave more than one charge, so the fold carries every
    // partial result it reaches.
    std::vector<Tensor> along(const std::vector<std::vector<std::string>> &inputs,
                              const std::vector<std::string> &output,
                              const std::vector<Tensor> &operands)
    {
      // Every leg of a tensor, in the order the tensor reads them.
      const auto legs_of = [](const Tensor &tensor)
      {
        std::vector<int64_t> out;
        for (int64_t leg = 0; leg < tensor.nlegs(); ++leg) out.push_back(leg);
        return out;
      };
      // The sectors the two charges can close to, each listed once.
      const auto closings_of = [](const FusionCategory &cat, const Sector &first, const Sector &second)
      {
        std::vector<Sector> out;
        for (const Outcome &outcome : cat.outcomes(first, second))
          if (std::find(out.begin(), out.end(), outcome.sector) == out.end())
            out.push_back(outcome.sector);
        return out;
      };

      std::vector<std::string> labels = inputs.front();
      std::vector<Tensor> states{operands.front()};
      for (size_t at = 1; at < operands.size(); ++at)
      {
        const std::vector<std::string> &next = inputs.at(at);
        std::vector<std::string> shared;
        for (const std::string &label : labels)
          if (holds(next, label)) shared.push_back(label);

        if (shared.empty())
        {
          // Nothing is summed by this step: one operand keeps every leg above, the other brings
          // every leg below, and no leg has to be recoupled.
          std::vector<Tensor> joined;
          for (const Tensor &state : states)
          {
            const Tensor head = state.repartition(legs_of(state));
            const Tensor tail = operands.at(at).repartition({});
            for (const Sector &closed : closings_of(*state.cat(), state.qtotal(),
                                                    operands.at(at).qtotal()))
              joined.push_back(outer(head, tail, closed));
          }
          labels.insert(labels.end(), next.begin(), next.end());
          states = std::move(joined);
          if (states.size() > 4096)
            throw std::invalid_argument("the products of the equation leave more than 4096 charges "
                                        "to sum, which is more than this takes");
          continue;
        }

        // The state keeps its unsummed legs above and brings the summed ones below; the operand
        // brings the summed ones above and keeps the rest below.  One contraction then sums every
        // shared label, the operand's codomain being exactly the rows of the state's domain.
        std::vector<int64_t> kept;
        for (int64_t leg = 0; leg < static_cast<int64_t>(labels.size()); ++leg)
          if (std::find(shared.begin(), shared.end(), labels.at(leg)) == shared.end())
            kept.push_back(leg);
        std::vector<int64_t> summed;
        std::vector<int64_t> order;
        for (const std::string &label : shared)
        {
          summed.push_back(position_of(labels, label));
          order.push_back(position_of(next, label));
        }
        std::vector<int64_t> loose = legs_of(operands.at(at));
        for (int64_t leg : order) loose.erase(std::find(loose.begin(), loose.end(), leg));
        std::vector<int64_t> rows;
        for (size_t leg = 0; leg < shared.size(); ++leg) rows.push_back(static_cast<int64_t>(leg));

        std::vector<Tensor> folded;
        for (const Tensor &state : states)
        {
          const Tensor head = place(state, kept, summed);
          const Tensor tail = place(operands.at(at), order, loose);
          for (size_t one = 0; one < shared.size(); ++one)
            if (!(head.domain().at(one) == tail.codomain().at(one)))
              throw std::invalid_argument("the label '" + shared.at(one) +
                                          "' of the equation names two different spaces in the "
                                          "operands " + std::to_string(at - 1) + " and " +
                                          std::to_string(at) +
                                          ", and only one space can be summed over a label");
          for (const Sector &closed : closings_of(*state.cat(), state.qtotal(),
                                                  operands.at(at).qtotal()))
            folded.push_back(contract(head, tail, rows, rows, closed));
        }

        // The summed labels are gone: the state keeps what it kept above and the operand brings
        // whatever it did not sum below.
        std::vector<std::string> folded_labels;
        for (int64_t leg : kept) folded_labels.push_back(labels.at(static_cast<size_t>(leg)));
        for (int64_t leg : loose) folded_labels.push_back(next.at(static_cast<size_t>(leg)));
        labels = std::move(folded_labels);
        states = std::move(folded);
        if (states.size() > 4096)
          throw std::invalid_argument("the loops of the equation leave more than 4096 charges to "
                                      "sum, which is more than this takes");
        continue;
      }

      std::vector<Tensor> out;
      for (Tensor &state : states)
      {
        std::vector<int64_t> order;
        for (const std::string &label : output) order.push_back(position_of(labels, label));
        out.push_back(place(state, order, {}));
      }
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
          // Two legs of one label on the same side pair through the invariant form of their space:
          // bend one across, which is the recoupling that reads a leg on the other side.
          const bool opposite = (first < codomain) != (second < codomain);
          int64_t moved = second;
          if (!opposite)
          {
            for (Tensor &alternative : alternatives)
              alternative = alternative.move_across(second, first < codomain);
            moved = alternatives.front().nlegs() - 1;
          }
          const int64_t above_leg = first < codomain ? first : moved;
          const int64_t below_leg = first < codomain ? moved : first;
          const int64_t sides = static_cast<int64_t>(alternatives.front().codomain().size());
          std::vector<int64_t> above;
          std::vector<int64_t> below;
          for (int64_t leg = 0; leg < alternatives.front().nlegs(); ++leg)
            if (leg != above_leg && leg != below_leg) (leg < sides ? above : below).push_back(leg);
          above.push_back(above_leg);
          below.push_back(below_leg);
          std::vector<Tensor> traced;
          for (const Tensor &alternative : alternatives)
            for (auto &answer : partial_trace(place(alternative, above, below)))
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

    std::vector<std::vector<std::string>> inputs;
    std::string left = spec.substr(0, at_arrow);
    size_t start = 0;
    while (true)
    {
      const size_t comma = left.find(',', start);
      inputs.push_back(letters(left.substr(start, comma - start)));
      if (comma == std::string::npos) break;
      start = comma + 1;
    }
    const std::vector<std::string> output = letters(spec.substr(at_arrow + arrow.size()));

    if (inputs.size() != operands.size())
      throw std::invalid_argument("the equation '" + spec + "' names " +
                                  std::to_string(inputs.size()) + " operands but got " +
                                  std::to_string(operands.size()));
    const FusionCategory &cat = *operands.front().cat();
    for (size_t at = 0; at < operands.size(); ++at)
    {
      if (!same_group(*operands.at(at).cat(), cat))
        throw std::invalid_argument("an einsum needs every operand over one category");
      if (inputs.at(at).size() != static_cast<size_t>(operands.at(at).nlegs()))
        throw std::invalid_argument("the group " + std::to_string(at) + " of '" + spec +
                                    "' names " + std::to_string(inputs.at(at).size()) +
                                    " legs but the tensor has " +
                                    std::to_string(operands.at(at).nlegs()));
    }

    // A label is summed over at most two legs, and it is either summed away or kept, not both.
    std::vector<std::string> seen;
    for (const std::vector<std::string> &group : inputs)
      for (const std::string &label : group)
      {
        if (std::count(seen.begin(), seen.end(), label) >= 2)
          throw std::invalid_argument("the label '" + label + "' of '" + spec +
                                      "' names more than two legs");
        seen.push_back(label);
      }
    for (const std::string &label : seen)
    {
      const auto count = std::count(seen.begin(), seen.end(), label);
      if (count == 1 && !holds(output, label))
        throw std::invalid_argument("the label '" + label + "' of '" + spec +
                                    "' names one leg but is not kept in the output, which would "
                                    "leave it open");
      if (count == 2 && holds(output, label))
        throw std::invalid_argument("the label '" + label + "' of '" + spec +
                                    "' is summed over and still kept");
    }
    for (const std::string &label : output)
      if (std::count(seen.begin(), seen.end(), label) != 1)
        throw std::invalid_argument("the output label '" + label + "' of '" + spec +
                                    "' does not name a leg of the operands");

    const Prepared prepared = self_contract(operands, inputs, spec);

    // The legs the operands keep have to span the charge the result carries.
    std::vector<Leg> output_legs;
    for (const std::string &label : output)
      for (size_t at = 0; at < prepared.labels.size(); ++at)
        if (holds(prepared.labels.at(at), label))
        {
          output_legs.push_back(prepared.operands.at(at).front().legs().at(static_cast<size_t>(
            position_of(prepared.labels.at(at), label))));
          break;
        }
    if (!output.empty() && fusion_trees(cat, output_legs, charge).empty())
      throw std::invalid_argument("the output legs of '" + spec + "' cannot fuse to " +
                                  render(charge));

    // A self-contracted leg leaves the charge its kept roots close to, so an operand can come with
    // more than one charge and the sum runs over every combination of them.
    size_t combinations = 1;
    for (const std::vector<Tensor> &alternatives : prepared.operands)
    {
      if (alternatives.empty())
        throw std::invalid_argument("the traces of '" + spec + "' leave nothing to sum");
      if (alternatives.size() > 4096 / combinations)
        throw std::invalid_argument("the traces of '" + spec +
                                    "' leave more than 4096 combinations of charges, which is more "
                                    "than this sums");
      combinations *= alternatives.size();
    }

    std::optional<Tensor> total;
    std::vector<size_t> choice(prepared.operands.size(), 0);
    while (true)
    {
      std::vector<Tensor> operands;
      for (size_t at = 0; at < choice.size(); ++at)
        operands.push_back(prepared.operands.at(at).at(choice.at(at)));
      for (const Tensor &result : along(prepared.labels, output, operands))
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
