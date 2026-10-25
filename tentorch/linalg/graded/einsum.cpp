#include "einsum.h"

#include <algorithm>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
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

    // Every sequence of intermediate charges that fuses the operand charges, one per step.
    std::vector<std::vector<Sector>> charge_paths(const std::vector<Sector> &charges,
                                                  const FusionCategory &cat)
    {
      std::vector<std::vector<Sector>> paths;
      std::vector<Sector> trail;
      std::function<void(size_t, const Sector &)> walk = [&](size_t at, const Sector &current) {
        if (at == charges.size())
        {
          if (paths.size() >= 4096)
            throw std::invalid_argument("the charges of these operands fuse in more than 4096 ways, "
                                        "which is more than this sums");
          paths.push_back(trail);
          return;
        }
        for (const Outcome &outcome : cat.outcomes(current, charges.at(at)))
          for (int64_t copy = 0; copy < outcome.multiplicity; ++copy)
          {
            trail.push_back(outcome.sector);
            walk(at + 1, outcome.sector);
            trail.pop_back();
          }
      };
      walk(1, charges.front());
      return paths;
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
    // list in the order it is to read, every leg of the tensor named by one of them.
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

    // One path of intermediate charges, contracted left to right and brought to the output order.
    Tensor along(const std::vector<std::vector<std::string>> &inputs,
                 const std::vector<std::string> &output,
                 const std::vector<Tensor> &operands,
                 const std::vector<Sector> &path)
    {
      std::vector<std::string> labels = inputs.front();
      Tensor acc = operands.front();
      for (size_t at = 1; at < operands.size(); ++at)
      {
        const std::vector<std::string> &next = inputs.at(at);
        std::vector<std::string> shared;
        for (const std::string &label : labels)
          if (holds(next, label)) shared.push_back(label);
        if (shared.empty())
          throw std::invalid_argument("the operands " + std::to_string(at - 1) + " and " +
                                      std::to_string(at) + " of the equation share no leg, which is "
                                      "an outer product and not a sum");

        // An operand handed in as a bra has the conjugate space on every leg, so the two legs that
        // meet are different spaces; reading it as the ket it is conjugates the entries and swaps
        // the two sides, and its labels swap with them.
        Tensor body = operands.at(at);
        std::vector<std::string> body_labels = next;
        bool flipped = false;
        for (const std::string &label : shared)
          if (!(acc.legs().at(position_of(labels, label)) ==
                body.legs().at(position_of(body_labels, label))))
          {
            if (!(body.qtotal() == body.cat()->trivial()))
              throw std::invalid_argument("the operand " + std::to_string(at) + " of '" + output[0] +
                                          "' meets the accumulator as a bra, and reading it as a ket "
                                          "needs the charge path over dual charges, which this does "
                                          "not sum yet");
            if (flipped)
              throw std::invalid_argument("the legs named '" + label + "' do not index the same space "
                                          "in the two operands that meet there");
            body = body.conj();
            body_labels.assign(next.begin() + static_cast<int64_t>(body.domain().size()), next.end());
            body_labels.insert(body_labels.end(), next.begin(),
                               next.begin() + static_cast<int64_t>(body.codomain().size()));
            flipped = true;
          }

        // The accumulator keeps its own legs above and hands the shared ones to the domain, in their
        // own order; the other operand comes up with the shared legs in exactly that order.
        std::vector<int64_t> kept;
        std::vector<int64_t> given;
        std::vector<int64_t> loose;
        for (size_t position = 0; position < labels.size(); ++position)
          (holds(shared, labels.at(position)) ? given : kept).push_back(
            static_cast<int64_t>(position));
        std::vector<int64_t> held;
        for (const std::string &label : shared) held.push_back(position_of(body_labels, label));
        for (size_t position = 0; position < body_labels.size(); ++position)
          if (!holds(shared, body_labels.at(position))) loose.push_back(static_cast<int64_t>(position));

        Tensor head = place(acc, kept, given);
        Tensor tail = place(body, held, loose);

        std::vector<int64_t> steps;
        for (size_t step = 0; step < shared.size(); ++step) steps.push_back(static_cast<int64_t>(step));
        acc = contract(head, tail, steps, steps, path.at(at - 1));

        labels.clear();
        for (const int64_t position : kept) labels.push_back(inputs.at(0).empty() ? "" : "");
        labels.clear();
        for (size_t position = 0; position < inputs.size(); ++position) break;
        labels.clear();
        for (const int64_t position : kept) labels.push_back(previous(at, position));
      }

      std::vector<int64_t> order;
      for (const std::string &label : output) order.push_back(position_of(labels, label));
      return place(acc, order, {});
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
        throw std::invalid_argument("the group '" + std::string(1, static_cast<char>('a' + at)) +
                                    "' of '" + spec + "' names " +
                                    std::to_string(inputs.at(at).size()) + " legs but the tensor has " +
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
                                    "' names one leg, which would be a trace");
      if (count == 2 && holds(output, label))
        throw std::invalid_argument("the label '" + label + "' of '" + spec +
                                    "' is summed over and still kept");
    }
    for (const std::string &label : output)
      if (std::count(seen.begin(), seen.end(), label) != 1)
        throw std::invalid_argument("the output label '" + label + "' of '" + spec +
                                    "' does not name a leg of the operands");

    std::vector<Sector> charges;
    for (const Tensor &operand : operands) charges.push_back(operand.qtotal());

    std::optional<Tensor> total;
    for (const std::vector<Sector> &path : charge_paths(charges, cat))
    {
      if (!(path.back() == charge)) continue;
      const Tensor result = along(inputs, output, operands, path);
      if (!total)
      {
        total = result;
        continue;
      }
      // A path whose intermediate charges the surviving legs cannot carry is the zero tensor.
      if (result.blocks().empty()) continue;
      if (total->blocks().empty()) total = result;
      else *total += result;
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
    std::vector<Sector> charges;
    for (const Tensor &operand : operands) charges.push_back(operand.qtotal());
    std::vector<Sector> ends;
    for (const std::vector<Sector> &path : charge_paths(charges, *operands.front().cat()))
      if (std::find(ends.begin(), ends.end(), path.back()) == ends.end()) ends.push_back(path.back());
    if (ends.size() != 1)
      throw std::invalid_argument("the operand charges of '" + spec + "' fuse to " +
                                  std::to_string(ends.size()) + " charges, so the charge to keep "
                                  "has to be named");
    return einsum(spec, operands, ends.front());
  }

} // namespace sym
