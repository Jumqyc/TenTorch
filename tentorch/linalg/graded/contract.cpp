#include "contract.h"

#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../symmetry/tree/tree.h"

namespace sym
{

  namespace
  {

    // The legs a list leaves out, in their original order.
    std::vector<int64_t> without(int64_t total, const std::vector<int64_t> &taken)
    {
      std::vector<bool> named(static_cast<size_t>(total), false);
      for (const int64_t leg : taken) named.at(static_cast<size_t>(leg)) = true;
      std::vector<int64_t> out;
      out.reserve(static_cast<size_t>(total) - taken.size());
      for (int64_t leg = 0; leg < total; ++leg)
        if (!named.at(static_cast<size_t>(leg))) out.push_back(leg);
      return out;
    }

    // One side's list of summed legs: in range and named once.
    void check_side(const Tensor &tensor, const std::vector<int64_t> &legs, const char *side)
    {
      std::vector<bool> named(static_cast<size_t>(tensor.nlegs()), false);
      for (const int64_t leg : legs)
      {
        if (leg < 0 || leg >= tensor.nlegs())
          throw std::invalid_argument("the leg " + std::to_string(leg) + " of the " + side +
                                      " tensor is out of range of its " +
                                      std::to_string(tensor.nlegs()) + " legs");
        if (named.at(static_cast<size_t>(leg)))
          throw std::invalid_argument("the leg " + std::to_string(leg) + " of the " + side +
                                      " tensor is summed twice");
        named.at(static_cast<size_t>(leg)) = true;
      }
    }

    // Everything about the two tensors that the splitting of a contraction does not change.
    void check_pairs(const Tensor &left,
                     const Tensor &right,
                     const std::vector<int64_t> &left_legs,
                     const std::vector<int64_t> &right_legs)
    {
      if (!same_group(*left.cat(), *right.cat()))
        throw std::invalid_argument("a contraction needs both tensors over one category");
      if (left.dtype() != right.dtype())
        throw std::invalid_argument("a contraction needs both tensors of one scalar type");
      if (left_legs.size() != right_legs.size())
        throw std::invalid_argument("the two legs of a contraction must be given in pairs, got " +
                                    std::to_string(left_legs.size()) + " and " +
                                    std::to_string(right_legs.size()));
      check_side(left, left_legs, "first");
      check_side(right, right_legs, "second");
      // A summed leg is one index of both factors, so the two legs of a pair index one space.
      const std::vector<Leg> left_all = left.legs();
      const std::vector<Leg> right_all = right.legs();
      for (size_t at = 0; at < left_legs.size(); ++at)
      {
        const Leg &one = left_all.at(static_cast<size_t>(left_legs.at(at)));
        const Leg &other = right_all.at(static_cast<size_t>(right_legs.at(at)));
        if (!(one == other))
          throw std::invalid_argument(
            "the pair at position " + std::to_string(at) + " joins the leg " +
            std::to_string(left_legs.at(at)) + " of the first tensor, over " +
            render_list(one.sectors()) + ", to the leg " + std::to_string(right_legs.at(at)) +
            " of the second, over " + render_list(other.sectors()) +
            ", and the two do not name one space");
      }
    }

    // The two operands at the layout the block product reads; `check_pairs` has to have run.
    std::pair<Tensor, Tensor> paired(const Tensor &left,
                                     const Tensor &right,
                                     const std::vector<int64_t> &left_legs,
                                     const std::vector<int64_t> &right_legs)
    {
      // A block product sums the columns of the left factor against the rows of the right one, so the
      // summed legs are brought to the last legs of the domain of the first tensor and the first legs
      // of the codomain of the second, in the order the pairs come in.  A tensor that already sits
      // that way is not rewritten.
      return {left.place(without(left.nlegs(), left_legs), left_legs),
              right.place(right_legs, without(right.nlegs(), right_legs))};
    }

  } // namespace

  void require_single_closing(const FusionCategory &cat,
                              const Sector &above,
                              const Sector &below,
                              const Sector &charge,
                              const std::string &subject,
                              const std::string &what)
  {
    if (cat.n_symbol(above, cat.dual(below), charge) > 1)
      throw std::invalid_argument(subject + " " + render(above) + " and " + render(below) +
                                  " close to " + render(charge) +
                                  " in more than one way, which " + what + " does not recouple yet");
  }

  std::pair<Tensor, Tensor> align(const Tensor &left,
                                  const Tensor &right,
                                  const std::vector<int64_t> &left_legs,
                                  const std::vector<int64_t> &right_legs)
  {
    check_pairs(left, right, left_legs, right_legs);
    return paired(left, right, left_legs, right_legs);
  }

  Tensor contract(const Tensor &left,
                  const Tensor &right,
                  const std::vector<int64_t> &left_legs,
                  const std::vector<int64_t> &right_legs,
                  const Sector &charge)
  {
    check_pairs(left, right, left_legs, right_legs);
    const FusionCategory &cat = *left.cat();
    const int64_t outcomes = cat.n_symbol(left.qtotal(), right.qtotal(), charge);
    if (outcomes == 0)
      throw std::invalid_argument("the charge " + render(charge) + " is not an outcome of " +
                                  render(left.qtotal()) + " and " + render(right.qtotal()));
    require_single_closing(cat, left.qtotal(), right.qtotal(), charge, "the charges", "a contraction");

    const auto [above, under] = paired(left, right, left_legs, right_legs);
    // The product of a tensor of charge q with one of charge r carries q (x) r, and a sum over the
    // legs between them keeps that charge.
    Tensor out(above.cat(), above.codomain(), under.domain(), charge, left.dtype(), left.device());
    for (const auto &key : above.keys())
      require_single_closing(cat, key.codomain, key.domain, above.qtotal());
    for (const auto &key : under.keys())
      require_single_closing(cat, key.codomain, key.domain, under.qtotal());
    for (const auto &key : out.keys())
      require_single_closing(cat, key.codomain, key.domain, charge);

    // The blocks of the second tensor, grouped by the root its codomain fuses to: that root is the
    // one the summed legs carry, so it has to equal the root of the domain of the first tensor.
    std::map<Sector, std::vector<std::pair<Sector, const at::Tensor *>>> rows;
    for (const auto &[key, block] : under.blocks())
      rows[key.codomain].push_back({key.domain, &block});

    for (const auto &[key, left_block] : above.blocks())
    {
      const auto found = rows.find(key.domain);
      if (found == rows.end()) continue;
      for (const auto &[domain_root, right_block] : found->second)
      {
        // The charges decide whether a pair of blocks can land in the result at all; the pairs that
        // cannot are zero, and a product of zero is what a fusion path that is not allowed carries.
        if (cat.n_symbol(key.codomain, cat.dual(domain_root), charge) == 0) continue;
        if (left_block.size(1) != right_block->size(0))
          throw std::logic_error("the two blocks over the roots " + render(key.domain) +
                                 " do not line up: " + std::to_string(left_block.size(1)) +
                                 " against " + std::to_string(right_block->size(0)));
        out.block(key.codomain, domain_root).addmm_(left_block, *right_block);
      }
    }
    return out;
  }

  Tensor contract(const Tensor &left,
                  const Tensor &right,
                  const std::vector<int64_t> &left_legs,
                  const std::vector<int64_t> &right_legs)
  {
    if (!(right.qtotal() == left.cat()->trivial()))
      throw std::invalid_argument("the second tensor of a contraction must carry the trivial charge");
    return contract(left, right, left_legs, right_legs, left.qtotal());
  }

} // namespace sym
