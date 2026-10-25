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

    // A contracted side is the leading run of legs of its tensor, in order.
    void check_prefix(const std::vector<int64_t> &legs, const char *side)
    {
      for (size_t at = 0; at < legs.size(); ++at)
      {
        if (legs.at(at) != static_cast<int64_t>(at))
        {
          throw std::invalid_argument(std::string("the contracted ") + side +
                                      " legs must be the leading ones, got " +
                                      std::to_string(legs.at(at)) + " at " +
                                      std::to_string(at));
        }
      }
    }

    // Everything about the two tensors that the splitting of a contraction does not change.
    void check_layouts(const Tensor &first,
                       const Tensor &other,
                       const std::vector<int64_t> &first_domain,
                       const std::vector<int64_t> &other_codomain)
    {
      if (!same_group(*first.cat(), *other.cat()))
        throw std::invalid_argument("a contraction needs both tensors over one category");
      if (first.dtype() != other.dtype())
        throw std::invalid_argument("a contraction needs both tensors of one scalar type");
      if (first_domain.size() != other_codomain.size())
        throw std::invalid_argument("the two legs of a contraction must be given in pairs");
      if (first_domain.empty())
        throw std::invalid_argument("a contraction needs at least one summed leg");
      // A leg that is summed leaves both blocks, so the summed legs are the whole of the domain of
      // `first` and the whole of the codomain of `other`; a leg left out would vanish silently.
      if (first_domain.size() != first.domain().size())
        throw std::invalid_argument("a contraction sums every leg of the domain of the first tensor, "
                                    "which has " +
                                    std::to_string(first.domain().size()) + " but got " +
                                    std::to_string(first_domain.size()));
      if (other_codomain.size() != other.codomain().size())
        throw std::invalid_argument("a contraction sums every leg of the codomain of the second tensor, "
                                    "which has " +
                                    std::to_string(other.codomain().size()) + " but got " +
                                    std::to_string(other_codomain.size()));
      check_prefix(first_domain, "domain");
      check_prefix(other_codomain, "codomain");
      for (size_t at = 0; at < first_domain.size(); ++at)
        if (!(first.domain().at(at) == other.codomain().at(at)))
          throw std::invalid_argument("the legs that meet at position " + std::to_string(at) + " differ");
    }

    // A pair of roots that closes in more than one way has a copy index per closing, and a product of
    // blocks would have to mix the copies through a recoupling.
    void check_single(const FusionCategory &cat, const Tensor &tensor, const Sector &charge,
                      const char *where)
    {
      for (const auto &key : tensor.keys())
      {
        const int64_t copies = cat.n_symbol(key.codomain, cat.dual(key.domain), charge);
        if (copies > 1)
          throw std::invalid_argument("the roots " + render(key.codomain) + " and " +
                                      render(key.domain) + " of the " + where + " close to " +
                                      render(charge) + " in more than one way, which a contraction "
                                      "does not recouple yet");
      }
    }

  } // namespace

  Tensor contract(const Tensor &first,
                  const Tensor &other,
                  const std::vector<int64_t> &first_domain,
                  const std::vector<int64_t> &other_codomain,
                  const Sector &charge)
  {
    check_layouts(first, other, first_domain, other_codomain);

    const FusionCategory &cat = *first.cat();
    const int64_t outcomes = cat.n_symbol(first.qtotal(), other.qtotal(), charge);
    if (outcomes == 0)
      throw std::invalid_argument("the charge " + render(charge) + " is not an outcome of " +
                                  render(first.qtotal()) + " and " + render(other.qtotal()));
    if (outcomes > 1)
      throw std::invalid_argument("the charges " + render(first.qtotal()) + " and " +
                                  render(other.qtotal()) + " close to " + render(charge) +
                                  " in more than one way, which a contraction does not recouple yet");

    // The product of a tensor of charge q with one of charge r carries q (x) r, and a sum over the
    // legs between them keeps that charge.
    Tensor out(first.cat(), first.codomain(), other.domain(), charge, first.dtype(), first.device());
    check_single(cat, first, first.qtotal(), "first tensor");
    check_single(cat, other, other.qtotal(), "second tensor");
    check_single(cat, out, charge, "product");

    // The blocks of the second tensor, grouped by the root its codomain fuses to.
    std::map<Sector, std::vector<std::pair<Sector, const at::Tensor *>>> rows;
    for (const auto &[key, block] : other.blocks())
      rows[key.domain].push_back({key.codomain, &block});

    for (const auto &[key, left] : first.blocks())
    {
      const auto found = rows.find(key.domain);
      if (found == rows.end()) continue;
      for (const auto &[domain_root, right] : found->second)
      {
        // The charges decide whether a pair of blocks can land in the result at all; the pairs that
        // cannot are zero, and a product of zero is what a fusion path that is not allowed carries.
        if (cat.n_symbol(key.codomain, cat.dual(domain_root), charge) == 0) continue;
        if (left.size(1) != right->size(0))
          throw std::logic_error("the two blocks over the roots " + render(key.domain) +
                                 " do not line up: " + std::to_string(left.size(1)) + " against " +
                                 std::to_string(right->size(0)));
        out.block(key.codomain, domain_root).addmm_(left, *right);
      }
    }
    return out;
  }

  Tensor contract(const Tensor &first,
                  const Tensor &other,
                  const std::vector<int64_t> &first_domain,
                  const std::vector<int64_t> &other_codomain)
  {
    check_layouts(first, other, first_domain, other_codomain);
    if (!(other.qtotal() == first.cat()->trivial()))
      throw std::invalid_argument("the second tensor of a contraction must carry the trivial charge");
    return contract(first, other, first_domain, other_codomain, first.qtotal());
  }

} // namespace sym
