#include "svd.h"

#include <algorithm>
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

    // One block of the tensor, its singular triples, and where its kept ones land on the bond.
    struct Factor
    {
      Sector codomain;
      Sector domain;
      int64_t copy = 0;
      at::Tensor u;
      at::Tensor values;
      at::Tensor vh;
      int64_t kept = 0;
      int64_t offset = 0;
    };

    // A singular value, with the block it came from and its rank inside that block.
    struct Candidate
    {
      double value;
      int64_t factor;
      int64_t index;
    };

  } // namespace

  static Split split_over(const Tensor &tensor,
                          std::optional<std::vector<int64_t>> left,
                          std::optional<int64_t> max_bond = std::nullopt,
                          double tol = 0.0,
                          int64_t absorb = 0)
  {
    if (absorb < 0 || absorb > 3)
      throw std::invalid_argument("absorb has to be 0, 1, 2 or 3, got " + std::to_string(absorb));
    if (max_bond.has_value() && *max_bond < 1)
      throw std::invalid_argument("a bond needs at least one multiplet, got " +
                                  std::to_string(*max_bond));

    const Tensor arranged = left ? tensor.repartition(*left) : tensor;
    const std::shared_ptr<const FusionCategory> cat = arranged.cat();
    std::vector<Factor> factors;
    for (const auto &[key, matrix] : arranged.blocks())
    {
      if (matrix.numel() == 0) continue;
      auto [u, values, vh] = at::linalg_svd(matrix, false);
      factors.push_back(Factor{key.codomain, key.domain, key.copy, u, values, vh});
    }

    // The bond is a single leg, so its multiplets are counted per sector over every block: rank all
    // the values, drop the ones at or below the tolerance, then cap the survivors from the outside.
    // The values of a block descend, so whatever survives of it is a leading run.
    // The values of every block are read back once, on the host: `item` per element would sync the
    // device once per element.
    std::vector<at::Tensor> host_values;
    host_values.reserve(factors.size());
    for (const Factor &factor : factors)
      host_values.push_back(factor.values.to(at::kCPU).to(at::kDouble).contiguous());
    std::vector<Candidate> candidates;
    for (int64_t at = 0; at < static_cast<int64_t>(factors.size()); ++at)
    {
      const double *values = host_values.at(static_cast<size_t>(at)).data_ptr<double>();
      for (int64_t index = 0; index < static_cast<int64_t>(host_values.at(static_cast<size_t>(at)).numel()); ++index)
        candidates.push_back(Candidate{values[index], at, index});
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate &left, const Candidate &right) { return left.value > right.value; });
    candidates.erase(std::find_if(candidates.begin(), candidates.end(),
                                  [tol](const Candidate &candidate) { return candidate.value <= tol; }),
                     candidates.end());
    if (max_bond.has_value() && static_cast<int64_t>(candidates.size()) > *max_bond)
      candidates.resize(static_cast<size_t>(*max_bond));
    for (const Candidate &candidate : candidates)
      factors.at(static_cast<size_t>(candidate.factor)).kept =
          std::max(factors.at(static_cast<size_t>(candidate.factor)).kept, candidate.index + 1);

    std::map<Sector, int64_t> multiplicities;
    for (const Factor &factor : factors)
    {
      if (factor.kept == 0) continue;
      multiplicities[factor.domain] += factor.kept;
    }

    // A leg is bent across a tensor by reading the block of the dual sector, so a sector and its
    // dual have to carry the same number of multiplets. Only the truncation can break the pairing,
    // so it is restored here: the values of a sector are ranked with the others, hence keeping the
    // leading `paired` of them keeps a leading run of every block that survives.
    std::map<Sector, int64_t> paired;
    for (const auto &[sector, count] : multiplicities)
    {
      const auto other = multiplicities.find(cat->dual(sector));
      if (other != multiplicities.end()) paired[sector] = std::min(count, other->second);
    }
    if (!paired.empty() && paired != multiplicities)
    {
      for (Factor &factor : factors) factor.kept = 0;
      std::map<Sector, int64_t> used;
      for (const Candidate &candidate : candidates)
      {
        Factor &factor = factors.at(static_cast<size_t>(candidate.factor));
        const auto room = paired.find(factor.domain);
        if (room == paired.end() || used[factor.domain] >= room->second) continue;
        ++used[factor.domain];
        factor.kept = std::max(factor.kept, candidate.index + 1);
      }
      multiplicities = std::move(paired);
    }
    // The offsets inside a block start over with every sector; the flat array of values follows the
    // bond, whose multiplets run through the sectors in order.
    std::map<Sector, int64_t> next;
    for (Factor &factor : factors)
    {
      factor.offset = next[factor.domain];
      next[factor.domain] += factor.kept;
    }
    std::map<Sector, int64_t> placed;
    int64_t run = 0;
    for (const auto &[sector, count] : multiplicities)
    {
      placed[sector] = run;
      run += count;
    }

    // A tensor whose values are all gone keeps the trivial sector, so the bond is still a leg.
    if (multiplicities.empty()) multiplicities.emplace(cat->trivial(), 1);
    const std::vector<std::pair<Sector, int64_t>> blocks(multiplicities.begin(), multiplicities.end());
    const Leg bond = Leg::from_blocks(cat, blocks).first;
    const at::TensorOptions options = at::TensorOptions().dtype(arranged.dtype()).device(arranged.device());

    Split out{
        Tensor(cat, arranged.codomain(), {bond}, arranged.qtotal(), arranged.dtype(), arranged.device()),
        at::zeros({bond.multiplets()}, options),
        Tensor(cat, {bond}, arranged.domain(), cat->trivial(), arranged.dtype(), arranged.device())};

    for (const Factor &factor : factors)
    {
      if (factor.kept == 0) continue;
      at::Tensor u = factor.u.narrow(1, 0, factor.kept);
      at::Tensor vh = factor.vh.narrow(0, 0, factor.kept);
      const at::Tensor values = factor.values.narrow(0, 0, factor.kept);
      // Splitting the values leaves each factor carrying half, so a single factor takes them whole
      // and either way the two factors multiply back to the tensor.
      if (absorb == 3)
      {
        u = u * values.sqrt();
        vh = values.sqrt().unsqueeze(1) * vh;
      }
      else if (absorb == 1)
      {
        u = u * values;
      }
      else if (absorb == 2)
      {
        vh = values.unsqueeze(1) * vh;
      }
      out.first.block(factor.codomain, factor.domain, factor.copy).narrow(1, factor.offset, factor.kept).copy_(u);
      out.second.block(factor.domain, factor.domain).narrow(0, factor.offset, factor.kept).copy_(vh);
      // The singular values are real; the bond array carries the tensor's scalar type, like every
      // block does, so the widening is written out rather than left to `copy_`.
      out.values.narrow(0, placed.at(factor.domain) + factor.offset, factor.kept)
          .copy_(values.to(arranged.dtype()));
    }
    return out;
  }

  at::Tensor svdvals(const Tensor &tensor) { return split_over(tensor, std::nullopt).values; }

  at::Tensor svdvals(const Tensor &tensor, const std::vector<int64_t> &left)
  {
    return split_over(tensor, left).values;
  }

  namespace
  {

    // One block of the tensor, its factors, and where the kept part lands on the bond.
    struct Piece
    {
      Sector codomain;
      Sector domain;
      int64_t copy = 0;
      at::Tensor left;
      at::Tensor right;
      int64_t kept = 0;
      int64_t offset = 0;
    };

    // Splits every block of the tensor and packs the factors along a bond that carries the domain
    // root of the block, so that first over the bond contracts with second back to the tensor.
    Decomposition split_blocks(const Tensor &tensor, const std::vector<Piece> &pieces)
    {
      const std::shared_ptr<const FusionCategory> cat = tensor.cat();
      std::map<Sector, int64_t> multiplicities;
      std::map<Sector, int64_t> next;
      std::vector<Piece> placed = pieces;
      for (Piece &piece : placed)
      {
        if (piece.kept == 0) continue;
        piece.offset = next[piece.domain];
        next[piece.domain] += piece.kept;
        multiplicities[piece.domain] += piece.kept;
      }
      if (multiplicities.empty()) multiplicities.emplace(cat->trivial(), 1);
      const std::vector<std::pair<Sector, int64_t>> blocks(multiplicities.begin(), multiplicities.end());
      const Leg bond = Leg::from_blocks(cat, blocks).first;
      Decomposition out{
          Tensor(cat, tensor.codomain(), {bond}, tensor.qtotal(), tensor.dtype(), tensor.device()),
          Tensor(cat, {bond}, tensor.domain(), cat->trivial(), tensor.dtype(), tensor.device())};
      for (const Piece &piece : placed)
      {
        if (piece.kept == 0) continue;
        out.first.block(piece.codomain, piece.domain, piece.copy).narrow(1, piece.offset, piece.kept).copy_(piece.left);
        out.second.block(piece.domain, piece.domain).narrow(0, piece.offset, piece.kept).copy_(piece.right);
      }
      return out;
    }

  } // namespace

  static Decomposition qr_over(const Tensor &tensor, std::optional<std::vector<int64_t>> left)
  {
    const Tensor arranged = left ? tensor.repartition(*left) : tensor;
    std::vector<Piece> pieces;
    for (const auto &[key, matrix] : arranged.blocks())
    {
      if (matrix.numel() == 0) continue;
      auto [q, r] = at::linalg_qr(matrix, "reduced");
      pieces.push_back(Piece{key.codomain, key.domain, key.copy, q, r, q.size(1)});
    }
    return split_blocks(arranged, pieces);
  }

  static Decomposition rq_over(const Tensor &tensor, std::optional<std::vector<int64_t>> left)
  {
    const Tensor arranged = left ? tensor.repartition(*left) : tensor;
    std::vector<Piece> pieces;
    for (const auto &[key, matrix] : arranged.blocks())
    {
      if (matrix.numel() == 0) continue;
      auto [q, r] = at::linalg_qr(matrix.transpose(0, 1), "reduced");
      pieces.push_back(Piece{key.codomain, key.domain, key.copy, r.transpose(0, 1), q.transpose(0, 1), q.size(1)});
    }
    return split_blocks(arranged, pieces);
  }

  Split svd(const Tensor &tensor, std::optional<int64_t> max_bond, double tol, int64_t absorb)
  {
    return split_over(tensor, std::nullopt, max_bond, tol, absorb);
  }

  Split svd(const Tensor &tensor,
            const std::vector<int64_t> &left,
            std::optional<int64_t> max_bond,
            double tol,
            int64_t absorb)
  {
    return split_over(tensor, left, max_bond, tol, absorb);
  }

  Decomposition qr(const Tensor &tensor) { return qr_over(tensor, std::nullopt); }

  Decomposition qr(const Tensor &tensor, const std::vector<int64_t> &left)
  {
    return qr_over(tensor, left);
  }

  Decomposition rq(const Tensor &tensor) { return rq_over(tensor, std::nullopt); }

  Decomposition rq(const Tensor &tensor, const std::vector<int64_t> &left)
  {
    return rq_over(tensor, left);
  }

} // namespace sym
