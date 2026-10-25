// The group layer of the category interface: the associator and the braiding are the overlaps of
// the Clebsch-Gordan tensors, so a group supplies neither of them.
#include "group.h"

#include <cstdint>
#include <vector>

namespace sym
{

  at::Tensor Group::clebsch_gordan(const Sector &a, const Sector &b, const Sector &c,
                                   int64_t copy) const
  {
    const std::tuple<Sector, Sector, Sector, int64_t> key{a, b, c, copy};
    const auto found = cg_cache_.find(key);
    if (found != cg_cache_.end())
    {
      return found->second;
    }
    // Every vertex is stored in the category's scalar type, so a product with a block never mixes
    // dtypes: `matmul` does not promote.
    return cg_cache_.emplace(key, compute_clebsch_gordan(a, b, c, copy).to(dtype())).first->second;
  }

  void Group::clear_cache() const
  {
    cg_cache_.clear();
    FusionCategory::clear_cache();
  }

  at::Tensor Group::F_symbol(const Sector &a, const Sector &b, const Sector &c,
                             const Sector &d) const
  {
    // The left tree ((a b) c) -> d, one dense vector per row.
    std::vector<at::Tensor> left;
    for (const FIndex &row : f_rows(a, b, c, d))
    {
      left.push_back(
        at::einsum("abe,ecd->abcd",
                   {clebsch_gordan(a, b, row.sector, row.inner),
                    clebsch_gordan(row.sector, c, d, row.outer)}));
    }
    // The right tree (a (b c)) -> d, the same way through f = (b c).
    std::vector<at::Tensor> right;
    for (const FIndex &column : f_columns(a, b, c, d))
    {
      right.push_back(
        at::einsum("bce,aed->abcd",
                   {clebsch_gordan(b, c, column.sector, column.inner),
                    clebsch_gordan(a, column.sector, d, column.outer)}));
    }
    const int64_t rows = static_cast<int64_t>(left.size());
    const int64_t cols = static_cast<int64_t>(right.size());
    if (rows == 0 || cols == 0)
    {
      return at::zeros({rows, cols}, at::TensorOptions().dtype(dtype()).device(device()));
    }
    // Each tree is an isometry into d, so the overlap carries q_dim(d) once.
    const at::Tensor overlap = at::matmul(at::stack(left, 0).flatten(1),
                                          at::stack(right, 0).conj().flatten(1).transpose(0, 1));
    return (overlap / q_dim(d)).reshape({rows, cols});
  }

  at::Tensor Group::R_symbol(const Sector &b, const Sector &c) const
  {
    const std::vector<RIndex> rows = r_rows(b, c);
    const int64_t size = static_cast<int64_t>(rows.size());
    at::Tensor value = at::zeros({size, size}, at::TensorOptions().dtype(dtype()).device(device()));
    for (int64_t start = 0; start < size;)
    {
      // One block per outcome, so an outcome owns a stretch of consecutive rows.
      const Sector &e = rows.at(static_cast<size_t>(start)).sector;
      int64_t end = start;
      while (end < size && rows.at(static_cast<size_t>(end)).sector == e) ++end;
      const int64_t copies = end - start;
      // The exchange only exists where both orders fuse to the same sectors with the same
      // multiplicity, which is what makes every block square.
      TORCH_CHECK(n_symbol(c, b, e) == copies,
                  "R_symbol needs N^e_bc == N^e_cb for every outcome e of b (x) c");
      // One vertex per copy, straight and exchanged; the coupled sector fixes their shape.
      std::vector<at::Tensor> straight;
      std::vector<at::Tensor> swapped;
      for (int64_t copy = 0; copy < copies; ++copy)
      {
        straight.push_back(clebsch_gordan(b, c, e, copy));
        // The exchanged vertex is stored with its own axis order, so bring it back to (b, c, e).
        swapped.push_back(clebsch_gordan(c, b, e, copy).permute({1, 0, 2}));
      }
      // Each vertex is an isometry into its coupled sector, so the block carries that q_dim once.
      const at::Tensor block = at::matmul(at::stack(straight, 0).flatten(1),
                                          at::stack(swapped, 0).conj().flatten(1).transpose(0, 1)) /
                               q_dim(e);
      value.slice(0, start, end).slice(1, start, end).copy_(block.reshape({copies, copies}));
      start = end;
    }
    return value;
  }

} // namespace sym
