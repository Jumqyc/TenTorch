// The group layer of the category interface: the associator and the braiding are the overlaps of
// the Clebsch-Gordan tensors, so a group supplies neither of them.
#include "group.h"

#include <pybind11/stl.h>

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
    return cg_cache_.emplace(key, compute_clebsch_gordan(a, b, c, copy)).first->second;
  }

  void Group::clear_cache() const
  {
    cg_cache_.clear();
    FusionCategory::clear_cache();
  }

  at::Tensor Group::F_symbol(const Sector &a, const Sector &b, const Sector &c,
                             const Sector &d) const
  {
    // The left tree ((a b) c) -> d, one dense vector per pair of vertex copies.
    std::vector<at::Tensor> left;
    for (const Outcome &e : outcomes(a, b))
    {
      for (int64_t outer = 0; outer < e.multiplicity; ++outer)
      {
        for (int64_t inner = 0; inner < n_symbol(e.sector, c, d); ++inner)
        {
          left.push_back(
            at::einsum("abe,ecd->abcd",
              {clebsch_gordan(a, b, e.sector, outer),
              clebsch_gordan(e.sector, c, d, inner)}));
        }
      }
    }
    // The right tree (a (b c)) -> d, the same way through f = (b c).
    std::vector<at::Tensor> right;
    for (const Outcome &f : outcomes(b, c))
    {
      for (int64_t outer = 0; outer < f.multiplicity; ++outer)
      {
        for (int64_t inner = 0; inner < n_symbol(a, f.sector, d); ++inner)
        {
          right.push_back(
            at::einsum("bce,aed->abcd",
                      {clebsch_gordan(b, c, f.sector, outer),
                      clebsch_gordan(a, f.sector, d, inner)}));
        }
      }
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
    const std::vector<Outcome> &fused = outcomes(b, c);
    int64_t size = 0;
    for (const Outcome &e : fused)
    {
      size += e.multiplicity;
    }
    at::Tensor value = at::zeros({size, size}, at::TensorOptions().dtype(dtype()).device(device()));
    int64_t offset = 0;
    for (const Outcome &e : fused)
    {
      const int64_t copies = e.multiplicity;
      // The exchange only exists where both orders fuse to the same sectors with the same
      // multiplicity, which is what makes every block square.
      TORCH_CHECK(n_symbol(c, b, e.sector) == copies,
                  "R_symbol needs N^e_bc == N^e_cb for every outcome e of b (x) c");
      // One vertex per copy, straight and exchanged; the coupled sector fixes their shape.
      std::vector<at::Tensor> straight;
      std::vector<at::Tensor> swapped;
      for (int64_t copy = 0; copy < copies; ++copy)
      {
        straight.push_back(clebsch_gordan(b, c, e.sector, copy));
        // The exchanged vertex is stored with its own axis order, so bring it back to (b, c, e).
        swapped.push_back(clebsch_gordan(c, b, e.sector, copy).permute({1, 0, 2}));
      }
      // Each vertex is an isometry into its coupled sector, so the block carries that q_dim once.
      const at::Tensor block = at::matmul(at::stack(straight, 0).flatten(1),
                                          at::stack(swapped, 0).conj().flatten(1).transpose(0, 1)) /
                               q_dim(e.sector);
      value.slice(0, offset, offset + copies)
          .slice(1, offset, offset + copies)
          .copy_(block.reshape({copies, copies}));
      offset += copies;
    }
    return value;
  }

  double Group::q_dim(Sector a) const { return static_cast<double>(dim(a)); }

  void bind_group(py::module_ &m)
  {
    // The group adds its vertex to the category API; the rest of its surface is the category's.
    py::class_<Group, FusionCategory, std::shared_ptr<Group>>(m, "Group")
      .def("clebsch_gordan", &Group::clebsch_gordan)
      .def("clear_cache", &Group::clear_cache);
  }

} // namespace sym
