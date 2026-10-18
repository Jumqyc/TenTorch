// The group subclass of the category interface: its Clebsch-Gordan calculus realizes the
// associator and the braiding, so no other categorical data has to be supplied.
#include "group.h"

#include <utility>
#include <vector>

namespace sym {

Group::Group(std::string type_name, FusionStyle fusion_style, BraidingStyle braiding_style,
             Sector trivial_sector, std::string name)
    : FusionCategory(std::move(type_name), fusion_style, braiding_style, std::move(trivial_sector),
                     std::move(name)) {}

at::Tensor Group::associator(const Sector& a, const Sector& b, const Sector& c, const Sector& d,
                             const Sector& e, const Sector& f, const c10::Device& device) const {
  // One amplitude per combination of the multiplicities of the four vertices that meet in the
  // move: ((a b) c) -> d through e = (a b), against (a (b c)) -> d through f = (b c).  The result
  // is shaped (N^e_ab, N^d_ec, N^f_bc, N^d_af).
  const int64_t copies_ab = n_symbol(a, b, e);
  const int64_t copies_ec = n_symbol(e, c, d);
  const int64_t copies_bc = n_symbol(b, c, f);
  const int64_t copies_af = n_symbol(a, f, d);
  const auto options = at::TensorOptions().dtype(fusion_dtype()).device(device);
  if (copies_ab == 0 || copies_ec == 0 || copies_bc == 0 || copies_af == 0) {
    return at::zeros({copies_ab, copies_ec, copies_bc, copies_af}, options);
  }
  // The two orders of one three-sector tree, each expanded into the dense basis (a, b, c, d).
  std::vector<at::Tensor> left;
  for (int64_t one = 0; one < copies_ab; ++one) {
    for (int64_t two = 0; two < copies_ec; ++two) {
      left.push_back(at::einsum("abe,ecd->abcd", {fusion_tensor(a, b, e, device, one),
                                                  fusion_tensor(e, c, d, device, two)}));
    }
  }
  std::vector<at::Tensor> right;
  for (int64_t one = 0; one < copies_bc; ++one) {
    for (int64_t two = 0; two < copies_af; ++two) {
      right.push_back(at::einsum("bce,aed->abcd", {fusion_tensor(b, c, f, device, one),
                                                   fusion_tensor(a, f, d, device, two)}));
    }
  }
  // The dense tree of the engine is normalized to the quantum dimension of its coupling, so the
  // overlap of two trees carries that dimension once.
  const at::Tensor overlap = at::matmul(at::stack(left, 0).flatten(1),
                                        at::stack(right, 0).conj().flatten(1).transpose(0, 1));
  return (overlap / quantum_dim(d)).reshape({copies_ab, copies_ec, copies_bc, copies_af});
}

at::Tensor Group::braiding(const Sector& a, const Sector& b, const Sector& c,
                           const c10::Device& device) const {
  const auto options = at::TensorOptions().dtype(fusion_dtype()).device(device);
  if (!can_fuse_to(a, b, c) || !can_fuse_to(b, a, c)) {
    return at::zeros({n_symbol(a, b, c), n_symbol(b, a, c)}, options);
  }
  std::vector<at::Tensor> straight;
  for (int64_t copy = 0; copy < n_symbol(a, b, c); ++copy) {
    straight.push_back(fusion_tensor(a, b, c, device, copy));
  }
  std::vector<at::Tensor> swapped;
  for (int64_t copy = 0; copy < n_symbol(b, a, c); ++copy) {
    // the exchanged vertex is stored with its own axis order, so bring it back to (a, b, c)
    swapped.push_back(fusion_tensor(b, a, c, device, copy).permute({1, 0, 2}));
  }
  // The dense tree of the engine is normalized to the quantum dimension of its coupling, as in
  // the associator.
  return at::matmul(at::stack(straight, 0).flatten(1),
                    at::stack(swapped, 0).conj().flatten(1).transpose(0, 1)) / quantum_dim(c);
}

void bind_group(py::module_& m) {
  // The group carries its own associator and braiding; the rest of its API is the category's.
  py::class_<sym::Group, sym::FusionCategory, std::shared_ptr<sym::Group>>(m, "Group");
}

}  // namespace sym
