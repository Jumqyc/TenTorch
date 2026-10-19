// The fusion categories of the engine: the labels, their fusion, the associator and the braiding
// that recouple them, and the dense realization of one vertex.  The tensor engine includes this
// header only, so what a concrete symmetry is stays behind the interface.
#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace sym {

using Sector = std::vector<int64_t>;

// How many outcomes the fusion of two sectors has.
enum class FusionStyle { single = 0, multiple_unique = 10, general = 20 };

// How sectors behave when two legs carrying them are exchanged.
enum class BraidingStyle { bosonic = 0, fermionic = 10, anyonic = 20 };

// Raised where the Python API raises ValueError.
class value_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Raised where the Python API raises NotImplementedError.
class not_implemented_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// One integer naming the device a cached tensor lives on.
inline int64_t device_code(const c10::Device& device) {
  return static_cast<int64_t>(device.type()) * 1000 + device.index();
}

// The sector as a Python tuple, so it stays a hashable key on the Python side.
inline py::tuple to_python(const Sector& sector) {
  py::tuple out(sector.size());
  for (size_t i = 0; i < sector.size(); ++i) {
    out[i] = py::int_(sector[i]);
  }
  return out;
}

// The sectors as a tuple of Python tuples.
inline py::tuple to_python(const std::vector<Sector>& sectors) {
  py::tuple out(sectors.size());
  for (size_t i = 0; i < sectors.size(); ++i) {
    out[i] = to_python(sectors[i]);
  }
  return out;
}

class FusionCategory {
 public:
  FusionCategory(std::string type_name, FusionStyle fusion_style, BraidingStyle braiding_style,
                 Sector trivial_sector, std::string category_name);
  virtual ~FusionCategory() = default;

  // Whether the fusion of two sectors has a single outcome.
  bool is_abelian() const { return fusion_style_ == FusionStyle::single; }

  // Whether a fusion vertex can be realized in more than one way.
  bool carries_multiplicity() const { return fusion_style_ == FusionStyle::general; }

  // Whether this is the trivial category: one label, and no charge.
  virtual bool is_trivial() const { return false; }

  const std::string& type_name() const { return type_name_; }
  const std::string& name() const { return name_; }
  const Sector& trivial_sector() const { return trivial_sector_; }
  FusionStyle fusion_style() const { return fusion_style_; }
  BraidingStyle braiding_style() const { return braiding_style_; }

  virtual bool is_valid_sector(const Sector& a) const = 0;
  virtual Sector dual_sector(const Sector& a) const = 0;
  virtual std::vector<Sector> fusion_outcomes(const Sector& a, const Sector& b) const = 0;
  virtual int64_t sector_dim(const Sector& a) const = 0;
  virtual bool is_same_symmetry(const FusionCategory& other) const = 0;

  // The quantum dimension of a sector as a real number: `sector_dim` is the integer extent the
  // dense code needs, while the dimension that weighs a recoupling is real.
  virtual double quantum_dim(const Sector& a) const;

  // Fuse sectors into the single outcome of an abelian category.
  Sector fuse(const std::vector<Sector>& sectors) const;
  // Whether `a` and `b` can fuse to `c`.
  bool can_fuse_to(const Sector& a, const Sector& b, const Sector& c) const;
  // Multiplicity of the fusion of `a` and `b` to `c`.
  virtual int64_t n_symbol(const Sector& a, const Sector& b, const Sector& c) const;
  // The dense realization of one fusion vertex, one block per channel.  `copy` selects the
  // coupling when the vertex couples more than once.
  virtual at::Tensor fusion_tensor(const Sector& a, const Sector& b, const Sector& c,
                                   const c10::Device& device, int64_t copy = 0) const;
  // The associator: the amplitudes that rewrite the coupling ((a b) c) -> d into (a (b c)) -> d
  // through the channels e = (a b) and f = (b c).  The result carries one amplitude per
  // combination of the multiplicities, in the shape (N^e_ab, N^d_ec, N^f_bc, N^d_af), so that a
  // graph contraction reads the amplitude of every copy combination on its own; it is an empty
  // tensor when the move is impossible.
  virtual at::Tensor associator(const Sector& a, const Sector& b, const Sector& c, const Sector& d,
                                const Sector& e, const Sector& f,
                                const c10::Device& device) const;
  // The braiding: what exchanging a and b in one vertex costs, one amplitude per copy of a b -> c
  // against every copy of b a -> c, in the shape (N^c_ab, N^c_ba).
  virtual at::Tensor braiding(const Sector& a, const Sector& b, const Sector& c,
                              const c10::Device& device) const;
  // The sign that conjugating a leg in this sector costs; it is the Frobenius-Schur indicator.
  virtual double dual_sign(const Sector& sector) const;
  virtual double dual_phase(const Sector& sector) const;
  // The conjugate basis of a sector against its own basis: entry (row, column) maps the
  // column index of the leg onto the row index of the leg that stores its dual.
  virtual at::Tensor duality_matrix(const Sector& a, const c10::Device& device) const;
  // The conjugate basis of a sector paired with its own: the fully contracted one-leg tree,
  // which is how an inner product is read without a Clebsch-Gordan tensor.
  virtual at::Tensor invariant_form(const Sector& a, const c10::Device& device) const;

  // Drop whatever memo tables the category keeps; a category with none need not override it.
  virtual void clear_caches() const {}
  /// The component of a coupled multiplet the reduced blocks are stored at.
  virtual int64_t reference_index(const Sector& a) const;
  /// The dtype of the fusion tensors of this category.
  virtual c10::ScalarType fusion_dtype() const;

 private:
  FusionStyle fusion_style_;
  BraidingStyle braiding_style_;
  Sector trivial_sector_;
  std::string name_;
  std::string type_name_;
};

// The trivial category: one label, and no charge.
const std::shared_ptr<FusionCategory>& no_symmetry();

// Bind the enums and the base category.
void bind_category(py::module_& m);

}  // namespace sym
