// The charge groups of the engine: their sectors, duality, fusion and quantum dimensions.
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
enum class BraidingStyle { bosonic = 0, fermionic = 10 };

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

class Group {
 public:
  Group(std::string type_name, FusionStyle fusion_style, BraidingStyle braiding_style,
        Sector trivial_sector, std::string group_name, double num_sectors);
  virtual ~Group() = default;

  // Whether the fusion of two sectors has a single outcome.
  bool is_abelian() const { return fusion_style_ == FusionStyle::single; }

  // Whether a fusion vertex can be realized in more than one way.
  bool carries_multiplicity() const { return fusion_style_ == FusionStyle::general; }

  const std::string& type_name() const { return type_name_; }
  const std::string& group_name() const { return group_name_; }
  double num_sectors() const { return num_sectors_; }
  int64_t sector_ind_len() const { return static_cast<int64_t>(trivial_sector_.size()); }
  const Sector& trivial_sector() const { return trivial_sector_; }
  FusionStyle fusion_style() const { return fusion_style_; }
  BraidingStyle braiding_style() const { return braiding_style_; }

  virtual bool is_valid_sector(const Sector& a) const = 0;
  virtual Sector dual_sector(const Sector& a) const = 0;
  virtual std::vector<Sector> fusion_outcomes(const Sector& a, const Sector& b) const = 0;
  virtual int64_t sector_dim(const Sector& a) const = 0;
  virtual bool is_same_symmetry(const Group& other) const = 0;

  // Fuse sectors into the single outcome of an abelian group.
  Sector fuse(const std::vector<Sector>& sectors) const;
  // Whether `a` and `b` can fuse to `c`.
  bool can_fuse_to(const Sector& a, const Sector& b, const Sector& c) const;
  // Multiplicity of the fusion of `a` and `b` to `c`.
  virtual int64_t n_symbol(const Sector& a, const Sector& b, const Sector& c) const;
  // The structural tensor of one fusion vertex, one block per channel.  `copy` selects the
  // coupling when the vertex couples more than once.
  virtual at::Tensor fusion_tensor(const Sector& a, const Sector& b, const Sector& c,
                                   const c10::Device& device, int64_t copy = 0) const;
  // Every sector of a group that enumerates them, in order.
  virtual std::vector<Sector> all_sectors() const;
  // The sector of the multiplet of dimension `dim`, for a group whose multiplets are told
  // apart by their dimension alone.
  virtual Sector sector_of_dimension(int64_t dim) const;
  // The sign that conjugating a leg in this sector costs; it is the Frobenius-Schur indicator.
  virtual double dual_sign(const Sector& sector) const;
  virtual double dual_phase(const Sector& sector) const;
  // The conjugate basis of a sector against its own basis: entry (row, column) maps the
  // column index of the leg onto the row index of the leg that stores its dual.
  virtual at::Tensor duality_matrix(const Sector& a, const c10::Device& device) const;
  virtual at::Tensor invariant_form(const Sector& a, const c10::Device& device) const;
  /// The component of a coupled multiplet the reduced blocks are stored at.
  virtual int64_t reference_index(const Sector& a) const;
  /// The dtype of the fusion tensors of this group.
  virtual c10::ScalarType fusion_dtype() const;

 private:
  FusionStyle fusion_style_;
  BraidingStyle braiding_style_;
  Sector trivial_sector_;
  std::string group_name_;
  std::string type_name_;
  double num_sectors_;
};

class AbelianGroup : public Group {
 public:
  AbelianGroup(std::string type_name, Sector trivial_sector, std::string group_name,
               double num_sectors, BraidingStyle braiding_style = BraidingStyle::bosonic);

  // Every sector of an abelian group has quantum dimension one.
  int64_t sector_dim(const Sector& a) const override;

  // The fusion of an abelian group has one outcome, so its Clebsch-Gordan vertex is one
  // whenever the sectors fuse and does not exist otherwise.
  at::Tensor fusion_tensor(const Sector& a, const Sector& b, const Sector& c,
                           const c10::Device& device, int64_t copy = 0) const override;
  c10::ScalarType fusion_dtype() const override;

  // A one-dimensional sector needs no basis conversion, and its invariant form is the
  // one-by-one identity: the abelian vertex already carries the pairing.
  at::Tensor duality_matrix(const Sector& a, const c10::Device& device) const override;
  at::Tensor invariant_form(const Sector& a, const c10::Device& device) const override;
};

const std::shared_ptr<Group>& no_symmetry();
const std::shared_ptr<Group>& u1_symmetry();
const std::shared_ptr<Group>& z_n_symmetry(int64_t modulus);
const std::shared_ptr<Group>& su2_symmetry();
const std::shared_ptr<Group>& fermion_parity();

// Bind the enums and the base group.
void bind_group(py::module_& m);

}  // namespace sym
