// The block-sparse tensor of the engine: a graded space, the map over it, and the contraction.
#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <torch/extension.h>

#include "../symmetry/cat/cat.h"
#include "../symmetry/tree/tree.h"

#include <map>
#include <memory>
#include <vector>

#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <torch/extension.h>


namespace sym::engine {

// One charge block of a leg, as the tree enumeration sees it: the sector it carries and whether the
// side stores that sector dualised.
std::vector<LegCharge> leg_charges(const Leg& leg);

// A leg from its summands, each carrying the multiplicity it appears with, and the direction it
// points: the slices are the running sum of the summand dimensions, so the leg's index is the
// concatenation of its summands.
Leg leg_of(std::shared_ptr<FusionCategory> group,
           const std::vector<std::pair<Sector, int64_t>>& summands, int64_t qconj);

// The two sides of a tensor: the legs it comes out of and the legs it goes into.
struct Space {
  std::shared_ptr<FusionCategory> group;
  std::vector<Leg> codomain;
  std::vector<Leg> domain;

  std::shared_ptr<FusionCategory> category() const { return group; }
  int64_t size() const { return static_cast<int64_t>(codomain.size() + domain.size()); }
  bool operator==(const Space& other) const;
};

// A tensor: one matrix per coupled charge, whose rows and columns are the tree pairs of that charge
// in the canonical enumeration of ``tree_pairs``.
struct TensorMap {
  Space space;
  // The charge the tensor as a whole carries: only the block of this charge is filled, and a tensor
  // with a trivial total charge is the one a closed diagram can weigh.
  Sector qtotal;
  std::map<Sector, at::Tensor> blocks;

  std::vector<LegCharge> codomain_charges() const;
  std::vector<LegCharge> domain_charges() const;
  // The tree pairs of one block, in row-major order of that block.
  std::vector<FusionTreePair> pairs(const Sector& coupled) const;
};

// The component array of one tree: the contraction of one fusion vertex per coupling, divided by
// the square root of the product of the quantum dimensions it fuses, so that the tree basis is
// orthonormal. A leg that carries duals stores its index in the conjugate basis, so it is dressed
// by the duality matrix.
at::Tensor tree_tensor(const FusionCategory& group, const FusionTree& tree,
                       const c10::Device& device, bool dress = false);

// The legs of a tensor, rearranged so that the codomain carries `keep` of them: a leg that crosses
// changes its direction, and both sides are re-expressed in each other's tree basis.
// Args:
//   tensor: a tensor map.
//   keep: the number of legs the codomain carries afterwards.
//   device: where the associator is read from.
// Returns:
//   The same map over rearranged legs, its blocks rebuilt in the new tree basis.
// Raises:
//   std::out_of_range: if `keep` does not fit the legs of the tensor.
TensorMap repartition_tensor(const TensorMap& tensor, int64_t keep, const c10::Device& device);

// The rows (or columns) one side of a block spans: every fusion tree of that charge, and for each
// of its legs the multiplicity index the row picks along that leg.
struct SideLayout {
  // The trees of the side, in the order the block enumerates them.
  std::vector<FusionTree> trees;
  // base.at(i) is the first row of trees.at(i); base.back() is the number of rows.
  std::vector<int64_t> base;
  // mults.at(i).at(leg) is the multiplicity of that leg along the i-th tree.
  std::vector<std::vector<int64_t>> mults;

  int64_t rows() const { return base.empty() ? 0 : base.back(); }
};

// The layout of one side of a block, given the legs of that side and the trees of one charge.
SideLayout side_layout(const std::vector<Leg>& legs, const std::vector<FusionTree>& trees);

// The tree index of a row and the multiplicity index that row picks along every leg.
std::pair<int64_t, std::vector<int64_t>> row_parts(const SideLayout& layout, int64_t row);

// The row that carries the given tree and multiplicity indices.
int64_t row_index(const SideLayout& layout, int64_t tree, const std::vector<int64_t>& mults);

// The component array of a tensor whose legs all point out of it: the sum of its block entries times
// the tree tensors. This is the only place a Clebsch-Gordan tensor is needed, and it is what the
// checks compare against.

// Bind the engine's tree and layout entry points for a check to compare against the legacy route.
void bind_engine(py::module_& m);

}  // namespace sym::engine

#include <vector>


namespace sym::engine {

// Move the legs of a tensor into a new arrangement: the first ``keep`` legs of ``order`` become the
// codomain, the rest the domain, and ``order`` names the legs of the tensor counting from one, its
// codomain legs first. A leg that changes side changes its direction and its tree carries the dual.
//
// A leg move recouples the trees of a block, so an entry of the source block is spread over the
// destination blocks of the charges the new root couples to; the result is therefore a fresh tensor
// whose blocks are accumulated from every source entry.
//
// Args:
//   tensor: the tensor to move, its legs numbered codomain first.
//   keep: how many of the first legs of ``order`` stay on the codomain.
//   order: a permutation of the leg numbers of ``tensor``.
//   device: where the category symbols are read from.
// Returns:
//   The tensor with its legs in the requested arrangement.
TensorMap permute_tensor(const TensorMap& tensor, int64_t keep, const std::vector<int64_t>& order,
                         const c10::Device& device);

// The same map written over any arrangement of its legs: each leg may cross to the other side, so
// `keep` names how many of them end up in the codomain and `order` lists the legs in their wanted
// order. The blocks are rebuilt in the tree basis of the new sides.
// Args:
//   tensor: a tensor map.
//   keep: the number of legs the codomain carries afterwards.
//   order: the legs in their wanted order, as one-based positions in ``tensor``.
//   device: where the category symbols are read from.
// Returns:
//   The same map over rearranged legs.
// Raises:
//   std::invalid_argument: if `keep` or `order` does not describe an arrangement of the legs.
TensorMap arrange_tensor(const TensorMap& tensor, int64_t keep, const std::vector<int64_t>& order,
                         const c10::Device& device);

// Contract two tensors whose middle legs meet: every codomain leg of ``right`` is glued to a domain
// leg of ``left``, so ``left`` carries the free codomain legs and ``right`` the free domain legs.
// The tree bases are orthonormal, so the contraction of one charge is the matrix product of the two
// blocks and needs no further weight.
//
// Args:
//   left: the tensor whose codomain legs survive on the left of the result.
//   right: the tensor whose domain legs survive on the right of the result, with as many codomain
//     legs as ``left`` has domain legs.
// Returns:
//   The contracted tensor, one block per charge the two operands share.
TensorMap contract(const TensorMap& left, int64_t left_glue, const TensorMap& right,
                   int64_t right_glue, const c10::Device& device);

// The two directions between a tensor and the dictionary Python hands it around as: its legs per
// side, each a list of (sector, multiplicity) summands plus the direction the leg points, and one
// block per charge.
TensorMap tensor_of(const std::shared_ptr<FusionCategory>& group, const py::dict& data);
py::dict tensor_as(const TensorMap& tensor);

// The device a Python argument names, defaulting to the project device.
c10::Device device_of(const py::object& value);

// Bind the moves above for a check to compare them with the legacy route.
void bind_contract(py::module_& m);

}  // namespace sym::engine

#ifndef MPSDMRG_ENGINE_ALGEBRA_H
#define MPSDMRG_ENGINE_ALGEBRA_H
/**
 * The pointwise and trace constructions of a block-sparse tensor of the engine.
 *
 * ``conj`` conjugates the coefficients, ``adjoint`` also turns the tensor around, and ``norm`` and
 * ``inner`` reduce a tensor to the scalar its diagram evaluates to. Every block entry is a
 * coefficient in the orthonormal tree basis, so these sums need no further weight.
 */

namespace sym::engine {

/**
 * The complex conjugate of every coefficient, the legs unchanged.
 * Args:
 *   tensor: the tensor to conjugate.
 * Returns:
 *   The tensor with every block conjugated.
 */
TensorMap conj(const TensorMap& tensor);

/**
 * The adjoint: every block conjugated and transposed, the two sides of the space exchanged.
 * Args:
 *   tensor: the tensor to turn around.
 * Returns:
 *   The tensor with the domain as its codomain and the codomain as its domain.
 */
TensorMap adjoint(const TensorMap& tensor);

/**
 * The Frobenius norm of a tensor, summed over every coefficient the tree basis carries.
 * Args:
 *   tensor: the tensor to measure.
 * Returns:
 *   The square root of the sum of the squared moduli of its components.
 */
double norm(const TensorMap& tensor);

/**
 * The inner product of two tensors over the same space, the second one conjugated.
 * Args:
 *   left: the tensor whose coefficients are not conjugated.
 *   right: the tensor that is conjugated.
 * Returns:
 *   The sum of ``left * conj(right)`` over every component of the space.
 * Raises:
 *   std::invalid_argument: if the two tensors do not cover the same space.
 */
std::complex<double> inner(const TensorMap& left, const TensorMap& right);

// Closes the legs pairwise and returns the scalar they leave: a map is traced over its own
// codomain and domain legs, and a diagram whose legs all point in is closed over the two bond legs
// of every tree, keeping the entries whose indices agree.
std::complex<double> trace(const TensorMap& tensor);

/// Expose the algebra of the engine to Python.
void bind_algebra(py::module_& m);

}  // namespace sym::engine

#endif

#include <optional>


namespace sym::engine {

// A categorical tensor: a map between graded spaces, stored as one matrix per coupled charge over
// the canonical tree basis of that charge. A tensor may also carry no graded structure at all, in
// which case it holds one plain array and no categorical operation applies to it.
class Tensor {
 public:
  Tensor() = default;
  explicit Tensor(TensorMap map) : map_(std::move(map)) {}
  explicit Tensor(at::Tensor plain) : plain_(std::move(plain)) {}

  bool graded() const { return map_.has_value(); }

  // The bare array of a tensor that carries no category.
  const at::Tensor& plain() const {
    if (map_) throw std::logic_error("this tensor is graded, it has no bare array");
    return plain_;
  }
  const TensorMap& map() const;
  TensorMap& map();

  // Build every block the legs allow: a tensor with a domain sums over the coupled charges, one
  // that is only a ket carries the single charge it is labelled with. kind: 0 zero, 1 one, 2 random.
  static Tensor filled(const std::vector<Leg>& codomain, const std::vector<Leg>& domain,
                       const Sector& qtotal, int kind, at::ScalarType dtype,
                       const c10::Device& device);

  // A dense array with graded legs of dimension one per summand, which is the only case where the
  // component array and the blocks are the same data. A symmetry-adapted leg needs a projection and
  // is left to the checks.
  static Tensor from_dense(const at::Tensor& data, std::optional<std::vector<Leg>> legs,
                           std::optional<Sector> qtotal);

  std::shared_ptr<FusionCategory> group() const;
  std::vector<Leg> legs() const;
  Sector qtotal() const;
  std::vector<at::Tensor> blocks() const;
  std::vector<Sector> charges() const;
  std::vector<int64_t> shape() const;
  at::ScalarType dtype() const;
  c10::Device device() const;
  bool is_dense() const;

  // The row and column one entry occupies in the component array of a block: `blocks` names, per
  // leg, the block that leg carries and `members` the member inside it, and the block is the one
  // labelled by `charge`. Throws std::invalid_argument when no tree spans those blocks or the
  // charge is no coupling of the side.
  std::pair<int64_t, int64_t> slot(const Sector& charge, const std::vector<int64_t>& blocks,
                                   const std::vector<int64_t>& members) const;

  std::complex<double> item() const;
  Tensor clone() const;
  Tensor detach() const;
  Tensor conj() const;
  // The adjoint: every block conjugated and transposed, and the two sides of the space exchanged, so
  // that a bra is the dual object of the ket it was built from and not merely its complex conjugate.
  Tensor adjoint() const;
  Tensor to(const py::kwargs& options) const;

  // Fuse the given groups of legs into one leg each; the groups are lists of leg indices.
  Tensor combine_legs(const std::vector<std::vector<int64_t>>& groups) const;

  // Sum of every block entry times its tree tensor: the component array of a tensor whose legs all
  // point out of it. This needs the Clebsch-Gordan tensor and exists for the checks only.

  Tensor& operator+=(const Tensor& other);
  Tensor& operator-=(const Tensor& other);
  Tensor& operator*=(const std::complex<double>& scalar);
  Tensor& operator/=(const std::complex<double>& scalar);
  Tensor operator-() const;

 private:
  std::optional<TensorMap> map_;
  at::Tensor plain_;
};

// Contract the operands in the order the labels give, into the legs the output names. Every label
// of every operand appears either once in the output or twice among the operands.
Tensor einsum(const std::string& spec, const std::vector<Tensor>& operands);

void bind_tensor(py::module_& m);

}  // namespace sym::engine
