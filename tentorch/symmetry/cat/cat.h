// The fusion categories of the engine: the labels, their fusion, the F symbol that recouples a
// fusion tree and the R symbol that exchanges two neighbouring labels.  The tensor engine includes
// this header only, so what a concrete symmetry is stays behind the interface.
#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace sym
{

  /**
   * @brief One simple object of the category, as the integers that name it in the basis the concrete
   * category chooses.
   * @details SU(2) uses one entry, the twice-spin $2j$; SU(3) uses two, the Dynkin labels $(p, q)$;
   * $Z_N$ uses one residue in $[0, N)$.  Two sectors are the same object exactly when the vectors
   * are equal, so the vector doubles as the key of a block.
   */
  using Sector = std::vector<int64_t>;

  /**
   * @brief One term of a fusion: the sector the pair fuses to, and how many distinct vertices
   * couple the pair into it.
   * @details The multiplicity belongs to the triple, not to the sector.  The SU(3) octet occurs
   * twice in $8 \otimes 8$ but once in $8 \otimes 10$, so no integer stored on the sector itself can
   * serve as its multiplicity.
   */
  struct Outcome
  {
    Sector sector;
    int multiplicity;
  };

  // One integer naming the device a cached tensor lives on.
  inline int64_t device_code(const c10::Device &device)
  {
    return static_cast<int64_t>(device.type()) * 1000 + device.index();
  }

  // The sector as a Python tuple, so it stays a hashable key on the Python side.
  inline py::tuple to_python(const Sector &sector)
  {
    py::tuple out(sector.size());
    for (size_t i = 0; i < sector.size(); ++i)
    {
      out[i] = py::int_(sector[i]);
    }
    return out;
  }

  // The sectors as a tuple of Python tuples.
  inline py::tuple to_python(const std::vector<Sector> &sectors)
  {
    py::tuple out(sectors.size());
    for (size_t i = 0; i < sectors.size(); ++i)
    {
      out[i] = to_python(sectors[i]);
    }
    return out;
  }

  /**
   * @brief The device a category materialises its symbols on when it is not told otherwise.
   * @return The CUDA device when the build can use one, and the CPU otherwise.
   */
  c10::Device default_device();

  /**
   * @brief A fusion category: the labels, how they fuse, and the two symbols that recouple them.
   * @details The conventions below are the contract between a category and the engine, because
   * every tensor of the engine is stored in the tree basis this interface defines.
   *
   * A block of a tensor carries one sector per leg and one multiplicity index per vertex of its
   * fusion tree.  The entry of a block is the coefficient of that basis vector: each vertex tensor
   * is normalised to unit Frobenius norm, and a tree is built by contracting them, so the tree basis
   * is orthonormal.  Three consequences the engine relies on:
   *
   * - The block of a tensor is the list of coefficients, not a matrix of overlap integrals.
   * - `F_symbol` and `R_symbol` are unitary matrices, so a move is a lossless basis change.
   * - A move multiplies by a matrix, not by a phase, as soon as a multiplicity exceeds one.
   */
  class FusionCategory
  {
  public:
    /**
     * @brief Attach a name and a device to the category.
     * @param name The name of the category, for error messages and for the hash of a fusion tree.
     * @param device The device the symbols and the cached tables are materialised on.
     */
    FusionCategory(std::string name, c10::Device device = default_device());
    virtual ~FusionCategory() = default;

    /**
     * @brief The sectors $a \otimes b$ fuses to, each listed once with its multiplicity.
     * @param a The first sector.
     * @param b The second sector.
     * @return The outcomes, in the order that fixes the block order of `F_symbol` and `R_symbol`.
     * @details The order is part of the contract, not an implementation detail: two categories that
     * return the same set in a different order describe different bases, and their symbols differ by
     * a permutation.
     *
     * The table of a pair is computed on the first request and then held, so a lookup allocates
     * nothing and the reference stays valid until `clear_cache`.  The cache is an optimisation only:
     * every answer is the one the concrete category computes.
     */
    const std::vector<Outcome> &outcomes(const Sector &a, const Sector &b) const;

    /**
     * @brief Drop the fusion tables this object holds.
     * @details A reference `outcomes` returned before the call must not be used after it.
     */
    virtual void clear_cache() const;

    /**
     * @brief Whether $a \otimes b$ contains $c$ at all.
     * @param a The first sector.
     * @param b The second sector.
     * @param c The candidate outcome.
     * @return True when the multiplicity of $c$ is positive.
     */
    bool can_fuse_to(const Sector &a, const Sector &b, const Sector &c) const;

    /**
     * @brief The multiplicity $N^c_{ab}$, the number of distinct vertices coupling $a \otimes b$ to
     * $c$.
     * @param a The first sector.
     * @param b The second sector.
     * @param c The outcome sector.
     * @return The multiplicity, and zero when the pair does not fuse to $c$.
     */
    int64_t n_symbol(const Sector &a, const Sector &b, const Sector &c) const;

    /**
     * @brief The associator $F^{d}_{abc}$: the matrix that recouples $((a \otimes b) \otimes c)
     * \to d$ into $(a \otimes (b \otimes c)) \to d$.
     * @param a The first sector, the leftmost of the left tree.
     * @param b The second sector.
     * @param c The third sector.
     * @param d The outer sector both trees fuse to.
     * @return A square matrix of side
     * $\sum_e N^e_{ab} N^d_{ec} = \sum_f N^f_{bc} N^d_{af} = \dim \operatorname{Hom}(a \otimes b
     * \otimes c, d)$, empty when $d$ is not an outcome.
     * @details The rows are the left tree and the columns the right tree.  Both bases are
     * orthonormal in the tree basis, so the matrix is unitary.
     *
     * A row is addressed by three indices: the inner sector $e$ of $a \otimes b$, taken in the order
     * `outcomes(a, b)` returns it; then the vertex $a \otimes b \to e$ in $[0, N^e_{ab})$; then the
     * vertex $e \otimes c \to d$ in $[0, N^d_{ec})$.  A column is addressed the same way through $f$
     * from `outcomes(b, c)`, $b \otimes c \to f$ and $a \otimes f \to d$.
     *
     * The gauge is fixed by the tree basis: every vertex tensor has unit Frobenius norm, and the
     * phase of each multiplicity index is the one the concrete category chose.  A change of gauge
     * multiplies the matrix by diagonal unitaries and leaves the pentagon equation invariant.
     */
    virtual at::Tensor F_symbol(const Sector &a,
                                const Sector &b,
                                const Sector &c,
                                const Sector &d) const = 0;

    /**
     * @brief The braiding $R_{bc}$: the matrix that exchanges two neighbouring sectors.
     * @param b The left sector before the exchange.
     * @param c The right sector before the exchange.
     * @return A square matrix of side $\sum_e N^e_{bc} = \sum_e N^e_{cb}$, empty when the two
     * sectors do not fuse at all.
     * @details The rows are addressed by $e$ from `outcomes(b, c)` and its vertex index, the columns
     * by $e$ from `outcomes(c, b)` and its vertex index.  The matrix is block diagonal in $e$, one
     * block of shape $N^e_{bc} \times N^e_{cb}$ per outcome, and it is unitary in the standard
     * normalisation.  For a multiplicity-free category every block is a single entry, which is the
     * usual phase $R^c_{bc}$.
     */
    virtual at::Tensor R_symbol(const Sector &b, const Sector &c) const = 0;

    /**
     * @brief The quantum dimension $d_a$, the Perron eigenvalue of the fusion matrix of $a$.
     * @param a The sector.
     * @return A real, positive, multiplicative dimension: $d_a d_b = \sum_c N^c_{ab} d_c$ and
     * $d_{\bar a} = d_a$, with $d_1 = 1$.
     * @details It is the Frobenius-Perron dimension and not the dimension of a representation space:
     * the Fibonacci category has $d_\tau = (1 + \sqrt{5}) / 2$, irrational, which is exactly why the
     * two are kept apart.
     */
    virtual double q_dim(Sector a) const = 0;

    /**
     * @brief The number of components of the sector in the dense realization.
     * @param a The sector.
     * @return The dimension of the vector space the sector stands for, and 1 when the category is
     * not realized as a category of representations.
     * @details Every simple object of a fusion category is one-dimensional.  `dim` exceeds 1 only
     * because a concrete category comes with a fibre functor to vector spaces, which the Fibonacci
     * category does not have.  The engine sizes a dense block with it, so it must satisfy
     * $\sum_c N^c_{ab} \dim c = \dim a \cdot \dim b$ wherever both it and $N$ are defined.
     */
    virtual int dim(Sector a) const = 0;

    /**
     * @brief The dual sector $\bar a$.
     * @param a The sector.
     * @return The sector that fuses with $a$ to the trivial one.
     * @details An involution, $\overline{\bar a} = a$, preserving the dimensions, $d_{\bar a} = d_a$
     * and $\dim \bar a = \dim a$.  A leg carrying $\bar a$ is the bra of the leg carrying $a$: the
     * same space with the arrow reversed.
     */
    virtual Sector dual(Sector a) const = 0;

    /**
     * @brief The trivial sector, the unit of the fusion.
     * @return The sector $1$ with $1 \otimes a = a \otimes 1 = a$ and $d_1 = \dim 1 = 1$.
     */
    virtual Sector trivial() const = 0;

    /**
     * @brief The scalar type of the matrices `F_symbol` and `R_symbol` return.
     * @return A real type when every symbol is real in the gauge the category chose, and a complex
     * one otherwise.
     */
    virtual c10::ScalarType dtype() const = 0;

    /**
     * @brief The device the symbols and the cached tables of the category live on.
     * @return The device `F_symbol` and `R_symbol` allocate on.
     * @details A symbol is the same mathematical object on any device, so the device names only
     * where the tables are materialised; it is object state here because the tables are cached.
     * The category takes it in its constructor and every concrete category inherits it.
     */
    virtual c10::Device device() const;

    /**
     * @brief The name the category was constructed with.
     * @return The name, for error messages and for the hash of a fusion tree.
     */
    const std::string &name() const;

   protected:
    /**
     * @brief The outcomes of $a \otimes b$, computed when the pair is first asked for.
     * @param a The first sector.
     * @param b The second sector.
     * @return The outcomes in the order that fixes the block order of `F_symbol` and `R_symbol`.
     * @details The one method a concrete category implements; `outcomes` wraps it with the cache.
     */
    virtual std::vector<Outcome> compute_outcomes(const Sector &a, const Sector &b) const = 0;

  private:
    std::string name_;
    c10::Device device_;
    // One fusion table per pair, keyed by the sectors themselves so that a lookup copies nothing.
    mutable std::map<Sector, std::map<Sector, std::vector<Outcome>>> outcomes_cache_;
  };

  /**
   * @brief Bind the fusion outcome and the base category into the extension module.
   * @param m The module to bind into.
   */
  void bind_category(py::module_ &m);

} // namespace sym
