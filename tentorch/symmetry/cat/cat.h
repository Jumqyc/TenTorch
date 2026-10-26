// The fusion categories of the engine: the labels, their fusion, the F symbol that recouples a
// fusion tree and the R symbol that exchanges two neighbouring labels.  The tensor engine includes
// this header only, so what a concrete symmetry is stays behind the interface.
#pragma once

#include <ATen/ATen.h>

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

  /**
   * @brief One row or column of an $F$ symbol: the inner sector and the two vertices it joins.
   * @details A row of $F^{d}_{abc}$ is an $e$ from `outcomes(a, b)` together with the vertex index of
   * $a \otimes b \to e$ and the vertex index of $e \otimes c \to d$; a column is the same through an
   * $f$ from `outcomes(b, c)`.  The address list is the only thing that says which entry of the
   * matrix is which basis vector, so it belongs to the interface and not to one implementation.
   */
  struct FIndex
  {
    Sector sector;   ///< the sector the inner vertex couples to
    int64_t inner;   ///< vertex index of the inner fusion
    int64_t outer;   ///< vertex index of the outer fusion
  };

  /**
   * @brief One row or column of an $R$ symbol: the coupled sector and the vertex index.
   */
  struct RIndex
  {
    Sector sector;   ///< the sector the vertex couples to
    int64_t copy;    ///< vertex index of that fusion
  };

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
   * fusion tree.  The entry of a block is the coefficient of that basis vector.  A vertex is
   * normalised as an isometry onto the sector it couples to, $\|C^{c}_{ab}\|^{2}_{F} = d_c$ for each
   * copy, so a tree over legs with root $r$ has $\|\text{tree}\|^{2} = d_r$ and two different trees
   * are orthogonal: the tree basis is orthonormal up to the dimension of the root.  Three
   * consequences the engine relies on:
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
     * @details The rows are the left tree and the columns the right tree, and the matrix is the
     * relation between them: a left tree is the combination
     * $\text{left}_r = \sum_c F^{d}_{abc}[r, c] \, \text{right}_c$.  Both bases are orthonormal up to
     * the root dimension, so the matrix is unitary.
     *
     * A row is addressed by three indices: the inner sector $e$ of $a \otimes b$, taken in the order
     * `outcomes(a, b)` returns it; then the vertex $a \otimes b \to e$ in $[0, N^e_{ab})$; then the
     * vertex $e \otimes c \to d$ in $[0, N^d_{ec})$.  A column is addressed the same way through $f$
     * from `outcomes(b, c)`, $b \otimes c \to f$ and $a \otimes f \to d$.
     *
     * The gauge is fixed by the tree basis: every vertex is an isometry onto its coupled sector,
     * $\|C^{c}_{ab}\|^2_F = d_c$ for each copy, so a tree over legs with root $r$ has
     * $\|\text{tree}\|^2 = d_r$ and two different trees are orthogonal.  The phase of each
     * multiplicity index is the one the concrete category chose; a change of gauge multiplies the
     * matrix by diagonal unitaries and leaves the pentagon equation invariant.
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
     * by $e$ from `outcomes(c, b)` and its vertex index.  The matrix is the relation between the two
     * vertices, $\text{straight}_r = \sum_c R_{bc}[r, c] \, \text{swapped}_c$, both read in the
     * $(b, c, e)$ axis order.  It is block diagonal in $e$, one block of shape
     * $N^e_{bc} \times N^e_{cb}$ per outcome, and it is unitary in the standard normalisation.  For a
     * multiplicity-free category every block is a single entry, which is the usual phase $R^c_{bc}$.
     */
    virtual at::Tensor R_symbol(const Sector &b, const Sector &c) const = 0;

    /**
     * @brief The rows of $F^{d}_{abc}$: the left tree $((a \otimes b) \otimes c) \to d$.
     * @param a The first sector of the left tree.
     * @param b The second sector.
     * @param c The third sector.
     * @param d The outer sector both trees fuse to.
     * @return One entry per row, in the order the matrix lists them.
     * @details Not virtual: the order follows from `outcomes` alone, so it is written once here and
     * every `F_symbol` reads it.  An inner sector that cannot reach $d$ contributes no rows.
     */
    std::vector<FIndex> f_rows(const Sector &a, const Sector &b, const Sector &c, const Sector &d) const;

    /**
     * @brief The columns of $F^{d}_{abc}$: the right tree $(a \otimes (b \otimes c)) \to d$.
     * @param a The first sector of the right tree.
     * @param b The second sector.
     * @param c The third sector.
     * @param d The outer sector both trees fuse to.
     * @return One entry per column, in the order the matrix lists them.
     */
    std::vector<FIndex> f_columns(const Sector &a, const Sector &b, const Sector &c, const Sector &d) const;

    /**
     * @brief The rows of $R_{bc}$, and its columns with $b$ and $c$ exchanged.
     * @param b The left sector.
     * @param c The right sector.
     * @return One entry per row: the sector `outcomes(b, c)` yields and its vertex index.
     */
    std::vector<RIndex> r_rows(const Sector &b, const Sector &c) const;

    /**
     * @brief The quantum dimension $d_a$, the Perron eigenvalue of the fusion matrix of $a$.
     * @param a The sector.
     * @return A real, positive, multiplicative dimension: $d_a d_b = \sum_c N^c_{ab} d_c$ and
     * $d_{\bar a} = d_a$, with $d_1 = 1$.
     * @details For a category of representations it is the dimension of the space the sector stands
     * for, an integer, and the number of components the Clebsch-Gordan tensor of a vertex carries
     * per leg.  A general fusion category has no such space and the dimension is the Frobenius-Perron
     * eigenvalue: the Fibonacci category has $d_\tau = (1 + \sqrt{5}) / 2$, irrational.
     *
     * The engine never turns a dimension into a block size.  A block is sized by the multiplicities
     * of the trees alone, because the tree basis already carries the components; a dense array that
     * a dimension would size does not exist anywhere in the engine.
     */
    virtual double q_dim(Sector a) const = 0;

    /**
     * @brief The dual sector $\bar a$.
     * @param a The sector.
     * @return The sector that fuses with $a$ to the trivial one.
     * @details An involution, $\overline{\bar a} = a$, preserving the dimension, $d_{\bar a} = d_a$.
     * A leg carrying $\bar a$ is the bra of the leg carrying $a$: the same space with the arrow
     * reversed.
     */
    virtual Sector dual(Sector a) const = 0;

    /**
     * @brief The Frobenius-Schur indicator $\nu_a$ of a sector.
     * @param a The sector.
     * @return $+1$ or $-1$, with $\nu_1 = +1$ and $\nu_{\bar a} = \nu_a$.
     * @details The invariant form of a simple sector is the map from $a \otimes \bar a$ to the
     * trivial sector, and on the space of $a$ it is symmetric ($\nu_a = +1$, real type) or
     * antisymmetric ($\nu_a = -1$, quaternionic type).  It is the phase a leg picks up when it is
     * turned around, $\nu_a = R^{1}_{a\bar a}$: a sector whose pair braids with a sign, like the
     * spin-$1/2$ of SU(2) or the odd sector of the fermion parity, carries $\nu_a = -1$.
     */
    virtual double frobenius_schur_sign(const Sector& a) const = 0;

    /**
     * @brief The trivial sector, the unit of the fusion.
     * @return The sector $1$ with $1 \otimes a = a \otimes 1 = a$ and $d_1 = 1$.
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

} // namespace sym
