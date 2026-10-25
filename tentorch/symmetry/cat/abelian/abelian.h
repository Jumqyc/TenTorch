// The abelian groups: no symmetry, U(1), Z_N and fermion parity.
#pragma once

#include "../group.h"

namespace sym {

/**
 * @brief A group whose fusion is its group law.
 * @details Every sector is one-dimensional and every outcome has multiplicity one, so the
 * Clebsch-Gordan tensor of a vertex is the single entry 1 and the group layer derives $F \equiv 1$
 * and $R \equiv 1$ from it.  A concrete group supplies only how two sectors combine and how one
 * inverts.
 */
class AbelianGroup : public Group {
 public:
  using Group::Group;

  Sector dual(Sector a) const override;
  Sector trivial() const override;
  double q_dim(Sector a) const override;
  double frobenius_schur_sign(const Sector& a) const override;
  c10::ScalarType dtype() const override;

 protected:
  std::vector<Outcome> compute_outcomes(const Sector& a, const Sector& b) const override;
  at::Tensor compute_clebsch_gordan(const Sector& a, const Sector& b, const Sector& c,
                                    int64_t copy) const override;

  /**
   * @brief The sector $a \otimes b$ fuses to.
   * @param a The first sector.
   * @param b The second sector.
   * @return The sector their product names, in the coordinates of the group.
   */
  virtual Sector combine(const Sector& a, const Sector& b) const = 0;

  /**
   * @brief The sector that fuses with $a$ to the trivial one.
   * @param a The sector.
   * @return The inverse of $a$.
   */
  virtual Sector inverse(const Sector& a) const = 0;
};

/** @brief The category of a single sector: everything fuses to the trivial one. */
class NoSymmetry : public AbelianGroup {
 public:
  explicit NoSymmetry(c10::Device device = default_device());

 protected:
  Sector combine(const Sector& a, const Sector& b) const override;
  Sector inverse(const Sector& a) const override;
};

/** @brief $U(1)$, with the charge as its single coordinate. */
class U1Symmetry : public AbelianGroup {
 public:
  explicit U1Symmetry(c10::Device device = default_device());

 protected:
  Sector combine(const Sector& a, const Sector& b) const override;
  Sector inverse(const Sector& a) const override;
};

/** @brief $Z_N$, with the residue in $[0, N)$ as its single coordinate. */
class ZNSymmetry : public AbelianGroup {
 public:
  // Raises:
  //   std::invalid_argument: If `modulus` is not positive.
  explicit ZNSymmetry(int64_t modulus, c10::Device device = default_device());
  int64_t modulus() const { return modulus_; }

 protected:
  Sector combine(const Sector& a, const Sector& b) const override;
  Sector inverse(const Sector& a) const override;

 private:
  int64_t modulus_;
};

/**
 * @brief Fermion parity, $\mathbb{Z}_2$ with the Koszul sign of the exchange.
 * @details It is not an `AbelianGroup`: its associator is trivial but its braiding is not, so the
 * all-ones vertex of an abelian group would give $R \equiv 1$ and lose the sign a fermion picks up
 * when it passes another fermion.
 */
class FermionParity : public FusionCategory {
 public:
  explicit FermionParity(c10::Device device = default_device());

  at::Tensor F_symbol(const Sector& a, const Sector& b, const Sector& c,
                      const Sector& d) const override;
  at::Tensor R_symbol(const Sector& b, const Sector& c) const override;
  double q_dim(Sector a) const override;
  Sector dual(Sector a) const override;
  Sector trivial() const override;
  double frobenius_schur_sign(const Sector& a) const override;
  c10::ScalarType dtype() const override;

 protected:
  std::vector<Outcome> compute_outcomes(const Sector& a, const Sector& b) const override;
};

// The singletons behind the module-level names; the same object every call.
const std::shared_ptr<FusionCategory>& no_symmetry();
const std::shared_ptr<Group>& u1_symmetry();
const std::shared_ptr<Group>& z_n_symmetry(int64_t modulus);
const std::shared_ptr<FusionCategory>& fermion_parity();

// Drop the cache of every Z_N singleton, including the ones no longer reachable by a call.
void clear_z_n_caches();

// Bind the abelian groups and their singletons.

}  // namespace sym
