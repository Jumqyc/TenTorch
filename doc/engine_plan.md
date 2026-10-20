# A tensor engine built from the fusion category up

This is the design of a new engine and tensor layer, written from zero. Nothing below
depends on the current `mpsdmrg/tensor/` implementation; the last section lists what of
the existing code can be reused unchanged.

## 1. Layers

```
sector      Sector, FusionCategory                      (topological data: F, R, d, duality)
tree        FusionTree, FusionTreePair, enumeration     (one side / two sides of a tensor)
space       GradedSpace, Leg                            (which sectors, which duality, charge)
tensor      TensorMap, blocks                           (the data)
moves       bend, braid, permute, repartition           (change of tree basis)
algebra     contract, tensorproduct, svd, qr, eigh      (operations on the data)
```

The first layer is the only one that survives from today's code. Everything above it is
new, because today's tensor stores one fusion tree per block while TensorKit stores one
fusion tree **pair** per block, and that difference is what makes the closed diagram
impossible today.

## 2. Sector layer

Interface (unchanged from `mpsdmrg/symmetry/cat/cat.h`):

```cpp
struct FusionCategory {
  virtual Sector dual_sector(const Sector&) const = 0;
  virtual std::vector<Sector> fusion_outcomes(const Sector& a, const Sector& b) const = 0;
  virtual int64_t fusion_multiplicity(const Sector& a, const Sector& b, const Sector& c) const = 0;
  virtual at::Tensor associator(a, b, c, d, e, f, device) const = 0;   // shape (N^e_ab, N^d_ec, N^f_bc, N^d_af)
  virtual at::Tensor braiding(a, b, c, device) const = 0;              // shape (N^c_ab, N^c_ba)
  virtual double quantum_dim(const Sector&) const = 0;
  virtual int64_t sector_dim(const Sector&) const = 0;
  virtual double frobenius_schur_phase(const Sector&) const = 0;       // (+-1)
  virtual FusionStyle fusion_style() const = 0;                        // unique | simple | generic
  virtual BraidingStyle braiding_style() const = 0;                    // bosonic | fermionic | anyonic
};
```

No change is needed: pentagon and hexagon already hold to 1e-16 for U(1), SU(2), SU(3) and
Fibonacci (`test_category_axioms.py`).

## 3. Tree layer

```cpp
struct FusionTree {                     // one side of a tensor
  std::vector<Sector> sectors;          // leg sectors, dualised for a leg that carries duals
  std::vector<bool> isdual;             // one per leg
  std::vector<Sector> inner;            // N-2 couplings, `inner[k]` couples legs 0..k+1
  std::vector<int64_t> vertices;        // N-1 vertex copies
  Sector coupled;                       // the charge of the whole side
};

struct FusionTreePair {
  FusionTree codomain;                  // the legs the map comes out of
  FusionTree domain;                    // the legs the map goes into
};
```

Invariants, checked in an `assert` build and relied on by every move:

* `verticts.size() == sectors.size() - 1` for `N > 1`; both empty for `N == 0`, where
  `coupled` is the trivial sector.
* `inner.size() == sectors.size() - 2` for `N > 1`.
* `codomain.coupled == domain.coupled` — that equality *is* the block label.
* `sectors[i]` is the sector the leg carries in the direction it points; `isdual[i]` says
  the leg was written in the conjugate basis.

Signatures and helpers:

```cpp
bool operator==(const FusionTree&, const FusionTree&);            // componentwise
bool operator<(const FusionTree&, const FusionTree&);              // so it can key a map
std::string repr(const FusionTree&);                              // "(a, b) -> c  inner (x,) copies (0, 1)"
std::vector<FusionTreePair> tree_pairs(int64_t n_codomain, int64_t n_domain);   // per coupled sector
int64_t tree_dim(const FusionTree&);                              // product of the vertex multiplicities
```

`tree_pairs` enumerates every pair whose sides carry the same coupled sector; it is the
block enumeration of a tensor, and nothing else in the engine enumerates blocks.

## 4. Space layer

```cpp
struct Leg {
  std::shared_ptr<FusionCategory> group;
  std::vector<std::pair<Sector, int64_t>> sectors;   // sector -> multiplicity
  int qconj;                                        // +1 codomain, -1 domain
};

struct GradedSpace {
  std::shared_ptr<FusionCategory> group;
  std::vector<Leg> codomain;                        // legs with qconj == +1
  std::vector<Leg> domain;                          // legs with qconj == -1
  Sector qtotal;                                    // carried as one extra 1-dimensional domain leg
};
```

The total charge is **data, not a leg**: a charged tensor is a map whose domain has one
extra leg of dimension one carrying `qtotal`. This removes every special case about
"which sector may appear" — it is the single biggest simplification over today's code,
where `qtotal` is a member that has to be threaded through every routine.

## 5. Tensor layer

```cpp
struct TensorMap {
  GradedSpace space;
  std::map<Sector, at::Tensor> blocks;   // coupled sector -> matrix
};
```

A block `blocks[c]` has shape

```
(prod over codomain trees of tree_dim)  x  (prod over domain trees of tree_dim)
```

where the trees are those of `tree_pairs` whose `coupled == c`, in the canonical order of
the enumeration (codomain trees lexicographic, then domain trees). The entries are
coefficients in the **orthonormal fusion tree basis**: the basis vector of a tree pair is
the normalised Clebsch-Gordan expansion of that pair. The normalisation is fixed by the
`A`/`B` symbols of §7, not by `fusion_tensor`, so the basis is orthonormal by construction
and a closed diagram is an ordinary matrix inner product.

The invariant, stated once: *a tensor is a collection of matrices, one per shared coupled
sector, in an orthonormal tree basis.* Every operation preserves it.

## 6. Moves

A move is implemented as a pair of index maps and a coefficient table, applied to the
block matrices:

```cpp
struct MoveResult {
  FusionTreePair tree;                 // the tree after the move
  double coefficient;                  // scalar in front
};
std::vector<MoveResult> bend_right(const FusionTreePair&, const c10::Device&);
std::vector<MoveResult> bend_left (const FusionTreePair&, const c10::Device&);
std::vector<MoveResult> braid     (const FusionTreePair&, int64_t leg, const c10::Device&);
std::vector<MoveResult> repartition(const FusionTreePair&, int64_t keep, const c10::Device&);
```

`bend_right` on a pair `(f1, f2)` moves the last codomain leg to the domain side:

```
c  = f1.coupled,  b = f1.sectors.back()
a  = f1.sectors.size() == 1 ? dual(b)
   : f1.sectors.size() == 2 ? f1.sectors[0]
   : f1.inner.back()
f1' = (f1.sectors[0..end-1], coupled = a)
f2' = (f2.sectors + [dual(b)], isdual + [!f1.isdual.back()],
       inner  + (f2.sectors.size() > 1 ? [c] : []),
       vertices + [nu])
coefficient = sqrt(d(c) / d(a)) * B^{a b}_c[mu, nu] * (f1.isdual.back() ? conj(fs(dual(b))) : 1)
```

with `mu` the copy of the bent vertex and `nu` the copy of the new domain vertex.
`bend_left` is the mirror: swap the sides, `bend_right`, swap back, conjugate.
`braid` exchanges the last two legs of a side through `R`; `permute` is a bubble sort of
braids, `repartition` a repeated `bend`.

Because the result of a move can be a sum over `nu`, every move returns a *list*; a
recoupling that reaches several trees produces several blocks that are added into the same
coupled sector.

## 7. Coefficients

Written out once, so that no call site has to remember them:

```
d_a      = quantum_dim(a)
A^{ab}_c = sqrt(d_a d_b / d_c) * conj(kappa_a * F^{a a b}_b[1, c])      kappa_a = +/-1
B^{ab}_c = sqrt(d_a d_b / d_c) * F^{a b dual(b)}_a[c, 1]
F        = associator,  R = braiding,  fs(a) = Frobenius-Schur phase (+-1)
```

`A` converts a fusion tree vertex into a splitting vertex (dualisation), `B` does the
inverse; both come from the associator, so no new topological data is needed. The `1` in
the index lists is the trivial sector's single copy.

## 8. Algebra

`contract(A, B, labels)`:

1. permute both operands so that the contracted legs sit last on the codomain of the
   first and first on the domain of the second;
2. bend them across so both operands end up with the contracted legs entirely on one side;
3. for each shared coupled sector `c`, `blocks_out[c] = blocks_A[c] * blocks_B[c]`
   (batched `at::matmul` over the tree-pair matrices);
4. the resulting pair is the output tree pair, already canonical.

A closed diagram is step 3 with empty codomain and domain: sum of the elementwise product
of the two block matrices, i.e. `(A * B).sum()` per sector. No special case, no weight, no
guard.

`svd`, `qr`, `eigh` run on each block matrix independently and keep the tree pairs on both
sides; the sector bookkeeping is the identity.

## 9. What gets deleted

* `mpsdmrg/tensor/diagram.{h,cpp}` entirely, with `diagram_weight`, `PairTree`, the weight
  cache, the guard codes and the bend/repartition/permute chain built on them.
* The `qtotal`-as-member plumbing in `tensor.cpp` (`homogeneous()`, `tree_totals`,
  `fusion_trees(..., candidate)` loops, `expandable`, `closed_blocks`).
* The dense route as a *route*: `stored_dense`, `cached_fold`, `cached_weight` remain only
  as `to_dense()`/`from_dense()` conversion used by tests.

## 10. What is reused unchanged

| Piece | Where | Why it survives |
| --- | --- | --- |
| sector classes, F/R tables, `quantum_dim`, `sector_dim`, duality | `mpsdmrg/symmetry/cat/**` | validated by pentagon/hexagon for four categories |
| `test/category_axioms.py` | `test/` | tests the layer that does not change |
| Python API of the tensor module | `mpsdmrg/tensor/__init__.py` | `Leg`, `Tensor`, `bond_leg`, `multiplet_leg`, `cached_einsum`, `svd`, `qr` signatures stay |
| MPS / MPO / DMRG / broomstick layer | `mpsdmrg/*.py` | talks only to the Python API above |
| `fusion_tensor` | sector layer | becomes the `to_dense()` conversion and the image-side oracle |
| `test/run_all.py` and the check scripts | `test/` | the acceptance gate; comparisons stay valid because both sides change together |

## 11. Stages and acceptance

| Stage | Deliverable | Acceptance |
| --- | --- | --- |
| 1 | tree layer: `FusionTree`, `FusionTreePair`, `tree_pairs`, `repr`, invariants | enumeration counts match the legacy `combos` for 12 layouts; `check_category_axioms` still green |
| 2 | space + `TensorMap` + `from_dense`/`to_dense` | round trip exact on random tensors of every category; block count identity `Σ_c N_cod(c) N_dom(c)` |
| 3 | moves: bend, braid, permute, repartition | each move equals the matrix built column by column from `to_dense` |
| 4 | `contract` + `einsum` | open and closed diagrams equal `to_dense` on all probe layouts, including a charged closed scalar |
| 5 | `svd`/`qr`/`eigh` on the new blocks | singular values equal the legacy engine's; MPS compression error unchanged |
| 6 | Python shim + switch MPS/DMRG over, delete the old route | `test/run_all.py`: same 21 pass, 3 pre-existing failures gone or unchanged |

Stage 4 is the one that answers the current defect: a charged closed scalar exists by
construction, because the tree basis is orthonormal and a closed diagram is a matrix inner
product.

## 12. Risks

* **No reference implementation available here.** TensorKit 0.17.2 and master both gate
  `bendright(::FusionTreePair)` behind `FusionStyle(I) === UniqueFusion`, which no standard
  sector has, and TensorKit has no SU(3) sector at all. The coefficients of §7 must be
  validated against the *algebra* (pentagon, hexagon, unit equations, and agreement with
  the CG expansion) rather than against a dump from TensorKit.
* **Two conventions must not be mixed.** The legacy `fusion_tensor` normalisation is
  `‖CG‖² = d(c)` for SU(3) and `1` for SU(2); the tree basis of §5 uses `A`/`B`. Any test
  that compares the two must go through `to_dense`, never compare block entries.
* **Fibonacci has no CG tensor**, so `to_dense` is unavailable there; its validation must be
  the axioms plus self-consistency (a closed diagram is a matrix inner product, so it is
  the norm and can be checked against `Σ|amplitude|²` of an explicit small expansion).
