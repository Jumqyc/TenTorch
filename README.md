# TenTorch

A Python library for one-dimensional tensor networks whose tensors are block-sparse in a fusion category, built on PyTorch.

- TeNPy is a mature DMRG package on numpy that is easy to use, but it runs on the CPU, and the
symmetry classes it releases are groups realised through Clebsch-Gordan coefficients, so its reach
stops where a CG tensor stops. 
- TensorKit.jl has the categorical picture in full, but it is Julia and CPU native, with GPU truncation only being stabilised in 2026. 

TenTorch keeps the categorical data — the $F$ and $R$ symbols, the fusion trees — and puts the symmetry-adapted tensor and the block multiplications on ordinary `torch` arrays, so a block can live on the GPU.

## Status


| layer | state |
| --- | --- |
| symmetry categories | done: no symmetry, $U(1)$, $Z_n$, $SU(2)$, $SU(3)$, Fibonacci, fermion parity |
| legs, fusion trees, blocks | done |
| `einsum`, `contract`, `inner`, `norm` | done, with a plan cache per shape |
| `svd`, `svdvals`, `qr`, `rq` | done, with the new bond carrying the charge sectors |
| `MPS`, mixed canonical form | done, `qr`/`rq` moves verified against the norm and the expectation |
| `MPO` | the container is done; the only builder is the Heisenberg chain, abelian categories only |
| `DMRG` | two site, verified against exact diagonalisation for $L = 4, 6, 8$ |
| non-abelian `MPO` | **not implemented**: `heisenberg()` raises for $SU(2)$, $SU(3)$ and Fibonacci, because a coefficient in the coupled tree basis is not a plain matrix element |



## Install

Requires Python 3.12 or newer and PyTorch. From the repository root:

```shell
python -m pip install -e .
```

The packages also live at the repository root, so running Python from the root works without
installing anything.

The engine is a C++ extension. It compiles itself on first import, or ahead of time with

```shell
.venv/bin/python tentorch/_ext/build.py --jobs 1
```

One compile job needs about 2.5 GiB of host memory and the build refuses to start below that rather
than swap the machine to death; `--jobs` above one multiplies the requirement.

## The layers

```
symmetry/     the categories, the legs and the fusion trees (C++ engine, Python surface)
tensors/      the block-sparse tensor, its block layout, and the einsum
linalg/       svd, svdvals, qr, rq and the Krylov eigensolver
networks/     MPS and MPO
models/       Model, and the Heisenberg automaton that builds its MPO
algorithms/   DMRG, dmrg, and exact diagonalisation
```

- `tentorch/symmetry` — a `Sector` is a tuple of charges; a `FusionCategory` gives the fusion
  outcomes, the associator, the braiding and the duality. Adding a symmetry means adding one
  subclass in `tentorch/symmetry/cat`.
- `tentorch/tensors` — a `Tensor` carries a codomain and a domain of `Leg`s and one block per pair of
  fusion trees; `einsum` plans the moves of an equation once per shape and caches the plan.
- `tentorch/linalg` — the splittings, and `lanczos` for the ground state of one window.
- `tentorch/networks/mps.py`, `mpo.py` — the state in mixed canonical form, and an operator whose
  sites carry `(bond left, out)` on the codomain and `(in, bond right)` on the domain.
- `tentorch/models` — `Model` (the physics, and the slots a driver writes back into) and the
  Heisenberg automaton that turns a chain into an `MPO`.
- `tentorch/algorithms` — the two site sweep and the reference diagonalisation.

## Performance

Measured on one CPU thread and on an 8 GiB RTX 4060, `u1`, `n_iter = 5`, open chain, the energy of
every run below matching the dense value to $10^{-11}$:

| chain | bond | CPU s/sweep | GPU s/sweep |
| --- | --- | --- | --- |
| $L=16$ | 32 | 0.30 / 0.32 / 0.28 | 2.90 / 2.63 / 2.39 |
| $L=32$ | 64 | 1.42 / 2.17 / 2.49 | 11.16 / 13.15 / 9.47 |

The CPU column is the current engine. The GPU column was measured before the leg moves were folded
into one pass and has not been measured since.

The GPU is six to nine times slower here, as a contraction of these bonds is a few hundred small
matrices, one per block pair, so a kernel launch costs more than the multiplication.
`TENTORCH_DEVICE=cpu` forces the CPU while the bond is small. 

- For reference, TeNPy on the same chains and the same settings takes
0.08/0.22/0.29 s ($L=16$, $\chi = 32$) and 0.15/0.39/0.52 s ($L=32$, $\chi = 64$). The difference is that TenPy uses Group theory Clebsch-Gordan coefficients and fully utilize block structure of tensor, where as the categorical approach creates sparser block, and permutes the trees sequentially, making it less computational friendly.  

`bench/` holds the measurement scripts: `bench_dmrg.py` for the sweep above and `build.py` for the
C++ engine benchmarks. `bench/README.md` records what each one measures.

## Tests

```shell
.venv/bin/python -m pytest tests -q                    # 26 tests, about two seconds
TENTORCH_SLOW=1 .venv/bin/python -m pytest tests -q    # adds the long chain, about 45 s
```

`tests/README.md` says what is covered and what is not.

## Documentation

`doc/main.pdf` (built from `doc/main.tex` with `pdflatex`) is the theory: the group picture, how it
becomes a category, a worked $SU(2)$ fusion example, the diagram algorithm, and charge flow.

The notebooks under `examples/` were written against an earlier API — `tentorch.dmrg(model)`, a
`tentorch.sites` module, an Ising and an AKLT model — and do not run against the current package.
They are kept for the pictures, not as examples.

## License

MIT, copyright (c) 2026 Yuchen Qian.
