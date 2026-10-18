# TenTorch

A Python library for one-dimensional tensor networks whose tensors are block-sparse in a
fusion category, built on PyTorch.

The two closest neighbours leave a gap this library fills. TeNPy is a mature DMRG package built
on numpy: it runs on the CPU, and the symmetry classes it releases are groups realised through
Clebsch-Gordan coefficients, so its reach stops where a Clebsch-Gordan tensor stops (a newer
fusion-category framework does exist on its development branch, with the same
`FusionStyle`/`BraidingStyle` vocabulary this library uses). TensorKit.jl has the categorical
formalism in full, but it is Julia, and although it can hold its dense blocks in CUDA arrays,
GPU support for its truncation routines was still being stabilised in 2026. TenTorch keeps the
categorical data — a sector, an associator and a braiding per block — and runs the dense blocks
on the GPU as ordinary `torch` operations.


## What it does

Density matrix renormalisation group (DMRG) ground states of one-dimensional Hamiltonians, with
the symmetry of the model carried exactly by the tensors: U(1), Z_n, SU(2), SU(3), Fibonacci,
fermion parity, or no symmetry at all. A block is keyed by a fusion tree of sectors, so the
block structure follows the category's fusion rules rather than a crystallographic group. The
Fibonacci category has no Clebsch-Gordan tensor, so it is reachable exactly where the
categorical route applies and nowhere else; the groups fall back to a dense contraction when the
categorical route declines a layout.


## Install

Requires Python 3.12 or newer and PyTorch. From the repository root:

```shell
python -m pip install -e .
```

The packages also live at the repository root, so running Python from the root works without
installing anything.

The engine is a C++ extension. It compiles itself on first import, or ahead of time with

```shell
.venv/bin/python mpsdmrg/ext/build.py --jobs 1
```

One compile job needs about 2.5 GiB of host memory and the build refuses to start below that
rather than swap the machine to death; `--jobs` above one multiplies the requirement.

## Getting started

The notebooks in `mpsdmrg/notebooks/` are the worked examples, and each one runs top to bottom
after a kernel restart:

| notebook | what it shows |
| --- | --- |
| `Heisenberg.ipynb` | the spin-1/2 Heisenberg chain, with and without SU(2) symmetry |
| `AKLT.ipynb` | the AKLT state, its exact energy and its correlations |
| `Charged.ipynb` | a Hamiltonian with a non-trivial total charge |
| `ULS.ipynb` | the SU(3) Uimin-Lai-Sutherland chain |

The object that drives a calculation is `Broomstick`: it owns the state, the Hamiltonian and
the sweep, and it is the only class a caller has to know.

## The layers

```
symmetry/cat   ->   symmetry/tree   ->   tensor   ->   linalg
 the category        legs and trees      blocks       splits
                                                    |
                                                    v
                                          mps / mpo / broomstick
```

- `mpsdmrg/symmetry/cat` — the category: sectors, fusion outcomes, the associator, the braiding,
  the duality. Adding a symmetry means adding one subclass here.
- `mpsdmrg/symmetry/tree` — a `Leg` carries one sector (or several, with multiplicity); a
  `FusionTree` is an ordered coupling path, and it is what keys one dense block.
- `mpsdmrg/tensor` — a `Tensor` holds one dense block per charge key, and `cached_einsum`
  contracts a whole network of them.
- `mpsdmrg/linalg` — `svd`, `qr`, `rq` and `svdvals` on block-sparse tensors; the split that
  produces new bond sectors.
- `mpsdmrg/mps.py`, `mpsdmrg/mpo.py`, `mpsdmrg/broomstick.py` — the state, the Hamiltonian and
  the sweep driver.

## Status and limits

A contraction has two routes for the weight of an output tree: a categorical one, which needs
nothing but associators and braidings, and a dense one, which builds the trees densely and
contracts them. The categorical route is exact on the layouts the library issues today — the 22
layouts of one L=6 SU(2) Heisenberg sweep reproduce the dense route to 8e-16. It declines the
shapes it does not model — a multiplicity pairing, a leg that is both contracted and kept, a leg
count the chain cannot express — and those fall back to the dense route, so a category without a
Clebsch-Gordan tensor still works wherever the route applies.

The dense route is also the faster of the two at present, by roughly 12 times, so it stays the
reference. A build with `DENSEPATH=1` keeps a Python switch that forces it
(`mpsdmrg.ext.set_dense_path`), which is how the two routes are compared on the same operands.

## Documentation and tests

`doc/main.pdf` (built from `doc/main.tex` with `pdflatex`) is the theory: the group picture,
how it becomes a category, a worked SU(2) fusion example, the diagram algorithm, charge flow,
and the anchors behind each section.

`test/run_all.py` runs every `test/check_*.py`; each check also runs on its own. Note that
`test/` is gitignored, so a fresh clone does not carry it.

## License

MIT, copyright (c) 2026 Yuchen Qian.
