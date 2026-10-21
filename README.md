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
.venv/bin/python tentorch/_ext/build.py --jobs 1
```

One compile job needs about 2.5 GiB of host memory and the build refuses to start below that
rather than swap the machine to death; `--jobs` above one multiplies the requirement.

## Getting started

The notebooks in `examples/` are the worked examples, and each one runs top to bottom
after a kernel restart:

| notebook | what it shows |
| --- | --- |
| `Heisenberg.ipynb` | the spin-1/2 Heisenberg chain, with and without SU(2) symmetry |
| `AKLT.ipynb` | the AKLT state, its exact energy and its correlations |
| `Charged.ipynb` | a Hamiltonian with a non-trivial total charge |
| `ULS.ipynb` | the SU(3) Uimin-Lai-Sutherland chain |

A calculation starts from a model and one function call:

```python
import tentorch
from tentorch.models import heisenberg

model = heisenberg(20, S=0.5)                      # 20 spin-1/2 sites, SU(2) symmetry
result = tentorch.dmrg(model, max_bond_dim=64, num_sweeps=10)
print(result.energy, result.state.bond_dim)
```

A `Model` is the physics only — the sites and the couplings; it holds no state and no numerical
options. `dmrg` builds the state, sweeps it, and returns the energy, the state and the largest
discarded weight of the run, so an observable is a call on `result.state` and a bond cap is an
argument, not a property of the model. `tentorch.algorithms.exact.exact_energy` diagonalises the
same model in its full Hilbert space, which is how the sweep is checked on short chains.

A site comes from `tentorch.sites` — `spin(S)` and `fundamental_su3()` — and carries the operator
table the couplings name. The model zoo lives in `tentorch/models` (Heisenberg, Ising, AKLT and
the SU(3) Uimin-Lai-Sutherland chain), and the notebooks below write their own where the point is
the Hamiltonian rather than the sweep.

## The layers

```
symmetry/cat   ->   symmetry/tree   ->   tensors   ->   linalg
 the category        legs and trees      blocks        splits
                                                     |
                                                     v
                               networks/mps, networks/mpo, algorithms/dmrg
```

- `tentorch/symmetry/cat` — the category: sectors, fusion outcomes, the associator, the braiding,
  the duality. Adding a symmetry means adding one subclass here.
- `tentorch/symmetry/tree` — a `Leg` carries one sector (or several, with multiplicity); a
  `FusionTree` is an ordered coupling path, and it is what keys one dense block.
- `tentorch/tensors` — a `Tensor` holds one dense block per charge key, and `cached_einsum`
  contracts a whole network of them.
- `tentorch/linalg` — `svd`, `qr`, `rq` and `svdvals` on block-sparse tensors; the split that
  produces new bond sectors.
- `tentorch/networks/mps.py`, `tentorch/networks/mpo.py` — the state and the Hamiltonian.
- `tentorch/sites.py`, `tentorch/model.py` — a site's operator table, and the physics of a chain:
  its sites and its couplings. A `Model` turns itself into an `MPO` on first use.
- `tentorch/models/` — the chains the examples and the benchmarks run on.
- `tentorch/algorithms/dmrg.py` — the sweep driver (`DMRG`, `dmrg`), and
  `tentorch/algorithms/exact.py` — exact diagonalisation, the reference it is checked against.

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
(`tentorch._ext.set_dense_path`), which is how the two routes are compared on the same operands.

## Documentation and tests

`doc/main.pdf` (built from `doc/main.tex` with `pdflatex`) is the theory: the group picture,
how it becomes a category, a worked SU(2) fusion example, the diagram algorithm, charge flow,
and the anchors behind each section.

`tests/` holds the pytest suite: the sites' operator tables, and the sweep energy of every model
against exact diagonalization. Run it with

```shell
.venv/bin/python -m pytest tests -q
```

`bench/bench_dmrg.py` times the sweep against the bond cap, one line per run:

```shell
.venv/bin/python bench/bench_dmrg.py --model heisenberg --L 20 --max-bond 16 32 64 --sweeps 5
```

## License

MIT, copyright (c) 2026 Yuchen Qian.
