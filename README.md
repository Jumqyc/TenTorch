# TenTorch

A Python library for one-dimensional tensor networks whose tensors are block-sparse in a fusion category, built on PyTorch.

There are several existing mature dmrg libraries. 

- TeNPy is a mature DMRG package built
on numpy and is generally easy to use, but: 
  - it runs on the CPU,
  - the symmetry classes it releases are groups realised through Clebsch-Gordan coefficients, so its reach stops where a Clebsch-Gordan tensor stops (a fusion-category framework does exist on its development branch). 

- TensorKit.jl is another library, with the categorical approach in full, but:
  - it is in Julia, whose ecosystem is not comparable with Python, 
  - It is CPU native, although it can hold its dense blocks in CUDA arrays, GPU support for its truncation routines was still being stabilised in 2026. 
  
This library, TenTorch keeps the categorical data, the $F$ and $R$ symbol, and store the symmetry-adapted tensor on the GPU and the matrix computation through ordinary `torch` operations.


## What it does

- Symmetry adapted tensor: currently the symmetry of the model carried exactly by the tensors: `U(1)`, `Z_n`, `SU(2)`, `SU(3)`, `Fibonacci`, `fermion parity`, or `no symmetry`. A block is keyed by a fusion tree of sectors, so the block structure follows the category's fusion rules rather than a crystallographic group. 
- A standard 2-site DMRG algorithm. Perform DMRG sweep on a 1d quantum chain. 
- An automated MPO construction: plug in the terms,  construct the MPO automatically. 
- To be added. 


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
argument, not a property of the model. `tests/exact.py` diagonalises the
same model in its full Hilbert space, which is how the sweep is checked on short chains.

A site comes from `tentorch.sites` — `spin(S)` — and carries the operator table the couplings
name. The model zoo lives in `tentorch/models` (Heisenberg, Ising and AKLT), and the notebooks
below write their own where the point is the Hamiltonian rather than the sweep.

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
  `tests/exact.py` — exact diagonalisation, the reference the sweep tests compare against.

## Status and limits

Currently the whole symmetry tensor layer is being rewritten thoroughly. If you wish to use this library, use the historical version on Github. 

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
