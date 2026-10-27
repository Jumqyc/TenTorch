'''
The compiled symmetry engine.

Exposes the categories with their legs and fusion trees, the block-sparse ``Tensor``, the
contraction ``contract``/``einsum``, the scalar products ``inner``/``norm``/``is_close`` and the
splittings ``svd``/``svdvals``/``qr``/``rq``.  The engine is compiled by ``tentorch/_ext/build.py``
into ``tentorch/_ext/build/symmetry_ext.so``; an import recompiles it when the shared object is
missing or older than one of the sources, the headers or the build script.

Everything here is the raw extension: the layers above build the user-facing objects on it and
reach the names a user does not need through this module.
'''

import importlib.util
import os
import sys
from typing import Any

# torch first: the extension hands `at::Tensor` back to Python, which needs torch's own type
# objects to already be in place
import torch  # noqa: F401
from tentorch._ext.build import (BUILD_DIR, GIB_PER_JOB, HEADERS, LIBRARY, SOURCES,
                                 available_gib, build_extension)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
# the build script is an input as well: it holds the list of sources
INPUTS = SOURCES + HEADERS + (os.path.join(HERE, 'build.py'),)

os.environ.setdefault('TORCH_EXTENSIONS_DIR', BUILD_DIR)
os.makedirs(BUILD_DIR, exist_ok = True)

_extension: Any
if os.path.exists(LIBRARY) and os.path.getmtime(LIBRARY) > max(
        os.path.getmtime(path) for path in INPUTS):
    _spec = importlib.util.spec_from_file_location('symmetry_ext', LIBRARY)
    if _spec is None or _spec.loader is None:
        raise ImportError(f'The extension at {LIBRARY} has no loader.')
    _extension = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_extension)
else:
    # the engine is older than one of its inputs, so this import has to compile it
    free = available_gib()
    if free < GIB_PER_JOB:
        stale = os.path.relpath(max(INPUTS, key = os.path.getmtime), ROOT)
        raise RuntimeError(
            f'{free:.1f} GiB of host memory is available but compiling the engine needs '
            f'{GIB_PER_JOB:.1f} GiB.  {stale} is newer than the built engine, so this import '
            f'has to recompile it: free some memory, then build with '
            f'`PYTHONPATH=. .venv/bin/python -u tentorch/_ext/build.py --jobs 1`.')
    _extension = build_extension(verbose = False, jobs = 1)

# the engine classes are pickled by name, so the module around them has to stay importable
sys.modules.setdefault('symmetry_ext', _extension)

_engine = _extension
"""The raw extension, for the layers that need a name this module does not re-export."""

Sector = tuple[int, ...]
"""One sector label: one integer per conserved charge of the group."""

# the categories
FusionCategory = _extension.FusionCategory
Group = _extension.Group
AbelianGroup = _extension.AbelianGroup
NoSymmetry = _extension.NoSymmetry
U1Symmetry = _extension.U1Symmetry
ZNSymmetry = _extension.ZNSymmetry
FermionParity = _extension.FermionParity
SU2Symmetry = _extension.SU2Symmetry
SU3Symmetry = _extension.SU3Symmetry
FibonacciCategory = _extension.FibonacciCategory
z_n = _extension.z_n
no_symmetry = _extension.no_symmetry
u1 = _extension.u1
fermion_parity = _extension.fermion_parity
su2 = _extension.su2
su3 = _extension.su3
fibonacci = _extension.fibonacci

# the legs, the trees and the tensors
Leg = _extension.Leg
FusionTree = _extension.FusionTree
Outcome = _extension.Outcome
BlockKey = _extension.BlockKey
Tensor = _extension.Tensor
validate = _extension.validate
reachable_charges = _extension.reachable_charges
fusion_trees = _extension.fusion_trees
block_keys = _extension.block_keys
codomain_legs = _extension.codomain_legs
domain_legs = _extension.domain_legs

# the contraction, the scalar products and the splittings
contract = _extension.contract
einsum = _extension.einsum
inner = _extension.inner
norm = _extension.norm
is_close = _extension.is_close
svd = _extension.svd
svdvals = _extension.svdvals
qr = _extension.qr
rq = _extension.rq

clear_caches = _extension.clear_caches
