'''
The C++ symmetry engine of ``mpsdmrg``.

Exposes the charge groups with their legs and fusion trees, the block-sparse ``Tensor`` with
``cached_einsum``, and the ``svd``/``qr``/``rq``/``svdvals``/``norm``/``inner``/``trace``
splittings.  The engine is compiled by ``mpsdmrg/ext/build.py`` into
``mpsdmrg/ext/build/symmetry_ext.so``; an import recompiles it when the shared object
is missing or older than one of the sources, the headers or the build script.
'''
from __future__ import annotations

import importlib.util
import os
import sys
from types import SimpleNamespace
from typing import Any

from base import COMPLEX, CUDA
from mpsdmrg.ext.build import (BUILD_DIR, GIB_PER_JOB, HEADERS, LIBRARY, SOURCES,
                                available_gib, build_extension)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
# the build script is an input as well: it holds the list of sources
INPUTS = SOURCES + HEADERS + (os.path.join(HERE, 'build.py'),)

os.environ.setdefault('TORCH_EXTENSIONS_DIR', BUILD_DIR)
os.makedirs(BUILD_DIR, exist_ok=True)

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
            f'`PYTHONPATH=. .venv/bin/python -u mpsdmrg/ext/build.py --jobs 1`.')
    _extension = build_extension(verbose = False, jobs = 1)

# the engine allocates a fresh block buffer with the project's device and dtype
_extension.set_defaults(str(CUDA), str(COMPLEX))

# the engine classes are pickled by name, so the module around them has to stay importable
sys.modules.setdefault('symmetry_ext', _extension)

# the raw extension: the engine's own layers and the contract checks reach the names that are
# not part of the public surface through it, so the surface below stays what a user needs
_engine = _extension

# the public surface of the engine
Sector = tuple[int, ...]
"""One sector label: one integer per conserved charge of the group."""

BraidingStyle = _extension.BraidingStyle
FusionStyle = _extension.FusionStyle
FusionCategory = _extension.FusionCategory
Group = _extension.Group
Tensor = _extension.Tensor
bond_leg = _extension.bond_leg
capped_leg = _extension.capped_leg
fuse_legs = _extension.fuse_legs
multiplet_leg = _extension.multiplet_leg
site_leg = _extension.site_leg
ZNSymmetry = _extension.ZNSymmetry
cached_einsum = _extension.cached_einsum
dense_path = _extension.dense_path


def set_dense_path(enabled: bool) -> None:
    '''
    Force every contraction onto the dense route, for comparing the two routes.
    Args:
      enabled: True to route every contraction through the dense path, False to release it.
    Raises:
      NotImplementedError: when this build has no dense-path switch.
    '''
    if not hasattr(_extension, 'set_dense_path'):
        raise NotImplementedError('this build has no dense-path switch; rebuild with DENSEPATH=1')
    _extension.set_dense_path(enabled)
fermion_parity = _extension.fermion_parity
reset_route_stats = _extension.reset_route_stats
route_stats = _extension.route_stats
fibonacci = _extension.fibonacci
inner = _extension.inner
no_symmetry = _extension.no_symmetry
norm = _extension.norm
qr = _extension.qr
rq = _extension.rq
su2_symmetry = _extension.su2_symmetry
su3_symmetry = _extension.su3_symmetry
svd = _extension.svd
svdvals = _extension.svdvals
trace = _extension.trace
u1_symmetry = _extension.u1_symmetry
z2_symmetry = _extension.z2_symmetry

# ``mpsdmrg.tensor`` refers to ``_expr_cache`` by name to drop the contraction plans; the engine
# keeps its plans inside the extension, so the name drops every cache the engine keeps.
_expr_cache = SimpleNamespace(clear = _extension.clear_caches)


def _restore_group(name: str) -> Group:
    '''
    The group a pickled group object refers to.

    Args:
      name: ``name`` of the group, and its only identity: the groups of one
        symmetry type are singletons of the engine.

    Returns:
      The singleton of that name, or a fresh ``ZNSymmetry`` for a modulus without one.
    '''
    if name.startswith('Z') and name[1:].isdigit():
        modulus = int(name[1:])
        return getattr(_extension, f'z{modulus}_symmetry', None) or ZNSymmetry(modulus)
    singletons = {'NoSymmetry': no_symmetry, 'U(1)': u1_symmetry, 'SU(2)': su2_symmetry,
                  'SU(3)': su3_symmetry, 'FermionParity': fermion_parity,
                  'Fibonacci': fibonacci}
    if name not in singletons:
        raise ValueError(f'No group of the engine is named {name!r}.')
    return singletons[name]
