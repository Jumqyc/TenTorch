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
from mpsdmrg.ext.build import (BUILD_DIR, HEADERS, LIBRARY, PCH, SOURCES, build_pch,
                                clear_stale_lock)
HERE = os.path.dirname(os.path.abspath(__file__))
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
    # the build needs ninja on PATH, which a bare .venv/bin/python does not add
    os.environ['PATH'] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get('PATH', '')
    clear_stale_lock(verbose = False)
    build_pch(verbose = False)
    from torch.utils.cpp_extension import load
    _extension = load(
        name = 'symmetry_ext',
        sources = list(SOURCES),
        extra_cflags = ['-O3', '-include', PCH],
        build_directory = BUILD_DIR,
        verbose = False)

# the engine allocates a fresh block buffer with the project's device and dtype
_extension.set_defaults(str(CUDA), str(COMPLEX))

# the engine classes are pickled by name, so the module around them has to stay importable
sys.modules.setdefault('symmetry_ext', _extension)

# the public names of the engine, re-exported so ``mpsdmrg.symmetry`` can import them
Sector = tuple[int, ...]
"""One sector label: one integer per conserved charge of the group."""

AbelianGroup = _extension.AbelianGroup
BraidingStyle = _extension.BraidingStyle
FermionParity = _extension.FermionParity
FusionStyle = _extension.FusionStyle
FusionTree = _extension.FusionTree
Group = _extension.Group
Leg = _extension.Leg
NoSymmetry = _extension.NoSymmetry
SU2Symmetry = _extension.SU2Symmetry
SU3Symmetry = _extension.SU3Symmetry
Tensor = _extension.Tensor
U1Symmetry = _extension.U1Symmetry
ZNSymmetry = _extension.ZNSymmetry
cached_einsum = _extension.cached_einsum
clear_caches = _extension.clear_caches
dual_sign = _extension.dual_sign
fermion_parity = _extension.fermion_parity
fusion_table = _extension.fusion_table
fusion_tensor = _extension.fusion_tensor
fusion_trees = _extension.fusion_trees
inner = _extension.inner
no_symmetry = _extension.no_symmetry
norm = _extension.norm
qr = _extension.qr
recoupling = _extension.recoupling
rq = _extension.rq
stored_dense = _extension.stored_dense
su2_symmetry = _extension.su2_symmetry
su3_casimir = _extension.su3_casimir
su3_clebsch = _extension.su3_clebsch
su3_clebsch_copies = _extension.su3_clebsch_copies
su3_dimension = _extension.su3_dimension
su3_fusion = _extension.su3_fusion
su3_metric = _extension.su3_metric
su3_symmetry = _extension.su3_symmetry
svd = _extension.svd
svdvals = _extension.svdvals
trace = _extension.trace
u1_symmetry = _extension.u1_symmetry
z2_symmetry = _extension.z2_symmetry
z3_symmetry = _extension.z3_symmetry
z4_symmetry = _extension.z4_symmetry
z5_symmetry = _extension.z5_symmetry
z6_symmetry = _extension.z6_symmetry
z7_symmetry = _extension.z7_symmetry
z8_symmetry = _extension.z8_symmetry
z9_symmetry = _extension.z9_symmetry

# ``mpsdmrg.tensor`` refers to ``_expr_cache`` by name to drop the contraction plans; the engine
# keeps its plans inside the extension, so the name drops every cache the engine keeps.
_expr_cache = SimpleNamespace(clear = clear_caches)


def _restore_group(group_name: str) -> Group:
    '''
    The group a pickled group object refers to.

    Args:
      group_name: ``group_name`` of the group, and its only identity: the groups of one
        symmetry type are singletons of the engine.

    Returns:
      The singleton of that name, or a fresh ``ZNSymmetry`` for a modulus without one.
    '''
    if group_name.startswith('Z') and group_name[1:].isdigit():
        modulus = int(group_name[1:])
        return globals().get(f'z{modulus}_symmetry') or ZNSymmetry(modulus)
    return {'NoSymmetry': no_symmetry, 'U(1)': u1_symmetry, 'SU(2)': su2_symmetry,
            'FermionParity': fermion_parity}[group_name]
