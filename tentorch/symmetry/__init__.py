'''The symmetry layer: the charge categories and their fusion trees.

``su2_symmetry``, ``su3_symmetry``, ``u1_symmetry``, ``z2_symmetry``, ``no_symmetry``,
``fermion_parity`` and ``fibonacci`` are the categories the engine contracts over.  Their
Python face — operator content and invariant data — is in ``tentorch.symmetry.cat``; the
fusion trees, their dense expansions and the block contractions over them are the C++
``tentorch/symmetry/tree`` layer.
'''
from tentorch.symmetry.cat.su3 import gell_mann

from tentorch._ext import (
    BraidingStyle,
    FusionCategory,
    FusionStyle,
    Group,
    ZNSymmetry,
    fermion_parity,
    fibonacci,
    no_symmetry,
    su2_symmetry,
    su3_symmetry,
    u1_symmetry,
    z2_symmetry,
)
