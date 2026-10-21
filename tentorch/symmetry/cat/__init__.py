"""
The multiplet content of a group.

One folder per group that carries multiplets, each holding its C++ core and the class that
answers for it; the builder of an invariant Hamiltonian asks here for the content of its group,
so a further group adds a folder instead of a branch.
"""
from __future__ import annotations

from tentorch._ext import (  # noqa: F401
    BraidingStyle, FusionStyle, Group, Sector, ZNSymmetry,
    fermion_parity, no_symmetry, su2_symmetry, su3_symmetry,
    u1_symmetry, z2_symmetry)
from tentorch.symmetry.cat.content import OperatorContent
from tentorch.symmetry.cat.su2 import SU2Content
from tentorch.symmetry.cat.su3 import SU3Content

# the content that answers for a group, keyed by the type name of its group
_CONTENTS: dict[str, OperatorContent] = {'SU2Symmetry': SU2Content(),
                                         'SU3Symmetry': SU3Content()}


def has_content(group: Group) -> bool:
    '''
    Whether the package implements the operator content of a group.

    Args:
        group: the symmetry group.
    Returns:
        True if a content class answers for the type of the group.
    '''
    return group.type_name in _CONTENTS


def content(group: Group) -> OperatorContent:
    '''
    The content that answers how a group acts on the operators of one site.

    Args:
        group: the symmetry group.
    Returns:
        The content holding ``tensor_operators``, ``channel`` and ``placement_phase``.
    Raises:
        NotImplementedError: If no content claims the type of the group.
    '''
    if not has_content(group):
        raise NotImplementedError(f'The multiplets of {group.name} have no content yet.')
    return _CONTENTS[group.type_name]
