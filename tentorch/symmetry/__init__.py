'''
The symmetry a tensor carries: the fusion categories, their legs and the fusion trees.

A leg is a list of sectors with a multiplicity each, so it is the space a bond or a site lives
in.  The categories are singletons of the compiled engine, one per conserved charge.
'''
from __future__ import annotations

from tentorch._ext import (
    AbelianGroup,
    BlockKey,
    FermionParity,
    FibonacciCategory,
    FusionCategory,
    FusionTree,
    Group,
    Leg,
    NoSymmetry,
    Outcome,
    SU2Symmetry,
    SU3Symmetry,
    Sector,
    U1Symmetry,
    ZNSymmetry,
    fermion_parity,
    fibonacci,
    no_symmetry,
    su2,
    su3,
    u1,
    z_n,
)

__all__ = [
    'AbelianGroup', 'BlockKey', 'FermionParity', 'FibonacciCategory', 'FusionCategory',
    'FusionTree', 'Group', 'Leg', 'NoSymmetry', 'Outcome', 'SU2Symmetry', 'SU3Symmetry', 'Sector',
    'U1Symmetry', 'ZNSymmetry', 'bond_leg', 'fermion_parity', 'fibonacci', 'no_symmetry', 'phys_leg',
    'su2', 'su3', 'trivial_leg', 'u1', 'z_n'
]


def phys_leg(cat: FusionCategory, sector: Sector, multiplicity: int = 1) -> Leg:
    '''
    The leg of one site, a single sector repeated `multiplicity` times.
    Args:
      cat: the category the leg is over.
      sector: the sector the site lives in, one label per conserved charge.
      multiplicity: how many copies of that sector the leg holds.
    Returns:
      The leg.
    '''
    return Leg(cat, [sector], (multiplicity,))


def bond_leg(cat: FusionCategory, sector: Sector, multiplicity: int = 1) -> Leg:
    '''
    A bond leg over one sector, the form a boundary bond of a state takes.
    Args:
      cat: the category the leg is over.
      sector: the charge the bond carries.
      multiplicity: how many copies of that charge the bond holds.
    Returns:
      The leg.
    '''
    return Leg(cat, [sector], (multiplicity,))


def trivial_leg(cat: FusionCategory, multiplicity: int = 1) -> Leg:
    '''
    The leg of a bond that carries the trivial charge.
    Args:
      cat: the category the leg is over.
      multiplicity: how many copies of the trivial charge the leg holds.
    Returns:
      The leg.
    '''
    return Leg(cat, [cat.trivial()], (multiplicity,))
