'''
The invariant initial state of a symmetric chain.
'''
from __future__ import annotations

import torch

from base import COMPLEX, CUDA
from mpsdmrg.symmetry import Group, Leg, Tensor


def cap_leg(group: Group, leg: Leg, max_dim: int) -> Leg:
    '''
    The leg cut down to at most ``max_dim`` states, keeping the lowest sectors.

    Args:
        group: the symmetry group of the leg.
        leg: the leg to cap, as returned by ``Leg.fuse``; its sectors are ordered by
            increasing charge, so the low-energy multiplets are the ones kept.
        max_dim: the largest dimension to keep; a leg that already fits is returned.
    Returns:
        A leg with the sectors of ``leg`` in the same order, each keeping as many of its
        copies as fit below ``max_dim``, and the same orientation.
    '''
    if leg.dim <= max_dim:
        return leg
    entries: list[tuple[tuple[int, ...], int]] = []
    total = 0
    for qi in range(leg.nblocks):
        room = (max_dim - total) // group.sector_dim(leg.sectors[qi])
        keep = min(leg.multiplicity(qi), room)
        if keep:
            size = keep * group.sector_dim(leg.sectors[qi])
            entries.append((leg.sectors[qi], size))
            total += size
    return Leg.from_blocks(group, entries, leg.qconj)[0]


def invariant_state(group: Group, length: int, phys_dim: int, max_bond_dim: int,
                    dtype: torch.dtype = COMPLEX, device: torch.device = CUDA,
                    ) -> tuple[list[Tensor], list[int]]:
    '''
    One singlet tensor per site of a chain with a symmetry group.

    The physical leg carries the sector of dimension ``phys_dim``, every tensor fuses its
    two bond legs and the physical leg to the trivial sector, and each bond carries every
    sector the physical legs to its left fuse to, capped at ``max_bond_dim``, so the state
    is a singlet and the sweep starts from a bond dimension it can afford.

    Args:
        group: the symmetry group of the chain.
        length: number of sites.
        phys_dim: dimension of one physical leg.
        max_bond_dim: largest bond dimension of the initial state.
        dtype: dtype of the tensors.
        device: device of the tensors.
    Returns:
        The tensors and the dimension of every bond, left boundary first.
    Raises:
        ValueError: If the chain of physical legs cannot close to the trivial sector, which
            is what an odd spin-1/2 chain and a spin-1 chain of a length that is not a
            multiple of three do.
    '''
    phys = Leg(group, (group.sector_of_dimension(phys_dim),), (0, phys_dim), qconj=1)
    bonds = [Leg.trivial(1, group)]
    for _ in range(length - 1):
        bonds.append(cap_leg(group, bonds[-1].fuse(phys), max_bond_dim))
    if not any(tuple(group.trivial_sector) in group.fusion_outcomes(sector, phys.sectors[0])
               for sector in bonds[-1].sectors):
        raise ValueError(
            f'A bond of {bonds[-1].dim} states cannot close {length} sites of dimension '
            f'{phys_dim} to the trivial sector of {group.group_name}.')
    bonds.append(Leg.trivial(1, group))
    tensors = [Tensor.rand((bonds[i].dim, phys.dim, bonds[i + 1].dim),
                           legs=(bonds[i], phys, bonds[i + 1].conj()),
                           qtotal=group.trivial_sector,
                           dtype=dtype, device=device) for i in range(length)]
    return tensors, [bond.dim for bond in bonds]


def initial_state(group: Group | None, length: int, phys_dim: int, max_bond_dim: int,
                  dtype: torch.dtype = COMPLEX, device: torch.device = CUDA,
                  ) -> tuple[list[Tensor], list[int]] | None:
    '''
    The default state of a chain, or None when the caller has to supply one.

    A group whose charges fuse in more than one way gets the invariant singlet of
    :func:`invariant_state`; an abelian group and no group at all get nothing, because the
    bond charges of such a chain belong to the model rather than to the symmetry.

    Args:
        group: the symmetry group of the chain, or None for no symmetry.
        length: number of sites.
        phys_dim: dimension of one physical leg.
        max_bond_dim: largest bond dimension of the initial state.
        dtype: dtype of the tensors.
        device: device of the tensors.
    Returns:
        ``(tensors, bond_dims)`` of one singlet tensor per site, or None when the group has
        no default state.
    Raises:
        ValueError: If the chain of physical legs cannot close to the trivial sector.
    '''
    if group is None or group.is_abelian:
        return None
    return invariant_state(group, length, phys_dim, max_bond_dim, dtype, device)
