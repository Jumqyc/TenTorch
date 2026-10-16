'''
The SU(3) multiplet content of an invariant Hamiltonian.

The tensor operators of one site and the invariant two-site channels they generate; the
builder that places them lives in :mod:`mpsdmrg.mpo` and asks here for the
content of a site.
'''
from __future__ import annotations

import torch

from mpsdmrg.ext import Group, Sector
# the type name of the group whose multiplets this module carries
GROUP = "SU3Symmetry"


def tensor_operators(group: Group, mapping: dict[str, torch.Tensor], dim: int,
                     dtype: torch.dtype, device: torch.device) -> dict[Sector, torch.Tensor]:
    r'''
    The tensor operators of one site, one multiplet per irrep of its operator content.

    A site in the irrep :math:`V` carries the operators of the irreps in :math:`V \otimes
    \bar V`; the invariant of :math:`V_s \otimes V \otimes V` reshaped to (dim s, d, d) holds the
    components of the multiplet :math:`T^{(s)}_m` in its first axis.

    Args:
        group: the symmetry group; it needs a Clebsch-Gordan tensor, so it must be non-abelian.
        mapping: operator names to their dense matrices; unused by SU(3).
        dim: physical dimension of one site.
        dtype, device: data type and device of the matrices.
    Returns:
        dict sector -> tensor in shape (dim s, d, d), the :math:`T^{(s)}_m` of that multiplet.
    Raises:
        NotImplementedError: If the group is abelian, so it carries no multiplet structure.
    '''
    if group.is_abelian:
        raise NotImplementedError(
            f'The tensor operators need a group with a multiplet structure, not {group.group_name}.')
    site = group.sector_of_dimension(dim)
    # the vertex already carries the leg bases, so its component is the operator on the site
    return {sector: group.fusion_tensor(sector, site, site, device=device).to(dtype)
            for sector in group.fusion_outcomes(site, group.dual_sector(site))}


def channel_operators(group: Group, operators: dict[Sector, torch.Tensor], dim: int,
                      ) -> dict[Sector, torch.Tensor]:
    r'''
    The invariant two-site operator of every multiplet, :math:`C_s = \sum_{m m'} D^s_{m m'}
    T^{(s)}_m \otimes T^{(s)}_{m'}`, keyed by the sector its multiplet lives in.

    The pairing :math:`D^s` is the invariant form of the multiplet, the conjugate basis the group
    hands out, so the contraction is the trivial channel of :math:`V_s \otimes V_s`.

    Args:
        group: the symmetry group; it must define the conjugate basis of its multiplets.
        operators: the site tensor operators, as :func:`tensor_operators` returns them.
        dim: physical dimension of one site.
    Returns:
        dict sector -> tensor in shape (d^2, d^2), the invariant channel of that multiplet.
    '''
    channels: dict[Sector, torch.Tensor] = {}
    for sector, tensor in operators.items():
        form = group.invariant_form(sector, tensor.device).to(tensor.dtype)
        # kron(T_m, T_m') has its rows in site order, so the physical axes are (a, b, c, e)
        total = torch.einsum('mn,mac,nbe->abce', form, tensor, tensor)
        channels[sector] = total.reshape(dim * dim, dim * dim)
    return channels
