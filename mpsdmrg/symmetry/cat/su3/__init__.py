'''
The SU(3) multiplet content of an invariant Hamiltonian.

The tensor operators of one site, as the action of SU(3) on the operators of a site.
'''
from __future__ import annotations

from typing import Any

import torch

from mpsdmrg.ext import Group, Sector
from mpsdmrg.symmetry.cat.content import OperatorContent


def gell_mann(dtype: torch.dtype = torch.complex128,
              device: torch.device | None = None) -> list[torch.Tensor]:
    r'''
    The eight Hermitian Gell-Mann generators of the defining representation.

    Args:
        dtype: data type of the returned matrices.
        device: device of the returned matrices; ``None`` keeps the default device.

    Returns:
        The eight 3x3 generators, normalized so that :math:`\mathrm{tr}(\lambda_a
        \lambda_b) = 2 \delta_{ab}`: the seven off-diagonal and diagonal ones as given and
        the eighth divided by :math:`\sqrt{3}`.
    '''
    raw = [
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
        [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, -2]],
    ]
    return [torch.tensor(matrix, dtype=dtype, device=device) / (1.0 if index < 7 else 3 ** 0.5)
            for index, matrix in enumerate(raw)]


class SU3Content(OperatorContent):
    r'''
    The action of SU(3) on the operators of a site.

    A site in the irrep :math:`V` carries the operators of the irreps in :math:`V \otimes
    \bar V`; the invariant of :math:`V_s \otimes V \otimes V` reshaped to (dim s, d, d) holds the
    components of the multiplet :math:`T^{(s)}_m` in its first axis.
    '''

    def tensor_operators(self, group: Group, mapping: dict[str, torch.Tensor], phys: Any,
                         dtype: torch.dtype, device: torch.device) -> dict[Sector, torch.Tensor]:
        r'''
        The multiplets of one site, keyed by the sector of the multiplet.

        Args:
            group: the symmetry group; it needs a Clebsch-Gordan tensor, so it must be non-abelian.
            mapping: operator names to their dense matrices; unused by SU(3).
            phys: the physical leg of one site, whose multiplet is the irrep the site lives in.
            dtype, device: data type and device of the matrices.
        Returns:
            dict sector -> tensor in shape (dim s, d, d), the :math:`T^{(s)}_m` of that multiplet.
        Raises:
            NotImplementedError: If the group is abelian, so it carries no multiplet structure.
        '''
        if group.is_abelian:
            raise NotImplementedError(
                f'The tensor operators need a group with a multiplet structure, not {group.name}.')
        site = phys.sectors[0]
        # the vertex already carries the leg bases, so its component is the operator on the site
        components = {sector: group.fusion_tensor(sector, site, site, device=device).to(dtype)
                      for sector in group.fusion_outcomes(site, group.dual_sector(site))}
        return {sector: self.normalize(group, sector, tensor)
                for sector, tensor in components.items()}
