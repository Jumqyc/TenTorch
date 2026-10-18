"""How a group acts on the operators of one site.

A site carries the operators of the irreps in the product of its own irrep with its conjugate.
The content of a group states which multiplets carry them and which invariant channel a pair of
multiplets forms, so the builder of an invariant Hamiltonian reads the action of the group
instead of inferring it, and a further group adds a class instead of a branch.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from mpsdmrg.ext import Group, Sector


class OperatorContent(ABC):
    '''
    The action of a group on the operators of one site.

    A multiplet is normalized so that its channel satisfies :math:`|C_s|^2 = \\dim(s)`, the
    convention of the engine's Clebsch-Gordan tensors, and a unit placement of the engine's
    fusion tree realizes :math:`C_s / \\dim(s)`; only the Frobenius-Schur phase of the sector is
    left to :meth:`placement_phase`.
    '''

    @abstractmethod
    def tensor_operators(self, group: Group, mapping: dict[str, torch.Tensor], phys: Any,
                         dtype: torch.dtype, device: torch.device
                         ) -> dict[Sector, torch.Tensor]:
        r'''
        The multiplet basis of one site, keyed by the sector of the multiplet.

        Args:
            group: the symmetry group; it needs a Clebsch-Gordan tensor, so it must be non-abelian.
            mapping: operator names to their dense matrices, the spin components among them.
            phys: the physical leg of one site, in the multiplet the site lives in.
            dtype, device: data type and device of the matrices.
        Returns:
            dict sector -> tensor in shape (dim s, d, d), the components of the multiplet.
        '''

    def channel(self, group: Group, sector: Sector, tensor: torch.Tensor) -> torch.Tensor:
        r'''
        The invariant two-site operator of a multiplet, :math:`C_s = \sum_{m m'} D^s_{m m'}
        T_m \otimes T_{m'}` with the invariant form :math:`D^s` of the group.

        Args:
            group: the symmetry group; it must define the conjugate basis of its multiplets.
            sector: the sector the multiplet lives in.
            tensor: the multiplet, in shape (dim s, d, d).
        Returns:
            The channel in shape (d^2, d^2).
        '''
        form = group.invariant_form(sector, tensor.device).to(tensor.dtype)
        # kron(T_m, T_m') has its rows in site order, so the physical axes are (a, b, c, e)
        return torch.einsum('mn,mac,nbe->abce', form, tensor, tensor).reshape(
            tensor.shape[1] ** 2, tensor.shape[1] ** 2)

    def normalize(self, group: Group, sector: Sector, tensor: torch.Tensor) -> torch.Tensor:
        '''
        Scale a multiplet so that its channel satisfies ``|C_s|^2 = dim(s)``.

        Args:
            group: the symmetry group.
            sector: the sector the multiplet lives in.
            tensor: the multiplet, in shape (dim s, d, d).
        Returns:
            The multiplet at the engine's scale, in shape (dim s, d, d).
        '''
        # the channel is quadratic in the multiplet, so its squared norm is quartic
        return tensor * (group.sector_dim(sector)
                         / self.channel(group, sector, tensor).abs().pow(2).sum()).pow(0.25)

    def placement_phase(self, group: Group, sector: Sector) -> complex:
        '''
        The phase the closing placement of a multiplet carries.

        Args:
            group: the symmetry group.
            sector: the sector the multiplet lives in.
        Returns:
            The Frobenius-Schur phase of the sector.
        '''
        return complex(group.dual_phase(sector))
