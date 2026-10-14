from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from math import inf

import torch

from mpsdmrg.symmetry.data.su2data import fusion_tensor as su2_fusion_tensor

Sector = tuple[int, ...]
"""One sector label: one integer per conserved charge of the group."""


class FusionStyle(Enum):
    """How many outcomes the fusion of two sectors has."""
    single = 0
    """One outcome, :math:`a \\otimes b = c`: the group is abelian."""
    multiple_unique = 10
    """Several outcomes, each with multiplicity at most one."""


class BraidingStyle(Enum):
    """How sectors behave when two legs carrying them are exchanged."""
    bosonic = 0
    fermionic = 10


class Group(ABC):
    '''
    Algebra of a symmetry group: sector labels, duality, fusion and quantum dimension.

    Args:
        fusion_style: how many outcomes the fusion of two sectors has.
        braiding_style: how sectors behave under exchange.
        trivial_sector: the trivial sector, in the same representation as every sector.
        group_name: short name, used in messages.
        num_sectors: number of sectors, or ``math.inf`` for infinitely many.
    '''

    def __init__(self,
                 fusion_style: FusionStyle,
                 braiding_style: BraidingStyle,
                 trivial_sector: Sector,
                 group_name: str,
                 num_sectors: int | float):
        self.fusion_style = fusion_style
        self.braiding_style = braiding_style
        self.trivial_sector = tuple(int(q) for q in trivial_sector)
        self.group_name = group_name
        self.num_sectors = num_sectors
        self.sector_ind_len = len(self.trivial_sector)

    @property
    def is_abelian(self) -> bool:
        '''Whether the fusion of two sectors has a single outcome.'''
        return self.fusion_style is FusionStyle.single

    @abstractmethod
    def is_valid_sector(self, a: Sector) -> bool:
        '''
        Whether ``a`` is a sector of this group.
        Args:
            a: the candidate sector.
        '''

    @abstractmethod
    def dual_sector(self, a: Sector) -> Sector:
        '''
        The sector dual to ``a``, with :math:`N^{a,\\bar a}_u = 1`.
        Args:
            a: a sector of this group.
        '''

    @abstractmethod
    def fusion_outcomes(self, a: Sector, b: Sector) -> tuple[Sector, ...]:
        '''
        The distinct sectors that ``a`` and ``b`` fuse to, each listed once.
        Args:
            a: a sector of this group.
            b: a sector of this group.
        '''

    @abstractmethod
    def sector_dim(self, a: Sector) -> int:
        '''
        Quantum dimension of ``a``.
        Args:
            a: a sector of this group.
        '''

    @abstractmethod
    def is_same_symmetry(self, other: Group) -> bool:
        '''
        Whether ``other`` is the same group as this one.
        Args:
            other: the group to compare with.
        '''

    def fuse(self, *sectors: Sector) -> Sector:
        '''
        Fuse sectors into the single outcome of an abelian group.
        Args:
            sectors: the sectors to fuse, none for the trivial sector.
        Returns:
            The fused sector.
        Raises:
            ValueError: If the group is not abelian.
        '''
        if not self.is_abelian:
            raise ValueError(f'Group {self.group_name} is not abelian; fusion has no single outcome.')
        fused = self.trivial_sector
        for a in sectors:
            fused = self.fusion_outcomes(fused, a)[0]
        return fused

    def can_fuse_to(self, a: Sector, b: Sector, c: Sector) -> bool:
        '''
        Whether ``a`` and ``b`` can fuse to ``c``.
        Args:
            a: a sector of this group.
            b: a sector of this group.
            c: the candidate outcome.
        '''
        return c in self.fusion_outcomes(a, b)

    def n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        '''
        Multiplicity :math:`N^{ab}_c` of the fusion of ``a`` and ``b`` to ``c``.
        Args:
            a: a sector of this group.
            b: a sector of this group.
            c: the candidate outcome.
        '''
        return 1 if self.can_fuse_to(a, b, c) else 0

    def fusion_tensor(self, a: Sector, b: Sector, c: Sector, device: torch.device) -> torch.Tensor:
        '''
        Structural tensor :math:`X^{ab}_c` of one fusion vertex, one block per channel.

        The amplitudes are the Clebsch-Gordan coefficients in the basis of the group, so
        :math:`X^{ab}_c` has shape ``(sector_dim(a), sector_dim(b), sector_dim(c))`` and is
        contracted at every vertex of a fusion tree.

        Args:
            a: the first fused sector.
            b: the second fused sector.
            c: the coupled sector, an outcome of fusing ``a`` and ``b``.
            device: the device of the returned tensor.
        Returns:
            The structural tensor, real float64, in shape ``(dim a, dim b, dim c)``.
        Raises:
            NotImplementedError: If the group has no structural tensor.
        '''
        raise NotImplementedError(f'Group {self.group_name} has no Clebsch-Gordan tensor.')

    def all_sectors(self) -> tuple[Sector, ...]:
        '''
        All sectors of the group, in order.
        Raises:
            ValueError: If the group has infinitely many sectors.
            NotImplementedError: If the group does not enumerate its sectors.
        '''
        if self.num_sectors == inf:
            raise ValueError(f'Group {self.group_name} has infinitely many sectors.')
        raise NotImplementedError(f'Group {self.group_name} does not enumerate its sectors.')

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Group):
            return NotImplemented
        return self.is_same_symmetry(other)

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.group_name))

    def __repr__(self) -> str:
        return self.group_name


class AbelianGroup(Group):
    '''A group whose fusion of two sectors has a single outcome.'''

    def __init__(self,
                 trivial_sector: Sector,
                 group_name: str,
                 num_sectors: int | float,
                 braiding_style: BraidingStyle = BraidingStyle.bosonic):
        super().__init__(FusionStyle.single, braiding_style, trivial_sector, group_name, num_sectors)

    def sector_dim(self, a: Sector) -> int:
        '''Every sector of an abelian group has quantum dimension one.'''
        return 1


class NoSymmetry(AbelianGroup):
    """No symmetry: one sector, so every charge block is dense."""

    def __init__(self):
        super().__init__((0,), 'NoSymmetry', 1)

    def is_valid_sector(self, a: Sector) -> bool:
        return a == (0,)

    def dual_sector(self, a: Sector) -> Sector:
        return a

    def fusion_outcomes(self, a: Sector, b: Sector) -> tuple[Sector, ...]:
        return (a,)

    def is_same_symmetry(self, other: Group) -> bool:
        return type(other) is NoSymmetry

    def all_sectors(self) -> tuple[Sector, ...]:
        return ((0,),)


class U1Symmetry(AbelianGroup):
    """Particle number or :math:`S^z`, with integer charges added on fusion."""

    def __init__(self):
        super().__init__((0,), 'U(1)', inf)

    def is_valid_sector(self, a: Sector) -> bool:
        return len(a) == 1

    def dual_sector(self, a: Sector) -> Sector:
        return (-a[0],)

    def fusion_outcomes(self, a: Sector, b: Sector) -> tuple[Sector, ...]:
        return ((a[0] + b[0],),)

    def is_same_symmetry(self, other: Group) -> bool:
        return type(other) is U1Symmetry


class ZNSymmetry(AbelianGroup):
    r""":math:`\mathbb{Z}_N`, with charges added modulo ``N``."""

    def __init__(self, N: int):
        if N <= 0:
            raise ValueError(f'The modulus of a Z_N group must be positive, not {N}.')
        self.N = int(N)
        super().__init__((0,), f'Z{self.N}', self.N)

    def is_valid_sector(self, a: Sector) -> bool:
        return len(a) == 1 and 0 <= a[0] < self.N

    def dual_sector(self, a: Sector) -> Sector:
        return ((-a[0]) % self.N,)

    def fusion_outcomes(self, a: Sector, b: Sector) -> tuple[Sector, ...]:
        return (((a[0] + b[0]) % self.N,),)

    def is_same_symmetry(self, other: Group) -> bool:
        return isinstance(other, ZNSymmetry) and other.N == self.N

    def all_sectors(self) -> tuple[Sector, ...]:
        return tuple((a,) for a in range(self.N))


class SU2Symmetry(Group):
    """SU(2) spin, with sectors ``(jj,)`` labelling spin ``jj / 2``."""

    def __init__(self):
        super().__init__(FusionStyle.multiple_unique, BraidingStyle.bosonic, (0,), 'SU(2)', inf)

    def is_valid_sector(self, a: Sector) -> bool:
        return len(a) == 1 and a[0] >= 0

    def dual_sector(self, a: Sector) -> Sector:
        return a

    def fusion_outcomes(self, a: Sector, b: Sector) -> tuple[Sector, ...]:
        return tuple((jj,) for jj in range(abs(a[0] - b[0]), a[0] + b[0] + 1, 2))

    def sector_dim(self, a: Sector) -> int:
        return a[0] + 1

    def fusion_tensor(self, a: Sector, b: Sector, c: Sector, device: torch.device) -> torch.Tensor:
        '''The SU(2) Clebsch-Gordan coefficients of one fusion vertex.'''
        return su2_fusion_tensor(a[0], b[0], c[0], device)

    def is_same_symmetry(self, other: Group) -> bool:
        return type(other) is SU2Symmetry


class FermionParity(AbelianGroup):
    """Fermion parity: the same fusion as :math:`\\mathbb{Z}_2`, with fermionic braiding."""

    def __init__(self):
        super().__init__((0,), 'FermionParity', 2, braiding_style=BraidingStyle.fermionic)

    def is_valid_sector(self, a: Sector) -> bool:
        return a in ((0,), (1,))

    def dual_sector(self, a: Sector) -> Sector:
        return a

    def fusion_outcomes(self, a: Sector, b: Sector) -> tuple[Sector, ...]:
        return (((a[0] + b[0]) % 2,),)

    def is_same_symmetry(self, other: Group) -> bool:
        return type(other) is FermionParity

    def all_sectors(self) -> tuple[Sector, ...]:
        return ((0,), (1,))


no_symmetry = NoSymmetry()
"""The trivial group, the default symmetry of a tensor."""

u1_symmetry = U1Symmetry()
"""The U(1) group of integer charges."""

z2_symmetry = ZNSymmetry(2)
z3_symmetry = ZNSymmetry(3)
z4_symmetry = ZNSymmetry(4)
z5_symmetry = ZNSymmetry(5)
z6_symmetry = ZNSymmetry(6)
z7_symmetry = ZNSymmetry(7)
z8_symmetry = ZNSymmetry(8)
z9_symmetry = ZNSymmetry(9)

su2_symmetry = SU2Symmetry()
"""The SU(2) group of spin sectors."""

fermion_parity = FermionParity()
"""Fermion parity, with fermionic braiding."""
