from typing import Iterator, Optional, Sequence

from tentorch._ext import Leg, Tensor
from tentorch.symmetry import FusionCategory, Sector


class MPO:
    '''
    A matrix product operator over a chain of sites.
    Args:
      sites: the site tensors, each over the codomain (bond left, out) and the domain (in, bond
        right).
      physical: the physical leg of one site, shared by every site.
      cat: the category of the legs.
      sector: the total charge of the operator, the trivial charge when it is None.
      dtype: the scalar type, the type of the category when it is None.
      device: the device, the device of the category when it is None.
    Raises:
      ValueError: when no site is given, or when a site does not have four legs.
    '''

    def __init__(self, sites: Sequence[Tensor], physical: Leg, cat: FusionCategory,
                 sector: Optional[Sector] = None, dtype = None, device = None) -> None:
        self.tensors = list(sites)
        if not self.tensors:
            raise ValueError('an operator needs at least one site')
        for at, site in enumerate(self.tensors):
            if site.nlegs() != 4:
                raise ValueError(f'site {at} has {site.nlegs()} legs, not the four of an operator')
        self.L = len(self.tensors)
        self.cat = cat
        self.physical = physical
        self.sector = cat.trivial() if sector is None else sector
        self.dtype = cat.dtype if dtype is None else dtype
        self.device = cat.device if device is None else device
        self.bond_dim = [site.codomain[0].multiplets() for site in self.tensors]
        self.bond_dim.append(self.tensors[-1].domain[1].multiplets())

    @property
    def physical_dim(self) -> int:
        '''
        The number of states of one physical leg.
        Returns:
          The sum of the multiplicities of the leg, which is its number of rows in a block.
        '''
        return sum(self.physical.multiplicity(block) for block in range(self.physical.nblocks()))

    def __len__(self) -> int:
        '''
        The number of sites.
        Returns:
          The length of the chain.
        '''
        return self.L

    def __iter__(self) -> Iterator[Tensor]:
        '''
        The site tensors.
        Returns:
          An iterator over the chain.
        '''
        return iter(self.tensors)

    def __getitem__(self, at: int) -> Tensor:
        '''
        One site tensor.
        Args:
          at: the position of the site.
        Returns:
          The site tensor.
        '''
        return self.tensors[at]

    def __repr__(self) -> str:
        '''
        A line naming the chain.
        Returns:
          The description.
        '''
        return (f'MPO(L={self.L}, physical_dim={self.physical_dim}, '
                f'bond_dim={self.bond_dim[1: self.L]})')
