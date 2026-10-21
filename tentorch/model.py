'''The physics of a chain: the sites, the couplings and the Hamiltonian operator they build.

A ``Model`` holds no state and no numerical options; an algorithm builds the state it optimizes
and leaves it on its result.
'''
import numpy as np
from numpy.typing import NDArray

from tentorch.config import COMPLEX, CUDA
from tentorch.networks.mpo import MPO
from tentorch.sites import Site
from tentorch.symmetry import FusionCategory


class Model:
    '''
    ``L`` sites of one kind, and the Hamiltonian written on them as a list of couplings.

    Args:
        L: number of sites.
        site: the local Hilbert space, from ``tentorch.sites``.
        dtype, device: data type and device of the tensors.
    Raises:
        ValueError: If ``L`` is not positive.
    '''
    def __init__(self,
                 L: int,
                 site: Site,
                 dtype = COMPLEX,
                 device = CUDA):
        if L < 1:
            raise ValueError(f'A chain needs at least one site, not {L}.')
        self.L = int(L)
        self.site = site
        self.dtype = dtype
        self.device = device
        self.couplings: list[tuple[tuple[str, ...], complex, tuple[int, ...]]] = []
        self._mpo: MPO | None = None

    @property
    def cat(self) -> FusionCategory:
        '''The fusion category the sites transform under.'''
        return self.site.cat

    @property
    def sector(self) -> tuple[int, ...] | None:
        '''The multiplet one site lives in, ``None`` for a category that needs none.'''
        return self.site.sector

    @property
    def physical_dim(self) -> int:
        '''Dimension of the local Hilbert space of one site.'''
        return self.site.phys_dim

    def add_couplings(self,
                      operators: str | tuple[str, ...],
                      coeff: complex,
                      locations: int | NDArray[np.int64] | tuple[int, ...]) -> None:
        '''
        Add one coupling to the Hamiltonian, as ``MPO.add_couplings`` does.

        Args:
            operators: the operator names, one per location; a single name may be given as a
                plain string, and a name of several letters is the product of those site
                operators on one site, e.g. ``'XY'`` is :math:`S_x S_y`.
            coeff: coefficient of the coupling.
            locations: the site indices the operators act on; one integer for a single site.
        '''
        self.couplings.append((operators if isinstance(operators, tuple) else (operators,),
                               coeff, locations if isinstance(locations, tuple) else (locations,)))

    @property
    def mpo(self) -> MPO:
        '''The Hamiltonian as a matrix product operator, built on first use from the couplings.'''
        if self._mpo is None:
            mpo = MPO(self.L, self.physical_dim, mapping = self.site.operators,
                      dtype = self.dtype, device = self.device,
                      cat = self.cat, site_sector = self.sector)
            for operators, coeff, locations in self.couplings:
                mpo.add_couplings(operators, coeff, locations)
            mpo.build()
            self._mpo = mpo
        return self._mpo

    def copy(self) -> 'Model':
        '''A deep copy that shares nothing with this model, Hamiltonian included.'''
        new = Model(self.L, self.site, dtype = self.dtype, device = self.device)
        new.couplings = list(self.couplings)
        return new

    def __repr__(self) -> str:
        '''The length, the sites and the number of couplings.'''
        return (f'Model(L = {self.L}, site = {self.site!r}, '
                f'couplings = {len(self.couplings)})')
