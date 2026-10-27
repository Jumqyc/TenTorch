from dataclasses import dataclass
from typing import Optional

from tentorch._ext import FusionCategory, Leg, Sector, u1

@dataclass
class Model:
    '''
    A chain and the couplings written on it. Currently unfinished.

    Args:
      length: the number of sites.
      coupling: the coefficient in front of every nearest neighbour term.
      cat: the category of the charges.
      physical: the physical leg of one site, the spin one half leg of `cat` when it is None.
      sector: the charge of the state, the trivial charge when it is None.
      state: the state the last run left behind, and None before one runs.
      energy: the variational energy the last run left behind, and None before one runs.
    '''

    length: int
    coupling: complex = 1.0
    cat: FusionCategory = u1
    physical: Optional[Leg] = None
    sector: Optional[Sector] = None
    state: Optional[object] = None
    energy: Optional[float] = None

    def __post_init__(self) -> None:
        '''Fill in the charge the model carries when the caller leaves it open.'''
        if self.length < 1:
            raise ValueError(f'a chain needs at least one site, not {self.length}')
        if self.sector is None:
            self.sector = self.cat.trivial()
