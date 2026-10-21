"""The model zoo: the chains the algorithms and the examples run on."""
from tentorch.models.aklt import aklt
from tentorch.models.heisenberg import heisenberg
from tentorch.models.ising import ising
from tentorch.models.uls import uls

__all__ = ["aklt", "heisenberg", "ising", "uls"]
