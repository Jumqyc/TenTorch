"""Checks of the Fibonacci category: its fusion data against the published tables.

The category has no Clebsch-Gordan tensor, so the dense route cannot serve as an oracle for it.
What is checked here is the data the categorical route reads: the F and R symbols against their
published closed forms, the twist relations of the R symbol, and that a closed diagram is weighed
by the route instead of being declined.  The coherence axioms of the category are checked in
``check_category_axioms.py``.
"""
import cmath
import math

import torch

from base import COMPLEX, CUDA
from mpsdmrg import cached_einsum
from mpsdmrg.ext import Tensor, bond_leg, fibonacci, reset_route_stats, route_stats

UNIT = fibonacci.trivial_sector
TAU = (1,)
PHI = (1.0 + math.sqrt(5.0)) / 2.0
TWIST = cmath.exp(4j * math.pi / 5.0)  # the topological spin of tau
TOL = 1e-13


def value(symbol: torch.Tensor) -> complex:
    '''The single entry of a category symbol.'''
    assert symbol.numel() == 1, f'symbol has {symbol.numel()} entries'
    return complex(symbol.reshape(-1)[0].cpu())


def block_value(tensor: Tensor) -> complex:
    '''The single entry of a zero-leg tensor.'''
    assert len(tensor.blocks) == 1, f'tensor has {len(tensor.blocks)} blocks'
    return complex(tensor.blocks[0].reshape(-1)[0].cpu())


def close(one: complex, two: complex) -> bool:
    '''Whether two complex numbers agree within the tolerance.'''
    return abs(one - two) < TOL


# the fusion rules: tau x tau = 1 + tau, and every leg carries a single state
assert fibonacci.sector_dim(UNIT) == 1 and fibonacci.sector_dim(TAU) == 1
assert fibonacci.fusion_outcomes(TAU, TAU) == (UNIT, TAU)
assert fibonacci.dual_sector(TAU) == TAU
assert close(fibonacci.quantum_dim(UNIT), 1.0) and close(fibonacci.quantum_dim(TAU), PHI)

# the F symbol F^{tau tau tau}_tau, in the basis of the (ab) and the (bc) channel
table = [[value(fibonacci.associator(TAU, TAU, TAU, TAU, one, two, CUDA))
          for two in (UNIT, TAU)] for one in (UNIT, TAU)]
published = [[1.0 / PHI, 1.0 / math.sqrt(PHI)], [1.0 / math.sqrt(PHI), -1.0 / PHI]]
for row in range(2):
    for column in range(2):
        assert close(table[row][column], published[row][column]), (row, column, table[row][column])

# the pentagon content of that symbol: it is symmetric and its own inverse
for row in range(2):
    for column in range(2):
        assert close(table[row][column], table[column][row])
        composed = sum(table[row][step] * table[step][column] for step in range(2))
        assert close(composed, 1.0 if row == column else 0.0), (row, column, composed)



# a triple with a unit leg recouples with the coefficient one
assert close(value(fibonacci.associator(TAU, UNIT, TAU, TAU, TAU, TAU, CUDA)), 1.0)
assert close(value(fibonacci.associator(UNIT, TAU, TAU, TAU, TAU, TAU, CUDA)), 1.0)

# the R symbol against its published closed form
assert close(value(fibonacci.braiding(TAU, TAU, UNIT, CUDA)), cmath.exp(-4j * math.pi / 5.0))
assert close(value(fibonacci.braiding(TAU, TAU, TAU, CUDA)), cmath.exp(3j * math.pi / 5.0))

# the twist relation R^{ab}_c R^{ba}_c = twist(c) / (twist(a) twist(b)), which the two braidings of
# a tau pair satisfy with the same symbol
for channel, spin in ((UNIT, 1.0 + 0j), (TAU, TWIST)):
    symbol = fibonacci.braiding(TAU, TAU, channel, CUDA)
    assert symbol.shape == (1, 1), symbol.shape
    assert close(value(symbol) ** 2, spin / TWIST ** 2), channel

# a closed diagram of the category is weighed by the route, and swapping the two operands of it
# relabels the same network, so both orders have to agree
torch.manual_seed(0)
legs = [bond_leg(fibonacci, [(TAU, 2)], 1), bond_leg(fibonacci, [(TAU, 2)], 1),
        bond_leg(fibonacci, [(TAU, 2)], -1)]
tensor = Tensor.rand(legs = legs, qtotal = UNIT, device = CUDA, dtype = COMPLEX, group = fibonacci)
reset_route_stats()
direct = block_value(cached_einsum('abc,abc->', tensor, tensor.conj()))
stats = route_stats()
assert stats.weigh_calls > 0, 'the diagram was declined instead of weighed by the route'
swapped = block_value(cached_einsum('bca,bca->', tensor.conj(), tensor))
assert abs(direct) > 1e-6, direct
assert abs(direct - swapped) < 1e-12, (direct, swapped)

print('FIBONACCI CHECK OK')
