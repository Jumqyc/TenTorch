"""Checks the coherence axioms of every category the categorical route reads symbols from.

The route composes associator and braiding symbols along a chain of fusion trees, so those symbols
have to satisfy the pentagon and the hexagon of a fusion category.  Each axiom is checked as the
agreement of two routes between one pair of states, summed over every intermediate channel.
"""
import torch

from base import CUDA
from category_axioms import Axioms
from mpsdmrg.ext import fibonacci, su2_symmetry, su3_symmetry, u1_symmetry

TOL = 1e-13

PENTAGON = [
    ('u1', u1_symmetry, [(1,), (1,), (-1,), (-1,)], (0,)),
    ('su2 spin-1', su2_symmetry, [(2,), (2,), (2,), (2,)], (0,)),
    ('su2 spin-1 to spin-2', su2_symmetry, [(2,), (2,), (2,), (2,)], (2,)),
    ('su3', su3_symmetry, [(1, 0), (1, 0), (0, 1), (0, 1)], (0, 0)),
    ('fibonacci to tau', fibonacci, [(1,), (1,), (1,), (1,)], (1,)),
    ('fibonacci to unit', fibonacci, [(1,), (1,), (1,), (1,)], (0,)),
]

HEXAGON = [
    ('u1', u1_symmetry, [(1,), (1,), (-1,)], (1,)),
    ('su2 spin-1', su2_symmetry, [(2,), (2,), (2,)], (0,)),
    ('fibonacci to tau', fibonacci, [(1,), (1,), (1,)], (1,)),
    ('fibonacci to unit', fibonacci, [(1,), (1,), (1,)], (0,)),
]

for name, group, sectors, total in PENTAGON:
    gap = Axioms(group, CUDA).pentagon(sectors, total)
    assert gap < TOL, f'{name}: the pentagon is off by {gap:.3e}'

for name, group, sectors, total in HEXAGON:
    gap = Axioms(group, CUDA).hexagon(sectors, total)
    assert gap < TOL, f'{name}: the hexagon is off by {gap:.3e}'

print('CATEGORY AXIOMS CHECK OK')
