r'''
The spin-$\tfrac12$ Heisenberg chain as a matrix product operator.

$$H = \sum_{i} \vec S_i \cdot \vec S_{i+1}
    = \sum_i \left(S^z_i S^z_{i+1} + \tfrac12 S^+_i S^-_{i+1} + \tfrac12 S^-_i S^+_{i+1}\right)$$

The operator is read as a finite automaton over the virtual bond: a state records which part of a
term has been written, so one pass along the chain spells out every term at once.
'''
from __future__ import annotations

from typing import Callable, Optional

from tentorch._ext import AbelianGroup, FusionTree, Leg, Tensor
from tentorch.networks.mpo import MPO
from tentorch.symmetry import FusionCategory, trivial_leg
from tentorch.tensors.tree import state

# The automaton: 0 nothing written yet, 1 the $S^z$ of a term, 2 its $S^+$, 3 its $S^-$, 4 closed.
START, CLOSED = 0, 4

# How the copies of a virtual bond stand for the states: the trivial charge carries the three states
# that do not move the charge, the charge $2$ carries $S^+$ and $-2$ carries $S^-$.
_STATE_OF_COPY = {(0,): {0: START, 1: 1, 2: CLOSED}, (2,): {0: 2}, (-2,): {0: 3}}

# The operator a transition writes, as a function of the physical sector it leaves and enters.
_OPERATOR: dict[tuple[int, int], Callable[[tuple, tuple], float]] = {
    (START, START): lambda out, into: 1.0 if out == into else 0.0,
    (CLOSED, CLOSED): lambda out, into: 1.0 if out == into else 0.0,
    (START, 1): lambda out, into: 0.5 if out == into == (1,) else (-0.5 if out == into else 0.0),
    (1, CLOSED): lambda out, into: 0.5 if out == into == (1,) else (-0.5 if out == into else 0.0),
    (START, 2): lambda out, into: 1.0 if out == (1,) and into == (-1,) else 0.0,
    (2, CLOSED): lambda out, into: 0.5 if out == (-1,) and into == (1,) else 0.0,
    (START, 3): lambda out, into: 1.0 if out == (-1,) and into == (1,) else 0.0,
    (3, CLOSED): lambda out, into: 0.5 if out == (1,) and into == (-1,) else 0.0,
}


def physical(cat: FusionCategory) -> Leg:
    '''
    The spin-$\tfrac12$ leg: the two sectors of one half unit of angular momentum.
    Args:
      cat: the category, whose sectors are tuples holding one integer charge.
    Returns:
      The leg, with the sector $(1)$ for spin up and $(-1)$ for spin down.
    '''
    return Leg(cat, [(1,), (-1,)], (1, 1))


def virtual(cat: FusionCategory) -> Leg:
    '''
    The virtual bond of the automaton: one block per charge it can carry.
    Args:
      cat: the category.
    Returns:
      The leg, with three copies of the trivial charge for the states that keep it.
    '''
    return Leg(cat, [(0,), (2,), (-2,)], (3, 1, 1))


def heisenberg(cat: FusionCategory, length: int, coupling: float = 1.0) -> MPO:
    '''
    The nearest neighbour Heisenberg chain.
    Args:
      cat: an abelian category whose sectors hold one integer charge.
      length: the number of sites.
      coupling: the coefficient in front of every bond term.
    Returns:
      The operator, each site tensor over legs (bond left, out, in, bond right) on the codomain.
    Raises:
      ValueError: when the category is not abelian, since its tree basis is the coupled basis and a
        coefficient then is not a plain matrix element.
    '''
    if not isinstance(cat, AbelianGroup):
        raise ValueError(f'the rows of {cat.name()} would need the recoupling to dress them')
    phys = physical(cat)
    bond = virtual(cat)
    edge = trivial_leg(cat, 1)
    sites = [_site(cat, phys, edge if at == 0 else bond, edge if at == length - 1 else bond,
                   at == 0, at == length - 1, coupling)
             for at in range(length)]
    return MPO(sites)


def _site(cat: FusionCategory, phys: Leg, w_left: Leg, w_right: Leg, first: bool, last: bool,
          coupling: float) -> Tensor:
    '''
    One site of the operator.
    Args:
      cat: the category.
      phys: the physical leg, once for the outgoing index and once for the incoming one.
      w_left: the virtual bond to the left.
      w_right: the virtual bond to the right.
      first: whether the site starts the chain, which fixes the state to `START`.
      last: whether the site ends the chain, which fixes the state to `CLOSED`.
      coupling: the coefficient of a bond term.
    Returns:
      The site tensor.
    '''
    def entry(tree: FusionTree, copy: tuple[int, ...]) -> float:
        before = START if first else _STATE_OF_COPY[tree.sectors[0]][copy[0]]
        after = CLOSED if last else _STATE_OF_COPY[tree.sectors[3]][copy[3]]
        write = _OPERATOR.get((before, after))
        return 0.0 if write is None else coupling * write(tree.sectors[1], tree.sectors[2])

    return state(cat, [w_left, phys, phys, w_right], entry)
