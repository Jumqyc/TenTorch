r'''
The spin-$\tfrac12$ Heisenberg chain as a matrix product operator.

$$H = \sum_{i} \vec S_i \cdot \vec S_{i+1}
    = \sum_i \left(S^z_i S^z_{i+1} + \tfrac12 S^+_i S^-_{i+1} + \tfrac12 S^-_i S^+_{i+1}\right)$$

The operator is read as a finite automaton over the virtual bond: a state records which part of a
term has been written, so one pass along the chain spells out every term at once.
'''

from typing import Callable

from tentorch._ext import AbelianGroup, FusionTree, Leg, NoSymmetry, Tensor
from tentorch.networks.mpo import MPO
from tentorch.symmetry import FusionCategory, Leg
from tentorch.tensors.tree import entered

# The automaton: 0 nothing written yet, 1 the $S^z$ of a term, 2 its $S^+$, 3 its $S^-$, 4 closed.
START, CLOSED = 0, 4

# How the copies of a virtual bond stand for the states of a category whose sectors are the charges:
# the trivial charge carries the three states that keep it, the charge $2$ carries $S^+$ and $-2$
# carries $S^-$.
_STATE_OF_CHARGED = {(0,): {0: START, 1: 1, 2: CLOSED}, (2,): {0: 2}, (-2,): {0: 3}}

# The operator a transition writes, as a function of the spin the physical leg leaves and the one it
# enters; the half of a raising and a lowering is carried by the closing transition.
_OPERATOR: dict[tuple[int, int], Callable[[int, int], float]] = {
    (START, START): lambda out, into: 1.0 if out == into else 0.0,
    (CLOSED, CLOSED): lambda out, into: 1.0 if out == into else 0.0,
    (START, 1): lambda out, into: 0.5 if out == into == 1 else (-0.5 if out == into else 0.0),
    (1, CLOSED): lambda out, into: 0.5 if out == into == 1 else (-0.5 if out == into else 0.0),
    (START, 2): lambda out, into: 1.0 if out == into + 2 else 0.0,
    (2, CLOSED): lambda out, into: 0.5 if out == into - 2 else 0.0,
    (START, 3): lambda out, into: 1.0 if out == into - 2 else 0.0,
    (3, CLOSED): lambda out, into: 0.5 if out == into + 2 else 0.0,
}


def _table(cat: FusionCategory) -> tuple[Leg, Leg, Callable[[tuple, int], int],
                                         Callable[[tuple, int], int]]:
    '''
    The legs of a site and the meaning of their blocks.
    Args:
      cat: an abelian category, either one whose sectors are charges or the one with a single
        sector, which then tells the spins and the states apart by the copies of that sector.
    Returns:
      The physical leg, the virtual leg, the state of a virtual block, and the spin of a physical
      block.
    '''
    if isinstance(cat, NoSymmetry):
        return (Leg(cat, [(0,)], (2,)), Leg(cat, [(0,)], (5,)),
                lambda sector, copy: copy, lambda sector, copy: 1 - 2 * copy)
    return (Leg(cat, [(1,), (-1,)], (1, 1)), Leg(cat, [(0,), (2,), (-2,)], (3, 1, 1)),
            lambda sector, copy: _STATE_OF_CHARGED[sector][copy], lambda sector, copy: sector[0])


def heisenberg(cat: FusionCategory, length: int, coupling: float = 1.0) -> MPO:
    '''
    The nearest neighbour Heisenberg chain.
    Args:
      cat: an abelian category whose sectors hold one integer charge, or the category with a single
        sector.
      length: the number of sites.
      coupling: the coefficient in front of every bond term.
    Returns:
      The operator, each site tensor over the codomain (bond left, out) and the domain (in, bond
      right).
    Raises:
      ValueError: when the category is not abelian, since its tree basis is the coupled basis and a
        coefficient then is not a plain matrix element.
    '''
    if not isinstance(cat, AbelianGroup):
        raise ValueError(f'the rows of {cat.name} would need the recoupling to dress them')
    phys, bond, state_of, spin_of = _table(cat)
    edge = Leg(cat, [cat.trivial()], (1,))
    sites = [_site(phys, edge if at == 0 else bond, edge if at == length - 1 else bond,
                   at == 0, at == length - 1, coupling, state_of, spin_of)
             for at in range(length)]
    return MPO(sites, phys, cat)


def _site(phys: Leg, w_left: Leg, w_right: Leg, first: bool, last: bool, coupling: float,
          state_of: Callable[[tuple, int], int],
          spin_of: Callable[[tuple, int], int]) -> Tensor:
    '''
    One site of the operator.
    Args:
      phys: the physical leg, once for the outgoing index and once for the incoming one.
      w_left: the virtual bond to the left, over the codomain.
      w_right: the virtual bond to the right, over the domain.
      first: whether the site starts the chain, which fixes the state before it.
      last: whether the site ends the chain, which fixes the state after it.
      coupling: the coefficient of a bond term.
      state_of: the automaton state of a block of the virtual leg.
      spin_of: the spin of a block of the physical leg, $+1$ up and $-1$ down.
    Returns:
      The site tensor, a map from the incoming leg and the right bond to the outgoing leg and the
      left one.
    '''
    cat = phys.cat

    def entry(tree_out: FusionTree, copy_out: tuple[int, ...],
              tree_in: FusionTree, copy_in: tuple[int, ...]) -> float:
        before = START if first else state_of(tree_out.sectors[0], copy_out[0])
        after = CLOSED if last else state_of(tree_in.sectors[1], copy_in[1])
        write = _OPERATOR.get((before, after))
        if write is None:
            return 0.0
        # a bond term is written by two sites, so its coefficient rides on the one that closes it
        weight = coupling if before not in (START, CLOSED) and after == CLOSED else 1.0
        return weight * write(spin_of(tree_out.sectors[1], copy_out[1]),
                              spin_of(tree_in.sectors[0], copy_in[0]))

    return entered(cat, [w_left, phys], [phys, w_right], entry)
