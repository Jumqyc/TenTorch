'''The mixed canonical form of a matrix product state: the norm and the contractions keep their
value wherever the centre sits.'''
import pytest

from tentorch import Leg, no_symmetry, u1
from tentorch._ext import NoSymmetry
from tentorch.tensors.tree import entered
from tentorch.models.heisenberg import heisenberg
from tentorch.networks.mps import MPS

CATEGORIES = [no_symmetry, u1]


def _product(cat, physical, spins):
    '''
    The product state of one spin per site.
    Args:
      cat: the category of the state.
      physical: the physical leg of one site.
      spins: the spin of every site, +1 or -1.
    Returns:
      The state, with bonds of one multiplet.
    '''
    if isinstance(cat, NoSymmetry):
        bonds = [Leg(cat, [(0,)], (1,)) for _ in range(len(spins) + 1)]
    else:
        bonds = [Leg(cat, [(sum(spins[:at]),)], (1,)) for at in range(len(spins) + 1)]

    def entry(at):
        def fill(tree_out, copy_out, tree_in, copy_in):
            if isinstance(cat, NoSymmetry):
                return 1.0 if 1 - 2 * copy_out[1] == spins[at] else 0.0
            return 1.0 if tree_out.sectors[1] == (spins[at],) else 0.0
        return fill

    tensors = [entered(cat, [bonds[at], physical], [bonds[at + 1]], entry(at))
               for at in range(len(spins))]
    return MPS(tensors, physical)


@pytest.mark.parametrize('cat', CATEGORIES)
def test_the_norm_does_not_depend_on_the_center(cat):
    '''Moving the centre around must not change the norm of the state.'''
    operator = heisenberg(cat, 4)
    state = MPS.random(cat, operator.physical, 4, seed=1, multiplets=2)
    state.normalize()
    for center in range(state.L):
        state.move_center(center)
        assert state.center == center
        assert abs(state.norm() - 1.0) < 1e-12
    for center in range(state.L - 1, -1, -1):
        state.move_center(center)
        assert abs(state.norm() - 1.0) < 1e-12


@pytest.mark.parametrize('cat', CATEGORIES)
def test_the_expectation_does_not_depend_on_the_center(cat):
    '''A contraction with an operator must not depend on where the centre sits.'''
    operator = heisenberg(cat, 4)
    state = MPS.random(cat, operator.physical, 4, seed=2, multiplets=2)
    state.normalize()
    values = []
    for center in range(state.L):
        state.move_center(center)
        values.append(complex(state.expectation(operator)))
    assert max(abs(value - values[0]) for value in values) < 1e-12


@pytest.mark.parametrize('cat', [no_symmetry])
def test_a_product_state_has_the_energy_of_its_spins(cat):
    r'''On a product state only the $S^z S^z$ terms contribute, one $-\tfrac14$ per aligned pair.'''
    operator = heisenberg(cat, 4)
    state = _product(cat, operator.physical, [1, -1, 1, -1])
    assert abs(state.norm() - 1.0) < 1e-12
    value = complex(state.expectation(operator))
    spins = [1, -1, 1, -1]
    pairs = sum(0.25 if left == right else -0.25 for left, right in zip(spins, spins[1:]))
    assert abs(value - pairs) < 1e-12
