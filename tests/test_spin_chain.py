'''The spin chains of the model zoo: the sweep energy against exact diagonalization.'''
import pytest

from tentorch import dmrg, no_symmetry, su2_symmetry
from tests.exact import exact_energy
from tentorch.models import aklt, heisenberg, ising

TOL = 1e-10


@pytest.mark.parametrize('L', [4, 6])
def test_heisenberg_su2_matches_exact(L: int):
    '''The SU(2) Heisenberg chain, whose ground state is a singlet, reaches the exact energy.'''
    model = heisenberg(L)
    result = dmrg(model, max_bond_dim=16, num_sweeps=6, verbose=False)
    exact = exact_energy(model)
    assert abs(result.energy - exact) < TOL


def test_heisenberg_symmetry_free_agrees():
    '''The same chain without a category finds the same energy, with larger bonds.'''
    model = heisenberg(4, cat=no_symmetry)
    result = dmrg(model, max_bond_dim=16, num_sweeps=6, verbose=False)
    assert abs(result.energy - exact_energy(model)) < TOL


def test_ising_matches_exact():
    '''The transverse-field Ising chain at criticality.'''
    model = ising(6)
    result = dmrg(model, max_bond_dim=16, num_sweeps=6, verbose=False)
    assert abs(result.energy - exact_energy(model)) < TOL


def test_aklt_ground_energy():
    '''The AKLT chain's valence-bond solid sits at -2/3 per bond.'''
    L = 6
    model = aklt(L)
    result = dmrg(model, max_bond_dim=16, num_sweeps=6, verbose=False)
    assert abs(result.energy - (-2.0 / 3.0) * (L - 1)) < TOL
    assert abs(result.energy - exact_energy(model)) < TOL


def test_aklt_rejects_the_adapted_builder():
    '''A biquadratic coupling is not an irreducible tensor, so SU(2) is refused.'''
    with pytest.raises(NotImplementedError):
        aklt(4, cat=su2_symmetry)
