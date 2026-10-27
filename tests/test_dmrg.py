'''The two site DMRG of the Heisenberg chain: the energy it reaches against exact diagonalisation, the
state a sweep leaves, and the truncation the bond causes.'''
import math
import os

import pytest

from tentorch import DMRG, Model, dmrg, no_symmetry, u1

from exact import exact_energy

CATEGORIES = [no_symmetry, u1]
SHORT = [4, 6, 8]


@pytest.mark.parametrize('cat', CATEGORIES)
@pytest.mark.parametrize('length', SHORT)
def test_the_energy_matches_exact_diagonalisation(cat, length):
    '''A chain whose bond is not truncated must reach the energy exact diagonalisation gives.'''
    model = Model(length, cat = cat)
    driver = DMRG(model, max_bond_dim = 64, n_iter = 5)
    driver.run(num_sweeps = 6, verbose = False)
    assert abs(driver.energy().real - exact_energy(model)) < 1e-10


@pytest.mark.parametrize('cat', CATEGORIES)
def test_the_driver_writes_the_state_and_the_energy_onto_the_model(cat):
    '''`dmrg` hands the model back with the state and the energy of the last sweep.'''
    model = Model(6, cat = cat)
    result = dmrg(model, max_bond_dim = 64, num_sweeps = 4, verbose = False)
    assert result is model
    assert result.state is not None
    assert abs(result.energy - exact_energy(model)) < 1e-10


@pytest.mark.parametrize('cat', CATEGORIES)
def test_a_sweep_leaves_a_normalised_state(cat):
    '''The split carries the singular values, so the centre is gauged back to unit norm.'''
    driver = DMRG(Model(6, cat = cat), max_bond_dim = 64, n_iter = 5)
    driver.sweep()
    assert abs(driver.state.norm() - 1.0) < 1e-12
    assert driver.center == driver.state.center


@pytest.mark.parametrize('cat', CATEGORIES)
def test_the_measured_energy_is_the_expectation_of_the_state(cat):
    '''The Rayleigh quotient of the window, once converged, is the energy of the whole chain.'''
    driver = DMRG(Model(6, cat = cat), max_bond_dim = 64, n_iter = 5)
    driver.run(num_sweeps = 6, verbose = False)
    assert abs(driver.energy() - driver.state.expectation(driver.operator)) < 1e-10


@pytest.mark.parametrize('cat', CATEGORIES)
def test_a_bigger_bond_reaches_a_lower_energy(cat):
    '''The truncation is variational: a bond of two multiplets cannot beat the exact energy.'''
    small = DMRG(Model(8, cat = cat), max_bond_dim = 2, n_iter = 5)
    small.run(num_sweeps = 6, verbose = False)
    large = DMRG(Model(8, cat = cat), max_bond_dim = 64, n_iter = 5)
    large.run(num_sweeps = 6, verbose = False)
    assert small.energy().real > large.energy().real - 1e-10
    assert abs(large.energy().real - exact_energy(Model(8, cat = cat))) < 1e-10


@pytest.mark.parametrize('cat', CATEGORIES)
def test_the_bond_stays_within_the_limit(cat):
    '''A capped bond keeps at most the multiplets the caller allows.'''
    driver = DMRG(Model(12, cat = cat), max_bond_dim = 4, n_iter = 5)
    driver.run(num_sweeps = 4, verbose = False)
    assert max(driver.state.bond_dims) <= 4


@pytest.mark.parametrize('cat', CATEGORIES)
def test_a_capped_bond_throws_weight_away(cat):
    '''A bond of two drops a measurable weight where the full bond drops none.'''
    capped = DMRG(Model(10, cat = cat), max_bond_dim = 2, n_iter = 5)
    _, capped_lost = capped.sweep()
    full = DMRG(Model(10, cat = cat), max_bond_dim = 64, n_iter = 5)
    full.sweep()
    _, full_lost = full.sweep()
    assert capped_lost > 1e-3
    assert full_lost < 1e-12


def test_the_energy_needs_a_sweep():
    '''The energy reads the environments of a sweep, so without one it says so.'''
    driver = DMRG(Model(4, cat = u1), max_bond_dim = 16)
    with pytest.raises(RuntimeError):
        driver.energy()


def test_a_run_needs_a_sweep():
    '''A run without a pass is a mistake, not a silent no-op.'''
    with pytest.raises(ValueError):
        DMRG(Model(4, cat = u1), max_bond_dim = 16).run(num_sweeps = 0)


@pytest.mark.skipif(not os.environ.get('TENTORCH_SLOW'), reason = 'set TENTORCH_SLOW=1 to run it')
def test_the_energy_density_of_a_long_chain_approaches_the_known_limit():
    r'''The bulk energy per site of the spin one half chain is $\tfrac14 - \ln 2$.'''
    length, bond = 64, 64
    model = Model(length, cat = u1)
    energy = dmrg(model, max_bond_dim = bond, num_sweeps = 5, verbose = False).energy
    assert abs(energy / length - (0.25 - math.log(2))) < 5e-3
