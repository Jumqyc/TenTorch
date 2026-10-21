'''The DMRG driver: what a run returns, and the invariants the sweep keeps.'''
import pytest

from tentorch import DMRG, MPS, dmrg, heisenberg


def test_result_carries_the_state():
    '''A run returns the energy, the state, the discarded weight and the sweep count.'''
    model = heisenberg(4)
    result = dmrg(model, max_bond_dim=16, num_sweeps=4, verbose=False)
    assert isinstance(result.state, MPS)
    assert result.sweeps == 4
    assert result.trunc_err >= 0.0
    assert abs(result.state.norm() - 1.0) < 1e-12


def test_energy_matches_the_state():
    '''The reported energy is the expectation of the model's Hamiltonian in the state.'''
    model = heisenberg(4)
    result = dmrg(model, max_bond_dim=16, num_sweeps=4, verbose=False)
    expectation = complex(result.state.expectation(model.mpo)).real
    assert abs(expectation - result.energy) < 1e-10


def test_bond_cap_is_respected():
    '''No bond of the state exceeds the cap the run was given.'''
    model = heisenberg(8)
    result = dmrg(model, max_bond_dim=6, num_sweeps=2, verbose=False)
    assert max(result.state.bond_dim) <= 6


def test_engine_energy_before_a_sweep():
    '''``DMRG.energy`` measures the start state instead of raising.'''
    engine = DMRG(heisenberg(4), max_bond_dim=16)
    assert isinstance(engine.energy, float)


def test_run_rejects_an_empty_sweep():
    '''A run without a pass is a mistake, not a silent no-op.'''
    engine = DMRG(heisenberg(4), max_bond_dim=16)
    with pytest.raises(ValueError):
        engine.run(num_sweeps=0)
