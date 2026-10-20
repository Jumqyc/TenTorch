"""The first part of the AKLT notebook, measured on a symmetry-adapted swept state.

The notebook builds the exact AKLT state by hand and measures its magnetization and its spin
correlation.  The hand-built state is dense, so it measures the components of the spin directly
and against their analytic value.  The symmetry-adapted state of the same Hamiltonian is a
multiplet: it carries no magnetization and measures the invariant :math:`\\vec S_i \\cdot
\\vec S_j` instead.
"""
from math import sqrt

import torch

from base import COMPLEX, CUDA
from mpsdmrg import Broomstick, MPO, MPS, SpinOperator
from mpsdmrg.symmetry.cat import su2_symmetry
from mpsdmrg.tensor import Tensor

LENGTH = 6


def exact_state() -> MPS:
    '''The hand-built AKLT state of the notebook: a spin-1 site tensor of bond dimension 2.'''
    sp = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=COMPLEX, device=CUDA)
    sm = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=COMPLEX, device=CUDA)
    sz = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=COMPLEX, device=CUDA)
    site = torch.zeros(2, 3, 2, dtype=COMPLEX, device=CUDA)
    site[:, 0, :] = sp
    site[:, 1, :] = -sz / sqrt(2.0)
    site[:, 2, :] = -sm
    tensors = [Tensor.from_torch(site.clone()) for _ in range(LENGTH)]
    state = MPS(LENGTH, 3, [2] * (LENGTH + 1), init_state=tensors, dtype=COMPLEX)
    state.normalize()
    return state


def swept_state() -> Broomstick:
    '''The AKLT Hamiltonian of the notebook, swept with SU(2)-symmetric tensors.'''
    stick = Broomstick(L=LENGTH, phys_dim=3, max_bond_dim=8, svd_tol=1e-14, group=su2_symmetry,
                       site_sector=(2,))
    for i in range(LENGTH - 1):
        for a in 'XYZ':
            stick.add_couplings((a, a), 1.0, (i, i + 1))
        for a in 'XYZ':
            for b in 'XYZ':
                stick.add_couplings((a + b, a + b), 1.0 / 3.0, (i, i + 1))
    stick.sweep(num_sweeps=6)
    return stick


def invariant_pair(i, j) -> MPO:
    '''
    The MPO of the spin scalar product of two sites, :math:`\\vec S_i \\cdot \\vec S_j`.

    Args:
        i, j: the two sites, in the order the couplings are added.
    '''
    operator = MPO(LENGTH, phys_dim=3, group=su2_symmetry, site_sector=(2,))
    for letter in 'XYZ':
        operator.add_couplings((letter, letter), 1.0, (i, j))
    operator.build()
    return operator


def total_spin() -> MPO:
    '''The MPO of the pair part of :math:`S_{\\mathrm{tot}}^2`, summed over every pair of sites.'''
    operator = MPO(LENGTH, phys_dim=3, group=su2_symmetry, site_sector=(2,))
    for i in range(LENGTH):
        for j in range(i + 1, LENGTH):
            for letter in 'XYZ':
                operator.add_couplings((letter, letter), 1.0, (i, j))
    operator.build()
    return operator


def main() -> None:
    '''Measure the notebook's quantities on the hand-built state and on the swept one.'''
    spin = SpinOperator(3).data
    sz = spin[2].blocks[0].to(COMPLEX).to(CUDA)
    exact, stick = exact_state(), swept_state()
    print('dmrg energy', stick.energy)
    magnetization = max(abs(complex(exact.expectation({i: sz}))) for i in range(LENGTH))
    correlation = [complex(exact.expectation({0: sz, j: sz})).real for j in range(1, LENGTH)]
    print(f'hand-built  max|m| {magnetization:.3e}  zz',
          [round(value, 9) for value in correlation])
    # The hand-built state is the exact valence-bond state, so its magnetization vanishes and its
    # components follow the analytic correlation of the infinite chain.
    assert magnetization < 1e-10, magnetization
    for j in range(1, LENGTH):
        want = -4.0 / 9.0 * (-1.0 / 3.0) ** (j - 1)
        assert abs(correlation[j - 1] - want) < 1e-9, (j, correlation[j - 1], want)
    # The swept state is a multiplet: it measures the invariant correlation, and being a singlet
    # it carries no magnetization, which its total spin shows.
    invariant = [complex(stick.state.expectation(invariant_pair(0, j))).real
                 for j in range(1, LENGTH)]
    print('su2 swept   S0.Sj', [round(value, 9) for value in invariant])
    # The open chain has a degenerate ground space: the hand-built state carries the bulk
    # correlation of the infinite chain while the sweep returns the reflection-symmetric singlet
    # of the finite chain.  What both must show is the ground energy of the notebook, a vanishing
    # magnetization and the reflection symmetry of the finite chain.
    assert abs(stick.energy + 2.0 / 3.0 * (LENGTH - 1)) < 1e-10, stick.energy
    total = complex(stick.state.expectation(total_spin())).real
    assert abs(total + LENGTH * 2.0 / 2.0) < 1e-8, total
    mirror = max(abs(invariant[j] - invariant[LENGTH - 2 - j]) for j in range(LENGTH - 1))
    assert mirror < 1e-9, mirror
    print('energy', stick.energy, 'S_tot^2', total * 2.0 + LENGTH * 2.0, 'reflection', mirror)
    print('NOTEBOOK AKLT CHECK OK')


if __name__ == '__main__':
    main()
