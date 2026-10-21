'''The local Hilbert spaces: the operators a site hands out, and the multiplets they carry.'''
import numpy as np
import torch

from tentorch import fundamental_su3, spin, su2_symmetry, su3_symmetry


def test_spin_operators_are_half_pauli():
    '''The three components of a spin-1/2 site are the Pauli matrices over two.'''
    site = spin(0.5)
    sigma_x = np.array([[0, 1], [1, 0]], dtype = np.complex128)
    expected = sigma_x / 2
    assert np.allclose(site.matrix('X').cpu().numpy(), expected)
    assert np.allclose(site.matrix('Z').cpu().numpy(), np.diag([0.5, -0.5]))


def test_spin_one_carries_the_triplet():
    '''A spin-1 site has dimension 3, the multiplet ``(2,)`` and ``S^2`` equal to 2.'''
    site = spin(1, su2_symmetry)
    assert site.phys_dim == 3
    assert site.sector == (2,)
    zero = torch.zeros(3, 3, dtype = torch.complex128, device = site.matrix('Z').device)
    squares = sum((site.matrix(component) @ site.matrix(component) for component in 'XYZ'), zero)
    identity = torch.eye(3, dtype = torch.complex128, device = zero.device)
    assert torch.allclose(squares, 2 * identity, atol = 1e-12)


def test_gell_mann_generators():
    '''The SU(3) site hands out the eight generators, whose structure constants close.'''
    site = fundamental_su3()
    assert site.phys_dim == 3
    assert site.sector == (1, 0)
    assert su3_symmetry is site.cat
    generators = [site.matrix(f'L{index}') for index in range(1, 9)]
    # Tr(L_a L_b) = 2 delta_ab for the Gell-Mann normalization, and every generator is
    # traceless and Hermitian
    for first, generator in enumerate(generators):
        assert abs(complex(torch.trace(generator).item())) < 1e-12
        assert torch.allclose(generator, generator.conj().T, atol = 1e-12)
        for second, other in enumerate(generators):
            trace = complex((generator @ other).trace().item())
            assert abs(trace - (2.0 if first == second else 0.0)) < 1e-12
