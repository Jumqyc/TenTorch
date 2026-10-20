"""Scratch check: an invariant Hamiltonian with a next-nearest-neighbour term."""
import torch

from base import COMPLEX, CUDA
from mpsdmrg import Broomstick
from mpsdmrg.operators import SpinOperator
from mpsdmrg.symmetry.cat import su2_symmetry

LENGTH = 6
J1, J2 = 1.0, 0.5


def dense_chain(length, dim, bond):
    '''The dense operator of a nearest-neighbour bond repeated along the chain.'''
    total = torch.zeros(dim ** length, dim ** length, dtype=COMPLEX, device=CUDA)
    for i in range(length - 1):
        piece = torch.eye(dim ** i, dtype=COMPLEX, device=CUDA)
        piece = torch.kron(piece, torch.kron(bond, torch.eye(dim ** (length - i - 2), dtype=COMPLEX, device=CUDA)))
        total = total + piece
    return total


def main() -> None:
    '''H = J1 sum S_i.S_{i+1} + J2 sum S_i.S_{i+2} against ED.'''
    spin = SpinOperator(2)
    components = [spin.data.blocks[0][i].to(COMPLEX).to(CUDA) for i in range(3)]
    dot = sum(torch.kron(a, a) for a in components)
    identity = torch.eye(4, dtype=COMPLEX, device=CUDA)
    total = torch.zeros(2 ** LENGTH, 2 ** LENGTH, dtype=COMPLEX, device=CUDA)
    for i in range(LENGTH - 1):
        piece = torch.eye(2 ** i, dtype=COMPLEX, device=CUDA)
        piece = torch.kron(piece, torch.kron(J1 * dot, torch.eye(2 ** (LENGTH - i - 2), dtype=COMPLEX, device=CUDA)))
        total = total + piece
    for i in range(LENGTH - 2):
        # S_i.S_{i+2}: the two operators are separated by the identity of site i+1
        for a in components:
            piece = torch.eye(2 ** i, dtype=COMPLEX, device=CUDA)
            piece = torch.kron(piece, torch.kron(a, torch.eye(2, dtype=COMPLEX, device=CUDA)))
            piece = torch.kron(piece, torch.kron(a, torch.eye(2 ** (LENGTH - i - 3), dtype=COMPLEX, device=CUDA)))
            total = total + J2 * piece
    reference = float(torch.linalg.eigvalsh(total).min())

    stick = Broomstick(L=LENGTH, phys_dim=2, max_bond_dim=32, svd_tol=1e-14, group=su2_symmetry,
                       site_sector=(1,))
    for i in range(LENGTH - 1):
        for a in 'XYZ':
            stick.add_couplings((a, a), J1, (i, i + 1))
    for i in range(LENGTH - 2):
        for a in 'XYZ':
            stick.add_couplings((a, a), J2, (i, i + 2))
    stick.build()
    stick.sweep(num_sweeps=10)
    print('exact ground energy', reference)
    print('dmrg  ground energy', stick.energy)
    print('difference', abs(stick.energy - reference))
    print('bond dims', stick.state.bond_dim)
    assert abs(stick.energy - reference) < 1e-8, stick.energy
    print('INVARIANT LONG CHECK OK')


if __name__ == '__main__':
    main()
