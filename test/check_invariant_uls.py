"""Scratch check: the Uimin-Lai-Sutherland chain through the engine and against ED."""
import torch

from base import COMPLEX, CUDA
from mpsdmrg import Broomstick
from mpsdmrg.operators import SpinOperator
from mpsdmrg.symmetry.cat import su2_symmetry

LENGTH = 6
DIM = 3


def dense_chain(length: int, dim: int, bond: torch.Tensor) -> torch.Tensor:
    '''
    The explicit matrix of a nearest-neighbour bond operator on an open chain.

    Args:
        length: number of sites.
        dim: physical dimension of one site.
        bond: the two-site operator in shape (dim**2, dim**2), Hermitian.
    Returns:
        The matrix in shape (dim**length, dim**length).
    '''
    eye = torch.eye(dim, dtype=bond.dtype, device=bond.device)
    total = torch.zeros((dim ** length, dim ** length), dtype=bond.dtype, device=bond.device)
    for i in range(length - 1):
        total += torch.kron(torch.kron(torch.eye(dim ** i, dtype=bond.dtype, device=bond.device),
                                       bond),
                            torch.eye(dim ** (length - 2 - i), dtype=bond.dtype,
                                      device=bond.device))
    return total


def main() -> None:
    '''The Lai-Sutherland chain: the ground energy of the engine against exact diagonalisation.'''
    spin = SpinOperator(DIM)
    components = [spin.data.blocks[0][i].to(COMPLEX).to(CUDA) for i in range(3)]
    dot = sum(torch.kron(a, a) for a in components)
    identity = torch.eye(DIM ** 2, dtype=COMPLEX, device=CUDA)
    # the site permutation P|a, b> = |b, a> is the SU(3) invariant interaction
    permutation = torch.zeros((DIM ** 2, DIM ** 2), dtype=COMPLEX, device=CUDA)
    for a in range(DIM):
        for b in range(DIM):
            permutation[a * DIM + b, b * DIM + a] = 1.0
    print('permutation residual', float((permutation - (dot + dot @ dot - identity)).abs().max()))
    reference = float(torch.linalg.eigvalsh(dense_chain(LENGTH, DIM, permutation)).min())

    stick = Broomstick(L=LENGTH, phys_dim=DIM, max_bond_dim=64, svd_tol=1e-14, group=su2_symmetry,
                       site_sector=(2,))
    for i in range(LENGTH - 1):
        stick.add_couplings(('I', 'I'), -1.0, (i, i + 1))
        for a in 'XYZ':
            stick.add_couplings((a, a), 1.0, (i, i + 1))
        for a in 'XYZ':
            for b in 'XYZ':
                stick.add_couplings((a + b, a + b), 1.0, (i, i + 1))
    stick.build()
    stick.sweep(num_sweeps=20)
    print('exact ground energy', reference)
    print('dmrg  ground energy', stick.energy)
    print('difference', abs(stick.energy - reference))
    print('state bond dims', [tensor.legs[0].dim for tensor in stick.state.tensors])
    assert abs(stick.energy - reference) < 1e-8, stick.energy
    print('INVARIANT ULS CHECK OK')


if __name__ == '__main__':
    main()
