"""Check: the Uimin-Lai-Sutherland chain in the SU(3) engine against exact diagonalisation."""
import torch

from base import COMPLEX, CUDA
from mpsdmrg import Broomstick
from mpsdmrg.tensor import Tensor
from mpsdmrg.symmetry.cat import su3_symmetry

LENGTH = 6
DIM = 3


def gell_mann(device: torch.device) -> list[torch.Tensor]:
    '''
    The eight Hermitian Gell-Mann generators of SU(3).

    Args:
        device: device of the matrices.
    Returns:
        The generators in shape (3, 3), normalized so that ``tr(T_a T_b) = 2 delta_ab``.
    '''
    imaginary = 1j
    generators = [
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, -imaginary, 0], [imaginary, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
        [[0, 0, -imaginary], [0, 0, 0], [imaginary, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 0, -imaginary], [0, imaginary, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, -2]],
    ]
    return [torch.tensor(matrix, dtype=COMPLEX, device=device) / (1.0 if index < 7 else 3 ** 0.5)
            for index, matrix in enumerate(generators)]


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
    total = torch.zeros((dim ** length, dim ** length), dtype=bond.dtype, device=bond.device)
    for i in range(length - 1):
        total += torch.kron(
            torch.kron(torch.eye(dim ** i, dtype=bond.dtype, device=bond.device), bond),
            torch.eye(dim ** (length - 2 - i), dtype=bond.dtype, device=bond.device))
    return total


def main() -> None:
    '''The permutation chain as an SU(3) invariant and the engine's ground energy.'''
    generators = gell_mann(CUDA)
    identity = torch.eye(DIM ** 2, dtype=COMPLEX, device=CUDA)
    # tr(T_a T_b) = 2 delta_ab makes sum_a T_a (x) T_a = P - I / 3, so P = I / 3 + sum_a T_a (x) T_a
    permutation = identity / 3.0 + 0.5 * sum(torch.kron(a, a) for a in generators)
    check = torch.zeros((DIM ** 2, DIM ** 2), dtype=COMPLEX, device=CUDA)
    for a in range(DIM):
        for b in range(DIM):
            check[a * DIM + b, b * DIM + a] = 1.0
    print('permutation residual', float((permutation - check).abs().max()))
    reference = float(torch.linalg.eigvalsh(dense_chain(LENGTH, DIM, permutation)).min())

    mapping = {f'L{index + 1}': Tensor.from_torch(generator) for index, generator in enumerate(generators)}
    mapping['I'] = Tensor.from_torch(torch.eye(DIM, dtype=COMPLEX, device=CUDA))
    stick = Broomstick(L=LENGTH, phys_dim=DIM, max_bond_dim=64, svd_tol=1e-14,
                       mapping=mapping, group=su3_symmetry, site_sector=(1, 0))
    for i in range(LENGTH - 1):
        stick.add_couplings(('I', 'I'), 1.0 / 3.0, (i, i + 1))
        for name in mapping:
            if name != 'I':
                stick.add_couplings((name, name), 0.5, (i, i + 1))
    stick.build()
    stick.sweep(num_sweeps=20)
    print('exact ground energy', reference)
    print('dmrg  ground energy', stick.energy)
    print('difference', abs(stick.energy - reference))
    print('state bond dims', [tensor.legs[0].dim for tensor in stick.state.tensors])
    assert abs(stick.energy - reference) < 1e-8, stick.energy
    print('INVARIANT ULS3 CHECK OK')


if __name__ == '__main__':
    main()
