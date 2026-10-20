"""Scratch check: the SU(2)-invariant Heisenberg MPO expands to the dense Heisenberg matrix."""
import torch

from base import COMPLEX, CUDA
from mpsdmrg import Broomstick
from mpsdmrg.operators import SpinOperator
from mpsdmrg.symmetry.cat import su2_symmetry

PHYS = 2


def dense_heisenberg(L: int) -> torch.Tensor:
    '''
    The explicit spin-1/2 Heisenberg matrix on an open chain of L sites.
    Args:
        L: number of sites.
    Returns:
        The matrix in shape ``(PHYS**L, PHYS**L)``.
    '''
    spin = SpinOperator(PHYS, dtype=COMPLEX, device=CUDA).data.blocks[0]
    eye = torch.eye(PHYS, dtype=COMPLEX, device=CUDA)
    total = torch.zeros(PHYS ** L, PHYS ** L, dtype=COMPLEX, device=CUDA)
    for i in range(L - 1):
        for s in range(3):
            term = torch.ones(1, 1, dtype=COMPLEX, device=CUDA)
            for k in range(L):
                term = torch.kron(term, spin[s] if k in (i, i + 1) else eye)
            total += term
    return total


def main() -> None:
    '''Sweep the SU(2)-adapted Heisenberg chain and compare its energy with the dense matrix.'''
    for L in (4, 6):
        stick = Broomstick(L=L, phys_dim=PHYS, max_bond_dim=32, svd_tol=1e-14,
                           group=su2_symmetry, site_sector=(1,))
        for i in range(L - 1):
            for op in ('X', 'Y', 'Z'):
                stick.add_couplings((op, op), 1.0, (i, i + 1))
        stick.build()
        stick.sweep(num_sweeps=6)
        reference = float(torch.linalg.eigvalsh(dense_heisenberg(L)).min())
        print(f'L = {L}  bond_dim = {stick.Hamiltonian.bond_dim}  energy = {stick.energy:.12f}'
              f'  dense = {reference:.12f}  diff = {abs(stick.energy - reference):.3e}')
        if abs(stick.energy - reference) > 1e-9:
            raise SystemExit(f'SU2 MPO CHECK FAILED at L = {L}')
    print('SU2 MPO CHECK OK')


if __name__ == '__main__':
    main()
