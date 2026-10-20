"""Scratch check: SU(2)-invariant DMRG on the spin-1/2 Heisenberg chain against exact diagonalisation."""
import torch

from base import COMPLEX, CUDA
from mpsdmrg import Broomstick
from mpsdmrg.operators import SpinOperator
from mpsdmrg.symmetry.cat import su2_symmetry

L, PHYS = 6, 2


def dense_heisenberg(length: int) -> torch.Tensor:
    '''
    The explicit spin-1/2 Heisenberg matrix on an open chain.
    Args:
        length: number of sites.
    Returns:
        The matrix in shape ``(PHYS**length, PHYS**length)``.
    '''
    spin = SpinOperator(PHYS, dtype=COMPLEX, device=CUDA).data.blocks[0]
    eye = torch.eye(PHYS, dtype=COMPLEX, device=CUDA)
    total = torch.zeros(PHYS ** length, PHYS ** length, dtype=COMPLEX, device=CUDA)
    for i in range(length - 1):
        for s in range(3):
            term = torch.ones(1, 1, dtype=COMPLEX, device=CUDA)
            for k in range(length):
                term = torch.kron(term, spin[s] if k in (i, i + 1) else eye)
            total += term
    return total


def main() -> None:
    exact = torch.linalg.eigvalsh(dense_heisenberg(L))[0].real.item()
    torch.manual_seed(0)
    stick = Broomstick(L=L, phys_dim=PHYS, max_bond_dim=16, svd_tol=1e-14, group=su2_symmetry,
                       site_sector=(1,))
    for i in range(L - 1):
        for op in ('X', 'Y', 'Z'):
            stick.add_couplings((op, op), 1.0, (i, i + 1))
    stick.build()
    stick.sweep(num_sweeps=10)
    energy = stick.energy
    print(f'exact ground energy  {exact:.12f}')
    print(f'dmrg  ground energy  {energy:.12f}')
    print(f'difference           {abs(energy - exact):.3e}')
    print(f'bond dims            {stick.state.bond_dim}')
    if abs(energy - exact) > 1e-9:
        raise SystemExit('SU2 DMRG CHECK FAILED')
    print('SU2 DMRG CHECK OK')


if __name__ == '__main__':
    main()
