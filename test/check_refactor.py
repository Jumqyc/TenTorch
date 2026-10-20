"""Scratch regression check: the same metric set under the working tree and at HEAD."""
import os

import torch

from base import CUDA
from mpsdmrg import Broomstick, MPO, MPS
from mpsdmrg.tensor import Tensor


def show(name: str, value) -> None:
    value = complex(value)
    print(f'{name:18s} {value.real:+.15f} {value.imag:+.15f}j')


torch.manual_seed(0)
L, phys = 8, 2
bond_dims = [1] + [4] * (L - 1) + [1]

mpo = MPO(L, phys_dim=phys)
for i in range(L - 1):
    for op in ('X', 'Y', 'Z'):
        mpo.add_couplings((op, op), 1.0, (i, i + 1))
mpo.build()

mps = MPS(L, phys_dim=phys, bond_dims=bond_dims)
show('norm env', mps.norm(mode='env'))
show('norm direct', mps.norm(mode='direct'))
show('Z0 env', mps.expectation({0: mpo.mapping['Z']}, mode='env'))
show('Z0 direct', mps.expectation({0: mpo.mapping['Z']}, mode='direct'))
show('H direct', mps.expectation(mpo))
print(f'{"entropy bond2":18s} {mps.entanglement_entropy(2):+.15f}')
print(f'{"entropy bond3 a2":18s} {mps.entanglement_entropy(3, alpha=2.0):+.15f}')

mps.move_center_to(0)
stick = Broomstick(mps, mpo, max_bond_dim=32, svd_tol=1e-14)
stick.sweep(num_sweeps=4, compute_variance=True)
print(f'{"energy":18s} {stick.energy:+.15f}')
print(f'{"variance":18s} {stick.compute_variance():+.15e}')
print('bond dims         ', mps.bond_dim)
print('2*mpo             ', (2.0 * mpo).bond_dim)
print('mpo*mpo           ', (mpo * mpo).bond_dim)
print('mpo+dict          ', (mpo + {0: Tensor.from_torch(torch.eye(phys, dtype=torch.complex128, device=CUDA))}).bond_dim)

path = os.path.join(os.path.dirname(__file__), 'check_state.pkl')
stick.save_as_pkl(path)
show('pickle energy', Broomstick.load_from_pkl(path).energy)
