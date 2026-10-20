"""Scratch check: the symmetry-invariant state that the engine builds on construction."""
import torch

from mpsdmrg import Broomstick
from mpsdmrg.symmetry.cat import su2_symmetry

torch.manual_seed(0)
L, PHYS = 6, 2
stick = Broomstick(L=L, phys_dim=PHYS, group=su2_symmetry, max_bond_dim=16, svd_tol=1e-14,
                   site_sector=(1,))
tensors = stick.state.tensors
print('group        ', tensors[0].group.name, 'is_dense', tensors[0].is_dense)
print('bond dims    ', stick.state.bond_dim)
print('qtotals      ', sorted({t.qtotal for t in tensors}))
print('leg sectors  ', [leg.sectors for leg in tensors[0].legs])
print('blocks       ', [len(t.blocks) for t in tensors])
print('shapes       ', [tuple(t.shape) for t in tensors])
print('norm direct  ', f'{stick.state.norm(mode="direct"):.12f}')
print('SU2 MPS CHECK DONE')
