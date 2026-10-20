"""Scratch check: the CGO consumer still works after the linalg unification."""
import torch

from CGO import mps as cgo  # noqa: F401  registers Broomstick.CGO
from mpsdmrg import Broomstick, MPO, MPS
import mpsdmrg.tensor as tensor

torch.manual_seed(0)
L, phys = 12, 2
mpo = MPO(L, phys_dim=phys)
for i in range(L - 1):
    for op in ('X', 'Y', 'Z'):
        mpo.add_couplings((op, op), 1.0, (i, i + 1))
mpo.build()
mps = MPS(L, phys_dim=phys, bond_dims=[1] + [8] * (L - 1) + [1])
mps.move_center_to(0)
stick = Broomstick(mps, mpo, max_bond_dim=8, svd_tol=1e-14)
stick.sweep(num_sweeps=1)

z = stick.Hamiltonian.mapping['Z']
M, B, X, info = cgo.CGO_projections(stick, l=2, operators={6: z}, num_sample=40, seed=0, return_info=True)
print('cache re-export', hasattr(tensor, '_expr_cache'))
print('M/B/X raw torch', bool(all(type(m) is torch.Tensor for m in M)
                            and type(B) is torch.Tensor and type(X) is torch.Tensor))
print('shapes', tuple(M[0].shape), tuple(B.shape), tuple(X.shape), 'q', info['q'], 'rank', info['rank'])
print('B hermitian', bool(torch.allclose(B, B.conj().T)))

exact = stick.state.expectation({6: z}, mode='direct')
lower, upper = stick.CGO(l=1, operators={6: z}, num_sample=4, seed=0)
print('exact', f'{float(exact.real):+.9f}', 'CGO', f'[{float(lower.real):+.9f}, {float(upper.real):+.9f}]')
print('brackets exact', bool(float(lower.real) <= float(exact.real) <= float(upper.real)))
print('CGO OK')
