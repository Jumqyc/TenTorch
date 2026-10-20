"""Scratch check: charged U(1) tensors — block structure, factorisations, contractions."""
import torch

from mpsdmrg.ext import _engine  # the leg type is engine-internal

Leg = _engine.Leg
from mpsdmrg.tensor import Tensor, cached_einsum
from mpsdmrg import linalg
from mpsdmrg.symmetry.cat import no_symmetry, su2_symmetry, u1_symmetry

torch.manual_seed(0)
DTYPE, DEVICE, u1 = torch.complex128, torch.device('cuda'), u1_symmetry

left = Leg(u1, ((0,),), (0, 1), qconj=1)
phys = Leg(u1, ((-1,), (1,)), (0, 1, 2), qconj=1)
right = Leg(u1, ((-1,), (1,)), (0, 1, 2), qconj=-1)

# only the entries whose charges fuse to the total are nonzero, so the charged engine and its
# charge-blind twin hold the same data and every comparison below is entry for entry
dense = torch.zeros((1, 2, 2), dtype=DTYPE, device=DEVICE)
dense[0, 0, 0], dense[0, 1, 1] = 0.7, -0.3
t = Tensor.from_torch(dense, legs=(left, phys, right), qtotal=(0,))
plain_legs = tuple(Leg(no_symmetry, ((0,),), (0, dim), qconj=leg.qconj)
                   for leg, dim in zip((left, phys, right), dense.shape))
plain = Tensor.from_torch(dense, legs=plain_legs, qtotal=(0,))


def entries(tensor: Tensor) -> torch.Tensor:
    """The magnitudes of every stored block entry, sorted: a charge-blind fingerprint."""
    return torch.sort(torch.cat([block.abs().reshape(-1) for block in tensor.blocks])).values


print('blocks         ', len(t.blocks), 'qdata', t.qdata)
print('stored entries ', sum(int(block.numel()) for block in t.blocks), 'of', dense.numel())
print('allowed kept   ', bool(t.blocks[0].reshape(-1)[0] == dense[0, 0, 0]
                              and t.blocks[1].reshape(-1)[0] == dense[0, 1, 1]))

u, s, vh = linalg.svd(t, left=(0, 1), right=(2,))
print('svd            ', 'K', u.shape[-1], 's', s.shape[0], 'qconj', u.legs[-1].qconj, vh.legs[0].qconj)
print('svd recon      ', f'{float(linalg.norm(cached_einsum("abk,kc->abc", u, vh * s.view(-1, 1)) - t)):.3e}')
q, r = linalg.qr(t, left=(0, 1), right=(2,))
print('qr recon       ', f'{float(linalg.norm(cached_einsum("abk,kc->abc", q, r) - t)):.3e}')
r2, q2 = linalg.rq(t, left=(0, 1), right=(2,))
print('rq recon       ', f'{float(linalg.norm(cached_einsum("abk,kc->abc", r2, q2) - t)):.3e}')
eye = Tensor.from_torch(torch.eye(q.shape[-1], dtype=DTYPE, device=DEVICE),
                        legs=(q.legs[-1].conj(), q.legs[-1]), qtotal=(0,))
print('qr orthonormal ', f'{float(linalg.norm(cached_einsum("abk,abj->kj", q.conj(), q) - eye)):.3e}')

u1t, s1, vh1 = linalg.svd(t, left=(0, 1), right=(2,), max_bond=1, tol=0.0)
print('truncated      ', 'K', u1t.shape[-1], 's', s1.shape[0],
      'norm', f'{float(linalg.norm(cached_einsum("abk,kc->abc", u1t, vh1 * s1.view(-1, 1)))):.12f}',
      '<=', f'{float(linalg.norm(t)):.12f}')
vals = torch.sort(linalg.svdvals(t, left=(0, 1), right=(2,))).values
plain_vals = torch.sort(linalg.svdvals(plain, left=(0, 1), right=(2,))).values
print('svdvals        ', bool(torch.allclose(vals, plain_vals)))
flat = t.combine_legs(((0, 1), (2,)))
print('combine_legs   ', flat.shape, 'norm equal', bool(torch.allclose(linalg.norm(flat), linalg.norm(t))),
      'entries equal', bool(torch.allclose(entries(flat), entries(t))))
print('norm / inner   ', bool(torch.allclose(linalg.norm(t), linalg.norm(plain))),
      f'{abs(cached_einsum("abc,abc->", t.conj(), t).item() - linalg.inner(t, t).item()):.3e}')
print('conj dualises  ', t.conj().legs[0].qconj, t.conj().qtotal, t.conj().shape)

x_dense = torch.zeros((2, 2), dtype=DTYPE, device=DEVICE)
x_dense[0, 0], x_dense[1, 1] = 0.4, -0.9
x = Tensor.from_torch(x_dense, legs=(right.conj(), right), qtotal=(0,))
x_plain = Tensor.from_torch(x_dense, legs=(plain_legs[2].conj(), plain_legs[2]), qtotal=(0,))
print('x stored       ', sum(int(block.numel()) for block in x.blocks), 'of 4 entries')
print('2-op charged   ', bool(torch.allclose(linalg.norm(cached_einsum('abc,cd->abd', t, x)),
                                             linalg.norm(cached_einsum('abc,cd->abd', plain, x_plain)))))
scalar_charged = cached_einsum('abc,cd,abd->', t, x, t.conj()).item()
scalar_plain = cached_einsum('abc,cd,abd->', plain, x_plain, plain.conj()).item()
print('3-op charged   ', f'{abs(scalar_charged - scalar_plain):.3e}')

for name, call in (('su2 rand       ', lambda: Tensor.rand((2, 2), group=su2_symmetry)),
                   ('mixed einsum   ', lambda: cached_einsum('ab,bc->ac', t, torch.eye(2, dtype=DTYPE, device=DEVICE))),
                   ('bad qtotal     ', lambda: Tensor([t.blocks[0]], (t.qdata[0],), t.legs, (1,), u1)),
                   ('no resolve     ', lambda: __import__('mpsdmrg.tensor', fromlist=['resolve']).resolve(t))):
    try:
        call()
        print(name, 'NO RAISE')
    except Exception as exc:
        print(name, type(exc).__name__, exc)

print('U1 CHECK OK')
