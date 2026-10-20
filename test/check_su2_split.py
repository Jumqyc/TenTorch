"""Scratch check: qr, rq and svd of multi-block, multi-sector SU(2) tensors.

Every case splits a tensor on an ordered (prefix) leg partition, contracts the factors back
into it and compares the dense Clebsch-Gordan expansion of the result with the original
tensor, then checks the canonical overlap of each isometry. The svd cases also contract the
three factors through the network, both with a bond diagonal and with the singular values
folded into either factor. ``NoSymmetry`` and ``U1Symmetry`` run through the same cases.
"""
from itertools import accumulate

import torch

from mpsdmrg.ext import _engine  # the leg type is engine-internal
from mpsdmrg.symmetry.cat import no_symmetry, su2_symmetry, u1_symmetry

Leg = _engine.Leg
from mpsdmrg.tensor import Tensor, cached_einsum
from mpsdmrg import linalg

torch.manual_seed(0)
DTYPE, DEVICE, TOL = torch.float64, torch.device('cuda'), 1e-12
ALPHA = 'abcdefghijklmnopqrstuvwxyz'


def spec(m: int, n: int) -> str:
    """The einsum subscripts that contract a left-factor/right-factor pair back into ``t``."""
    lefts, k, rights = ALPHA[:m], ALPHA[m], ALPHA[m + 1:m + 1 + n]
    return f'{lefts}{k},{k}{rights}->{lefts}{rights}'


def isometry(q: Tensor, m: int, n: int, side: str) -> torch.Tensor:
    """The overlap of an isometry factor with its conjugate, over the new leg."""
    if side == 'left':
        labels = ALPHA[:m]
        return cached_einsum(f'{labels}y,{labels}z->yz', q.conj(), q)
    labels = ALPHA[m + 1:m + 1 + n]
    return cached_einsum(f'y{labels},z{labels}->yz', q.conj(), q)


def check(t: Tensor, left: tuple[int, ...], right: tuple[int, ...]) -> tuple[float, float]:
    """Assert both factorisations rebuild ``t`` and that their isometry stays canonical.

    The overlap of the isometry with its conjugate contracts to the identity up to a sign
    per new-leg block, the phase the Clebsch-Gordan trees carry; an abelian tensor has no
    such phase and must contract to the identity exactly.
    """
    m, n = len(left), len(right)
    errors = []
    for tag, pair, side in (('qr', linalg.qr(t, left=left, right=right), 'left'),
                            ('rq', linalg.rq(t, left=left, right=right), 'right')):
        rebuilt = cached_einsum(spec(m, n), *pair)
        errors.append(float(linalg.norm(rebuilt - t)))
        iso = pair[0] if tag == 'qr' else pair[1]
        got = isometry(iso, m, n, side)
        for block in got.blocks:
            eye = torch.eye(block.shape[0], dtype=DTYPE, device=DEVICE)
            scalar = block.reshape(-1)[0]
            assert float((block - scalar * eye).abs().max()) < TOL
            assert float((scalar - 1.0).abs()) < TOL or not t.group.is_abelian
        assert errors[-1] < TOL
    assert len(t.qdata) > 1 or t.group is no_symmetry, 'every split case needs several blocks'
    return errors[0], errors[1]


def report(name: str, t: Tensor, left: tuple[int, ...], right: tuple[int, ...]) -> None:
    """Split one tensor, check it, and print the reconstruction errors."""
    qr_err, rq_err = check(t, left, right)
    print(f'{name:22s} blocks {len(t.qdata)}  legs {tuple(t.shape)}  '
          f'qr {qr_err:.2e}  rq {rq_err:.2e}')


def svd_spec(m: int, n: int) -> str:
    """The einsum subscripts that contract ``u``, the bond diagonal and ``vh`` into ``t``."""
    lefts, k, j, rights = ALPHA[:m], ALPHA[m], ALPHA[m + 1], ALPHA[m + 2:m + 2 + n]
    return f'{lefts}{k},{k}{j},{j}{rights}->{lefts}{rights}'


def bond_diagonal(leg: Leg, s: torch.Tensor) -> Tensor:
    """The bond operator ``diag(s)`` acting on the states of ``leg``.

    Its diagonal is the identity of the fusion tree, whose vertex carries one factor of the
    quantum dimension, so each multiplicity block is scaled by the square root of that
    dimension; for an abelian group the factor is one.
    """
    blocks, qdata, offset = [], [], 0
    for qi in range(leg.nblocks):
        multiplicity = leg.multiplicity(qi)
        blocks.append(leg.group.sector_dim(leg.sector(qi)) ** 0.5
                      * torch.diag(s[offset:offset + multiplicity]))
        qdata.append((qi, qi))
        offset += multiplicity
    return Tensor(blocks, tuple(qdata), (leg.conj(), leg), (0,), leg.group)


def scale_bond(factor: Tensor, s: torch.Tensor, at: int) -> Tensor:
    """Fold the singular values of each charge into the bond leg of ``factor``."""
    leg, at = factor.legs[at], at % len(factor.legs)
    starts = (0, *accumulate(leg.multiplicity(qi) for qi in range(leg.nblocks)))
    shape = [1] * len(factor.legs)
    shape[at] = -1
    return Tensor([b * s[starts[q[at]]:starts[q[at] + 1]].reshape(shape)
                   for b, q in zip(factor.blocks, factor.qdata)],
                  factor.qdata, factor.legs, factor.qtotal, factor.group)


def check_svd(t: Tensor, left: tuple[int, ...], right: tuple[int, ...]) -> tuple[float, float]:
    """Assert the svd factors rebuild ``t`` through the network, block by block.

    ``u`` carries the fusion vertex of the split, so contracting it with the bond diagonal
    and ``vh`` recovers ``t``, as does folding the singular values into either factor; the
    values themselves stay those of the partition, so their squares sum to ``||t||^2``.
    """
    m, n = len(left), len(right)
    u, s, vh = linalg.svd(t, left=left, right=right)
    rebuilt = cached_einsum(svd_spec(m, n), u, bond_diagonal(u.legs[-1], s), vh)
    table = dict(zip(t.qdata, t.blocks))
    block = max(float((b - table[q]).abs().max()) for q, b in zip(rebuilt.qdata, rebuilt.blocks))
    folded = (float(linalg.norm(cached_einsum(spec(m, n), u, scale_bond(vh, s, 0)) - t)),
              float(linalg.norm(cached_einsum(spec(m, n), scale_bond(u, s, -1), vh) - t)))
    values = linalg.svdvals(t, left=left, right=right)
    assert len(rebuilt.qdata) == len(t.qdata) and block < TOL
    assert float(linalg.norm(rebuilt - t)) < TOL and max(folded) < TOL
    assert abs(float((s * s).sum()) - float(linalg.norm(t)) ** 2) < TOL
    assert float((torch.sort(values).values - torch.sort(s).values).abs().max()) < TOL
    assert len(t.qdata) > 1 or t.group is no_symmetry, 'every split case needs several blocks'
    return float(linalg.norm(rebuilt - t)), max(folded)


def report_svd(name: str, t: Tensor, left: tuple[int, ...], right: tuple[int, ...]) -> None:
    """Split one tensor with ``svd``, check it, and print the reconstruction errors."""
    network, folded = check_svd(t, left, right)
    print(f'{name:22s} blocks {len(t.qdata)}  legs {tuple(t.shape)}  '
          f'svd {network:.2e}  folded {folded:.2e}')


def check_truncated(t: Tensor, left: tuple[int, ...], right: tuple[int, ...],
                    max_bond: int) -> float:
    """Assert a truncated svd leaves the discarded weight as the reconstruction error.

    The squared error of the rebuilt tensor is ``||t||^2`` minus its kept singular values
    squared, the bookkeeping the truncation error of a sweep relies on.
    """
    m, n = len(left), len(right)
    u, s, vh = linalg.svd(t, left=left, right=right, max_bond=max_bond)
    rebuilt = cached_einsum(spec(m, n), u, scale_bond(vh, s, 0))
    error = float(linalg.norm(rebuilt - t)) ** 2
    assert s.shape[0] <= max_bond
    assert abs(error - (float(linalg.norm(t)) ** 2 - float((s * s).sum()))) < TOL
    return error


for group, tag, bond, phys, right in (
        (su2_symmetry, 'su2', (((0,), (2,)), (0, 1, 4)), (((1,),), (0, 2)),
         (((1,), (3,)), (0, 2, 6))),
        (no_symmetry, 'none', (((0,),), (0, 4)), (((0,),), (0, 2)), (((0,),), (0, 6))),
        (u1_symmetry, 'u1', (((0,), (1,)), (0, 1, 4)), (((-1,), (1,)), (0, 1, 2)),
         (((-1,), (1,)), (0, 3, 6)))):
    # a dense tensor contracts on the dense path, which rebuilds its legs with qconj 1
    qconj = 1 if group is no_symmetry else -1
    bl = Leg(group, bond[0], bond[1])
    ph = Leg(group, phys[0], phys[1])
    br = Leg(group, right[0], right[1], qconj=qconj)
    three = Tensor.rand((4, 2, 6), legs=(bl, ph, br), qtotal=(0,), dtype=DTYPE, device=DEVICE)
    two_right = Leg(group, bond[0], bond[1], qconj=qconj)
    two = Tensor.rand((4, 4), legs=(bl, two_right), qtotal=(0,), dtype=DTYPE, device=DEVICE)
    report_svd(f'{tag} 2-leg left=(0)', two, (0,), (1,))
    report(f'{tag} 3-leg left=(0)', three, (0,), (1, 2))
    report_svd(f'{tag} 3-leg left=(0)', three, (0,), (1, 2))
    report(f'{tag} 3-leg left=(0,1)', three, (0, 1), (2,))
    report_svd(f'{tag} 3-leg left=(0,1)', three, (0, 1), (2,))
    four_right = Leg(group, bond[0], bond[1], qconj=qconj)
    four = Tensor.rand((4, 2, 2, 4), legs=(bl, ph, ph, four_right), qtotal=(0,), dtype=DTYPE,
                       device=DEVICE)
    report(f'{tag} 4-leg left=(0)', four, (0,), (1, 2, 3))
    report_svd(f'{tag} 4-leg left=(0)', four, (0,), (1, 2, 3))
    report(f'{tag} 4-leg left=(0,1)', four, (0, 1), (2, 3))
    report_svd(f'{tag} 4-leg left=(0,1)', four, (0, 1), (2, 3))
    report(f'{tag} 4-leg left=(0,1,2)', four, (0, 1, 2), (3,))
    report_svd(f'{tag} 4-leg left=(0,1,2)', four, (0, 1, 2), (3,))
    discarded = check_truncated(four, (0, 1), (2, 3), max_bond=2)
    print(f'{tag} 4-leg left=(0,1) max_bond 2  discarded {discarded:.2e}')

print('SU2 SPLIT CHECK OK')
