"""The splittings of the block-sparse tensor of ``tentorch``.

Exposes ``svd``, ``qr``, ``rq``, ``svdvals``, ``norm``, ``inner``, ``trace``, ``lanczos`` and
``absorb``; each acts on a ``Tensor`` of the tensor layer and returns the factors as ``Tensor``.
"""
from tentorch import _ext as _engine
from typing import Any, Callable, Sequence, cast

import torch

from tentorch._ext import (  # noqa: F401
    Tensor, cached_einsum, inner, norm, qr, rq, svdvals, trace)

__all__ = ['absorb', 'inner', 'lanczos', 'norm', 'qr', 'rq', 'svd', 'svdvals', 'trace']


def svd(t: Tensor, left: Sequence[int], right: Sequence[int],
        **kwargs: Any) -> tuple[Tensor, torch.Tensor, Tensor]:
    '''
    Singular value decomposition of a tensor over two leg groups.

    Args:
        t: the tensor to split.
        left: the legs of the first factor, in order.
        right: the legs of the second factor, in order.
        **kwargs: forwarded to the engine (``max_bond``, ``tol``, ``absorb``).
    Returns:
        tuple: (first factor, singular values, second factor).
    '''
    return cast(tuple[Tensor, torch.Tensor, Tensor],
                _engine.svd(t, left, right, **kwargs))


def lanczos(func: Callable[[Tensor], Tensor], v: Tensor,
            n_iter: int = 8) -> tuple[float, Tensor]:
    '''
    Ground state of a Hermitian operator by the Lanczos recursion.

    Args:
        func: the action of the operator on a tensor with the legs of *v*.
        v: starting vector.
        n_iter: number of Krylov steps.
    Returns:
        tuple: the lowest Ritz value and its vector, on the legs of *v*.
    '''
    def pair(a: Tensor, b: Tensor) -> float:
        return inner(a, b).real

    start = v / pair(v, v) ** 0.5

    vecs: list[Tensor] = [start]
    alphas = torch.zeros(n_iter, dtype=torch.float64, device=start.device)
    betas = torch.zeros(n_iter, dtype=torch.float64, device=start.device)
    for k in range(n_iter):
        current = vecs[k]
        w = func(current)
        alphas[k] = pair(current, w)
        w = w - alphas[k] * current
        if k > 0:
            w = w - betas[k - 1] * vecs[k - 1]
        # the three-term recurrence loses orthogonality once the Krylov space fills up, and a
        # tridiagonal built on a dependent basis gives Ritz values no better than the starting
        # vector; folding the residual against every vector already built keeps it orthonormal
        for _ in range(2):
            for previous in vecs[:k + 1]:
                w = w - pair(previous, w) * previous
        beta = pair(w, w) ** 0.5
        betas[k] = beta
        vecs.append(w / beta if float(beta) > 0.0 else w * 0.0)
    # a negligible beta closes the Krylov space: the vectors generated after it are round-off,
    # and folding them into the tridiagonal mixes in directions that are not orthogonal to the
    # ones already found, which loses the normalisation of the ground vector
    closed = (betas < 1e-12).nonzero()
    n_eff = n_iter if closed.numel() == 0 else min(n_iter, int(closed[0]) + 1)
    T = torch.diag(alphas[:n_eff])
    if n_eff > 1:
        T = T + torch.diag(betas[:n_eff - 1], diagonal=1) \
              + torch.diag(betas[:n_eff - 1], diagonal=-1)
    eigvals, eigvecs = torch.linalg.eigh(T)
    ground = vecs[0] * eigvecs[0, 0].to(dtype=start.dtype)
    for k in range(1, n_eff):
        ground = ground + eigvecs[k, 0].to(dtype=start.dtype) * vecs[k]
    return float(eigvals[0]), ground


def absorb(second: Tensor, values: torch.Tensor) -> Tensor:
    '''
    Fold the singular values of a split into the second factor.

    Args:
        second: the second factor of ``svd``, whose first leg is the new bond.
        values: the singular values, in the order the split returned them; one per copy kept of
            every charge of the new bond.
    Returns:
        Tensor: the second factor with every bond window scaled by its singular value.
    '''
    return cast(Tensor, _engine.absorb(second, values))
