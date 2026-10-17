"""The splittings of the block-sparse tensor of ``mpsdmrg``.

Exposes ``svd``, ``qr``, ``rq``, ``svdvals``, ``norm``, ``inner``, ``trace`` and ``absorb``; each
acts on a ``Tensor`` of the tensor layer and returns the factors as ``Tensor``.
"""
from mpsdmrg import ext as _engine
from typing import Any, Sequence, cast

import torch

from mpsdmrg.ext import Tensor, inner, norm, qr, rq, svdvals, trace  # noqa: F401

__all__ = ['absorb', 'inner', 'norm', 'qr', 'rq', 'svd', 'svdvals', 'trace']


def svd(t: Tensor, left: Sequence[int], right: Sequence[int],
        **kwargs: Any) -> tuple[Tensor, torch.Tensor, Tensor]:
    '''
    Singular value decomposition of a tensor over two leg groups.

    Args:
        t: the tensor to split.
        left: the legs of the first factor, in order.
        right: the legs of the second factor, in order.
        **kwargs: forwarded to the engine (``max_bond``, ``tol``, ``direction``).
    Returns:
        tuple: (first factor, singular values, second factor).
    '''
    if any(leg.resolved for leg in t.legs):
        first, values, second = _engine.svd(cast(Tensor, t).unresolve(), left, right, **kwargs)
        return cast(Tensor, first).resolve(), values, cast(Tensor, second).resolve()
    return cast(tuple[Tensor, torch.Tensor, Tensor],
                _engine.svd(t, left, right, **kwargs))


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
    leg = second.legs[0]
    if leg.resolved:
        # a resolved bond window stores each copy over its whole multiplet, so one singular value
        # scales sector_dim elements rather than one
        widths = [leg.slices[i + 1] - leg.slices[i] for i in range(len(leg.slices) - 1)]
        dims = [second.group.sector_dim(leg.sector(i)) for i in range(len(widths))]
        repeat = [dim for dim, width in zip(dims, widths) for _ in range(width // dim)]
        values = values.repeat_interleave(torch.tensor(repeat, device=values.device))
    return second * values.reshape(-1, 1, 1)
