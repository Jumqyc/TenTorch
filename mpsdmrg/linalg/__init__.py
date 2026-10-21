"""The splittings of the block-sparse tensor of ``mpsdmrg``.

Exposes ``svd``, ``qr``, ``rq``, ``svdvals``, ``norm``, ``inner``, ``trace``, ``absorb`` and
``scale``; each acts on a ``Tensor`` of the tensor layer and returns the factors as ``Tensor``.
"""
from mpsdmrg import ext as _engine
from typing import Any, Sequence, cast

import string

import torch

from mpsdmrg.ext import Tensor, cached_einsum, inner, norm, qr, rq, svdvals, trace  # noqa: F401

__all__ = ['absorb', 'inner', 'norm', 'qr', 'rq', 'scale', 'svd', 'svdvals', 'trace']


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
    return cast(tuple[Tensor, torch.Tensor, Tensor],
                _engine.svd(t, left, right, **kwargs))


def scale(t: Tensor, leg: int, values: torch.Tensor) -> Tensor:
    '''
    Multiply one leg of a tensor by a vector, one value per copy of the leg.

    The diagonal tensor that carries the values is contracted with *leg*, so the result has
    the same legs as *t* and every entry is scaled by the value of its copy along *leg*.

    Args:
        t: the tensor to scale.
        leg: the position of the leg to scale.
        values: one value per copy of that leg, in the block order the leg lists its sectors.
    Returns:
        Tensor: the scaled tensor, with the legs of *t*.
    Raises:
        ValueError: If *values* does not hold one entry per copy of *leg*.
    '''
    bond = t.legs[leg]
    blocks, at = [], 0
    for qi in range(bond.nblocks):
        copies = bond.multiplicity(qi)
        blocks.append(torch.diag(values[at:at + copies]))
        at += copies
    if at != values.numel():
        raise ValueError(f'scaling leg {leg} takes {at} values, one per copy of the leg, '
                         f'not {values.numel()}')
    dtype = t.blocks[0].dtype if t.blocks else torch.complex128
    # the diagonal carries the leg and its dual, so the output leg is the leg of *t* and
    # lands on the side that leg sits on
    codomain = 2 if leg < len(t.map()['codomain']) else 1
    diagonal = Tensor(
        charges=[tuple(bond.sector(qi)) for qi in range(bond.nblocks)],
        blocks=[block.to(dtype) for block in blocks],
        legs=(bond.conj(), bond),
        qtotal=bond.group.trivial_sector,
        codomain=codomain)
    count = len(t.legs)
    letters = string.ascii_lowercase
    kept = letters[count]
    spec = (f'{letters[:count]},{letters[leg]}{kept}'
            f'->{letters[:leg]}{kept}{letters[leg + 1:count]}')
    return cached_einsum(spec, t, diagonal)


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
    return scale(second, 0, values)
