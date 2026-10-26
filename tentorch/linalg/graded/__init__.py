'''
The splittings of the block-sparse tensor: svd, qr and rq, and the values they return.

Every splitting takes the legs of the first factor as `left`; the legs named there are brought to
the codomain of the first factor, and the rest go to the second.  A bond that is already where the
split wants it costs nothing.
'''
from __future__ import annotations

from typing import Optional, Sequence

import torch

from tentorch._ext import Tensor, qr, rq, svd, svdvals

__all__ = ['qr', 'rq', 'svd', 'svdvals']


def split(tensor: Tensor, left: Optional[Sequence[int]] = None, max_bond: Optional[int] = None,
          tol: float = 0.0, absorb: int = 0) -> tuple[Tensor, torch.Tensor, Tensor]:
    '''
    Singular value decomposition over two groups of legs.
    Args:
      tensor: the tensor to split.
      left: the legs of the first factor, a list of leg positions over the codomain and then the
        domain; every leg is kept when it is None.
      max_bond: the number of multiplets of the new bond to keep, every one when it is None.
      tol: singular values at or below it are dropped.
      absorb: 0 leaves the values apart, 1 folds them into the first factor, 2 into the second and
        3 splits the root into both.
    Returns:
      The first factor, the values and the second factor, as a tuple.
    '''
    if left is None:
        return svd(tensor, max_bond, tol, absorb)
    return svd(tensor, list(left), max_bond, tol, absorb)


def qr_split(tensor: Tensor, left: Optional[Sequence[int]] = None) -> tuple[Tensor, Tensor]:
    '''
    QR decomposition over two groups of legs.
    Args:
      tensor: the tensor to split.
      left: the legs of the first factor, every leg when it is None.
    Returns:
      The orthogonal first factor and the triangular second one.
    '''
    return qr(tensor) if left is None else qr(tensor, list(left))


def rq_split(tensor: Tensor, left: Optional[Sequence[int]] = None) -> tuple[Tensor, Tensor]:
    '''
    RQ decomposition over two groups of legs.
    Args:
      tensor: the tensor to split.
      left: the legs of the first factor, every leg when it is None.
    Returns:
      The triangular first factor and the orthogonal second one.
    '''
    return rq(tensor) if left is None else rq(tensor, list(left))
