"""The splittings of the block-sparse tensor of ``mpsdmrg``.

Exposes ``svd``, ``qr``, ``rq``, ``svdvals``, ``norm``, ``inner`` and ``trace``; each acts on a
``Tensor`` of the tensor layer and returns the factors as ``Tensor``.
"""
from mpsdmrg.ext import inner, norm, qr, rq, svd, svdvals, trace  # noqa: F401

__all__ = ['inner', 'norm', 'qr', 'rq', 'svd', 'svdvals', 'trace']
