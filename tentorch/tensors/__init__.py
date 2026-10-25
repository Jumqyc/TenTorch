'''
The block-sparse tensor: the container, its legs, its blocks and the contractions over it.

A tensor holds a block per pair of fusion trees over its codomain and its domain, so the
charge structure never shows up as a dense array.
'''
from __future__ import annotations

from tentorch._ext import (
    BlockKey,
    Tensor,
    block_keys,
    contract,
    einsum,
    inner,
    is_close,
    norm,
)
from tentorch.symmetry import Leg, phys_leg, trivial_leg  # noqa: F401

__all__ = [
    'BlockKey', 'Leg', 'Tensor', 'block_keys', 'contract', 'einsum', 'inner', 'is_close', 'norm',
    'phys_leg', 'trivial_leg'
]
