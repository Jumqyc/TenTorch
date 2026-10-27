'''
The block-sparse tensor: the container, its legs, its blocks and the contractions over it.

A tensor holds a block per pair of fusion trees over its codomain and its domain, so the
charge structure never shows up as a dense array.
'''

from tentorch._ext import (
    BlockKey,
    Tensor,
    block_keys,
    codomain_legs,
    contract,
    domain_legs,
    einsum,
    inner,
    is_close,
    norm,
)
from tentorch.symmetry import Leg

__all__ = [
    'BlockKey', 'Leg', 'Tensor', 'block_keys', 'codomain_legs', 'contract', 'domain_legs', 'einsum',
    'inner', 'is_close', 'norm'
]
