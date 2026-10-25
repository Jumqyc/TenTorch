'''
The row layout of a block: which fusion tree, and which mix of multiplicities, each row stands for.

A block holds one row block per fusion tree of that side, and inside a tree's block one row per
combination of the multiplicities its legs carry, the last leg varying fastest.  Writing a state
entry by entry needs that arithmetic, which is what this module does.
'''
from __future__ import annotations

import itertools
from typing import Callable, Iterator, Optional, Sequence

import torch

from tentorch._ext import BlockKey, FusionTree, Leg, Tensor, block_keys, fusion_trees
from tentorch.symmetry import FusionCategory, Sector


def block_of(leg: Leg, sector: Sector) -> int:
    '''
    The block of a leg that holds a sector.
    Args:
      leg: the leg to search.
      sector: the sector to find.
    Returns:
      The block index.
    Raises:
      ValueError: when the leg carries no block of that sector.
    '''
    for block in range(leg.nblocks()):
        if leg.sector(block) == sector:
            return block
    raise ValueError(f'the leg carries none of the sector {sector}')


def rows(cat: FusionCategory, legs: Sequence[Leg], root: Sector) -> Iterator[tuple[FusionTree, tuple[int, ...]]]:
    '''
    Every row of the block of a root, in the order the block stores them.
    Args:
      cat: the category of the legs.
      legs: the legs of the side, in order.
      root: the charge the side couples to.
    Returns:
      One entry per row: the tree, and the multiplicity copy of each leg.
    '''
    for tree in fusion_trees(cat, legs, root):
        multiplicities = [leg.multiplicity(block_of(leg, sector)) for leg, sector in zip(legs, tree.sectors)]
        yield from ((tree, copy) for copy in itertools.product(*[range(mult) for mult in multiplicities]))


def state(cat: FusionCategory, legs: Sequence[Leg],
          entry: Callable[[FusionTree, tuple[int, ...]], complex],
          root: Optional[Sector] = None) -> Tensor:
    '''
    A tensor with every leg on the codomain, written entry by entry.
    Args:
      cat: the category of the tensor.
      legs: the legs, in order.
      entry: the coefficient of a row, given its tree and the multiplicity copy of each leg.
      root: the charge the tensor carries, the trivial charge when it is None.
    Returns:
      The tensor.
    '''
    charge = cat.trivial() if root is None else root
    blocks: dict[BlockKey, torch.Tensor] = {}
    for key in block_keys(cat, list(legs), [], charge):
        data = torch.zeros((sum(1 for _ in rows(cat, legs, key.codomain)), 1),
                           dtype = cat.dtype,
                           device = cat.device)
        for row, (tree, copy) in enumerate(rows(cat, legs, key.codomain)):
            data[row, 0] = entry(tree, copy)
        blocks[key] = data
    return Tensor.from_tuple(cat, list(legs), [], charge, blocks)


def scalar(tensor: Tensor) -> complex:
    '''
    The value of a tensor that has no leg left, the result of closing a chain.
    Args:
      tensor: the tensor, with no leg.
    Returns:
      Its value.
    Raises:
      ValueError: when the tensor still has a leg.
    '''
    if tensor.nlegs() != 0:
        raise ValueError(f'a scalar needs a tensor with no leg, not {tensor.nlegs()}')
    blocks = tensor.to_tuple()[3]
    if len(blocks) != 1:
        raise ValueError(f'a scalar needs one block, not {len(blocks)}')
    return complex(next(iter(blocks.values())).item())
