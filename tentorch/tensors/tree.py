'''
The row layout of a block: which fusion tree, and which mix of multiplicities, each row stands for.

A block holds one row block per fusion tree of that side, and inside a tree's block one row per
combination of the multiplicities its legs carry, the last leg varying fastest.  Writing a state
entry by entry needs that arithmetic, which is what this module does.
'''

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


def entered(cat: FusionCategory, codomain: Sequence[Leg], domain: Sequence[Leg],
            entry: Callable[[FusionTree, tuple[int, ...], FusionTree, tuple[int, ...]], complex],
            root: Optional[Sector] = None) -> Tensor:
    '''
    A tensor with a codomain and a domain, written entry by entry.
    Args:
      cat: the category of the legs.
      codomain: the legs of the codomain, in order.
      domain: the legs of the domain, in order.
      entry: the coefficient of one entry, given the tree and the multiplicity copy of the codomain
        side and the same pair of the domain side.
      root: the charge the tensor carries, the trivial charge when it is None.
    Returns:
      The tensor.
    '''
    charge = cat.trivial() if root is None else root
    blocks: dict[BlockKey, torch.Tensor] = {}
    for key in block_keys(cat, list(codomain), list(domain), charge):
        out_rows = list(rows(cat, codomain, key.codomain))
        in_rows = list(rows(cat, domain, key.domain))
        data = torch.zeros((len(out_rows), len(in_rows)), dtype = cat.dtype, device = cat.device)
        for row, (tree_out, copy_out) in enumerate(out_rows):
            for column, (tree_in, copy_in) in enumerate(in_rows):
                data[row, column] = entry(tree_out, copy_out, tree_in, copy_in)
        blocks[key] = data
    return Tensor.from_tuple(cat, list(codomain), list(domain), charge, blocks)
