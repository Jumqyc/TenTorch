"""The block-sparse tensor of ``tentorch``.

Exposes ``Tensor``, the symmetry-adapted container whose blocks carry the fusion trees of the
group layer and the ``cached_einsum`` contraction over them.  The leg algebra a model
layer needs -- the leg of a bare dimension, the leg of one multiplet, a bond leg, fusion and
the cap -- is re-exported here, so nothing above the tensor layer names a leg type.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import opt_einsum
import torch

from tentorch._ext import _engine
from tentorch._ext import Group, Tensor  # noqa: F401
from tentorch._ext import bond_leg, capped_leg, fuse_legs, multiplet_leg, site_leg  # noqa: F401
from tentorch._ext import cached_einsum as _engine_einsum

_PLANS: dict[tuple, list] = {}


def phys_leg(group: Group, phys_dim: int, site_sector: tuple[int, ...] | None = None) -> Any:
    '''
    The physical leg of one site.

    Args:
        group: the symmetry group of the chain.
        phys_dim: the dimension one site has, the dimension the leg must have.
        site_sector: the multiplet one site lives in; a group whose multiplets are not told
            apart by their dimension needs it, an abelian group does not.
    Returns:
        The leg of that site.
    Raises:
        ValueError: If a multiplet-carrying group is given without ``site_sector``, or if the
            named multiplet does not have dimension ``phys_dim``.
    '''
    if site_sector is None:
        if not group.is_abelian:
            raise ValueError(
                f'{group.name} names a site by its multiplet, not by its dimension: pass '
                f'site_sector, e.g. (1,) for a spin-1/2 site of SU(2) or (1, 0) for the '
                f'fundamental of SU(3).')
        return site_leg(group, phys_dim)
    leg = multiplet_leg(group, site_sector)
    if leg.dim != phys_dim:
        raise ValueError(f'The multiplet {tuple(site_sector)} has dimension {leg.dim}, '
                         f'not {phys_dim}.')
    return leg


def _clear_caches() -> None:
    '''
    Drop every contraction plan this layer and the engine keep.
    Args:
      None.
    Returns:
      None.
    '''
    _PLANS.clear()
    _engine.clear_caches()


# the hook a profiling script calls to drop every cached contraction
_expr_cache = SimpleNamespace(clear = _clear_caches)


def _plan(spec: str, shapes: tuple[tuple[int, ...], ...]) -> list:
    '''
    The pairwise contraction plan of one equation.
    Args:
      spec: the einsum equation, with an explicit output.
      shapes: the dense shape of every operand; the blocks do not change the plan.
    Returns:
      The opt_einsum ``contraction_list``: one entry per two-operand step.
    '''
    key = (spec, shapes)
    plan = _PLANS.get(key)
    if plan is None:
        plan = opt_einsum.contract_path(spec, *shapes, shapes=True)[1].contraction_list
        _PLANS[key] = plan
    return plan


def _pairwise(spec: str, tensors: tuple[Any, ...]) -> Tensor:
    '''
    Contract three or more operands as a sequence of two-operand contractions.
    Args:
      spec: the einsum equation, with an explicit output.
      tensors: dense, charged, or raw operands with the shapes named by ``spec``.
    Returns:
      The contraction.
    '''
    operands = list(tensors)
    for inds, _, equation, _, _ in _plan(spec, tuple(tuple(t.shape) for t in tensors)):
        if not set(equation.split('->')[0].split(',')[0]) & set(
                equation.split('->')[0].split(',')[1]):
            # a step that shares no label is an outer product; the engine contracts only tensors
            # that glue a leg, so the whole equation goes through its own greedy order instead
            return _engine_einsum(spec, *tensors)
        # the plan lists a pair from the larger position down and names its operands in that
        # order, so popping in that order hands the engine the operands its equation expects
        pair = [operands.pop(position) for position in inds]
        # a step over raw arrays is a plain dense contraction; only a charged operand goes to
        # the engine, whose operands carry legs
        operands.append(_engine_einsum(equation, *pair) if isinstance(pair[0], Tensor)
                        else torch.einsum(equation, *pair))
    return operands[0]


def cached_einsum(spec: str, *tensors: Any) -> Tensor:
    '''
    Contract the operands with the given equation.
    Args:
      spec: the einsum equation, with an explicit output; every label appears once in the output
        or exactly twice among the operands.
      tensors: the operands.  A charged ``Tensor`` makes the contraction block by block.
        With three or more operands, charged calls and dense/raw calls of one dtype use the
        cached pairwise plan of opt_einsum to avoid large intermediates. Mixed-dtype dense calls
        keep the engine's original order to preserve operation-dependent dtype promotion.
    Returns:
      The contraction.
    '''
    if not any(isinstance(t, Tensor) for t in tensors):
        # a call made of bare arrays is a plain dense contraction
        return torch.einsum(spec, *tensors)
    if len(tensors) > 2:
        if any(isinstance(t, Tensor) and not t.is_dense for t in tensors):
            return _pairwise(spec, tensors)
        raw = [t.blocks[0] if isinstance(t, Tensor) else t for t in tensors]
        if all(t.dtype == raw[0].dtype for t in raw[1:]):
            return _pairwise(spec, tensors)
    return _engine_einsum(spec, *tensors)
