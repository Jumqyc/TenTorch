"""The block-sparse tensor of ``mpsdmrg``.

Exposes ``Tensor``, the symmetry-adapted container whose blocks carry the fusion trees of the
group layer and the ``cached_einsum`` contraction over them.  The leg algebra a model
layer needs -- the leg of a bare dimension, a bond leg, fusion and the cap -- is
re-exported here, so nothing above the tensor layer names a leg type.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import opt_einsum
import torch

from mpsdmrg.ext import _engine
from mpsdmrg.ext import Tensor  # noqa: F401
from mpsdmrg.ext import bond_leg, capped_leg, fuse_legs, site_leg  # noqa: F401
from mpsdmrg.ext import cached_einsum as _engine_einsum

_PLANS: dict[tuple, list] = {}


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
      tensors: the operands, at least one of them a charged ``Tensor``.
    Returns:
      The contraction.
    '''
    operands = list(tensors)
    for inds, _, equation, _, _ in _plan(spec, tuple(tuple(t.shape) for t in tensors)):
        # the plan lists a pair from the larger position down and names its operands in that
        # order, so popping in that order hands the engine the operands its equation expects
        pair = [operands.pop(position) for position in inds]
        operands.append(_engine_einsum(equation, *pair))
    return operands[0]


def cached_einsum(spec: str, *tensors: Any) -> Tensor:
    '''
    Contract the operands with the given equation.
    Args:
      spec: the einsum equation, with an explicit output; every label appears once in the output
        or exactly twice among the operands.
      tensors: the operands.  A charged ``Tensor`` makes the contraction block by block, and
        three or more operands go through the pairwise plan of opt_einsum, so that no step of a
        contraction touches more than two blocks at a time.
    Returns:
      The contraction.
    '''
    # the Clebsch-Gordan paths weigh reduced blocks, while a resolved operand stores its
    # components densely, so a mixed contraction brings the resolved side to reduced data first
    if not all(isinstance(t, Tensor) and all(leg.resolved for leg in t.legs) for t in tensors):
        tensors = tuple(t.unresolve() if isinstance(t, Tensor)
                        and any(leg.resolved for leg in t.legs) else t for t in tensors)
    if len(tensors) > 2 and any(isinstance(t, Tensor) and not t.is_dense for t in tensors):
        return _pairwise(spec, tensors)
    return _engine_einsum(spec, *tensors)


def resolve(tensor: Tensor) -> Tensor:
    '''
    The same tensor with every leg marked resolved.

    Args:
      tensor: a tensor whose total charge is known, so that its dense expansion exists.

    Returns:
      A tensor with the same legs up to the resolved flag and the same dense expansion, whose
      blocks now store the component data a bare operator matrix needs to act on it.
    '''
    legs = tuple(
        _engine.Leg(leg.group, leg.sectors, leg.slices, leg.qconj, True) for leg in tensor.legs)
    return Tensor.from_torch(tensor.to_torch(), legs=legs, qtotal=tensor.qtotal)
