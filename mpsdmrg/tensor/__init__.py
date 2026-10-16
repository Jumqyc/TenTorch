"""The block-sparse tensor of ``mpsdmrg``.

Exposes ``Tensor``, the symmetry-adapted container whose blocks carry the fusion trees of the
group layer and the ``cached_einsum`` contraction over them.
puts around every local update.
"""
from __future__ import annotations

from typing import Any

import opt_einsum
import torch

from mpsdmrg.ext import Tensor, _expr_cache  # noqa: F401
from mpsdmrg.ext import cached_einsum as _engine_einsum

# the guard the DMRG puts around every local update


_PLANS: dict[tuple, list] = {}


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
    if len(tensors) > 2 and any(isinstance(t, Tensor) and not t.is_dense for t in tensors):
        return _pairwise(spec, tensors)
    return _engine_einsum(spec, *tensors)
