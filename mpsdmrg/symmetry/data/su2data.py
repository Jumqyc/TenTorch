"""Clebsch-Gordan coefficients of SU(2), labelled by twice the spin.

Exposes ``fusion_tensor``: the structural tensor of one fusion vertex,
:math:`X^{j_1 j_2}_{j_3}`, with ``j = 2 * spin`` so that the labels match the sectors of
:class:`mpsdmrg.symmetry.SU2Symmetry`, and ``recoupling``: the overlap of the two coupling
schemes of three spins.

The coefficients are the Condon-Shortley ones, real float64, so
:math:`X[k_1, k_2, k_3] = \\langle j_1 m_1; j_2 m_2 | j_3 m_3 \\rangle` with
``m = 2 k - j``. For every allowed :math:`j_3` and every :math:`m_3` the vector over
:math:`(m_1, m_2)` has unit norm, and for every :math:`(m_1, m_2)` the entries over
:math:`(j_3, m_3)` are complete.

The coefficients come from the C++ symmetry engine, which caches them per spin pair and
device; this module only keeps the names of the data layer.
"""
from __future__ import annotations

from functools import lru_cache

import torch
from torch import Tensor

from base import CUDA
from mpsdmrg.symmetry import cpp


def _table(j1: int, j2: int, device: torch.device = CUDA) -> dict[int, Tensor]:
    '''
    Clebsch-Gordan coefficients of one spin pair.

    Args:
        j1: twice the first spin.
        j2: twice the second spin.
        device: device of the returned tensors.
    Returns:
        ``{j3: x}`` for every allowed ``j3``, with ``x`` of shape
        ``(j1 + 1, j2 + 1, j3 + 1)`` and ``x[k1, k2, k3] = <j1 m1; j2 m2 | j3 m3>``.
    '''
    return cpp.fusion_table(j1, j2, device)


@lru_cache(maxsize=None)
def _cached_table(j1: int, j2: int, device: torch.device) -> dict[int, Tensor]:
    '''The table of ``_table``, kept for reuse; the entries are read-only.'''
    return _table(j1, j2, device)


def fusion_tensor(j1: int, j2: int, j3: int, device: torch.device = CUDA) -> Tensor:
    '''
    Clebsch-Gordan coefficients of one fusion vertex.

    Args:
        j1: twice the first spin.
        j2: twice the second spin.
        j3: twice the fused spin.
        device: device of the returned tensor.
    Returns:
        ``x`` of shape ``(j1 + 1, j2 + 1, j3 + 1)`` with
        ``x[k1, k2, k3] = <j1 m1; j2 m2 | j3 m3>``, ``m = 2 k - j``.
    Raises:
        ValueError: If ``j1`` and ``j2`` do not fuse to ``j3``.
    '''
    return cpp.su2_symmetry.fusion_tensor((j1,), (j2,), (j3,), device)


@lru_cache(maxsize=None)
def _cached_recoupling(j1: int, j2: int, j3: int, j4: int, device: torch.device
                       ) -> tuple[tuple[int, ...], tuple[int, ...], Tensor]:
    '''The recoupling of ``recoupling``, kept for reuse; the tensor is read-only.'''
    rows, cols, f = cpp.recoupling(j1, j2, j3, j4)
    return tuple(rows), tuple(cols), f.to(device)


def recoupling(j1: int, j2: int, j3: int, j4: int, device: torch.device = CUDA
               ) -> tuple[tuple[int, ...], tuple[int, ...], Tensor]:
    '''
    Overlap of the two coupling schemes of three spins, :math:`j_1` with :math:`j_2`
    first against :math:`j_2` with :math:`j_3` first.

    Args:
        j1: twice the first spin.
        j2: twice the second spin.
        j3: twice the third spin.
        j4: twice the total spin.
        device: device of the returned tensor.
    Returns:
        ``(rows, cols, f)`` with ``rows`` the left intermediate spins, ``cols`` the right
        ones, and ``f[i, k]`` real float64, the overlap of ``rows[i]`` with ``cols[k]``.
    Raises:
        ValueError: If the spins do not couple to ``j4``.
    '''
    rows, cols, f = _cached_recoupling(j1, j2, j3, j4, device)
    if not rows or not cols:
        raise ValueError(f'Spins {j1 / 2}, {j2 / 2} and {j3 / 2} do not couple to {j4 / 2}.')
    return rows, cols, f
