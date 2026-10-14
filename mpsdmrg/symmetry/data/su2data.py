"""Clebsch-Gordan coefficients of SU(2), labelled by twice the spin.

Exposes ``fusion_tensor``: the structural tensor of one fusion vertex,
:math:`X^{j_1 j_2}_{j_3}`, with ``j = 2 * spin`` so that the labels match the sectors of
:class:`mpsdmrg.symmetry.SU2Symmetry`.

The coefficients are the Condon-Shortley ones, real float64, so
:math:`X[k_1, k_2, k_3] = \\langle j_1 m_1; j_2 m_2 | j_3 m_3 \\rangle` with
``m = 2 k - j``. For every allowed :math:`j_3` and every :math:`m_3` the vector over
:math:`(m_1, m_2)` has unit norm, and for every :math:`(m_1, m_2)` the entries over
:math:`(j_3, m_3)` are complete.

The coefficients come from ``su2data.cpp``, built just in time with
``torch.utils.cpp_extension.load`` and cached in ``mpsdmrg/symmetry/data/build``, so only
the first import pays for the compilation; ``mpsdmrg/symmetry/data/build.py`` builds it
without importing this module.  The compiled functions work in float64 on the host, and
the tensors they return move to the requested device.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from functools import lru_cache
from typing import Any

import torch
from torch import Tensor

from base import CUDA

_HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(_HERE, 'build')
LIBRARY = os.path.join(BUILD_DIR, 'su2data_ext.so')

if os.environ.get('TORCH_EXTENSIONS_DIR') is None:
    os.environ['TORCH_EXTENSIONS_DIR'] = BUILD_DIR
os.makedirs(BUILD_DIR, exist_ok=True)

_extension: Any
if os.path.exists(LIBRARY):
    _spec = importlib.util.spec_from_file_location('su2data_ext', LIBRARY)
    if _spec is None or _spec.loader is None:
        raise ImportError(f'The extension at {LIBRARY} has no loader.')
    _extension = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_extension)
else:
    # the build needs ninja on PATH, which a bare .venv/bin/python does not add
    os.environ['PATH'] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get('PATH', '')
    from torch.utils.cpp_extension import load
    _extension = load(name='su2data_ext',
                      sources=[os.path.join(_HERE, 'su2data.cpp')],
                      extra_cflags=['-O3'],
                      build_directory=BUILD_DIR,
                      verbose=False)


def _table(j1: int, j2: int, device: torch.device) -> dict[int, Tensor]:
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
    return {j3: x.to(device) for j3, x in _extension.table(j1, j2).items()}


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
    table = _cached_table(j1, j2, device)
    if j3 not in table:
        raise ValueError(f'Spins {j1 / 2} and {j2 / 2} do not fuse to {j3 / 2}.')
    return table[j3]


def _recoupling(j1: int, j2: int, j3: int, j4: int,
                device: torch.device) -> tuple[tuple[int, ...], tuple[int, ...], Tensor]:
    '''
    Overlap of the two ways to couple three spins to a fourth.

    The left scheme couples :math:`j_1` with :math:`j_2` first, the right scheme
    :math:`j_2` with :math:`j_3` first; the entries are
    :math:`\\langle (j_1 j_2) j_{12}, j_3 | j_1, (j_2 j_3) j_{23} \\rangle`, evaluated
    from the same Clebsch-Gordan coefficients the fusion tensors hold.

    Args:
        j1: twice the first spin.
        j2: twice the second spin.
        j3: twice the third spin.
        j4: twice the total spin.
        device: device of the returned tensors.
    Returns:
        ``(rows, cols, f)`` with ``rows`` the intermediate spins of the left scheme,
        ``cols`` those of the right scheme, and ``f[i, k]`` the overlap of ``rows[i]``
        with ``cols[k]``.
    '''
    rows, cols, f = _extension.recoupling(j1, j2, j3, j4)
    return tuple(rows), tuple(cols), f.to(device)


@lru_cache(maxsize=None)
def _cached_recoupling(j1: int, j2: int, j3: int, j4: int,
                       device: torch.device) -> tuple[tuple[int, ...], tuple[int, ...], Tensor]:
    '''The recoupling of ``_recoupling``, kept for reuse; the tensor is read-only.'''
    return _recoupling(j1, j2, j3, j4, device)


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
