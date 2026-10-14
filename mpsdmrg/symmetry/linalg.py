from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from itertools import accumulate
from math import prod
from typing import Callable, Concatenate, NamedTuple, ParamSpec, TypeVar

import torch

from mpsdmrg.symmetry.groups import Group, Sector
from mpsdmrg.symmetry.tensor import (BlockKey, FusionTree, Leg, Tensor, _couplings,
                                     _fused_charge, _leg_indices)

P = ParamSpec('P')
R = TypeVar('R')


class _Sector(NamedTuple):
    '''One fused charge of a split tensor: the charge, the matrix of its states, their
    dimensions, and the charge-block key the matrix came from.'''
    charge: Sector
    matrix: torch.Tensor
    left: tuple[tuple, ...]
    right: tuple[tuple, ...]
    left_dims: tuple[int, ...]
    right_dims: tuple[int, ...]
    qdata: BlockKey


def _split_charge(group: Group, legs: tuple[Leg, ...], qdata: BlockKey,
                  qtotal: Sector, left: tuple[int, ...]) -> Sector:
    '''
    Charge of the left legs of one block at a contiguous split.

    A non-abelian tensor keeps the couplings of its fusion tree in the block index, so the
    charge above the first legs is that stored coupling, not a fusion of their sectors.

    Args:
        group: the group of the tensor.
        legs: the legs of the tensor.
        qdata: the charge-block index and tree couplings of the block.
        qtotal: the total charge of the tensor.
        left: leg positions of the left part.
    Returns:
        The trivial sector for no left leg, the coupling above a proper prefix of the legs,
        and the total charge for all legs.
    Raises:
        NotImplementedError: If the left legs are not a contiguous prefix, since the flat
            fusion tree couples the legs in their order.
    '''
    if not left:
        return group.trivial_sector
    if len(left) == len(legs):
        return tuple(qtotal)
    if tuple(left) != tuple(range(len(left))):
        raise NotImplementedError(f'A {group.group_name} split needs the first legs in order, '
                                  f'not {left}.')
    return FusionTree.from_blocks(group, legs, qdata, qtotal).chain[len(left) - 1]


def _split_scale(group: Group, charge: Sector) -> float:
    '''
    Weight the isometry factor of a split at ``charge`` carries.

    The two factors hold one fusion vertex fewer than the tensor they rebuild, so the dense
    contraction of the pair recovers the tensor divided by the square root of the quantum
    dimension of that vertex, and the overlap of the isometry with its conjugate recovers
    the identity divided by the same dimension. Weighting the isometry by the square root
    of the dimension restores both.

    Args:
        group: the group of the tensor.
        charge: the fused charge of the new leg.
    Returns:
        The square root of the quantum dimension of ``charge``, one for an abelian group.
    '''
    return group.sector_dim(charge) ** 0.5


def _sector_matrices(t: Tensor, left: tuple[int, ...], right: tuple[int, ...]) -> list[_Sector]:
    '''
    Split a charged tensor into one dense matrix per fused charge of its left legs.

    The rows of a charge are the distinct states of its left legs, in the order they first
    appear, and its columns the same for its right legs. A state is the charge-block index
    of the legs on its side and, for a non-abelian group, the couplings of their fusion
    tree, so the two states of a block recover its whole key. Two stored blocks sharing
    their left state share its rows, and two sharing their right state share its columns;
    every other entry of the matrix is zero.

    Args:
        t: the tensor to split.
        left: leg positions forming the matrix rows, in order.
        right: leg positions forming the matrix columns, in order.
    Returns:
        Per charge, sorted by charge: the matrix and its row and column states with their
        dimensions.
    '''
    group, n, cut = t.group, len(t.legs), max(len(left) - 1, 0)
    by_charge: dict[Sector, list[int]] = {}
    for bi, q in enumerate(t.qdata):
        charge = (_fused_charge(group, t.legs, q, left) if group.is_abelian
                  else _split_charge(group, t.legs, q, t.qtotal, left))
        by_charge.setdefault(charge, []).append(bi)
    out = []
    for charge in sorted(by_charge):
        bis = by_charge[charge]
        left_at: dict[tuple, int] = {}
        right_at: dict[tuple, int] = {}
        states = {}
        for bi in bis:
            q = _leg_indices(t.qdata[bi], n)
            inner = _couplings(t.qdata[bi], n) if not group.is_abelian else ()
            states[bi] = (tuple(q[l] for l in left) + inner[:cut],
                          tuple(q[r] for r in right) + inner[cut:])
            for key, at in zip(states[bi], (left_at, right_at)):
                at.setdefault(key, len(at))
        left_states, right_states = tuple(left_at), tuple(right_at)
        left_dims = tuple(prod(t.legs[l].multiplicity(c[i]) for i, l in enumerate(left))
                          for c in left_states)
        right_dims = tuple(prod(t.legs[r].multiplicity(c[i]) for i, r in enumerate(right))
                           for c in right_states)
        left_slices = (0, *accumulate(left_dims))
        right_slices = (0, *accumulate(right_dims))
        mat = torch.zeros((left_slices[-1], right_slices[-1]), dtype=t.dtype, device=t.device)
        for bi in bis:
            lc, rc = left_at[states[bi][0]], right_at[states[bi][1]]
            rdim, cdim = left_dims[lc], right_dims[rc]
            mat[left_slices[lc]:left_slices[lc] + rdim, right_slices[rc]:right_slices[rc] + cdim] = (
                t.blocks[bi].permute(tuple(left) + tuple(right)).reshape(rdim, cdim))
        out.append(_Sector(charge, mat, left_states, right_states, left_dims, right_dims,
                           t.qdata[bis[0]]))
    return out


def _factor_blocks(t: Tensor,
                   left: tuple[int, ...],
                   right: tuple[int, ...],
                   part: _Sector,
                   first: torch.Tensor,
                   second: torch.Tensor,
                   kept: slice | torch.Tensor,
                   rank: int) -> tuple[list[torch.Tensor], list[tuple[int, ...]],
                                       list[torch.Tensor], list[tuple[int, ...]]]:
    '''
    Blocks of the two factors of one fused charge.

    The rows of ``first`` and the columns of ``second`` are the sector's left and right
    index. One block is emitted per distinct state, so each factor keeps its own block
    structure and the two share the new leg.

    Args:
        t: the tensor being split.
        left: leg positions of the first factor's kept legs, in order.
        right: leg positions of the second factor's kept legs, in order.
        part: the sector split of ``t``.
        first: the row-indexed factor matrix.
        second: the column-indexed factor matrix.
        kept: the columns of ``first`` and the rows of ``second`` to keep, a slice or indices.
        rank: charge-block index of the new leg within this sector.
    Returns:
        The blocks and charge-block indices of the first and of the second factor.
    '''
    left_slices, right_slices = (0, *accumulate(part.left_dims)), (0, *accumulate(part.right_dims))
    first_kept, second_kept = first[:, kept], second[kept, :]
    k = first_kept.shape[1]
    first_blocks, first_qdata, second_blocks, second_qdata = [], [], [], []
    for lc, combo in enumerate(part.left):
        row, rdim = left_slices[lc], part.left_dims[lc]
        first_blocks.append(first_kept[row:row + rdim].reshape(
            *[t.legs[l].multiplicity(combo[i]) for i, l in enumerate(left)], k))
        first_qdata.append(combo[:len(left)] + (rank,) + combo[len(left):])
    for rc, combo in enumerate(part.right):
        col, cdim = right_slices[rc], part.right_dims[rc]
        second_blocks.append(second_kept[:, col:col + cdim].reshape(
            k, *[t.legs[r].multiplicity(combo[i]) for i, r in enumerate(right)]))
        second_qdata.append((rank,) + combo)
    return first_blocks, first_qdata, second_blocks, second_qdata


def _assemble_split(t: Tensor,
                    left: tuple[int, ...],
                    right: tuple[int, ...],
                    parts: list[_Sector],
                    factors: list[tuple[torch.Tensor, torch.Tensor]],
                    charges: list[Sector],
                    kept: Sequence[slice | torch.Tensor] | None,
                    first_qtotal: Sector,
                    second_qtotal: Sector) -> tuple[Tensor, Tensor]:
    '''
    Assemble the two factors of a per-charge split into tensors sharing the new leg.

    Args:
        t: the tensor being split.
        left: leg positions of the first factor's kept legs, in order.
        right: leg positions of the second factor's kept legs, in order.
        parts: the sector splits of ``t``, in the order of ``charges``.
        factors: the row-indexed and column-indexed factor matrices of every sector.
        charges: the charge of every sector's new leg.
        kept: the kept columns of each first factor and rows of each second factor, a
            slice or index tensor per sector; all if None.
        first_qtotal: total charge of the first factor.
        second_qtotal: total charge of the second factor.
    Returns:
        ``(first, second)`` with ``first`` carrying ``(*left legs, new leg)`` and
        ``second`` carrying ``(new leg, *right legs)``; empty sectors are dropped.
    '''
    if kept is None:
        kept = [slice(None)] * len(parts)
    entries = []
    for charge, k, (first, _) in zip(charges, kept, factors):
        if isinstance(k, torch.Tensor) and k.shape[0] == 0:
            continue
        multiplicity = int(k.shape[0]) if isinstance(k, torch.Tensor) else int(first.shape[1])
        entries.append((charge, multiplicity * t.group.sector_dim(charge)))
    new_leg, rank = Leg.from_blocks(t.group, entries, -1)
    block_of_charge = {charge: rank[i] for i, (charge, _) in enumerate(entries)}
    first_blocks, first_qdata, second_blocks, second_qdata = [], [], [], []
    for part, charge, (first_mat, second_mat), k in zip(parts, charges, factors, kept):
        if isinstance(k, torch.Tensor) and k.shape[0] == 0:
            continue
        blocks = _factor_blocks(
            t, left, right, part, first_mat, second_mat, k, block_of_charge[charge])
        first_blocks += blocks[0]
        first_qdata += blocks[1]
        second_blocks += blocks[2]
        second_qdata += blocks[3]
    first = Tensor(first_blocks, tuple(first_qdata), tuple(t.legs[l] for l in left) + (new_leg,),
                   first_qtotal, t.group)
    second = Tensor(second_blocks, tuple(second_qdata),
                    (new_leg.conj(), *[t.legs[r] for r in right]), second_qtotal, t.group)
    return first, second


def _with_sectors(
        func: Callable[Concatenate[Tensor, tuple[int, ...], tuple[int, ...], list[_Sector], P], R]
) -> Callable[Concatenate[Tensor, tuple[int, ...], tuple[int, ...], P], R]:
    '''
    Check a leg partition and hand a split primitive the sector matrices of that partition.

    The decorated function takes the tensor, the two leg partitions, the list of
    ``_Sector`` splits of that partition, and then its own arguments.
    Raises:
        ValueError: If the two groups do not cover each leg index exactly once.
    '''
    @wraps(func)
    def wrapper(t: Tensor,
                left: tuple[int, ...],
                right: tuple[int, ...],
                *args: P.args,
                **kwargs: P.kwargs) -> R:
        if sorted(tuple(left) + tuple(right)) != list(range(len(t.legs))):
            raise ValueError(f'Legs {tuple(left)} and {tuple(right)} do not partition a '
                             f'{len(t.legs)}-leg tensor.')
        return func(t, left, right, _sector_matrices(t, left, right), *args, **kwargs)
    return wrapper


@_with_sectors
def svd(t: Tensor,
        left: tuple[int, ...],
        right: tuple[int, ...],
        parts: list[_Sector],
        max_bond: int | None = None,
        tol: float = 0.0,
        direction: str = 'right') -> tuple[Tensor, torch.Tensor, Tensor]:
    '''
    Split ``t`` between its left and right legs and factor it as :math:`U S V^\\dagger`.

    Each fused charge of the left legs is factorised on its own, and the singular values
    of every charge are truncated jointly to ``max_bond`` and ``tol``. A tensor without
    charges is a single fused charge, so both cases take this one path. ``s`` stays the
    singular values of the partition, so their squares sum to the squared norm of ``t``,
    while one factor carries the fusion vertex the factors do not hold (see
    ``_split_scale``), so contracting the three factors in the network recovers ``t``.

    Args:
        t: the tensor to split.
        left: leg positions forming the matrix rows, in order.
        right: leg positions forming the matrix columns, in order.
        parts: the per-charge splits of that partition, filled in by the decorator.
        max_bond: maximum number of singular values to keep; all if None.
        tol: discard singular values not greater than this.
        direction: which factor carries that vertex weight, ``'right'`` on ``u`` for a
            sweep that keeps ``u`` as the site tensor, ``'left'`` on ``vh`` for the
            opposite one.
    Returns:
        ``(u, s, vh)`` with ``u`` legs ``(*left dims, K)``, ``s`` the kept singular
        values as a plain 1-D torch tensor ordered by fused charge, and ``vh`` legs
        ``(K, *right dims)``.
    Raises:
        ValueError: If ``direction`` is neither ``'left'`` nor ``'right'``.
    '''
    if direction not in ('left', 'right'):
        raise ValueError(f'A split is prepared for "left" or "right", not {direction!r}.')
    factors = [torch.linalg.svd(part.matrix, full_matrices=False) for part in parts]
    flat = torch.cat([s for _, s, _ in factors])
    selected = flat > tol
    if max_bond is not None:
        allowed = torch.zeros_like(selected)
        allowed[torch.argsort(flat, descending=True)[:max_bond]] = True
        selected = selected & allowed
    kept, offset = [], 0
    for s in [s for _, s, _ in factors]:
        kept.append(selected[offset:offset + s.shape[0]].nonzero().squeeze(-1))
        offset += s.shape[0]
    weighted = [(u_mat, _split_scale(t.group, part.charge) * vh_mat) if direction == 'left'
                else (_split_scale(t.group, part.charge) * u_mat, vh_mat)
                for part, (u_mat, _, vh_mat) in zip(parts, factors)]
    u, vh = _assemble_split(t, left, right, parts, weighted, [part.charge for part in parts],
                            kept, t.group.trivial_sector, t.qtotal)
    values = [s[k] for (_, s, _), k in zip(factors, kept) if k.shape[0] > 0]
    return u, torch.cat(values) if values else torch.zeros(0, dtype=t.dtype, device=t.device), vh


@_with_sectors
def qr(t: Tensor,
       left: tuple[int, ...],
       right: tuple[int, ...],
       parts: list[_Sector]) -> tuple[Tensor, Tensor]:
    '''
    Factor ``t`` as :math:`Q R` with the isometry on the left, one fused charge at a time.

    Args:
        t: the tensor to split.
        left: leg positions forming the matrix rows, in order.
        right: leg positions forming the matrix columns, in order.
        parts: the per-charge splits of that partition, filled in by the decorator.
    Returns:
        ``(q, r)`` with ``q`` legs ``(*left dims, K)`` and ``r`` legs ``(K, *right dims)``.
    '''
    factors = []
    for part in parts:
        q_mat, r_mat = torch.linalg.qr(part.matrix)
        factors.append((_split_scale(t.group, part.charge) * q_mat, r_mat))
    return _assemble_split(t, left, right, parts, factors, [part.charge for part in parts], None,
                           t.group.trivial_sector, t.qtotal)


@_with_sectors
def rq(t: Tensor,
       left: tuple[int, ...],
       right: tuple[int, ...],
       parts: list[_Sector]) -> tuple[Tensor, Tensor]:
    '''
    Factor ``t`` as :math:`R Q` with the isometry on the right, one fused charge at a time.

    Args:
        t: the tensor to split.
        left: leg positions forming the matrix rows, in order.
        right: leg positions forming the matrix columns, in order.
        parts: the per-charge splits of that partition, filled in by the decorator.
    Returns:
        ``(r, q)`` with ``r`` legs ``(*left dims, K)`` and ``q`` legs ``(K, *right dims)``.
    '''
    abelian = t.group.is_abelian
    charges = [(t.group.dual_sector(_fused_charge(t.group, t.legs, part.qdata, right))
                if abelian else part.charge)
               for part in parts]
    factors = []
    for part in parts:
        q_mat, r_mat = torch.linalg.qr(part.matrix.mH)
        factors.append((r_mat.mH, _split_scale(t.group, part.charge) * q_mat.mH))
    # the first factor couples its legs to the trivial sector, the second to the total charge
    return _assemble_split(t, left, right, parts, factors, charges, None,
                           t.qtotal if abelian else t.group.trivial_sector,
                           t.group.trivial_sector if abelian else t.qtotal)


@_with_sectors
def svdvals(t: Tensor,
            left: tuple[int, ...],
            right: tuple[int, ...],
            parts: list[_Sector]) -> torch.Tensor:
    '''
    Singular values of the matrix formed by splitting ``t`` between its left and right legs.

    Args:
        t: the tensor to split.
        left: leg positions forming the matrix rows, in order.
        right: leg positions forming the matrix columns, in order.
        parts: the per-charge splits of that partition, filled in by the decorator.
    Returns:
        The singular values as a plain 1-D torch tensor, concatenated by fused charge.
    '''
    return torch.cat([torch.linalg.svdvals(part.matrix) for part in parts])


def norm(t: Tensor, dim: tuple[int, ...] | None = None) -> torch.Tensor:
    '''
    Frobenius norm of the tensor, optionally reduced over ``dim``.

    The flat fusion tree expands a block through the unitary Clebsch-Gordan maps of its
    vertices, so distinct blocks expand orthogonally and the plain sum of the squared
    blocks is the norm of the dense expansion.
    Args:
        t: the tensor.
        dim: leg positions to reduce; all legs if None.
    Raises:
        NotImplementedError: If the group is not abelian and the reduction keeps more than
            one leg, since a block covers one tree state of the kept leg.
    '''
    if t.is_dense or not t.legs:
        return torch.linalg.norm(t.to_torch(), dim=dim)
    if dim is None:
        return torch.sqrt(torch.stack([torch.sum(b.abs() ** 2) for b in t.blocks]).sum())
    keep = tuple(i for i in range(len(t.legs)) if i not in dim)
    if not t._group.is_abelian:
        if len(keep) != 1:
            raise NotImplementedError(f'Reducing the legs of a {t._group.group_name} tensor '
                                      f'keeps one leg, not {len(keep)}.')
        # One block covers one tree state of the kept leg. Its dense expansion carries the
        # Clebsch-Gordan factor of that sector, so the reduced norm is the block norm over
        # ``sqrt(sector_dim)``; the blocks come in qdata order, the kept leg's blocks first.
        reduced = []
        for block, q in zip(t.blocks, t.qdata):
            qdata = _leg_indices(q, len(t.legs))
            charge_dim = float(t._group.sector_dim(t.legs[keep[0]].sector(qdata[keep[0]]))) ** 0.5
            reduced.append((block.abs() ** 2).sum(dim=tuple(dim)).sqrt() / charge_dim)
        return torch.cat(reduced)
    out = torch.zeros(tuple(t.legs[i].dim for i in keep), dtype=t.blocks[0].real.dtype,
                      device=t.device)
    for block, q in zip(t.blocks, t.qdata):
        qdata = _leg_indices(q, len(t.legs))
        index = tuple(slice(t.legs[i].slices[qdata[i]], t.legs[i].slices[qdata[i] + 1])
                      for i in keep)
        out[index] += (block.abs() ** 2).sum(dim=tuple(dim.index(i) for i in dim))
    return torch.sqrt(out)


def inner(a: Tensor, b: Tensor) -> torch.Tensor:
    '''
    Inner product :math:`\\sum \\overline{a} b` over all legs.

    A non-abelian block expands through the unitary Clebsch-Gordan maps of its fusion
    tree, so matching blocks pair up without further weight, as in ``norm``.
    Args:
        a: the bra tensor.
        b: the ket tensor.
    '''
    if a.is_dense and b.is_dense:
        return torch.sum(a.to_torch().conj() * b.to_torch())
    if a.legs != b.legs:
        raise ValueError(f'An inner product needs equal legs, not {a.legs} and {b.legs}.')
    table = dict(zip(b.qdata, b.blocks))
    total = torch.zeros((), dtype=a.dtype, device=a.device)
    for q, block in zip(a.qdata, a.blocks):
        if q in table:
            total = total + torch.sum(block.conj() * table[q])
    return total


def trace(t: Tensor) -> torch.Tensor:
    '''
    Trace over the two legs of a two-leg tensor.
    Args:
        t: the two-leg tensor.
    Raises:
        NotImplementedError: If the two legs do not share their block boundaries.
    '''
    if t.is_dense:
        return torch.trace(t.to_torch())
    if len(t.legs) != 2 or t.legs[0].slices != t.legs[1].slices:
        raise NotImplementedError(f'A trace needs two legs with equal block boundaries, not {t.legs}.')
    if not t._group.is_abelian:
        # the Clebsch-Gordan tensors of the two trees of a dual pair of legs expand it to
        # a dense square matrix, and the invariant contraction of the pair is its trace
        return torch.trace(t.to_torch())
    total = torch.zeros((), dtype=t.dtype, device=t.device)
    for block in t.blocks:
        if block.shape[0] == block.shape[1]:
            total = total + torch.einsum('aa->', block)
    return total
