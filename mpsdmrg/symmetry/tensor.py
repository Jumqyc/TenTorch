from __future__ import annotations

from collections import Counter
from functools import reduce
from itertools import accumulate, product
from math import prod
from typing import Callable, cast

import torch
from opt_einsum import contract_expression

from base import COMPLEX, CUDA
from mpsdmrg.symmetry.data.su2data import fusion_tensor
from mpsdmrg.symmetry.groups import Group, Sector, no_symmetry

# One charge-block key: the charge-block index on each leg, then, for a non-abelian group,
# the coupled sector of every intermediate vertex of its fusion tree.
BlockKey = tuple[int | Sector, ...]


class Leg:
    '''
    Charge structure of one leg: its group, one block per charge sector and the
    boundaries of every block.

    The blocks are kept sorted by sector, so two legs with the same group and the same
    sectors in the same multiplicities are equal.

    Args:
        group: the group the sectors belong to.
        sectors: one sector per charge block.
        slices: block boundaries, ``len(sectors) + 1`` strictly increasing entries
            starting at 0.
        qconj: +1 if the leg carries its charges, -1 if it carries their duals.
    '''

    def __init__(self,
                 group: Group,
                 sectors: tuple[Sector, ...],
                 slices: tuple[int, ...],
                 qconj: int = 1):
        assert qconj in (1, -1), f'The charge direction of a leg must be +1 or -1, not {qconj}.'
        sectors = tuple(tuple(int(q) for q in s) for s in sectors)
        slices = tuple(int(s) for s in slices)
        assert len(slices) == len(sectors) + 1 and slices[0] == 0, \
            f'Slices {slices} do not describe {len(sectors)} blocks.'
        assert all(b > a for a, b in zip(slices, slices[1:])), \
            f'Slices {slices} are not strictly increasing.'
        assert len(set(sectors)) == len(sectors), f'Leg sectors {sectors} repeat a sector.'
        assert all(group.is_valid_sector(s) for s in sectors), \
            f'A sector of {sectors} is not a sector of group {group.group_name}.'
        dims = tuple(b - a for a, b in zip(slices, slices[1:]))
        order = sorted(range(len(sectors)), key=lambda i: sectors[i])
        self.group = group
        self.qconj = int(qconj)
        self.sectors = tuple(sectors[i] for i in order)
        self.slices = (0, *accumulate(dims[i] for i in order))

    @classmethod
    def trivial(cls, dim: int, group: Group | None = None, qconj: int = 1) -> Leg:
        '''
        One block of ``dim`` indices in the trivial sector.
        Args:
            dim: number of indices on the leg.
            group: the group of the leg; the group without symmetry if None.
            qconj: +1 if the leg carries its charges, -1 if it carries their duals.
        '''
        group = no_symmetry if group is None else group
        return cls(group, (group.trivial_sector,), (0, dim), qconj)

    @classmethod
    def from_blocks(cls,
                    group: Group,
                    entries: list[tuple[Sector, int]],
                    qconj: int = 1) -> tuple[Leg, tuple[int, ...]]:
        '''
        Build a leg from ``(sector, block dimension)`` entries, sorted by sector.
        Args:
            group: the group of the leg.
            entries: one pair per block, in block order.
            qconj: +1 if the leg carries its charges, -1 if it carries their duals.
        Returns:
            The leg and, for entry ``k``, the block index it ends up at.
        '''
        order = sorted(range(len(entries)), key=lambda k: entries[k][0])
        rank = [0] * len(order)
        for position, k in enumerate(order):
            rank[k] = position
        sectors = tuple(entries[k][0] for k in order)
        dims = tuple(entries[k][1] for k in order)
        return cls(group, sectors, (0, *accumulate(dims)), qconj), tuple(rank)

    @property
    def dim(self) -> int:
        '''Number of indices on the leg.'''
        return self.slices[-1]

    @property
    def nblocks(self) -> int:
        '''Number of charge blocks on the leg.'''
        return len(self.sectors)

    def sector(self, qi: int, dual: bool = False) -> Sector:
        '''
        Sector of one charge block.
        Args:
            qi: the charge-block index.
            dual: whether to return the sector the leg contributes to a fusion; the dual
                of the stored sector for a leg that carries the duals of its charges.
        '''
        return self.group.dual_sector(self.sectors[qi]) if dual and self.qconj == -1 \
            else self.sectors[qi]

    def block_dim(self, qi: int) -> int:
        '''
        Dimension of one charge block, the quantum dimension times its multiplicity.
        Args:
            qi: the charge-block index.
        '''
        return self.slices[qi + 1] - self.slices[qi]

    def multiplicity(self, qi: int) -> int:
        '''
        Multiplicity of one charge block on the leg.
        Args:
            qi: the charge-block index.
        '''
        return self.block_dim(qi) // self.group.sector_dim(self.sectors[qi])

    def fuse(self, other: Leg) -> Leg:
        '''
        The leg carrying every outcome of fusing this leg's sectors with ``other``'s.

        One block per outcome holds every copy of the pairs that reach it, so its
        multiplicity is the sum of the pair multiplicities, weighted by the fusion
        multiplicity of the group.

        Args:
            other: the leg fused with this one; it carries the same group.
        '''
        assert self.group == other.group, \
            f'Cannot fuse legs of groups {self.group.group_name} and {other.group.group_name}.'
        counts: dict[Sector, int] = {}
        for i, first in enumerate(self.sectors):
            mi = self.multiplicity(i)
            for j, second in enumerate(other.sectors):
                for outcome in self.group.fusion_outcomes(first, second):
                    counts[outcome] = counts.get(outcome, 0) + (
                        mi * other.multiplicity(j)
                        * self.group.n_symbol(first, second, outcome))
        entries = [(outcome, self.group.sector_dim(outcome) * copies)
                   for outcome, copies in counts.items()]
        return Leg.from_blocks(self.group, entries, 1)[0]

    def conj(self) -> Leg:
        '''The same leg with the direction of its charges reversed.'''
        return Leg(self.group, self.sectors, self.slices, -self.qconj)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Leg):
            return NotImplemented
        return (self.group == other.group and self.sectors == other.sectors
                and self.slices == other.slices and self.qconj == other.qconj)

    def __hash__(self) -> int:
        return hash((self.group, self.sectors, self.slices, self.qconj))

    def __repr__(self) -> str:
        return (f'Leg(dim={self.dim}, qconj={self.qconj}, blocks={len(self.sectors)}, '
                f'sectors={self.sectors})')


class FusionTree:
    '''
    One left-nested fusion order ``((l0 l1) l2) ...`` of a tuple of legs: the sectors of
    the legs, the couplings of the successive prefixes and the sector they all fuse to.

    Args:
        group: the group the sectors belong to.
        sectors: one sector per leg, in leg order, dualised for a leg with qconj = -1.
        inner: couplings of the prefixes, ``inner[k]`` couples legs 0 to k+1; it has
            ``max(len(sectors) - 2, 0)`` entries.
        coupled: the sector every leg fuses to.
    '''

    def __init__(self,
                 group: Group,
                 sectors: tuple[Sector, ...],
                 inner: tuple[Sector, ...],
                 coupled: Sector):
        sectors = tuple(tuple(int(q) for q in s) for s in sectors)
        inner = tuple(tuple(int(q) for q in s) for s in inner)
        coupled = tuple(int(q) for q in coupled)
        assert sectors, 'A fusion tree needs at least one leg.'
        assert len(inner) == max(len(sectors) - 2, 0), \
            f'{len(inner)} inner couplings cannot fuse {len(sectors)} legs.'
        self.group = group
        self.sectors = sectors
        self.inner = inner
        self.coupled = coupled
        for k in range(1, len(sectors)):
            assert group.can_fuse_to(self.chain[k - 1], sectors[k], self.chain[k]), \
                f'Leg {k} of {sectors} does not fuse into the coupling {self.chain[k]}.'

    @classmethod
    def from_blocks(cls,
                    group: Group,
                    legs: tuple[Leg, ...],
                    qdata: tuple,
                    qtotal: Sector) -> FusionTree:
        '''
        The tree stored in one charge-block key of a tensor. A tensor without legs stores no
        couplings, so its tree is the single total charge.
        Args:
            group: the group of the legs.
            legs: the legs of the tensor.
            qdata: the charge-block index on each leg, then the stored intermediate
                couplings of the fusion tree.
            qtotal: the total charge of the tensor.
        Returns:
            The tree of the leg sectors of that block and its stored couplings.
        '''
        if not legs:
            return cls(group, (qtotal,), (), qtotal)
        sectors = tuple(leg.sector(qdata[i], dual=True) for i, leg in enumerate(legs))
        return cls(group, sectors, tuple(qdata[len(legs):]), qtotal)

    @classmethod
    def of_tensor(cls, tensor: Tensor, qi: int) -> FusionTree:
        '''
        The tree stored in one block of a tensor.
        Args:
            tensor: the tensor.
            qi: its block index.
        Returns:
            The tree of that block.
        '''
        return cls.from_blocks(tensor.group, tensor.legs, tensor.qdata[qi], tensor.qtotal)

    @property
    def chain(self) -> tuple[Sector, ...]:
        '''The couplings ``c_0, ..., c_{n-1}``: ``c_k`` couples legs 0 to k.'''
        if len(self.sectors) == 1:
            return (self.coupled,)
        return (self.sectors[0], *self.inner, self.coupled)

    def to_dense_block(self, device: torch.device = CUDA) -> torch.Tensor:
        '''
        The Clebsch-Gordan expansion of the tree, in the tree's own magnetic indices.
        Args:
            device: where the tensor is built.
        Returns:
            Shape ``(d_0, ..., d_{n-1}, d_coupled)`` with ``d_k = sector_dim(sectors[k])``;
            the last axis is the magnetic index of the coupled sector.
        '''
        # the expansion is the contraction of the Clebsch-Gordan tensor of every visited
        # vertex, its last axis the running magnetic index of the coupling chain
        spins = tuple(s[0] for s in self.sectors)
        out = torch.eye(spins[0] + 1, dtype=torch.float64, device=device)
        for i in range(1, len(spins)):
            table = fusion_tensor(self.chain[i - 1][0], spins[i], self.chain[i][0], device)
            out = torch.einsum('...m,mab->...ab', out, table)
        return out

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FusionTree):
            return NotImplemented
        return (self.group == other.group and self.sectors == other.sectors
                and self.inner == other.inner and self.coupled == other.coupled)

    def __hash__(self) -> int:
        return hash((self.group, self.sectors, self.inner, self.coupled))

    def __repr__(self) -> str:
        return (f'FusionTree(sectors={self.sectors}, inner={self.inner}, '
                f'coupled={self.coupled})')


def fusion_trees(group: Group,
                 sectors: tuple[Sector, ...],
                 coupled: Sector) -> tuple[FusionTree, ...]:
    '''
    Every left-nested fusion of the given leg sectors into ``coupled``.
    Args:
        group: the group the sectors belong to.
        sectors: one sector per leg, in leg order, dualised for a leg with qconj = -1.
        coupled: the sector every leg must fuse to.
    Returns:
        One tree per allowed combination of the prefix couplings.
    '''
    if not sectors:
        return (FusionTree(group, ((0,) * len(group.trivial_sector),), (), coupled),) \
            if tuple(coupled) == tuple(group.trivial_sector) else ()
    if len(sectors) == 1:
        return (FusionTree(group, sectors, (), coupled),) if sectors[0] == tuple(coupled) else ()
    states = [(sectors[0], ())]
    for k in range(1, len(sectors)):
        following = []
        for current, inner in states:
            for outcome in group.fusion_outcomes(current, sectors[k]):
                if k == len(sectors) - 1:
                    if outcome == tuple(coupled):
                        following.append((outcome, inner))
                else:
                    following.append((outcome, inner + (outcome,)))
        states = following
    return tuple(FusionTree(group, sectors, inner, coupled) for _, inner in states)


_LABELS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


def stored_dense(legs: tuple, tree: FusionTree, device: torch.device) -> torch.Tensor:
    '''
    Dense expansion of one tree in the stored indices of its legs.

    A leg with ``qconj = -1`` stores the magnetic index of the tree flipped and carries the
    phase ``(-1)^m``, which is what makes a contraction of two dual legs a plain index sum.

    Args:
        legs: the legs of the tree, in tree order.
        tree: the tree.
        device: where the tensor is built.
    Returns:
        Shape ``(d_0, ..., d_{n-1}, d_coupled)`` in the legs' stored indices.
    '''
    out = tree.to_dense_block(device)
    for i, leg in enumerate(legs):
        if leg.qconj == -1:
            spin = tree.sectors[i][0]
            phase = torch.ones(spin + 1, dtype=torch.float64, device=device)
            phase[1::2] = -1.0
            shape = [1] * out.dim()
            shape[i] = spin + 1
            out = torch.flip(out, dims=(i,)) * ((-1.0) ** spin * phase).reshape(shape)
    return out


def dual_sign(group: Group, sector: Sector) -> float:
    '''
    Sign a leg of ``sector`` picks up when its charge direction is reversed.

    Reversing the charge of a leg dualises it, which conjugates the invariant pairing of its
    sector. A self-dual sector whose singlet is antisymmetric contributes the Frobenius-Schur
    sign :math:`(-1)^{2j}`; for SU(2) that is every half-integer spin, since its quantum
    dimension is :math:`2j + 1`. Every other sector contributes one.

    Args:
        group: the group of the leg.
        sector: the stored sector of the leg.
    Returns:
        ``-1.0`` for a pseudo-real self-dual sector, ``1.0`` otherwise.
    '''
    return -1.0 if (group.dual_sector(sector) == tuple(sector)
                    and (group.sector_dim(sector) - 1) % 2) else 1.0


def tree_weight(out_legs: tuple,
                out_tree: FusionTree,
                operands: tuple,
                labels: tuple[str, ...],
                out_labels: str,
                device: torch.device = CUDA) -> float:
    '''
    Overlap of a contracted network of trees with the tree of the result.

    The reduced blocks of the operands are multiplied by this scalar, which carries all the
    Clebsch-Gordan structure of the contraction: every operand tree and the result tree are
    expanded densely and the indices the subscripts identify are contracted, the totals of
    the operands are summed and one component of the result total is kept (the overlap does
    not depend on which one).

    Args:
        out_legs: legs of the result, in subscript order.
        out_tree: the tree the result block is stored in.
        operands: ``(legs, tree)`` of every operand, in subscript order.
        labels: the subscript of every operand, its leg labels in leg order.
        out_labels: the subscript of the result.
        device: where the Clebsch-Gordan tensors are built.
    Returns:
        The weight of that output tree, a real scalar.
    '''
    arrays = [stored_dense(legs, tree, device) for legs, tree in operands]
    free = [c for c in _LABELS if c not in set(''.join(labels) + out_labels)]
    operands_labels = [labels[i] + free[i] for i in range(len(arrays))]
    if not out_labels:
        return float(torch.einsum(','.join(operands_labels) + '->', *arrays))
    output = stored_dense(out_legs, out_tree, device)[..., out_tree.coupled[0] // 2]
    subscript = ','.join([out_labels, *operands_labels])
    return float(torch.einsum(subscript + '->', output, *arrays))


Sym: Group = no_symmetry
"""The group of a tensor or leg created without an explicit group."""

# Every operation of this library is a tensor contraction without a gradient tape.
no_grad = torch.no_grad



def _data(x: Tensor | torch.Tensor | complex) -> torch.Tensor:
    '''
    Return the dense storage of an operand.
    Args:
        x: a ``Tensor``, a raw ``torch.Tensor`` or a scalar.
    Returns:
        The dense array of ``x``, in shape ``x.shape``.
    Raises:
        NotImplementedError: If a ``Tensor`` operand stores more than one charge block,
            in which case no single dense array represents it.
    '''
    if not isinstance(x, Tensor):
        return torch.as_tensor(x)
    if len(x.blocks) != 1:
        raise NotImplementedError('The dense operations need a single charge block, '
                                  f'but the tensor stores {len(x.blocks)}.')
    return x.blocks[0]


def _leg_indices(key: BlockKey, n: int) -> tuple[int, ...]:
    '''
    Charge-block index on each of the first ``n`` legs of a charge-block key.
    Args:
        key: the charge-block key.
        n: the number of legs.
    '''
    return tuple(int(k) for k in cast('tuple[int, ...]', key[:n]))


def _couplings(key: BlockKey, n: int) -> tuple[Sector, ...]:
    '''
    Coupled sector of every intermediate vertex of a charge-block key.
    Args:
        key: the charge-block key, whose first ``n`` entries index the legs.
        n: the number of legs.
    '''
    return cast('tuple[Sector, ...]', key[n:])


def _fused_charge(group: Group, legs: tuple[Leg, ...], key: BlockKey,
                  axes: tuple[int, ...] | range) -> Sector:
    '''
    Fused charge of one block, restricted to ``axes``.
    Args:
        group: the group of the legs.
        legs: the legs of the tensor.
        key: the charge-block key of the block.
        axes: the leg positions to fuse.
    '''
    qdata = _leg_indices(key, len(legs))
    fused = group.trivial_sector
    for i in axes:
        s = legs[i].sectors[qdata[i]]
        fused = group.fuse(fused, s if legs[i].qconj == 1 else group.dual_sector(s))
    return fused


def _fused_qtotal(group: Group, qtotals: list[Sector]) -> Sector:
    '''
    Total charge of a contraction of tensors.

    Every contracted subscript pairs a sector with its dual, so those legs drop out and the
    result carries the fusion of the operands' totals.

    Args:
        group: the group of the operands.
        qtotals: the total charges of the operands, in order.
    Returns:
        The total charge of the contracted tensor.
    Raises:
        NotImplementedError: If a non-abelian fusion has several outcomes, since the
            contracted tensor is then a sum of charge sectors.
    '''
    if group.is_abelian:
        return reduce(group.fuse, qtotals)
    total = group.trivial_sector
    for q in qtotals:
        outcomes = group.fusion_outcomes(total, tuple(q))
        if len(outcomes) != 1:
            raise NotImplementedError(f'Fusing {total} with {tuple(q)} has {len(outcomes)} '
                                      f'outcomes; the contracted tensor is a sum of them.')
        total = tuple(outcomes[0])
    return total


def _combos(legs: tuple[Leg, ...], qtotal: Sector) -> tuple[BlockKey, ...]:
    '''
    All charge-block keys whose fusion gives ``qtotal``.

    A non-abelian key carries the intermediate couplings of its tree after the leg
    blocks; an abelian fusion has a single outcome per pair, so its keys are the leg
    blocks alone.

    Args:
        legs: the legs of the tensor.
        qtotal: the total charge of the tensor.
    '''
    if not legs:
        return ((),)
    group = legs[0].group
    combos: list[BlockKey] = []
    for q in product(*[range(leg.nblocks) for leg in legs]):
        if group.is_abelian:
            if _fused_charge(group, legs, q, range(len(legs))) == qtotal:
                combos.append(q)
        else:
            sectors = tuple(leg.sector(q[i], dual=True) for i, leg in enumerate(legs))
            combos.extend(q + tree.inner for tree in fusion_trees(group, sectors, qtotal))
    return tuple(combos)


def _block_shape(legs: tuple[Leg, ...], key: BlockKey) -> tuple[int, ...]:
    '''
    Shape of the reduced block at charge-block key ``key``: the legs' multiplicities.

    The dimension of a sector itself comes from the structural tensors of the fusion tree,
    so only the number of copies of each sector is stored.
    '''
    qdata = _leg_indices(key, len(legs))
    return tuple(leg.multiplicity(qdata[i]) for i, leg in enumerate(legs))


def _legs_and_qtotal(shape: tuple, legs: tuple[Leg, ...] | None, group: Group | None,
                     qtotal: Sector | None) -> tuple[tuple[Leg, ...], Sector]:
    '''
    Resolve the legs and total charge of a tensor created from a shape.

    The shape is written either as one tuple or list, as in ``Tensor.zeros((2, 3))``, or as
    the dimensions themselves, as in ``Tensor.zeros(2, 3)``.

    Args:
        shape: the dimensions of the tensor.
        legs: the legs to use; trivial legs of ``shape`` if None.
        group: the group of the trivial legs; ``Sym`` if None and ``legs`` is None.
        qtotal: the total charge; the trivial sector of the group if None.
    Raises:
        ValueError: If ``legs`` does not match ``shape``.
    '''
    shape = tuple(int(d) for d in shape[0]) \
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)) \
        else tuple(int(d) for d in shape)
    if legs is None:
        group = Sym if group is None else group
        legs = tuple(Leg.trivial(d, group) for d in shape)
    else:
        legs = tuple(legs)
        if shape and tuple(leg.dim for leg in legs) != shape:
            raise ValueError(f'Legs of dimensions {tuple(leg.dim for leg in legs)} do not match '
                             f'the shape {shape}.')
    group = legs[0].group if legs else (Sym if group is None else group)
    return legs, group.trivial_sector if qtotal is None else tuple(qtotal)


class Tensor:
    '''
    Symmetry-adapted tensor: one dense block per allowed combination of leg charge
    blocks, the charge structure of each leg, and the total charge.

    The blocks are kept sorted by charge-block index and the legs by sector, so equal
    tensors have equal storage.

    Args:
        blocks: the stored reduced blocks.
        qdata: the charge-block index on each leg and, for a non-abelian group, the
            coupled sectors between the legs of the fusion tree; one tuple per block.
        legs: the charge structure of each leg.
        qtotal: the total charge of the tensor; the trivial sector if None.
        group: the group of a scalar tensor without legs; unused otherwise.
    '''

    def __init__(self,
                 blocks: list[torch.Tensor],
                 qdata: tuple[BlockKey, ...],
                 legs: tuple[Leg, ...],
                 qtotal: Sector | None = None,
                 group: Group | None = None):
        self.legs = tuple(legs)
        self.blocks = list(blocks)
        n = len(self.legs)
        self.qdata = tuple(_leg_indices(qi, n)
                           + tuple(tuple(int(s) for s in sector) for sector in _couplings(qi, n))
                           for qi in qdata)
        self._group = self.legs[0].group if self.legs else (Sym if group is None else group)
        self.qtotal = self._group.trivial_sector if qtotal is None else tuple(int(q) for q in qtotal)
        for leg in self.legs:
            assert leg.group == self._group, \
                f'All legs of a tensor share one group, not {leg.group} and {self._group}.'
        assert len(self.blocks) == len(self.qdata), \
            f'{len(self.blocks)} blocks need as many charge-block index tuples.'
        order = sorted(range(len(self.qdata)), key=lambda i: self.qdata[i])
        self.blocks = [self.blocks[i] for i in order]
        self.qdata = tuple(self.qdata[i] for i in order)
        assert len(set(self.qdata)) == len(self.qdata), \
            f'Charge-block indices {self.qdata} repeat a block.'
        key_length = n if self._group.is_abelian else n + max(n - 2, 0)
        for block, qidx in zip(self.blocks, self.qdata):
            assert len(qidx) == key_length, \
                f'Block indices {qidx} need {key_length} entries, not {len(qidx)}.'
            assert all(0 <= q < leg.nblocks for q, leg in zip(_leg_indices(qidx, n), self.legs)), \
                f'Block indices {qidx} are out of range for the legs {self.legs}.'
            dims = _block_shape(self.legs, qidx)
            assert tuple(block.shape) == dims, \
                f'Block shape {tuple(block.shape)} does not match the legs {dims}.'
            if self._group.is_abelian:
                charge = _fused_charge(self._group, self.legs, qidx, range(n))
                assert charge == self.qtotal, \
                    f'Block {qidx} carries charge {charge}, not the total charge {self.qtotal}.'
            else:
                FusionTree.from_blocks(self._group, self.legs, qidx, self.qtotal)

    @property
    def group(self) -> Group:
        '''The symmetry group, taken from the legs (from ``group`` for a scalar).'''
        return self._group

    @classmethod
    def from_torch(cls,
                   data: torch.Tensor,
                   legs: tuple[Leg, ...] | None = None,
                   qtotal: Sector | None = None) -> Tensor:
        '''
        Build a tensor from a dense ``torch.Tensor``, keeping only its allowed blocks.

        Entries outside the allowed blocks are dropped; the result is symmetric by
        construction.

        Args:
            data: the dense array.
            legs: the charge structure of each leg; one trivial block of the matching
                dimension per leg if None.
            qtotal: the total charge of the tensor; the trivial sector if None.
        '''
        legs, qtotal = _legs_and_qtotal(tuple(data.shape), legs, None, qtotal)
        combos = _combos(legs, qtotal)
        if len(combos) == 1 and all(leg.nblocks == 1 for leg in legs):
            return cls([data], combos, legs, qtotal)
        blocks = [data[tuple(slice(leg.slices[qi], leg.slices[qi + 1])
                             for leg, qi in zip(legs, _leg_indices(c, len(legs))))].clone()
                  for c in combos]
        return cls(blocks, combos, legs, qtotal)

    @staticmethod
    def zeros(*shape: int | tuple[int, ...] | list[int],
              dtype: torch.dtype = COMPLEX,
              device: torch.device = CUDA,
              legs: tuple[Leg, ...] | None = None,
              group: Group | None = None,
              qtotal: Sector | None = None) -> Tensor:
        '''
        Zero tensor.
        Args:
            shape: one tuple/list or the dimensions as separate arguments.
            dtype: storage data type.
            device: storage device.
            legs: the legs to use; trivial legs of ``shape`` under ``group`` if None.
            group: the group of the trivial legs; ``Sym`` if None.
            qtotal: the total charge; the trivial sector of the group if None.
        '''
        legs, qtotal = _legs_and_qtotal(shape, legs, group, qtotal)
        combos = _combos(legs, qtotal)
        blocks = [torch.zeros(_block_shape(legs, c), dtype=dtype, device=device) for c in combos]
        return Tensor(blocks, combos, legs, qtotal)

    @staticmethod
    def ones(*shape: int | tuple[int, ...] | list[int],
             dtype: torch.dtype = COMPLEX,
             device: torch.device = CUDA,
             legs: tuple[Leg, ...] | None = None,
             group: Group | None = None,
             qtotal: Sector | None = None) -> Tensor:
        '''
        One-filled tensor.
        Args:
            shape: one tuple/list or the dimensions as separate arguments.
            dtype: storage data type.
            device: storage device.
            legs: the legs to use; trivial legs of ``shape`` under ``group`` if None.
            group: the group of the trivial legs; ``Sym`` if None.
            qtotal: the total charge; the trivial sector of the group if None.
        '''
        legs, qtotal = _legs_and_qtotal(shape, legs, group, qtotal)
        combos = _combos(legs, qtotal)
        blocks = [torch.ones(_block_shape(legs, c), dtype=dtype, device=device) for c in combos]
        return Tensor(blocks, combos, legs, qtotal)

    @staticmethod
    def rand(*shape: int | tuple[int, ...] | list[int],
             dtype: torch.dtype = COMPLEX,
             device: torch.device = CUDA,
             legs: tuple[Leg, ...] | None = None,
             group: Group | None = None,
             qtotal: Sector | None = None) -> Tensor:
        '''
        Uniform random tensor on :math:`[0, 1)`.
        Args:
            shape: one tuple/list or the dimensions as separate arguments.
            dtype: storage data type.
            device: storage device.
            legs: the legs to use; trivial legs of ``shape`` under ``group`` if None.
            group: the group of the trivial legs; ``Sym`` if None.
            qtotal: the total charge; the trivial sector of the group if None.
        '''
        legs, qtotal = _legs_and_qtotal(shape, legs, group, qtotal)
        combos = _combos(legs, qtotal)
        blocks = [torch.rand(_block_shape(legs, c), dtype=dtype, device=device) for c in combos]
        return Tensor(blocks, combos, legs, qtotal)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(leg.dim for leg in self.legs)

    @property
    def dtype(self) -> torch.dtype:
        return self.blocks[0].dtype

    @property
    def device(self) -> torch.device:
        return self.blocks[0].device

    @property
    def is_dense(self) -> bool:
        '''Whether a single block of the trivial group stores the whole tensor, so the
        dense shortcuts apply.'''
        return len(self.blocks) == 1 and self.group == no_symmetry

    @property
    def real(self) -> Tensor:
        '''The real part of every block, on the same legs and blocks.'''
        return Tensor([b.real for b in self.blocks], self.qdata, self.legs, self.qtotal, self._group)

    def numel(self) -> int:
        return sum(b.numel() for b in self.blocks)

    def element_size(self) -> int:
        return self.blocks[0].element_size()

    def to_torch(self) -> torch.Tensor:
        '''
        The tensor as one dense ``torch.Tensor`` of the full leg dimensions.

        A tensor of the trivial group is its stored block. Otherwise an abelian block is
        written into its slices, so for a tensor returned by ``combine_legs`` the fused
        index is ordered by sector, a permutation of the dense ``reshape`` of the original
        array. A non-abelian block stores only the copies of each leg sector, so every
        vertex of its fusion tree contributes a structural tensor; inside a charge block
        the entries of a sector run fastest, copy by copy.

        Returns:
            The dense array, in shape ``self.shape``.
        Raises:
            NotImplementedError: If the group is not abelian and the total charge is not
                the trivial sector, which needs one dense component per multiplet entry.
        '''
        if self.is_dense:
            return self.blocks[0]
        out = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        if self._group.is_abelian:
            for block, q in zip(self.blocks, self.qdata):
                if not self.legs:
                    out[()] = block
                    continue
                out[tuple(slice(leg.slices[i], leg.slices[i + 1])
                          for leg, i in zip(self.legs, _leg_indices(q, len(self.legs))))] = block
            return out
        if self.qtotal != self._group.trivial_sector:
            raise NotImplementedError(f'Expanding the multiplet of charge {self.qtotal} needs one '
                                      f'component per entry, which this array does not have.')
        n = len(self.legs)
        if n == 0:
            out[()] = self.blocks[0]
            return out
        for bi, (block, q) in enumerate(zip(self.blocks, self.qdata)):
            qdata = _leg_indices(q, n)
            tree = FusionTree.of_tensor(self, bi)
            x = stored_dense(self.legs, tree, self.device)[..., tree.coupled[0] // 2]
            for mu in product(*[range(leg.multiplicity(qdata[i])) for i, leg in enumerate(self.legs)]):
                window = tuple(slice(self.legs[i].slices[qdata[i]] + mu[i] * x.shape[i],
                                     self.legs[i].slices[qdata[i]] + (mu[i] + 1) * x.shape[i])
                               for i in range(n))
                out[window] += block[mu] * x
        return out

    def item(self):
        return self.to_torch().item()

    def clone(self) -> Tensor:
        return Tensor([b.clone() for b in self.blocks], self.qdata, self.legs, self.qtotal, self._group)

    def detach(self) -> Tensor:
        return Tensor([b.detach() for b in self.blocks], self.qdata, self.legs, self.qtotal, self._group)

    def conj(self) -> Tensor:
        '''
        Complex conjugate, with the charges of every leg and of the total reversed.

        A reversed leg is dualised, so it picks up the Frobenius-Schur sign of its sector.
        The stored expansion of the result is then the conjugate of the stored expansion of
        the tensor, which the contracted norms and environments rely on.
        '''
        signs = []
        for q in self.qdata:
            qi = _leg_indices(q, len(self.legs))
            signs.append(prod(dual_sign(self._group, self.legs[i].sectors[qi[i]])
                              for i, leg in enumerate(self.legs) if leg.qconj == -1))
        return Tensor([b.conj() * sign for b, sign in zip(self.blocks, signs)],
                      self.qdata,
                      tuple(leg.conj() for leg in self.legs),
                      self._group.dual_sector(self.qtotal),
                      self._group)

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> Tensor:
        '''
        Move the blocks to ``device`` and/or cast them to ``dtype``.
        Args:
            device: target device; unchanged if None.
            dtype: target data type; unchanged if None.
        '''
        return Tensor([b.to(device=device, dtype=dtype) for b in self.blocks],
                      self.qdata, self.legs, self.qtotal, self._group)

    def combine_legs(self, groups: tuple[tuple[int, ...], ...]) -> Tensor:
        '''
        Fuse each group of legs into one leg, in the order of ``groups``.

        The fused index is ordered by the fused sector, so for a charged tensor
        ``to_torch`` differs from the dense ``reshape`` by a permutation of that index.

        Args:
            groups: leg positions, each leg appearing in exactly one group.
        Returns:
            The tensor with one leg per group, of the product of its dimensions.
        Raises:
            ValueError: If the groups do not partition the legs.
        '''
        flat = tuple(i for group in groups for i in group)
        if sorted(flat) != list(range(len(self.legs))):
            raise ValueError(f'Leg groups {tuple(groups)} do not partition the legs.')
        if self.is_dense:
            dims = tuple(prod(self.legs[i].dim for i in group) for group in groups)
            return Tensor.from_torch(self.to_torch().permute(flat).reshape(dims))
        new_legs, ranks = [], []
        for group in groups:
            entries = []
            for key in self.qdata:
                q = _leg_indices(key, len(self.legs))
                entries.append((_fused_charge(self._group, self.legs, key, group),
                                prod(self.legs[i].block_dim(q[i]) for i in group)))
            leg, rank = Leg.from_blocks(self._group, entries, 1)
            new_legs.append(leg)
            ranks.append(rank)
        blocks, qdata = [], []
        local = tuple(flat.index(i) for i in range(len(self.legs)))
        for bi, key in enumerate(self.qdata):
            q = _leg_indices(key, len(self.legs))
            dims = tuple(prod(self.legs[i].block_dim(q[i]) for i in group) for group in groups)
            blocks.append(self.blocks[bi].permute(local).reshape(dims))
            qdata.append(tuple(ranks[g][bi] for g in range(len(groups))))
        return Tensor(blocks, tuple(qdata), tuple(new_legs), self.qtotal, self._group)

    def __getitem__(self, key) -> Tensor:
        if self.is_dense:
            return Tensor.from_torch(self.to_torch()[key])
        out = self.to_torch()[key]
        if out.dim() != 0:
            raise NotImplementedError('Indexing a charged tensor supports scalar selections only.')
        return Tensor.from_torch(out)

    def __setitem__(self, key, value: Tensor | torch.Tensor | complex) -> None:
        if self.is_dense:
            self.to_torch()[key] = _data(value)
            return
        if torch.is_tensor(key) and key.dtype == torch.bool and key.numel() == 1:
            if bool(key.item()):
                self.blocks = [torch.zeros_like(b) for b in self.blocks]
            return
        raise NotImplementedError('Assigning into a charged tensor supports a scalar boolean mask only.')

    def _binary_add(self, other: Tensor | torch.Tensor | complex) -> Tensor:
        '''Sum with a tensor of the same legs, a raw array or a scalar.'''
        if isinstance(other, Tensor):
            if other.legs != self.legs:
                raise ValueError(f'Adding tensors of different legs {self.legs} and {other.legs}.')
            table = dict(zip(other.qdata, other.blocks))
            qdata = list(self.qdata)
            blocks = [b + table[q] if q in table else b.clone() for q, b in zip(self.qdata, self.blocks)]
            for q, b in zip(other.qdata, other.blocks):
                if q not in self.qdata:
                    qdata.append(q)
                    blocks.append(b.clone())
            return Tensor(blocks, tuple(qdata), self.legs, self.qtotal, self._group)
        if isinstance(other, torch.Tensor) and other.dim() > 0:
            return Tensor([b + other for b in self.blocks], self.qdata, self.legs, self.qtotal, self._group)
        if bool(torch.as_tensor(other) == 0):
            return self.clone()
        raise NotImplementedError('A constant offset cannot be represented on a charged tensor.')

    def __add__(self, other: Tensor | torch.Tensor | complex) -> Tensor:
        return self._binary_add(other)

    def __radd__(self, other: Tensor | torch.Tensor | complex) -> Tensor:
        return self._binary_add(other)

    def __sub__(self, other: Tensor | torch.Tensor | complex) -> Tensor:
        return self._binary_add(-other if isinstance(other, Tensor) else -torch.as_tensor(other))

    def _scale(self, other: Tensor | torch.Tensor | complex, divide: bool = False) -> Tensor:
        '''
        Scale every block by a scalar or by a raw array along the legs.

        A raw array of the tensor's rank scales leg ``i`` by ``other`` along that leg
        where ``other.shape[i]`` matches the leg's stored multiplicity, and leaves it
        alone where that entry is 1. A non-abelian block stores one index per
        multiplicity, not per leg dimension, so the window of a charge block comes from
        the running multiplicities. This is the block-wise meaning of
        ``s.view(-1, 1, 1) * vh``.
        '''
        if isinstance(other, Tensor):
            if other.legs != self.legs:
                raise ValueError(f'Scaling tensors of different legs {self.legs} and {other.legs}.')
            table = dict(zip(other.qdata, other.blocks))
            blocks = [(b / table[q] if divide else b * table[q]) for q, b in zip(self.qdata, self.blocks)
                      if q in table]
            return Tensor(blocks, tuple(q for q in self.qdata if q in table),
                          self.legs, self.qtotal, self._group)
        if not isinstance(other, torch.Tensor) or other.numel() == 1 or other.dim() == 0:
            op = torch.Tensor.__truediv__ if divide else torch.Tensor.__mul__
            return Tensor([op(b, other) for b in self.blocks], self.qdata, self.legs,
                          self.qtotal, self._group)
        if other.dim() != len(self.legs):
            raise ValueError(f'A raw factor of shape {tuple(other.shape)} does not broadcast over '
                             f'{len(self.legs)} legs.')
        # Stored offset of every charge block: blocks keep one index per multiplicity.
        offsets = []
        for leg in self.legs:
            running = [0]
            for qi in range(leg.nblocks):
                running.append(running[-1] + leg.multiplicity(qi))
            offsets.append(tuple(running))
        blocks = []
        for b, key in zip(self.blocks, self.qdata):
            q = _leg_indices(key, len(self.legs))
            index = tuple(slice(None) if other.shape[i] == 1 else
                          slice(offsets[i][q[i]], offsets[i][q[i] + 1])
                          for i in range(len(self.legs)))
            factor = other[index]
            blocks.append(b / factor if divide else b * factor)
        return Tensor(blocks, self.qdata, self.legs, self.qtotal, self._group)

    def __mul__(self, other: Tensor | torch.Tensor | complex) -> Tensor:
        return self._scale(other)

    def __rmul__(self, other: Tensor | torch.Tensor | complex) -> Tensor:
        return self._scale(other)

    def __truediv__(self, other: Tensor | torch.Tensor | complex) -> Tensor:
        return self._scale(other, divide=True)

    def __neg__(self) -> Tensor:
        return Tensor([-b for b in self.blocks], self.qdata, self.legs, self.qtotal, self._group)

    def __iadd__(self, other: Tensor | torch.Tensor | complex) -> Tensor:
        if self.is_dense and not isinstance(other, Tensor):
            self.to_torch().__iadd__(_data(other))
            return self
        new = self._binary_add(other)
        self.blocks, self.qdata = new.blocks, new.qdata
        return self

    def __imul__(self, other: Tensor | torch.Tensor | complex) -> Tensor:
        if self.is_dense and not isinstance(other, Tensor):
            self.to_torch().__imul__(_data(other))
            return self
        new = self._scale(other)
        self.blocks, self.qdata = new.blocks, new.qdata
        return self

    def __itruediv__(self, other: Tensor | torch.Tensor | complex) -> Tensor:
        if self.is_dense and not isinstance(other, Tensor):
            self.to_torch().__itruediv__(_data(other))
            return self
        new = self._scale(other, divide=True)
        self.blocks, self.qdata = new.blocks, new.qdata
        return self


_expr_cache: dict[tuple, Callable[..., torch.Tensor]] = {}


def cached_einsum(einsum_str: str, *tensors: Tensor) -> Tensor:
    '''
    Contract symmetry-adapted tensors with an ``opt_einsum`` expression cached by subscript
    string and shapes.

    Args:
        einsum_str: the einsum subscript string.
        tensors: the operands, each a ``Tensor``; a raw ``torch.Tensor`` is taken as the
            dense tensor that stores it.
    Returns:
        The contracted tensor.
    Raises:
        NotImplementedError: If a symmetry-adapted operand is contracted with a raw array.
    '''
    raw = any(not isinstance(t, Tensor) for t in tensors)
    if any(isinstance(t, Tensor) and not t.is_dense for t in tensors):
        if raw:
            raise NotImplementedError('A symmetry-adapted contraction needs every operand to be '
                                      'a Tensor, not a mix with raw arrays.')
        return _blocked_einsum(einsum_str, tensors)
    operands = tuple(t if isinstance(t, Tensor) else Tensor.from_torch(t) for t in tensors)
    shapes = tuple(t.shape for t in operands)
    key = (einsum_str,) + shapes
    expr = _expr_cache.get(key)
    if expr is None:
        expr = _expr_cache[key] = contract_expression(einsum_str, *shapes, optimize='auto')
    return Tensor.from_torch(expr(*[_data(t) for t in operands]))


def _blocked_einsum(einsum_str: str, tensors: tuple[Tensor, ...]) -> Tensor:
    '''
    Contract symmetry-adapted tensors block by block, over the shared sector of every
    contracted subscript.
    Args:
        einsum_str: the einsum subscript string.
        tensors: the operands, all ``Tensor``.
    Returns:
        The contracted tensor, with the legs of the subscripts it keeps.
    Raises:
        NotImplementedError: If a subscript appears more than twice, if one appears twice in
            the output, or if a non-abelian group fuses the operands to more than one sector.
        ValueError: If a contracted leg of two operands does not match its partner.
    '''
    lhs, _, rhs = einsum_str.replace(' ', '').partition('->')
    labels = tuple(tuple(s) for s in lhs.split(','))
    out = tuple(rhs)
    if len(labels) != len(tensors):
        raise ValueError(f'{len(labels)} subscript groups do not match {len(tensors)} operands.')
    counts = Counter(l for ls in labels for l in ls)
    for l in out:
        if counts[l] > 1:
            raise NotImplementedError(f'Subscript {l!r} appears {counts[l]} times in the output.')
    summed = tuple(l for l in counts if l not in out)
    for l in summed:
        if counts[l] != 2:
            raise NotImplementedError(f'Subscript {l!r} is contracted {counts[l]} times; only '
                                      'a pair of operands is supported.')
    axes = [{l: labels[o].index(l) for l in labels[o]} for o in range(len(tensors))]
    for l in summed:
        o1, o2 = (o for o in range(len(tensors)) if l in labels[o])
        leg1, leg2 = tensors[o1].legs[axes[o1][l]], tensors[o2].legs[axes[o2][l]]
        if (leg1.sectors != leg2.sectors or leg1.slices != leg2.slices
                or leg1.qconj != -leg2.qconj):
            raise ValueError(f'Contracted subscript {l!r} pairs {leg1} with {leg2}.')
    table = []
    for o, t in enumerate(tensors):
        by_sector = {}
        for bi, q in enumerate(t.qdata):
            qi = _leg_indices(q, len(t.legs))
            key = {l: t.legs[axes[o][l]].sectors[qi[axes[o][l]]] for l in summed if l in labels[o]}
            by_sector.setdefault(tuple(key.items()), []).append(bi)
        table.append(by_sector)
    carrier = {l: next(o for o in range(len(tensors)) if l in labels[o]) for l in out}
    summed_axes = {l: [o for o in range(len(tensors)) if l in labels[o]] for l in summed}
    options = {l: next(tensors[o].legs[axes[o][l]] for o in summed_axes[l]).sectors for l in summed}
    out_legs = tuple(tensors[carrier[l]].legs[axes[carrier[l]][l]] for l in out)
    group = tensors[0].group
    qtotal = _fused_qtotal(group, [t.qtotal for t in tensors])
    label_strings = tuple(''.join(ls) for ls in labels)
    out_labels = ''.join(out)
    blocks: dict[tuple[int, ...], torch.Tensor] = {}
    for assignment in product(*[options[l] for l in summed]):
        sectors = dict(zip(summed, assignment))
        candidates = []
        for o in range(len(tensors)):
            key = tuple((l, sectors[l]) for l in summed if l in labels[o])
            candidates.append(table[o].get(key, ()))
        for combo in product(*candidates):
            value = torch.einsum(einsum_str, *[tensors[o].blocks[combo[o]] for o in range(len(tensors))])
            if group.is_abelian:
                qkey = tuple(tensors[o].qdata[combo[o]][axes[o][l]] for l, o in carrier.items())
                blocks[qkey] = value if qkey not in blocks else blocks[qkey] + value
                continue
            # every tree the result can carry gets the overlap of the operand trees with it
            operands = tuple((tensors[o].legs, FusionTree.of_tensor(tensors[o], combo[o]))
                             for o in range(len(tensors)))
            out_sectors = tuple(
                out_leg.sector(tensors[carrier[l]].qdata[combo[carrier[l]]][axes[carrier[l]][l]],
                               dual=True)
                for l, out_leg in zip(out, out_legs))
            legs = tuple(tensors[o].qdata[combo[o]][axes[o][l]] for l, o in carrier.items())
            for tree in fusion_trees(group, out_sectors, qtotal):
                weight = tree_weight(out_legs, tree, operands, label_strings, out_labels,
                                     tensors[0].device)
                if weight == 0.0:
                    continue
                qkey = legs + tree.inner
                term = value * weight
                blocks[qkey] = term if qkey not in blocks else blocks[qkey] + term
    return Tensor(list(blocks.values()), tuple(blocks), out_legs, qtotal, group)


