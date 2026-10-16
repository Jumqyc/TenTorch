"""
The matrix product operator of a Hamiltonian.
"""
from __future__ import annotations

from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray

from base import COMPLEX, CUDA
from mpsdmrg import linalg
from mpsdmrg.symmetry.group import Group, Leg, Sector
from mpsdmrg.symmetry.group import content
from mpsdmrg.operators import SpinOperator
from mpsdmrg.tensor import Tensor, cached_einsum

START, END = 0, 1


class Bond:
    r'''
    The invariant channels of one nearest-neighbour bond.

    ``terms`` maps the sector of a multiplet to the coefficient of the channel
    :math:`C_s = \sum_m (-1)^m T^{(k)}_m \otimes T^{(k)}_{-m}`, so the bond operator is
    :math:`\sum_s terms[s] C_s`.
    '''

    def __init__(self, site: int, terms: dict[Sector, complex]) -> None:
        '''Store the left site of the bond and its channel coefficients.'''
        self.site = site
        self.terms = terms


def bond_channels(group: Group,
                  phys_dim: int,
                  mapping: dict[str, torch.Tensor],
                  couplings: list[tuple[tuple[str, ...], complex, tuple[int, ...]]],
                  dtype = COMPLEX,
                  device = CUDA,
                  tol: float = 1e-10) -> tuple[dict[int, torch.Tensor], dict[Sector, torch.Tensor],
                                               list[Bond]] | None:
    r'''
    Reduce nearest-neighbour couplings to the invariant channels of every bond.

    Every bond term is assembled into one dense two-site operator and expanded in the invariant
    channels of the site; a sum that is not invariant leaves a residual and is refused.

    Args:
        group: the symmetry group; it needs a Clebsch-Gordan tensor, so it must be non-abelian.
        phys_dim: physical dimension of one site.
        mapping: operator names to their dense matrices; it must hold 'X', 'Y' and 'Z'.
        couplings: the terms :meth:`~mpsdmrg.mpo.MPO.add_couplings` recorded, each on two
            nearest neighbours.
        dtype, device: data type and device of the operators.
        tol: relative residual that still counts as invariant.
    Returns:
        ``None`` when the group is abelian, because it carries no multiplet structure; otherwise
        the site tensor operators, the invariant channels of a site pair, and the channel terms
        of every bond in bond order.
    Raises:
        ValueError: If the mapping misses a spin component, a term is not two-site, or an operator
            is not in the mapping.
        NotImplementedError: If a term is not nearest-neighbour, or the couplings of a bond are
            not invariant.
    '''
    if group.is_abelian:
        return None
    module = content(group)
    operators = module.tensor_operators(group, mapping, phys_dim, dtype, device)
    channels = module.channel_operators(group, operators, phys_dim)
    terms: dict[int, torch.Tensor] = {}
    for ops, coeff, locs in couplings:
        if len(ops) != 2 or len(locs) != 2:
            raise ValueError(f'An invariant coupling needs two operators, not {tuple(ops)} '
                             f'at {tuple(locs)}.')
        if locs[1] != locs[0] + 1:
            raise NotImplementedError(f'The invariant builder needs nearest-neighbour '
                                      f'couplings, not {tuple(locs)}.')
        for op in ops:
            if op not in mapping:
                raise ValueError(f'Operator {op!r} is not defined in the mapping.')
        pair = torch.kron(mapping[ops[0]].to(dtype=dtype, device=device).contiguous(),
                          mapping[ops[1]].to(dtype=dtype, device=device).contiguous())
        terms[locs[0]] = terms[locs[0]] + coeff * pair if locs[0] in terms else coeff * pair
    bonds: list[Bond] = []
    for site in sorted(terms):
        target = terms[site]
        coefficients = {sector: complex(torch.vdot(channel.reshape(-1), target.reshape(-1))
                                        / torch.vdot(channel.reshape(-1), channel.reshape(-1)))
                        for sector, channel in channels.items()}
        rebuilt = sum(coefficients[sector] * channels[sector] for sector in channels)
        residual = float((rebuilt - target).abs().max())
        if residual > tol * max(1.0, float(target.abs().max())):
            raise NotImplementedError(
                f'The couplings at sites ({site}, {site + 1}) are not invariant: their invariant '
                f'part misses the sum by {residual:.3e}.')
        bonds.append(Bond(site, {sector: value for sector, value in coefficients.items()
                                 if abs(value) > tol}))
    return operators, channels, bonds


def _bond_leg(group: Group, sectors: list[Sector]) -> Leg:
    '''
    The bond states of the invariant automaton: the start and the end of an interaction in the
    trivial sector, and one multiplet per channel.

    Args:
        group: the symmetry group; it needs a Clebsch-Gordan tensor.
        sectors: the multiplets carried by the bond, each appearing once.
    Returns:
        Leg whose trivial sector has multiplicity two and whose other sectors are the multiplets.
    '''
    entries = [(group.trivial_sector, 2)] + [(sector, group.sector_dim(sector))
                                             for sector in sectors]
    return Leg.from_blocks(group, entries, 1)[0]


def _pair_probe(group: Group, phys_dim: int, leg: Leg, sector: Sector | None, jump: bool,
                dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    r'''
    The dense operator a two-site chain expands to when one bond closes after one site.

    Args:
        group: the symmetry group.
        phys_dim: physical dimension of one site.
        leg: the bond of the two-site chain.
        sector: the multiplet inserted at the first site; None inserts nothing.
        jump: whether the trivial rail jumps from its start to its end at the first site,
            which is how a constant term closes; otherwise the sector is inserted.
        dtype, device: data type and device of the tensors.
    Returns:
        tensor in shape (d^2, d^2), the operator the pair expands to.
    '''
    phys_sector = group.sector_of_dimension(phys_dim)
    phys = Leg(group, (phys_sector,), (0, phys_dim), qconj=1)
    trivial = Leg.trivial(1, group)
    tensors: list[Tensor] = []
    for site, (left, right) in enumerate(((trivial, leg.conj()), (leg, trivial.conj()))):
        tensor = Tensor.zeros(left.dim, right.dim, phys_dim, phys_dim,
                              legs=(left, right, phys, phys.conj()), qtotal=group.trivial_sector,
                              dtype=dtype, device=device)
        for block, q in zip(tensor.blocks, tensor.qdata):
            if site == 0:
                if jump and q[0] == 0 and q[1] == 0:
                    block[START, right.multiplicity(0) - 1, 0, 0] = 1.0
                elif not jump and q[0] == 0 and right.sector(cast(int, q[1])) == sector:
                    block[START, START, 0, 0] = 1.0
            else:
                if q[1] != 0:
                    continue
                if jump and q[0] == 0:
                    block[END, 0, 0, 0] = 1.0
                elif not jump and left.sector(cast(int, q[0])) == sector:
                    block[START, right.multiplicity(0) - 1, 0, 0] = 1.0
        tensors.append(tensor)
    left, right = tensors
    # contract the shared bond; the axes are (wl, bra, ket, wr, bra, ket) afterwards
    joined = torch.einsum('abpq,bcrs->apqcrs', left.to_torch(), right.to_torch())
    # join the physical indices in site order: (bra0, bra1, ket0, ket1)
    operator = joined[START, :, :, 0, :, :].permute(0, 2, 1, 3)
    dim = operator.shape[0]
    return operator.reshape(dim * dim, dim * dim)


def _product(products: torch.Tensor, order: list[Sector], sector: Sector,
             terms: dict[Sector, complex]) -> complex:
    r'''
    The placement of one channel that realises a bond's channel coefficients.

    The placement is linear in the coefficients: :math:`\lambda_k = \sum_j (M^{-1})_{kj} c_j`.

    Args:
        products: the inverse calibration matrix, as :func:`calibrate` returns it.
        order: the calibrated sectors, the trivial one first.
        sector: the sector whose placement is wanted.
        terms: the channel coefficients of the bond.
    Returns:
        The placement, complex because the coefficients may be.
    '''
    row = order.index(sector)
    return sum((float(products[row, order.index(other)]) * terms.get(other, 0.0)
                for other in order), 0.0j)


def calibrate(group: Group, phys_dim: int, leg: Leg, sectors: list[Sector],
              channels: dict[Sector, torch.Tensor], dtype: torch.dtype, device: torch.device,
              tol: float = 1e-10) -> tuple[torch.Tensor, list[Sector], torch.Tensor]:
    r'''
    The placement values that make the automaton expand to the invariant channels.

    The opening and closing of one multiplet are one number each and only their product fixes
    the channel, so the opening is the positive square root of that product; the constant
    channel is the trivial rail jumping from its start to its end at one site and is calibrated
    as one more placement, because a multiplet probe is not traceless and leaks into it.

    Args:
        group: the symmetry group; it needs a Clebsch-Gordan tensor.
        phys_dim: physical dimension of one site.
        leg: the bond of the automaton.
        sectors: the multiplets carried by the bond, in bond order.
        channels: the invariant channel operators, as they are keyed by sector.
        dtype, device: data type and device of the probe.
        tol: relative tolerance of the calibration.
    Returns:
        ``(identity, order, products)``, where ``order`` lists the calibrated sectors with the
        trivial one first and ``products[k, j]`` is the placement of ``order[k]`` that yields
        channel ``order[j]``; ``identity`` is the value of one rail transfer.
    Raises:
        NotImplementedError: If the automaton cannot reproduce the channels.
    '''
    unit = torch.eye(phys_dim * phys_dim, dtype=torch.complex128, device=device)
    trivial = tuple(group.trivial_sector)
    order = [trivial, *sectors]
    probes = {sector: _pair_probe(group, phys_dim, leg, sector, sector == trivial, dtype,
                                  device).to(torch.complex128) for sector in order}
    scale = float(torch.vdot(unit.reshape(-1), probes[trivial].reshape(-1))
                  / torch.vdot(unit.reshape(-1), unit.reshape(-1))).real
    identity = scale ** -0.5

    flat = [probes[sector].reshape(-1) for sector in order]
    kinds = [channels[sector].to(torch.complex128).reshape(-1) for sector in order]
    # the automaton realizes sum_j value_j probe_j, so the probes are solved onto the channels in
    # the least-squares sense: their own Gram matrix need not be invertible
    gram = torch.stack([torch.stack([torch.vdot(a, b) for b in flat]) for a in flat])
    moments = torch.stack([torch.stack([torch.vdot(a, b) for b in kinds]) for a in flat])
    products = torch.linalg.pinv(gram) @ moments
    for sector in order:
        rebuilt = sum(products[order.index(other), order.index(sector)] * probes[other]
                      for other in order)
        residual = float((rebuilt - channels[sector].to(torch.complex128)).abs().max())
        if residual > tol * max(1.0, float(channels[sector].abs().max())):
            raise NotImplementedError(
                f'The invariant automaton misses the channel of {sector} by {residual:.3e}.')
    return identity, order, products


def invariant_mpo(group: Group, length: int, phys_dim: int, mapping: dict[str, torch.Tensor],
                  couplings, dtype: torch.dtype = COMPLEX,
                  device: torch.device = CUDA, tol: float = 1e-10
                  ) -> tuple[list[Tensor], int] | None:
    r'''
    The MPO tensors of an invariant nearest-neighbour Hamiltonian.

    Every site carries one tensor of the bond states ``(start, end, one multiplet per channel)``;
    an interaction opens on the bond to its right and closes from the bond to its left, so the
    only nonzero paths pair one nearest-neighbour bond with the bond operator.

    Args:
        group: the symmetry group.
        length: number of sites.
        phys_dim: physical dimension of one site.
        mapping: operator names to their dense matrices.
        couplings: the terms :meth:`~mpsdmrg.mpo.MPO.add_couplings` recorded.
        dtype, device: data type and device of the tensors.
        tol: relative tolerance of the calibration and of the invariance check.
    Returns:
        ``(tensors, bond_dim)``, or ``None`` when the group is abelian, so the caller's own
        automaton applies.
    Raises:
        ValueError: If the mapping misses a spin component or a coupling names a plain operator.
        NotImplementedError: If a coupling is not nearest-neighbour, a bond is not invariant, or
            the automaton cannot reproduce its channels.
    '''
    prepared = bond_channels(group, phys_dim, mapping, couplings, dtype, device, tol)
    if prepared is None:
        return None
    operators, channels, bonds = prepared
    trivial = tuple(group.trivial_sector)
    sectors = sorted({sector for bond in bonds for sector in bond.terms if sector != trivial})
    leg = _bond_leg(group, sectors)
    identity, order, products = calibrate(group, phys_dim, leg, sectors, channels, dtype, device,
                                          tol)
    strength = {bond.site: bond.terms for bond in bonds}

    phys = Leg(group, (group.sector_of_dimension(phys_dim),), (0, phys_dim), qconj=1)
    triv = Leg.trivial(1, group)
    tensors: list[Tensor] = []
    for i in range(length):
        if i == 0:
            left, right, role = triv, leg.conj(), 'left'
        elif i == length - 1:
            left, right, role = leg, triv.conj(), 'right'
        else:
            left, right, role = leg, leg.conj(), 'bulk'
        tensor = Tensor.zeros(left.dim, right.dim, phys_dim, phys_dim,
                              legs=(left, right, phys, phys.conj()), qtotal=group.trivial_sector,
                              dtype=dtype, device=device)
        for block, q in zip(tensor.blocks, tensor.qdata):
            if q[0] == 0 and q[1] == 0:
                if role == 'right':
                    block[END, 0, 0, 0] = identity
                else:
                    block[START, START, 0, 0] = identity
                    # the constant of this bond closes without an interaction; the jump probe
                    # carries one rail transfer, the bond to its right carries a second
                    constant = _product(products, order, trivial, strength.get(i, {})) / identity
                    block[START, right.multiplicity(0) - 1, 0, 0] = constant
                    if role == 'bulk':
                        block[END, END, 0, 0] = identity
            elif q[0] == 0 and q[1] != 0 and role != 'right':
                sector = right.sector(cast(int, q[1]))
                if sector in strength.get(i, {}):
                    product = _product(products, order, sector, strength[i])
                    block[START, START, 0, 0] = abs(product) ** 0.5
            elif q[0] != 0 and q[1] == 0 and role != 'left':
                sector = left.sector(cast(int, q[0]))
                if sector in strength.get(i - 1, {}):
                    product = _product(products, order, sector, strength[i - 1])
                    block[START, right.multiplicity(0) - 1, 0, 0] = \
                        (1.0 if product.real >= 0 else -1.0) * abs(product) ** 0.5
        tensors.append(tensor)
    return tensors, leg.dim


def operator_mpo(length: int, phys_dim: int, mapping: dict[str, torch.Tensor], couplings,
                 dtype: torch.dtype = COMPLEX, device: torch.device = CUDA
                 ) -> tuple[list[Tensor], int]:
    r'''
    Build the MPO tensors whose bond states are the operator labels of the couplings.

    Intermediate states are identified by ``(op_str, rel_pos)`` pairs where *op_str* is the
    operator just placed and *rel_pos* is the distance in sites to the next operator.
    Nearest-neighbour states are shared across couplings; long-range states are kept unique
    per coupling so that a shared identity segment cannot open a spurious path.

    Args:
        length: number of sites.
        phys_dim: physical dimension of one site.
        mapping: operator names to their dense matrices.
        couplings: the terms :meth:`~mpsdmrg.mpo.MPO.add_couplings` recorded.
        dtype, device: data type and device of the tensors.
    Returns:
        ``(tensors, bond_dim)``, with one tensor per site and one for the right boundary.
    '''
    first_site_counts: dict[tuple, int] = {}
    for c_idx, (ops, _, locs) in enumerate(couplings):
        if len(ops) > 1 and locs[1] - locs[0] == 1:
            key = (ops[0], 1, locs[0])  # (op_str, rel_pos=1, site)
            first_site_counts[key] = first_site_counts.get(key, 0) + 1
    state_ranges: dict[tuple, set[tuple[int, int]]] = {}
    coupling_keys: list[list[tuple]] = [[] for _ in couplings]
    for c_idx, (ops, _, locs) in enumerate(couplings):
        for n in range(len(ops) - 1):
            rel_pos = locs[n + 1] - locs[n]
            if n == 0 and rel_pos == 1:
                site_key = (ops[0], 1, locs[0])
                if first_site_counts.get(site_key, 0) > 1:
                    key = (c_idx, ops[n], rel_pos)  # unique per coupling
                else:
                    key = (ops[n], rel_pos)          # shared
            else:
                # long-range states stay unique per coupling, so no shared identity segment
                # can open a spurious path between two couplings
                key = (ops[n], rel_pos) if rel_pos == 1 else (c_idx, ops[n], rel_pos)
            coupling_keys[c_idx].append(key)
            state_ranges.setdefault(key, set()).add((locs[n], locs[n + 1]))

    sorted_keys = sorted(state_ranges, key=lambda x: (len(x), x))
    state_idx = {k: i + 1 for i, k in enumerate(sorted_keys)}
    bond_dim = len(state_idx) + 2  # start (0) + end (bond_dim - 1)

    p = phys_dim
    tensors = [Tensor.zeros((1, bond_dim, p, p), dtype=dtype, device=device)]
    tensors += [Tensor.zeros((bond_dim, bond_dim, p, p), dtype=dtype, device=device)
                for _ in range(1, length)]
    tensors += [Tensor.zeros((bond_dim, 1, p, p), dtype=dtype, device=device)]

    eye = torch.eye(p, dtype=dtype, device=device)
    tensors[0][0, 0] = eye                             # site-0 start
    tensors[length][bond_dim - 1, 0] = eye             # right boundary
    for i in range(1, length):
        tensors[i][0, 0] = eye                         # bulk start
        tensors[i][bond_dim - 1, bond_dim - 1] = eye   # bulk end

    # identity on skipped sites for long-range couplings
    for key, ranges in state_ranges.items():
        rel_pos = key[-1]
        if rel_pos <= 1:
            continue
        s = state_idx[key]
        for from_site, to_site in ranges:
            for k in range(from_site + 1, to_site):
                if 0 < k < length:
                    tensors[k][s, s] = eye

    # ── place operator matrices ──
    for c_idx, (ops, coeff, locs) in enumerate(couplings):
        keys = coupling_keys[c_idx]
        for n, (op_str, site) in enumerate(zip(ops, locs)):
            mat = mapping[op_str]
            if n == 0:
                mat = coeff * mat

            state_in = 0 if n == 0 else state_idx[keys[n - 1]]
            state_out = bond_dim - 1 if n == len(ops) - 1 else state_idx[keys[n]]

            if site == 0:
                tensors[0][0, state_out] += mat
            elif site == length:
                tensors[length][state_in, 0] += mat
            else:
                tensors[site][state_in, state_out] += mat
    return tensors, bond_dim


def build_mpo(group: Group | None, length: int, phys_dim: int, mapping: dict[str, torch.Tensor],
              couplings, dtype: torch.dtype = COMPLEX, device: torch.device = CUDA,
              tol: float = 1e-10) -> tuple[list[Tensor], int]:
    r'''
    Build the MPO tensors of a Hamiltonian; the group answers which automaton applies.

    A non-abelian group walks the invariant channels of every nearest-neighbour bond
    (:func:`mpsdmrg.symmetry.invariant.invariant_mpo`); every other group, including no group
    at all, walks the operator labels of the couplings.

    Args:
        group: the symmetry group of the Hamiltonian, or None for no symmetry.
        length: number of sites.
        phys_dim: physical dimension of one site.
        mapping: operator names to their dense matrices.
        couplings: the terms :meth:`~mpsdmrg.mpo.MPO.add_couplings` recorded.
        dtype, device: data type and device of the tensors.
        tol: relative residual that still counts as invariant.
    Returns:
        ``(tensors, bond_dim)``, with one tensor per site and one for the right boundary.
    Raises:
        ValueError: If a coupling names a letter that is not in the mapping.
        NotImplementedError: If a non-abelian group's couplings have no invariant
            nearest-neighbour realisation.
    '''
    if group is not None:
        built = invariant_mpo(group, length, phys_dim, mapping, couplings, dtype, device, tol)
        if built is not None:
            return built
    return operator_mpo(length, phys_dim, mapping, couplings, dtype, device)


class MPO:
    '''
    A class representing a Matrix Product Operator (MPO) in quantum physics and tensor networks.
    '''
    def __init__(self,
                 L:int,
                 phys_dim:int,
                 mapping:dict[str,Tensor]|None = None,
                 dtype = COMPLEX,
                 device = CUDA):
        '''
        Args:
            L (int): Length of the MPO (number of sites).
            phys_dim (int): Physical dimension of the MPO.
            mapping (dict[str, Tensor]): A dictionary mapping operator names to their dense operator tensors. If None, the spin components and the identity are used.
        '''
        self.dtype = dtype
        self.device = device
        self.L = L
        self.group: Group | None = None  # set by the engine that owns the Hamiltonian
        self.couplings: list[tuple[tuple[str, ...], complex, tuple[int, ...]]] = []
        self.physical_dim = phys_dim
        self.tensors: list[Tensor] = []
        self.bond_dim = 0

        if mapping is None:
            spin_op = SpinOperator(phys_dim=phys_dim, dtype=dtype, device=device).data
            mapping = {
                'X': spin_op[0],
                'Y': spin_op[1],
                'Z': spin_op[2],
                'I': Tensor.from_torch(torch.eye(phys_dim, dtype=dtype, device=device))
            }
        self.mapping = dict(mapping)
        if not all(isinstance(matrix, Tensor) for matrix in self.mapping.values()):
            raise TypeError('The mapping of an MPO holds mpsdmrg.tensor.Tensor matrices.')

    @property
    def _buffers(self) -> dict[str, torch.Tensor]:
        '''The dense buffers of the mapping, for the contractions that build the tensors.'''
        return {name: matrix.to_torch() for name, matrix in self.mapping.items()}

    def to(self, device):
        '''
        Move the MPO tensors to the specified device (e.g., 'cuda' or 'cpu').
        '''
        self.device = device
        for i in range(len(self.tensors)):
            self.tensors[i] = self.tensors[i].to(device)
        for k in self.mapping:
            self.mapping[k] = self.mapping[k].to(device)

    def copy(self):
        '''
        Create a deep copy of the MPO instance.
        '''
        new = MPO(self.L,
                  self.physical_dim,
                  mapping={k: v.clone() for k, v in self.mapping.items()},
                  dtype=self.dtype,
                  device=self.device)
        new.couplings = [(ops, coeff, locs) for ops, coeff, locs in self.couplings]
        new.group = self.group
        if self.tensors:
            new.tensors = [t.clone().detach() for t in self.tensors]
        new.bond_dim = self.bond_dim
        return new

    def __repr__(self):
        return f"MPO(L={self.L}, physical_dim={self.physical_dim}, bond_dim={self.bond_dim})"
    # ── Arithmetic dunders ──────────────────────────────────────────────

    def __add__(self, other):
        '''
        MPO + MPO: merge couplings.   MPO + dict: add single-site operators.
        '''
        if isinstance(other, MPO):
            new = self.copy()
            # Copy any missing operator matrices from other into new.mapping
            for op_name in set(op for ops, _, _ in other.couplings for op in ops):
                new._register_merged_op(op_name)
            # Directly merge couplings (they are already merged by add_couplings)
            new.couplings.extend(other.couplings)
            new.build()
            return new
        elif isinstance(other, dict):
            new = self.copy()
            for site, op_mat in other.items():
                name = f"__dict_{len(new.mapping)}"
                new.mapping[name] = op_mat.to(dtype=new.dtype, device=new.device)
                new.add_couplings((name,), 1.0, (site,))
            new.build()
            return new
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, dict):
            return self.__add__(other)
        return NotImplemented

    def __mul__(self, other):
        '''
        MPO * scalar: scale couplings.
        MPO * MPO:   operator composition (contract physical indices).
        '''
        if isinstance(other, (int, float, complex)):
            new = self.copy()
            if new.tensors:
                # scale the tensors directly (simpler for already-built MPOs)
                for i in range(len(new.tensors)):
                    new.tensors[i] = new.tensors[i] * other
            else:
                new.couplings = [(ops, coeff * other, locs) for ops, coeff, locs in new.couplings]
                new.build()
            return new
        elif isinstance(other, MPO):
            # MPO * MPO: contract physical indices, outer-product the bond dims
            new = MPO(self.L, self.physical_dim,
                      mapping={k: v.clone() for k, v in self.mapping.items()},
                      dtype=self.dtype, device=self.device)
            new.tensors = []
            for i in range(len(self.tensors)):
                t1 = self.tensors[i]
                t2 = other.tensors[i]
                # t1: (wL1, wR1, p, q), t2: (wL2, wR2, q, r)  →  (wL1*wL2, wR1*wR2, p, r)
                new_t = cached_einsum('abpq,cdqr->acbdpr', t1, t2).combine_legs(
                    ((0, 1), (2, 3), (4,), (5,)))
                new.tensors.append(new_t)
            new.bond_dim = self.bond_dim * other.bond_dim
            return new
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            return self.__mul__(other)
        return NotImplemented

    def __neg__(self):
        return self.__mul__(-1.0)

    def __sub__(self, other):
        return self.__add__((-1.0) * other)

    # ── Coupling management ─────────────────────────────────────────────

    def add_couplings(self,operators:tuple[str,...],coeff:complex,locations:NDArray[np.int64]|tuple[int,...]):
        '''
        Add new couplings to the MPO.
        Args:
            operators (tuple[str]): A tuple of operator names (e.g., ('X', 'Y')).  A name of
                several letters is the product of those mapping entries on one site, e.g.
                ``'XY'`` is :math:`S_x S_y`.
            coeffs (complex): A complex coefficient for the coupling.
            locations (tuple[int]): A tuple of site indices where the operators act.
        Raises:
            ValueError: If the length of operators does not match the number of locations,
                        or if any site index is out of bounds for the MPO length,
                        or if any operator letter is not defined in the mapping.
        '''
        if len(operators) != len(locations):
            raise ValueError(f"Length of operators {len(operators)} does not match the number of locations {len(locations)}.")
        if any(loc < 0 or loc >= self.L for loc in locations):
            raise ValueError(f"Site indices {locations} are out of bounds for MPO of length {self.L}.")
        if not all(op in self.mapping or all(letter in self.mapping for letter in op)
                   for op in operators):
            raise ValueError(f"Operators {operators} are not defined in the mapping. Available operators: {list(self.mapping.keys())}.")
        operators = tuple(self._register_merged_op(op) for op in operators)

        # Merge consecutive operators on the same site
        merged_ops = []
        merged_locs = []
        current_op = operators[0]
        current_loc = locations[0]
        for op, loc in zip(operators[1:], locations[1:]):
            if loc == current_loc:
                current_op = current_op + op
            else:
                merged_ops.append(self._register_merged_op(current_op))
                merged_locs.append(current_loc)
                current_op = op
                current_loc = loc
        merged_ops.append(self._register_merged_op(current_op))
        merged_locs.append(current_loc)
        self.couplings.append((tuple(merged_ops), coeff, tuple(merged_locs)))

    def _register_merged_op(self, op_string: str) -> str:
        '''
        Register a merged operator matrix if not already present, return its name.
        '''
        if op_string not in self.mapping:
            base = op_string.split('@')[0]  # e.g. "XY@0" → "XY"
            buffers = self._buffers
            mat = buffers[base[0]]
            for ch in base[1:]:
                mat = mat @ buffers[ch]
            self.mapping[op_string] = Tensor.from_torch(mat.to(dtype=self.dtype, device=self.device))
        return op_string

    @staticmethod
    def preprocess(group: Group,
                   phys_dim: int,
                   mapping: dict[str, torch.Tensor],
                   couplings: list[tuple[tuple[str, ...], complex, tuple[int, ...]]],
                   dtype = COMPLEX,
                   device = CUDA,
                   tol: float = 1e-10
                   ) -> tuple[dict[int, torch.Tensor], dict[Sector, torch.Tensor],
                              list[Bond]] | None:
        r'''
        Reduce nearest-neighbour couplings to the invariant channels of every bond.

        A coupling names its operators by string; a name of several letters is the product of
        the one-letter entries of the mapping, so a caller can write ``('XY', 'XY')`` as the
        composite :math:`S_x S_y` without extending the mapping himself.  The group theory
        lives in :mod:`mpsdmrg.symmetry.invariant`; this is the entry point of the Hamiltonian
        into it.

        Args:
            group: the symmetry group of the Hamiltonian.
            phys_dim: physical dimension of one site.
            mapping: operator names to their dense matrices.
            couplings: the terms :meth:`add_couplings` recorded.
            dtype, device: data type and device of the operators.
            tol: relative residual that still counts as invariant.
        Returns:
            ``(operators, channels, bonds)`` as :func:`mpsdmrg.symmetry.invariant.bond_channels`
            returns them, or None when the couplings have no invariant realisation.
        Raises:
            ValueError: If a coupling names a letter that is not in the mapping.
        '''
        return bond_channels(group, phys_dim, mapping, couplings, dtype, device, tol)

    def build(self):
        '''
        Build the MPO tensors from the current couplings and mapping.

        The bond automaton and every group question live in
        :mod:`mpsdmrg.mpo`; an operator name of several letters is the product of those
        mapping entries on one site.
        '''
        self.tensors, self.bond_dim = build_mpo(self.group, self.L, self.physical_dim,
                                                self._buffers, self.couplings, self.dtype,
                                                self.device)

    def __sizeof__(self):
        return sum(t.numel() * t.element_size() for t in self.tensors)
