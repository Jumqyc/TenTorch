'''
The invariant MPO of a group with a multiplet structure.

The automaton whose bond carries one multiplet per channel of the operator content, the
calibration that fixes its placement values from those channels, and the reduction of the
couplings to them.  The builder holds no group theory of its own: the multiplet content comes
from the symmetry-specific module and every question about the group is asked of the group.
'''
from __future__ import annotations

import torch

from base import COMPLEX, CUDA
from typing import cast
from mpsdmrg.symmetry import Group, Leg, Sector, Tensor
from mpsdmrg.symmetry.multiplet import content

# the two states of the trivial sector in an MPO bond: the start and the end of an interaction
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
        couplings: the terms :meth:`~mpsdmrg.dmrg.MPO.add_couplings` recorded, each on two
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
    missing = [name for name in ('X', 'Y', 'Z') if name not in mapping]
    if missing:
        raise ValueError(f'The invariant builder needs the spin components {missing} '
                         f'in the mapping.')
    spins = torch.stack([mapping[name].to(dtype=dtype, device=device).contiguous()
                         for name in ('X', 'Y', 'Z')])
    module = content(group)
    operators = module.tensor_operators(group, spins)
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
        couplings: the terms :meth:`~mpsdmrg.dmrg.MPO.add_couplings` recorded.
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
