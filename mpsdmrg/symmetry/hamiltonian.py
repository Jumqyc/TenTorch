"""The MPO tensors of a Hamiltonian.

The bond automaton whose states are the operator labels of the couplings, and the one entry
point :func:`build_mpo` that builds the tensors of any group; the group itself answers which
automaton applies, so no caller has to know whether its charges fuse uniquely.
"""
from __future__ import annotations

import torch

from base import COMPLEX, CUDA
from mpsdmrg.symmetry import Group, Tensor
from mpsdmrg.symmetry.invariant import invariant_mpo


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
        couplings: the terms :meth:`~mpsdmrg.dmrg.MPO.add_couplings` recorded.
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
        couplings: the terms :meth:`~mpsdmrg.dmrg.MPO.add_couplings` recorded.
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
