'''
The SU(2) multiplet content of an invariant Hamiltonian.

The tensor operators of one site and the invariant two-site channels they generate; the
builder that places them lives in :mod:`mpsdmrg.mpo` and asks here for the
content of a site.
'''
from __future__ import annotations

import torch

from mpsdmrg.ext import Group, Sector
# the type name of the group whose multiplets this module carries
GROUP = "SU2Symmetry"


def tensor_operators(group: Group, mapping: dict[str, torch.Tensor], dim: int,
                     dtype: torch.dtype, device: torch.device) -> dict[int, torch.Tensor]:
    r'''
    The rank-k tensor operators of one site, from the adjoint action of the spin algebra.

    The adjoint action ``X -> [S_a, X]`` has the Casimir ``sum_a ad_a^2`` with eigenvalue
    :math:`k(k+1)` on the rank-k operators, and a single-multiplet site holds one multiplet per
    rank, so its eigenvectors give the :math:`T^{(k)}_m` up to the ladder normalization.

    Args:
        group: the symmetry group; it needs a Clebsch-Gordan tensor, so it must be non-abelian.
        mapping: operator names to their dense matrices; the spin components ``X``, ``Y`` and
            ``Z`` must be in it.
        dim: physical dimension of one site.
        dtype, device: data type and device of the matrices.
    Returns:
        dict rank -> tensor in shape (2k + 1, d, d), the :math:`T^{(k)}_m` for m from k down to
        -k, normalized by ``ad_+ T_m = sqrt((k - m)(k + m + 1)) T_{m + 1}``.
    Raises:
        ValueError: If a spin component is missing from the mapping.
        NotImplementedError: If the group is abelian, so it carries no multiplet structure.
    '''
    if group.is_abelian:
        raise NotImplementedError(
            f'The tensor operators need a group with a multiplet structure, not {group.group_name}.')
    missing = [name for name in ('X', 'Y', 'Z') if name not in mapping]
    if missing:
        raise ValueError(f'The SU(2) tensor operators need the spin components {missing} '
                         f'in the mapping.')
    spins = torch.stack([mapping[name].to(dtype=dtype, device=device).contiguous()
                         for name in ('X', 'Y', 'Z')])
    eye = torch.eye(dim, dtype=spins.dtype, device=spins.device)
    # row-major vectorization of a commutator: ad_a(X) = a X - X a -> a x I - I x a^T
    adjoint = [torch.kron(a.contiguous(), eye) - torch.kron(eye, a.mT.contiguous()) for a in spins]
    values, vectors = torch.linalg.eigh(sum(a @ a for a in adjoint))
    ladder = adjoint[0] + 1j * adjoint[1]
    operators: dict[int, torch.Tensor] = {}
    for value in torch.unique(values.round(decimals=8)):
        rank = int(round((-1.0 + (1.0 + 4.0 * float(value)) ** 0.5) / 2.0))
        block = vectors[:, (values - value).abs() < 1e-6]
        _, rotation = torch.linalg.eigh(block.conj().T @ adjoint[2] @ block)
        block = block @ rotation
        order = torch.argsort((block.conj().T @ adjoint[2] @ block).diagonal().real)
        multiplet = [block[:, i] for i in order]
        for i in range(len(multiplet) - 1):
            m = -rank + i
            multiplet[i + 1] = ((ladder @ multiplet[i]).reshape(-1)
                                / ((rank - m) * (rank + m + 1)) ** 0.5)
        operators[rank] = torch.stack([t.reshape(dim, dim) for t in multiplet])
    return operators


def channel_operators(group: Group, operators: dict[int, torch.Tensor], dim: int,
                      ) -> dict[Sector, torch.Tensor]:
    r'''
    The invariant two-site operator of every rank, :math:`C_k = \sum_m (-1)^m T^{(k)}_m \otimes
    T^{(k)}_{-m}`, keyed by the sector its multiplet lives in; on a single-multiplet site these
    span the invariant operators.

    Args:
        group: the symmetry group; it must label its multiplets by their dimension.
        operators: the site tensor operators, as :func:`tensor_operators` returns them.
        dim: physical dimension of one site.
    Returns:
        dict sector -> tensor in shape (d^2, d^2), the invariant channel of that multiplet.
    '''
    channels: dict[Sector, torch.Tensor] = {}
    for rank, tensor in operators.items():
        total = torch.zeros((dim * dim, dim * dim), dtype=tensor.dtype, device=tensor.device)
        for m in range(-rank, rank + 1):
            total += (-1.0 if (rank - m) % 2 else 1.0) * torch.kron(tensor[rank + m],
                                                                    tensor[rank - m])
        channels[group.sector_of_dimension(tensor.shape[0])] = total
    return channels
