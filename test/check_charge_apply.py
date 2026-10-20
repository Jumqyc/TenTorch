"""Check the charge-changing MPO contract: the charged channel and the invariant operator.

A charge-changing operator mixes bond charges inside one window.  The block-sparse contraction keeps
the fusion path of every block, so the charged channel of a two-site operator adds up with the
trivial one to the invariant application of the same operator, and a one-site component of a
multiplet, which needs a member axis of its own, stays refused.
"""
import torch

from base import COMPLEX, CUDA
from mpsdmrg import Broomstick, MPO, MPS
from mpsdmrg.tensor import Tensor
from mpsdmrg.symmetry.cat import no_symmetry, su2_symmetry, su3_symmetry


def gell_mann(device: torch.device) -> list[torch.Tensor]:
    '''
    The eight Hermitian Gell-Mann generators of SU(3).

    Args:
        device: device of the matrices.
    Returns:
        The generators in shape (3, 3), normalized so that ``tr(T_a T_b) = 2 delta_ab``.
    '''
    imaginary = 1j
    generators = [
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, -imaginary, 0], [imaginary, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
        [[0, 0, -imaginary], [0, 0, 0], [imaginary, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 0, -imaginary], [0, imaginary, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, -2]],
    ]
    return [torch.tensor(matrix, dtype=COMPLEX, device=device) / (1.0 if index < 7 else 3 ** 0.5)
            for index, matrix in enumerate(generators)]


def bond_sectors(state: MPS) -> set[tuple[int, ...]]:
    '''Every sector carried by a bond of the state, the trivial edge included.'''
    return {tuple(sector) for tensor in state.tensors for leg in (tensor.legs[0], tensor.legs[2])
            for sector in leg.sectors}


def heisenberg(group, length: int, phys_dim: int, site_sector: tuple[int, ...]) -> Broomstick:
    '''
    The Heisenberg chain of one group, swept to its ground state.

    Args:
        group: the symmetry group of the chain.
        length: number of sites.
        phys_dim: dimension of one site.
        site_sector: the multiplet one site lives in.
    Returns:
        The swept stick, its state at the ground state.
    '''
    stick = Broomstick(L=length, phys_dim=phys_dim, max_bond_dim=32, svd_tol=1e-14, group=group,
                       site_sector=site_sector)
    for site in range(length - 1):
        for name in 'XYZ':
            stick.add_couplings((name, name), 1.0, (site, site + 1))
    stick.build()
    stick.sweep(num_sweeps=4)
    return stick


def su2_case() -> None:
    '''The channels of a bond operator on the Heisenberg ground state, against the trivial group.'''
    length = 4
    couplings = [((name, name), 1.0, (0, 1)) for name in 'XYZ']
    stick = heisenberg(su2_symmetry, length, 2, (1,))
    norms: dict[tuple[int, ...] | None, float] = {}
    applied: MPS | None = None
    for total in (None, (0,), (2,)):
        operator = MPO(L=length, phys_dim=2, group=su2_symmetry, total=total, site_sector=(1,))
        for names, coefficient, sites in couplings:
            operator.add_couplings(names, coefficient, sites)
        operator.build()
        state = stick.state.apply(operator, max_bond_dim=32, tol=1e-14)
        norms[total] = float(state.norm().real)
        if total == (2,):
            applied = state
    sectors = bond_sectors(applied)

    # the same operator on the same chain without symmetry, where every tensor is dense
    dense_stick = heisenberg(no_symmetry, length, 2, None)
    reference = MPO(L=length, phys_dim=2)
    for names, coefficient, sites in couplings:
        reference.add_couplings(names, coefficient, sites)
    reference.build()
    reference_norm = float(dense_stick.state.apply(reference, max_bond_dim=32, tol=1e-14).norm().real)

    print('SU(2) invariant', norms[None], 'trivial', norms[(0,)], 'triplet', norms[(2,)])
    print('SU(2) dense reference', reference_norm, 'bond sectors', sorted(sectors))
    assert abs(norms[(2,)] - reference_norm) < 1e-10, (norms[(2,)], reference_norm)
    # the total charge selects which fusion pairs the operator keeps; every channel of this bond
    # operator is a two-site pair of the same generators, so all three totals build it
    assert abs(norms[None] - norms[(0,)]) < 1e-10, norms
    assert abs(norms[None] - norms[(2,)]) < 1e-10, norms
    assert (2,) in sectors, sectors


def su3_case() -> None:
    '''An adjoint charge transfer on the ULS state, whose bonds carry a multiplicity of two.'''
    length = 6
    mapping = {f'L{index + 1}': Tensor.from_torch(generator)
               for index, generator in enumerate(gell_mann(CUDA))}
    mapping['I'] = Tensor.from_torch(torch.eye(3, dtype=COMPLEX, device=CUDA))
    stick = Broomstick(L=length, phys_dim=3, max_bond_dim=32, svd_tol=1e-14, mapping=mapping,
                       group=su3_symmetry, site_sector=(1, 0))
    for site in range(length - 1):
        stick.add_couplings(('I', 'I'), 1.0 / 3.0, (site, site + 1))
        for name in mapping:
            if name != 'I':
                stick.add_couplings((name, name), 0.5, (site, site + 1))
    stick.build()
    stick.sweep(num_sweeps=4)
    multiplicities = [leg.multiplicity(index) for tensor in stick.state.tensors for leg in tensor.legs
                      for index, _ in enumerate(leg.sectors)]
    energy = float(stick.energy.real)
    print('SU(3) energy', energy, 'bond dims', stick.state.bond_dim,
          'bond multiplicities', sorted(set(multiplicities)))
    # the ground energy of the six-site ULS chain at the SU(3) symmetric point
    assert abs(energy - (-4.069784138805)) < 1e-9, energy

    charge = MPO(L=length, phys_dim=3, group=su3_symmetry, total=(1, 1), mapping=mapping,
                 site_sector=(1, 0))
    charge.add_couplings(('L1', 'L1'), 1.0, (0, 1))
    charge.build()
    applied = stick.state.apply(charge, max_bond_dim=32, tol=1e-14)
    transferred = float(applied.norm().real)
    sectors = bond_sectors(applied)
    print('SU(3) adjoint channel norm squared', transferred, 'bond sectors', sorted(sectors))
    assert max(multiplicities) >= 2, multiplicities
    assert (1, 1) in sectors, sectors


def one_site_case() -> None:
    '''A one-site component of a multiplet is refused: it needs a member axis of its own.'''
    charge = MPO(L=4, phys_dim=2, group=su2_symmetry, total=(2,), site_sector=(1,))
    charge.add_couplings(('X',), 1.0, (0,))
    try:
        charge.build()
    except NotImplementedError as refusal:
        print('one-site component refused:', refusal)
    else:
        raise AssertionError('a one-site component of a multiplet was built.')


def main() -> None:
    '''The channels of a bond operator in both groups, and the one-site refusal.'''
    su2_case()
    su3_case()
    one_site_case()
    print('CHARGE APPLY CHECK OK')


if __name__ == '__main__':
    main()
