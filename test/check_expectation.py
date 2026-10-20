"""Check MPS.expectation on invariant combinations, against the dense components of the same state.

A symmetry-adapted state is a multiplet: it carries no definite magnetic member, so a single
component of an operator is not one of its observables.  Only the invariant combinations can be
measured on it, and Wigner and Eckart turn those into the components.  The dense state of the same
physical state measures the components directly, and the two routes have to agree.
"""
from mpsdmrg import Broomstick, MPO
from mpsdmrg.operators import SpinOperator, wigner_eckart_factor
from mpsdmrg.symmetry.cat import no_symmetry, su2_symmetry

LENGTH = 6
BOND = 8
FACTOR = wigner_eckart_factor(su2_symmetry, (2,)) # <Sz Sz> = FACTOR * <S . S>


def heisenberg(group, length=LENGTH, site_sector=None):
    '''
    The L=6 Heisenberg ground state with the given group.

    Args:
        group: symmetry group of the chain.
        length: number of sites.
        site_sector: the multiplet one site lives in, as in ``MPS``; needed by a group whose
            multiplets are not told apart by their dimension.
    '''
    stick = Broomstick(L=length, phys_dim=2, max_bond_dim=BOND, svd_tol=1e-14, group=group,
                       site_sector=site_sector)
    for i in range(length - 1):
        for letter in 'XYZ':
            stick.add_couplings((letter, letter), 0.25, (i, i + 1))
    stick.sweep(num_sweeps=8)
    return stick


def invariant_pair(i, j):
    '''
    The MPO of the spin scalar product of two sites, :math:`\\vec S_i \\cdot \\vec S_j`.

    Args:
        i, j: the two sites, in the order the couplings are added.
    '''
    operator = MPO(LENGTH, phys_dim=2, group=su2_symmetry, site_sector=(1,))
    for letter in 'XYZ':
        operator.add_couplings((letter, letter), 1.0, (i, j))
    operator.build()
    return operator


def main() -> None:
    '''Compare the adapted invariants with the dense components of the same physical state.'''
    spin = SpinOperator(2).data
    X, Y, Z = (spin[i].blocks[0] for i in range(3))
    symmetry, dense = heisenberg(su2_symmetry, site_sector=(1,)), heisenberg(no_symmetry)
    assert abs(symmetry.energy - dense.energy) < 1e-10

    # A rank-1 operator has no matrix element between two singlets, so every component of a
    # single site vanishes.  The dense route measures that; the adapted route can only state it,
    # a component not being one of its observables, and measures the invariant S(S+1) instead.
    for site in (0, 3):
        for letter, matrix in (('Sx', X), ('Sy', Y), ('Sz', Z)):
            want = dense.state.expectation({site: matrix})
            assert abs(want) < 1e-10, (site, letter, want)
        squared = dense.state.expectation({site: Z @ Z})
        assert abs(squared - FACTOR * 0.75) < 1e-10, (site, squared)

    # The invariant pair correlator of the adapted state against the dense components: the three
    # components of a singlet are equal, and each is the factor times the invariant.
    for i, j in ((0, 3), (0, 5), (1, 4)):
        invariant = complex(symmetry.state.expectation(invariant_pair(i, j))).real
        for letter, matrix in (('Sx', X), ('Sy', Y), ('Sz', Z)):
            component = complex(dense.state.expectation({i: matrix, j: matrix})).real
            assert abs(component - FACTOR * invariant) < 1e-10, ((i, j), letter, component,
                                                                 invariant)

    got = symmetry.state.expectation(symmetry.Hamiltonian)
    assert abs(got - symmetry.energy) < 1e-10, (got, symmetry.energy)
    print('EXPECTATION CHECK OK')


if __name__ == '__main__':
    main()
