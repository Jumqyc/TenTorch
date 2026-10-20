'''
Check that ``MPS.apply`` realises an invariant operator: for the ground state of a
Hamiltonian the applied state is the same state scaled by the energy.
'''
from mpsdmrg import Broomstick
from mpsdmrg.symmetry.cat import su2_symmetry
from mpsdmrg.tensor import cached_einsum

import torch

LENGTH = 4
LABELS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


def dense_vector(tensors):
    '''
    Contract an MPS tensor list into one dense vector, site 0 the most significant.

    Args:
        tensors (list[Tensor]): legs (left bond, physical, right bond).
    Returns:
        torch.Tensor: the vector in shape (phys_dim ** L,).
    '''
    n = len(tensors)
    expr = [f'{LABELS[i]}{LABELS[n + 1 + i]}{LABELS[i + 1]}' for i in range(n)]
    return torch.einsum(','.join(expr) + '->' + ''.join(LABELS[n + 1 + i] for i in range(n)),
                        *[t.blocks[0] for t in tensors]).reshape(-1)


def dense_matrix(tensors, phys_dim):
    '''
    Contract an MPO tensor list into one dense operator, its two edges closed.

    Args:
        tensors (list[Tensor]): legs (left bond, right bond, physical, physical.conj()).
        phys_dim (int): the physical dimension of a site.
    Returns:
        torch.Tensor: the operator in shape (phys_dim ** L, phys_dim ** L), ket index last.
    '''
    n = len(tensors)
    expr = [f'{LABELS[i]}{LABELS[i + 1]}{LABELS[2 * n + 1 + i]}{LABELS[n + 1 + i]}'
            for i in range(n)]
    output = ''.join(LABELS[n + 1 + i] for i in range(n)) + ''.join(LABELS[2 * n + 1 + i]
                                                                   for i in range(n))
    return torch.einsum(','.join(expr) + '->' + output,
                        *[t.blocks[0] for t in tensors]).reshape(phys_dim ** n, phys_dim ** n)


def braket(bra, ket):
    '''
    The overlap <bra|ket> of two MPS tensor lists.

    Args:
        bra (list[Tensor]): the conjugated side, legs (left bond, physical, right bond).
        ket (list[Tensor]): the other side, same legs.
    Returns:
        complex: the overlap.
    '''
    env = cached_einsum('lsr,lsq->rq', bra[0].conj(), ket[0])
    for i in range(1, len(ket)):
        env = cached_einsum('rq,rst,qsu->tu', env, bra[i].conj(), ket[i])
    return complex(env.blocks[0].reshape(-1)[0])


def main() -> None:
    '''Apply the Heisenberg Hamiltonian to its own ground state and check the scaling.'''
    stick = Broomstick(L=LENGTH, phys_dim=2, max_bond_dim=16, svd_tol=1e-14, group=su2_symmetry,
                       site_sector=(1,))
    for i in range(LENGTH - 1):
        for a in 'XYZ':
            stick.add_couplings((a, a), 1.0, (i, i + 1))
    stick.build()
    stick.sweep(num_sweeps=4)
    state = stick.state
    applied = state.apply(stick.Hamiltonian)
    energy = stick.energy
    overlap = braket(state.tensors, applied.tensors)
    norm = braket(applied.tensors, applied.tensors).real
    print('energy   ', energy)
    print('overlap  ', overlap)
    print('norm^2   ', norm)
    assert abs(overlap - energy) < 1e-10, overlap
    assert abs(norm - energy ** 2) < 1e-10, norm
    print('APPLY CHECK OK')
    dense_case()


def dense_case() -> None:
    '''
    Apply a one-site matrix in the dense engine: the norm of ``S_z|psi>`` is ``<S_z^2>``.

    A dense leg has no multiplet structure, so a bare matrix is the whole operator.  The result
    is not canonical, so its norm has to contract the state instead of reading the centre tensor.
    '''
    from mpsdmrg.operators import SpinOperator
    from mpsdmrg.symmetry.cat import no_symmetry

    length = LENGTH + 2
    stick = Broomstick(L=length, phys_dim=2, max_bond_dim=32, svd_tol=1e-14, group=no_symmetry)
    for i in range(length - 1):
        for a in 'XYZ':
            stick.add_couplings((a, a), 1.0, (i, i + 1))
    stick.build()
    stick.sweep(num_sweeps=6)
    state = stick.state
    sz = SpinOperator(2).data[2].blocks[0]
    applied = state.apply(sz, 3)
    expected = complex(state.expectation({3: sz @ sz})).real
    print('dense state norm    ', complex(state.norm()).real)
    print('dense applied norm  ', complex(applied.norm()).real)
    print('<S_z^2>             ', expected)
    assert abs(expected - 0.25) < 1e-10, expected
    assert abs(complex(state.norm()).real - 1.0) < 1e-10, state.norm()
    assert abs(complex(applied.norm()).real - expected) < 1e-10, applied.norm()
    print('APPLY MATRIX CHECK OK')
    plain_mpo_case()


def plain_mpo_case():
    '''
    Apply a ``no_symmetry`` MPO: its dense chain holds one tensor per site and applies exactly.

    A dense MPO needs no charge bookkeeping, so the chain is the operator itself: the two ends
    are trivial, a term that skips a site passes through it on the identity rail, and a term
    that ends on the last site closes on that site's single closing slot.
    '''
    from mpsdmrg.mpo import MPO
    from mpsdmrg.symmetry.cat import no_symmetry

    length = LENGTH
    stick = Broomstick(L=length, phys_dim=2, max_bond_dim=32, svd_tol=1e-14, group=no_symmetry)
    for i in range(length - 1):
        for a in 'XYZ':
            stick.add_couplings((a, a), 1.0, (i, i + 1))
    stick.build()
    stick.sweep(num_sweeps=4)
    state = stick.state

    operator = MPO(L=length, phys_dim=2, group=no_symmetry)
    operator.add_couplings(('X', 'X'), 1.0, (0, 1))
    operator.add_couplings(('Z',), 0.7, (2,))
    operator.add_couplings(('Y', 'Y'), -0.3, (2, 3))
    operator.build()
    reference = dense_matrix(operator.tensors, 2) @ dense_vector(state.tensors)
    applied = dense_vector(state.apply(operator).tensors)
    print('plain tensor count  ', len(operator.tensors))
    print('plain |O psi|^2    ', float(torch.vdot(reference, reference).real))
    print('plain applied      ', float(torch.vdot(applied, applied).real))
    assert len(operator.tensors) == length, len(operator.tensors)
    assert float(torch.max(torch.abs(reference - applied))) < 1e-10
    print('APPLY DENSE MPO CHECK OK')


if __name__ == '__main__':
    main()
