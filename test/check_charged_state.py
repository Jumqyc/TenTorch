"""Scratch check: a state whose tensors carry a non-trivial total charge.

The two spin-1/2 sites in the triplet component: the bond between them carries the running
spin-1/2, and each tensor sits in the total sector (2,) rather than in the trivial one.
"""
from mpsdmrg.ext import su2_symmetry
from mpsdmrg.mps import MPS
from mpsdmrg.tensor import Tensor, multiplet_leg


def charged_state() -> MPS:
    '''
    The two-site state whose every tensor carries the total charge (2,).

    Returns:
      The state, with a spin-1/2 running charge on the middle bond.
    '''
    half, trivial = multiplet_leg(su2_symmetry, (1,)), multiplet_leg(su2_symmetry, (0,))
    bond = multiplet_leg(su2_symmetry, (1,))
    left = Tensor.zeros(1, 2, 2, legs=(trivial, half, bond.conj()),
                        qtotal=(2,), group=su2_symmetry)
    right = Tensor.zeros(2, 2, 1, legs=(bond, half, trivial),
                         qtotal=(2,), group=su2_symmetry)
    left.blocks[0].fill_(1.0)
    right.blocks[0].fill_(1.0)
    return MPS(2, 2, init_state=[left, right], group=su2_symmetry)


def main() -> None:
    '''Build the charged state and check that it is normalized and carries its charge.'''
    state = charged_state()
    print('totals', [tensor.qtotal for tensor in state.tensors])
    print('blocks', [len(tensor.blocks) for tensor in state.tensors])
    norm = complex(state.norm()).real
    print('norm^2', norm)
    assert abs(norm - 1.0) < 1e-12, norm
    assert all(tensor.qtotal != su2_symmetry.trivial_sector for tensor in state.tensors)
    print('CHARGED STATE CHECK OK')


if __name__ == '__main__':
    main()
