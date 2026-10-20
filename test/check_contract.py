"""Judge every non-abelian contraction against another route through the engine.

A contraction is judged by regrouping a product of three tensors, by the cyclic trace identity of a
matrix product, and against a hand-made pairwise split of a three-operand contraction.
"""
import torch

from base import CUDA
from mpsdmrg import linalg
from mpsdmrg.ext import _engine  # the leg type is engine-internal
from mpsdmrg.symmetry.cat import su2_symmetry
from mpsdmrg.tensor import Tensor, cached_einsum

Leg = _engine.Leg


def rand(legs, qtotal):
    '''A random tensor of the group on the given legs, in the trivial total charge.'''
    return Tensor.rand(*[leg.dim for leg in legs], legs=legs, qtotal=qtotal,
                       group=su2_symmetry, device=CUDA)


def judge_associative(name, first, second, third, outer, inner):
    '''
    A product of three tensors, grouped left and right.

    Args:
        first, second, third: the operands, in the leg structure the specs assume.
        outer: the spec of the left grouping, contracting (first second) with third.
        inner: the spec of the right grouping, contracting first with (second third).
    '''
    left = cached_einsum(outer[0], cached_einsum(outer[1], first, second), third)
    right = cached_einsum(inner[0], first, cached_einsum(inner[1], second, third))
    delta = float(linalg.norm(left - right).real)
    print(f'{name}: dev {delta:.3e}')
    assert delta < 1e-12, name


def judge_trace(name, first, second):
    '''A matrix product, judged by the cyclic identity tr(A B) = tr(B A).'''
    left = cached_einsum('ab,bc->ac', first, second)
    right = cached_einsum('ab,bc->ac', second, first)
    delta = float((linalg.trace(left) - linalg.trace(right)).abs())
    print(f'{name}: trace {complex(linalg.trace(left).item()).real:.6f} dev {delta:.3e}')
    assert delta < 1e-12, name


def main():
    '''Every contraction shape the engine handles, each against a second route.'''
    half = Leg(su2_symmetry, ((1,),), (0, 2))
    one = Leg(su2_symmetry, ((2,),), (0, 3))
    thick = Leg(su2_symmetry, ((2,),), (0, 6))
    mixed = Leg(su2_symmetry, ((0,), (2,)), (0, 1, 4))
    torch.manual_seed(0)
    square = lambda leg: rand((leg, leg.conj()), (0,))
    product = (('ac,cd->ad', 'ab,bc->ac'), ('ab,bd->ad', 'bc,cd->bd'))
    judge_associative('half x half x half', square(half), square(half), square(half), *product)
    judge_associative('spin-1 x spin-1 x spin-1', square(one), square(one), square(one), *product)
    judge_associative('multiplicity 2 x 2 x 2', square(thick), square(thick), square(thick), *product)
    judge_associative('mixed x mixed x mixed', square(mixed), square(mixed), square(mixed), *product)
    judge_associative('three-leg tree', rand((one, one, one), (0,)), rand((one.conj(), one), (0,)),
                      rand((one.conj(), one), (0,)),
                      ('abd,de->abe', 'abc,cd->abd'), ('abc,ce->abe', 'cd,de->ce'))
    judge_associative('four-leg tree', rand((one, one, one, one), (0,)), rand((one.conj(), one), (0,)),
                      rand((one.conj(), one), (0,)),
                      ('abce,ef->abcf', 'abcd,de->abce'), ('abcd,df->abcf', 'de,ef->df'))
    judge_trace('half x half trace', square(half), square(half))
    judge_trace('spin-1 trace', square(one), square(one))
    judge_trace('multiplicity 2 trace', square(thick), square(thick))
    judge_trace('mixed trace', square(mixed), square(mixed))

    # a three-operand contraction against a hand-made pairwise split
    first = rand((half, half), (0,))
    second = rand((half, half), (0,))
    third = rand((half.conj(), half.conj()), (0,))
    automatic = cached_einsum('ab,cd,ad->bc', first, second, third)
    manual = cached_einsum('bd,cd->bc', cached_einsum('ab,ad->bd', first, third), second)
    delta = float(linalg.norm(automatic - manual).real)
    print(f'three operands: dev {delta:.3e}')
    assert delta < 1e-12, 'three operands'
    print('SU2 CONTRACT CHECK OK')


if __name__ == '__main__':
    main()
