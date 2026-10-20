"""Scratch check: the C++ block-sparse tensor against the Python engine."""
import torch

from mpsdmrg.ext import _engine  # the leg type is engine-internal
from mpsdmrg.symmetry.cat import no_symmetry, su2_symmetry, u1_symmetry

Leg = _engine.Leg
from mpsdmrg.tensor import Tensor, cached_einsum
from mpsdmrg import ext as cpp

DTYPE, DEVICE = torch.complex128, torch.device('cuda')
SU2 = su2_symmetry
GROUPS = {repr(group): group for group in (cpp.no_symmetry, cpp.u1_symmetry, cpp.z2_symmetry,
                                          cpp.su2_symmetry, cpp.fermion_parity)}


def mirror(legs):
    """The same legs in the C++ engine."""
    return tuple(_engine.Leg(GROUPS[repr(leg.group)], leg.sectors, leg.slices, qconj=leg.qconj)
                 for leg in legs)


def spin_leg(jj: int, qconj: int = 1) -> Leg:
    """One leg carrying a single spin sector."""
    return Leg(SU2, ((jj,),), (0, jj + 1), qconj=qconj)


def raw(shape: tuple[int, ...]) -> torch.Tensor:
    """A fixed dense buffer of ``shape``."""
    count = 1
    for dim in shape:
        count *= dim
    return torch.arange(1, 1 + count, dtype=torch.float64, device=DEVICE).reshape(shape).to(DTYPE)


def blocks(tensor) -> dict:
    """The blocks of a tensor keyed by their charge-block index."""
    return {key: block for key, block in zip(tensor.qdata, tensor.blocks)}


def compare(name: str, want, got, tol: float = 0.0) -> None:
    """Compare a Python tensor with a C++ tensor on every observable.

    Args:
      name: label of the case.
      want: the Python tensor.
      got: the C++ tensor.
      tol: tolerance of a block entry; a contraction may sum in another order.
    """
    assert isinstance(want, Tensor) and isinstance(got, cpp.Tensor), name
    assert want.shape == got.shape, (name, want.shape, got.shape)
    assert tuple(map(repr, want.legs)) == tuple(map(repr, got.legs)), name
    assert want.qdata == got.qdata, (name, want.qdata, got.qdata)
    assert want.qtotal == got.qtotal, (name, want.qtotal, got.qtotal)
    assert want.is_dense == got.is_dense and want.numel() == got.numel(), name
    assert repr(want.group) == repr(got.group), (name, repr(want.group), repr(got.group))
    left, right = blocks(want), blocks(got)
    assert set(left) == set(right), (name, set(left) ^ set(right))
    worst = 0.0
    for key in left:
        assert left[key].shape == right[key].shape, (name, key)
        worst = max(worst, float((left[key] - right[key]).abs().max()))
    assert worst <= tol, (name, 'block', worst)
    print(f'{name:24s} blocks {len(left):2d}  shape {want.shape}  max|diff| = {worst:.3e}')


def build(data: torch.Tensor, legs, qtotal=None):
    """The same tensor through both engines."""
    return Tensor.from_torch(data, legs, qtotal), cpp.Tensor.from_torch(data, mirror(legs), qtotal)


def random_pair(legs, qtotal=(0,), seed: int = 0):
    """The same random charged tensor through both engines, over the same block data."""
    options = dict(qtotal=qtotal, dtype=DTYPE, device=DEVICE)
    torch.manual_seed(seed)
    want = Tensor.rand(*[leg.dim for leg in legs], legs=legs, **options)
    got = cpp.Tensor([block.clone() for block in want.blocks], want.qdata, mirror(want.legs),
                     want.qtotal, GROUPS[repr(want.group)])
    return want, got


def zeros_pair(legs, qtotal=(0,)):
    """The same zero tensor through both engines."""
    options = dict(qtotal=qtotal, dtype=DTYPE, device=DEVICE)
    return (Tensor.zeros(*[leg.dim for leg in legs], legs=legs, **options),
            cpp.Tensor.zeros(*[leg.dim for leg in legs], legs=mirror(legs), **options))


half, one = spin_leg(1), spin_leg(2)
thick = Leg(SU2, ((2,),), (0, 6))
mixed = Leg(SU2, ((0,), (2,)), (0, 1, 4))
u1_link = Leg(u1_symmetry, ((-1,), (0,), (1,)), (0, 1, 3, 5))

compare('dense 2x3', Tensor.from_torch(raw((2, 3))), cpp.Tensor.from_torch(raw((2, 3))))
compare('u1 2 legs', *build(raw((5, 5)), (u1_link, u1_link)))
compare('u1 dual leg', *build(raw((5, 5)), (u1_link, u1_link.conj())))
compare('no symmetry', *build(raw((2, 3)), (Leg(no_symmetry, ((0,),), (0, 2)),
                                        Leg(no_symmetry, ((0,),), (0, 3)))))
compare('su2 random 2 legs', *random_pair((half, half.conj()), seed=1))
compare('su2 random 3 legs', *random_pair((one, one, one), seed=2))
compare('su2 random thick', *random_pair((thick, thick.conj()), seed=3))
compare('su2 zeros mixed', *zeros_pair((mixed, mixed.conj())))
print('construction ok')

for name, legs, seed in (('su2', (mixed, mixed.conj()), 4), ('u1', (u1_link, u1_link), 5)):
    want, got = random_pair(legs, seed=seed) if name == 'su2' else build(raw((5, 5)), legs)
    compare(f'{name} + self', want + want, got + got)
    compare(f'{name} - 2 * self', want - want * 2.0, got - got * 2.0)
    compare(f'{name} * 3.5', want * 3.5, got * 3.5)
    compare(f'{name} / 2', want / 2.0, got / 2.0)
    compare(f'{name} - self', -want, -got)
    compare(f'{name} conj', want.conj(), got.conj())
    compare(f'{name} clone', want.clone(), got.clone())
    compare(f'{name} real', want.real, got.real)
scalar = (Leg(u1_symmetry, ((0,),), (0, 1)), Leg(u1_symmetry, ((0,),), (0, 1)))
assert build(raw((1, 1)), scalar)[0].item() == build(raw((1, 1)), scalar)[1].item()
print('arithmetic ok')

def outcome(call):
    """The name of what a call raises, or 'ok'."""
    try:
        call()
        return 'ok'
    except Exception as exc:
        return type(exc).__name__


# a charged tensor keeps its magnetic indices in the fusion vertex, so one entry is not a
# stored number: both engines refuse the read instead of materialising the component
want, got = build(raw((5, 5)), (u1_link, u1_link))
assert outcome(lambda: want[0, 1]) == outcome(lambda: got[0, 1]) == 'NotImplementedError'
print('charged getitem          NotImplementedError: close the selected legs with unit vectors')


# a combined leg needs one distinct charge per block: the link pair repeats its charge
u1_left, u1_phys = Leg(u1_symmetry, ((0,),), (0, 1)), Leg(u1_symmetry, ((-1,), (1,)), (0, 1, 2))
u1_right = Leg(u1_symmetry, ((-1,), (1,)), (0, 1, 2), qconj=-1)
want, got = random_pair((u1_left, u1_phys, u1_right), seed=7)
compare('u1 combine', want.combine_legs(((0, 1), (2,))), got.combine_legs(((0, 1), (2,))))
want, got = random_pair((one, one, one), seed=8)
assert (outcome(lambda: want.combine_legs(((0, 1), (2,)))) ==
        outcome(lambda: got.combine_legs(((0, 1), (2,)))) == 'NotImplementedError')
print('su2 combine              NotImplementedError: non-abelian fusion of several legs')
want, got = build(raw((5, 5)), (u1_link, u1_link))
assert (outcome(lambda: want.combine_legs(((0, 1),))) ==
        outcome(lambda: got.combine_legs(((0, 1),))) == 'AssertionError')
print('indexing and fusion ok')

left, right, square = raw((4, 3)), raw((3, 5)), raw((3, 4))
for spec, second in (('ab,bc->ac', right), ('ab,bc->ca', right), ('ab,bc->', square)):
    want = cached_einsum(spec, Tensor.from_torch(left), Tensor.from_torch(second))
    got = cpp.cached_einsum(spec, cpp.Tensor.from_torch(left), cpp.Tensor.from_torch(second))
    compare(f'dense {spec}', want, got, tol=1e-12)
print('dense einsum ok')

CASES = (
    ('half leg', 'ab,bc->ac', ((half, half.conj()), (half, half.conj()))),
    ('half scalar', 'ab,ab->', ((half, half), (half.conj(), half.conj()))),
    ('spin-1 leg', 'ab,bc->ac', ((one, one.conj()), (one, one.conj()))),
    ('multiplicity 2', 'ab,bc->ac', ((thick, thick.conj()), (thick, thick.conj()))),
    ('mixed sectors', 'ab,bc->ac', ((mixed, mixed.conj()), (mixed, mixed.conj()))),
    ('three-leg tree', 'abc,cd->abd', ((one, one, one), (one.conj(), one))),
    ('four-leg tree', 'abcd,de->abce', ((one, one, one, one), (one.conj(), one))),
    ('three operands', 'ab,cd,ad->bc', ((half, half), (half, half), (half.conj(), half.conj()))),
)
for index, (name, spec, leg_sets) in enumerate(CASES):
    want_operands, got_operands = [], []
    for operand, legs in enumerate(leg_sets):
        want, got = random_pair(legs, seed=10 + 2 * index + operand)
        want_operands.append(want)
        got_operands.append(got)
    compare(f'su2 {name}', cached_einsum(spec, *want_operands),
            cpp.cached_einsum(spec, *got_operands), tol=1e-12)
print('su2 einsum ok')

charged, charged_cpp = random_pair((half, half.conj()), seed=9)
assert outcome(lambda: cached_einsum('ab,bc->ac', Tensor.from_torch(left), right)) == 'ok'
assert outcome(lambda: cpp.cached_einsum('ab,bc->ac', cpp.Tensor.from_torch(left), right)) == 'ok'
assert outcome(lambda: cached_einsum('ab,bc->ac', charged, right)) == 'NotImplementedError'
assert outcome(lambda: cpp.cached_einsum('ab,bc->ac', charged_cpp, right)) == 'NotImplementedError'
print('charged x raw            NotImplementedError on both engines')

for name, call in (
        ('no singlet', lambda: cpp.Tensor([torch.zeros(1, 1, 1, dtype=DTYPE, device=DEVICE)],
                                           ((0, 0, 0, (2,),),), mirror((spin_leg(1), spin_leg(1), spin_leg(1))))),
        ('bad block index', lambda: cpp.Tensor([torch.zeros(2, 2, dtype=DTYPE, device=DEVICE)],
                                                ((9, 9),), mirror((mixed, mixed))))):
    try:
        call()
        print(f'{name:20s} NO RAISE')
    except AssertionError as exc:
        print(f'{name:20s} AssertionError: {exc}')

cpp._engine.clear_caches()
print('CPP TENSOR CHECK OK')
