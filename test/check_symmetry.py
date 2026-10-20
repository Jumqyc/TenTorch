"""Scratch checks of the group layer: the singletons, the sector algebra and the fusion."""
from mpsdmrg.ext import _engine  # the group classes are engine-internal
from mpsdmrg.symmetry.cat import (
    BraidingStyle, FusionStyle, ZNSymmetry, fermion_parity, no_symmetry, su2_symmetry,
    u1_symmetry, z2_symmetry)

AbelianGroup = _engine.AbelianGroup
FermionParity = _engine.FermionParity
NoSymmetry = _engine.NoSymmetry
SU2Symmetry = _engine.SU2Symmetry
U1Symmetry = _engine.U1Symmetry

z3_symmetry = ZNSymmetry(3)

# singletons and equality with a fresh instance of the same class
assert no_symmetry == NoSymmetry()
assert u1_symmetry == U1Symmetry()
assert z2_symmetry == ZNSymmetry(2)
assert su2_symmetry == SU2Symmetry()
assert fermion_parity == FermionParity()

# NoSymmetry
assert no_symmetry.fuse() == (0,)
assert no_symmetry.fuse((0,), (0,)) == (0,)
assert no_symmetry.dual_sector((0,)) == (0,)
assert no_symmetry.sector_dim((0,)) == 1
assert no_symmetry.trivial_sector == (0,) and no_symmetry.type_name == 'NoSymmetry'
assert no_symmetry.is_abelian
assert no_symmetry.fusion_style is FusionStyle.single
assert no_symmetry.braiding_style is BraidingStyle.bosonic

# U(1)
assert u1_symmetry.fuse((1,), (-1,)) == (0,)
assert u1_symmetry.fuse((2,), (3,), (-1,)) == (4,)
assert u1_symmetry.dual_sector((2,)) == (-2,)
assert u1_symmetry.dual_sector((-2,)) == (2,)
assert u1_symmetry.fusion_outcomes((1,), (1,)) == ((2,),)
assert u1_symmetry.is_abelian and u1_symmetry.name == 'U(1)'

# Z_N
assert z2_symmetry.fuse((1,), (1,)) == (0,)
assert z2_symmetry.dual_sector((1,)) == (1,)
assert z3_symmetry.fuse((2,), (2,)) == (1,)
assert z3_symmetry.dual_sector((1,)) == (2,)
assert z2_symmetry.sector_dim((1,)) == 1 and z3_symmetry.name == 'Z3'
assert z3_symmetry.fusion_outcomes((2,), (2,)) == ((1,),)

# SU(2): sector (jj,) is spin jj/2
assert su2_symmetry.fusion_outcomes((1,), (1,)) == ((0,), (2,))
assert su2_symmetry.fusion_outcomes((2,), (2,)) == ((0,), (2,), (4,))
assert su2_symmetry.fusion_outcomes((0,), (2,)) == ((2,),)
assert su2_symmetry.sector_dim((0,)) == 1 and su2_symmetry.sector_dim((2,)) == 3
assert su2_symmetry.dual_sector((3,)) == (3,)
assert not su2_symmetry.is_abelian
assert su2_symmetry.fusion_style is FusionStyle.multiple_unique
try:
    su2_symmetry.fuse((1,), (1,))
except ValueError as exc:
    print('su2 fuse ->', exc)
else:
    raise AssertionError('SU(2) fusion has no unique outcome')

# FermionParity: Z2 fusion, fermionic braiding
assert fermion_parity.fuse((1,), (1,)) == (0,)
assert fermion_parity.dual_sector((1,)) == (1,)
assert fermion_parity.braiding_style is BraidingStyle.fermionic
assert fermion_parity.is_abelian and fermion_parity.sector_dim((1,)) == 1

# equality and hashing
assert z2_symmetry == ZNSymmetry(2) and hash(z2_symmetry) == hash(ZNSymmetry(2))
assert z2_symmetry != z3_symmetry
assert u1_symmetry != no_symmetry and su2_symmetry != u1_symmetry
assert fermion_parity != z2_symmetry
assert z2_symmetry != 'Z2'
assert isinstance(z2_symmetry, AbelianGroup) and isinstance(fermion_parity, AbelianGroup)
assert isinstance(no_symmetry, AbelianGroup) and isinstance(u1_symmetry, AbelianGroup)
assert not isinstance(su2_symmetry, AbelianGroup)

try:
    ZNSymmetry(0)
except ValueError as exc:
    print('ZN(0) ->', exc)
else:
    raise AssertionError('Z_0 must be rejected')

print('group names:', [g.name for g in
                       (no_symmetry, u1_symmetry, z2_symmetry, su2_symmetry, fermion_parity)])
print('SYMMETRY OK')
