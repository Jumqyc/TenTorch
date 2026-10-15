"""The symmetry-adapted tensor layer of ``mpsdmrg``.

Exposes the charge groups (the abstract ``Group`` with ``Sector``, ``FusionStyle`` and
``BraidingStyle``, its concrete ``NoSymmetry``, ``U1Symmetry``, ``ZNSymmetry``,
``SU2Symmetry``, ``SU3Symmetry`` and ``FermionParity``, and the module-level singletons), the
Clebsch-Gordan and recoupling data of the non-abelian groups (``su2data``), and the
block-sparse ``Tensor`` with its ``Leg``, the ``cached_einsum`` contraction and the
``svd``, ``qr``, ``rq``, ``svdvals``, ``norm``, ``inner`` and ``trace`` primitives.
"""
import torch

from mpsdmrg.symmetry.cpp import (  # noqa: F401
    SU2Symmetry, SU3Symmetry, U1Symmetry, ZNSymmetry, AbelianGroup, BraidingStyle, FermionParity,
    FusionStyle, FusionTree, Group, Leg, NoSymmetry, Sector, Tensor, _expr_cache,
    cached_einsum, dual_sign, fermion_parity, fusion_trees, inner, no_symmetry, norm, qr, rq,
    stored_dense, su2_symmetry, su3_symmetry, svd, svdvals, trace, u1_symmetry, z2_symmetry, z3_symmetry,
    z4_symmetry, z5_symmetry, z6_symmetry, z7_symmetry, z8_symmetry, z9_symmetry)

# the guard the DMRG puts around every local update
no_grad = torch.no_grad
