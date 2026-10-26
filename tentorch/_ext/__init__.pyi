'''
Type stubs for the compiled symmetry engine.

Mirrors the names ``tentorch/_ext/__init__.py`` re-exports from ``symmetry_ext.so``, so a type
checker sees the surface of the extension without importing it.  A sector is a tuple of integers,
one per conserved charge of the group.
'''
from __future__ import annotations

from typing import Optional, Sequence, overload

import torch

Sector = tuple[int, ...]

class Outcome:
    '''One way two sectors fuse: the resulting sector and how many times it appears.'''

    sector: Sector
    multiplicity: int

    def __repr__(self) -> str: ...

class FusionCategory:
    '''A fusion category: the sectors, how they fuse and the symbols that recouple them.'''

    name: str
    dtype: torch.dtype
    device: torch.device

    def outcomes(self, a: Sector, b: Sector) -> list[Outcome]:
        '''The sectors `a` and `b` fuse to, with their multiplicities.'''
        ...

    def n_symbol(self, a: Sector, b: Sector, c: Sector) -> int:
        '''The multiplicity of `c` in the fusion of `a` and `b`.'''
        ...

    def can_fuse_to(self, a: Sector, b: Sector, c: Sector) -> bool:
        '''Whether `a` and `b` fuse to `c`.'''
        ...

    def F_symbol(self, a: Sector, b: Sector, c: Sector, d: Sector) -> torch.Tensor:
        '''The recoupling matrix of the four sectors, over the outcomes of the pairs.'''
        ...

    def R_symbol(self, b: Sector, c: Sector) -> torch.Tensor:
        '''The braiding matrix of the two sectors.'''
        ...

    def q_dim(self, a: Sector) -> float:
        '''The quantum dimension of the sector.'''
        ...

    def dual(self, a: Sector) -> Sector:
        '''The charge of the conjugate sector.'''
        ...

    def frobenius_schur_sign(self, a: Sector) -> float:
        '''The Frobenius-Schur sign of the sector, `+1` or `-1`.'''
        ...

    def trivial(self) -> Sector:
        '''The charge of the trivial sector.'''
        ...

    def clear_cache(self) -> None:
        '''Drop the fusion table and the vertex table this category keeps.'''
        ...

class Group(FusionCategory):
    '''
    A category whose sectors are the irreps of a group, so a vertex can be dressed with its
    Clebsch-Gordan coefficients.
    '''

    def clebsch_gordan(self, a: Sector, b: Sector, c: Sector, copy: int) -> torch.Tensor:
        '''The Clebsch-Gordan matrix of the fusion `a` times `b` to `c`, one copy of it.'''
        ...

class AbelianGroup(Group):
    '''A group whose sectors fuse in exactly one way, so every vertex carries no coefficient.'''

class NoSymmetry(AbelianGroup):
    '''The group of one sector: nothing is conserved.'''

    def __init__(self) -> None: ...

class U1Symmetry(AbelianGroup):
    '''The group of one integer charge, conserved by addition.'''

    def __init__(self) -> None: ...

class ZNSymmetry(AbelianGroup):
    '''The cyclic group of order `modulus`, its charges conserved modulo that order.'''

    def __init__(self, modulus: int) -> None: ...

class FermionParity(AbelianGroup):
    '''The group of fermion parity: the odd sector braids to a sign.'''

    def __init__(self) -> None: ...

class SU2Symmetry(Group):
    '''The irreducible representations of SU(2), labelled by twice the spin.'''

    def __init__(self) -> None: ...

class SU3Symmetry(Group):
    '''The irreducible representations of SU(3), labelled by their Dynkin labels.'''

    def __init__(self) -> None: ...

class FibonacciCategory(FusionCategory):
    '''The Fibonacci category: the trivial sector and one of quantum dimension the golden ratio.'''

    def __init__(self) -> None: ...

class Leg:
    '''A direct sum of sectors, each with a number of multiplets.'''

    def __init__(self, cat: FusionCategory, sectors: Sequence[Sector],
                 multiplicities: Sequence[int]) -> None: ...

    cat: FusionCategory
    sectors: list[Sector]
    multiplicities: list[int]

    @staticmethod
    def trivial(multiplets: int, cat: FusionCategory) -> Leg:
        '''A leg of one sector, the trivial one, with `multiplets` multiplets.'''
        ...

    @staticmethod
    def from_blocks(cat: FusionCategory,
                    entries: Sequence[tuple[Sector, int]]) -> tuple[Leg, list[int]]:
        '''A leg of the given blocks, with where each entry landed once the sectors are sorted.'''
        ...

    def nblocks(self) -> int:
        '''The number of sectors the leg holds.'''
        ...

    def sector(self, block: int) -> Sector:
        '''The sector of one block.'''
        ...

    def multiplicity(self, block: int) -> int:
        '''The number of multiplets of one block.'''
        ...

    def multiplets(self) -> int:
        '''The number of multiplets over every block.'''
        ...

    def fuse(self, other: Leg) -> Leg:
        '''The leg of the fusion of the two legs.'''
        ...

    def conj(self) -> Leg:
        '''The leg of the conjugate spaces, with every sector dualised.'''
        ...

    def capped(self, max_multiplets: int) -> Leg:
        '''The same leg with at most `max_multiplets` multiplets, the largest block first.'''
        ...

class FusionTree:
    '''The legs of one side and the charges they fuse through, one copy per inner vertex.'''

    sectors: list[Sector]
    inner: list[Sector]
    copies: list[int]
    coupled: Sector

    def size(self) -> int:
        '''The number of legs, which is two more than the number of inner vertices.'''
        ...

    def chain(self) -> list[Sector]:
        '''The sectors from the first leg to the coupled charge, vertex by vertex.'''
        ...

class BlockKey:
    '''The two roots and the copy that name one block of a tensor.'''

    codomain: Sector
    domain: Sector
    copy: int

    def __init__(self, codomain: Sector, domain: Sector, copy: int) -> None: ...

class Tensor:
    '''A tensor in the tree basis: one dense matrix per block.'''

    def __init__(self, cat: FusionCategory, codomain: Sequence[Leg], domain: Sequence[Leg],
                 qtotal: Sector, dtype: Optional[torch.dtype] = ...,
                 device: Optional[torch.device] = ...) -> None: ...

    cat: FusionCategory
    codomain: list[Leg]
    domain: list[Leg]
    qtotal: Sector
    dtype: torch.dtype
    device: torch.device

    @staticmethod
    def from_tuple(cat: FusionCategory, codomain: Sequence[Leg], domain: Sequence[Leg],
                   qtotal: Sector,
                   blocks: Optional[dict[BlockKey, torch.Tensor]] = ...) -> Tensor:
        '''A tensor from its legs and its blocks; a block left out stays zero.'''
        ...

    @staticmethod
    def random(cat: FusionCategory, codomain: Sequence[Leg], domain: Sequence[Leg],
               qtotal: Sector) -> Tensor:
        '''A tensor with every block drawn from the standard normal distribution.'''
        ...

    def nlegs(self) -> int:
        '''The number of legs, the codomain ones and the domain ones together.'''
        ...

    def legs(self) -> list[Leg]:
        '''The legs, the codomain ones first and then the domain ones.'''
        ...

    def nblocks(self) -> int:
        '''The number of blocks.'''
        ...

    def keys(self) -> list[BlockKey]:
        '''The key of every block.'''
        ...

    def conj(self) -> Tensor:
        '''The conjugate tensor: the two sides swapped and every block conjugated.'''
        ...

    def clone(self) -> Tensor:
        '''A copy that shares no storage with this tensor.'''
        ...

    def repartition(self, codomain: Sequence[int]) -> Tensor:
        '''The same tensor with the named legs on the codomain, in that order; the other legs keep
        their relative order on the domain.'''
        ...

    def to_tuple(self) -> tuple[list[Leg], list[Leg], Sector, dict[BlockKey, torch.Tensor]]:
        '''The legs and the blocks, the inverse of `from_tuple`.'''
        ...

    def __iadd__(self, other: Tensor) -> Tensor:
        '''Add another tensor of the same legs and charge in place.'''
        ...

    def __isub__(self, other: Tensor) -> Tensor:
        '''Subtract another tensor of the same legs and charge in place.'''
        ...

    def __imul__(self, scalar: float) -> Tensor:
        '''Scale in place.'''
        ...

    def __itruediv__(self, scalar: float) -> Tensor:
        '''Divide in place.'''
        ...

    def __mul__(self, scalar: complex) -> Tensor: ...

    def __rmul__(self, scalar: complex) -> Tensor: ...

    def __repr__(self) -> str: ...

def validate(category: FusionCategory, tree: FusionTree) -> None:
    '''Raise when the tree does not fuse the way the category allows.'''
    ...

def reachable_charges(category: FusionCategory, legs: Sequence[Leg]) -> list[Sector]:
    '''The charges the legs fuse to.'''
    ...

def fusion_trees(category: FusionCategory, legs: Sequence[Leg], coupled: Sector) -> list[FusionTree]:
    '''Every tree over the legs that fuses to the coupled charge.'''
    ...

def block_keys(category: FusionCategory, codomain: Sequence[Leg], domain: Sequence[Leg],
               qtotal: Sector) -> list[BlockKey]:
    '''The key of every block a tensor over those legs and that charge holds.'''
    ...

@overload
def contract(first: Tensor, other: Tensor, first_domain: Sequence[int],
             other_codomain: Sequence[int]) -> Tensor: ...
@overload
def contract(first: Tensor, other: Tensor, first_domain: Sequence[int],
             other_codomain: Sequence[int], charge: Sector) -> Tensor: ...
def contract(first: Tensor, other: Tensor, first_domain: Sequence[int],
             other_codomain: Sequence[int], charge: Sector) -> Tensor:
    '''Contract the named legs of the two tensors, the charge of the result given explicitly.'''
    ...

def einsum(spec: str, *tensors: Tensor, outcharge: Optional[Sector] = ...) -> Tensor:
    '''Contract a whole network at once, summing over every path of charges that reaches
    `outcharge`, which defaults to the trivial charge.'''
    ...

def inner(first: Tensor, second: Tensor) -> complex:
    '''The scalar product of two tensors with the same legs.'''
    ...

def norm(tensor: Tensor) -> float:
    '''The Frobenius norm of a tensor.'''
    ...

def is_close(first: Tensor, second: Tensor, rtol: float = ..., atol: float = ...) -> bool:
    '''Whether two tensors agree block by block within the tolerances.'''
    ...

@overload
def svd(tensor: Tensor, max_bond: Optional[int] = ..., tol: float = ...,
        absorb: int = ...) -> tuple[Tensor, torch.Tensor, Tensor]: ...
@overload
def svd(tensor: Tensor, left: Sequence[int], max_bond: Optional[int] = ..., tol: float = ...,
        absorb: int = ...) -> tuple[Tensor, torch.Tensor, Tensor]: ...
def svd(tensor: Tensor, left: Optional[Sequence[int]] = ..., max_bond: Optional[int] = ...,
        tol: float = ..., absorb: int = ...) -> tuple[Tensor, torch.Tensor, Tensor]:
    '''Singular value decomposition over two groups of legs.'''
    ...

@overload
def svdvals(tensor: Tensor) -> torch.Tensor: ...
@overload
def svdvals(tensor: Tensor, left: Sequence[int]) -> torch.Tensor: ...
def svdvals(tensor: Tensor, left: Optional[Sequence[int]] = ...) -> torch.Tensor:
    '''The singular values of the splitting, without the factors.'''
    ...

@overload
def qr(tensor: Tensor) -> tuple[Tensor, Tensor]: ...
@overload
def qr(tensor: Tensor, left: Sequence[int]) -> tuple[Tensor, Tensor]: ...
def qr(tensor: Tensor, left: Optional[Sequence[int]] = ...) -> tuple[Tensor, Tensor]:
    '''QR decomposition over two groups of legs.'''
    ...

@overload
def rq(tensor: Tensor) -> tuple[Tensor, Tensor]: ...
@overload
def rq(tensor: Tensor, left: Sequence[int]) -> tuple[Tensor, Tensor]: ...
def rq(tensor: Tensor, left: Optional[Sequence[int]] = ...) -> tuple[Tensor, Tensor]:
    '''RQ decomposition over two groups of legs.'''
    ...

def clear_caches() -> None:
    '''Drop every table and cache the engine keeps.'''
    ...

def z_n(modulus: int) -> ZNSymmetry:
    '''The cyclic group of order `modulus`, shared by everything that asks for it.'''
    ...

no_symmetry: NoSymmetry
u1: U1Symmetry
fermion_parity: FermionParity
su2: SU2Symmetry
su3: SU3Symmetry
fibonacci: FibonacciCategory
