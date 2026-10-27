import math
from typing import Optional, Sequence

import torch

from tentorch._ext import (FusionCategory, FusionTree, Leg, Sector, Tensor, contract,
                           einsum, inner, reachable_charges)
from tentorch.linalg.graded import qr, rq
from tentorch.tensors.tree import entered


def _leg(cat: FusionCategory, sectors: Sequence[Sector], multiplets: int = 1) -> Leg:
    '''
    A bond over the named sectors, with the same number of multiplets in each of them.
    Args:
      cat: the category of the leg.
      sectors: the sectors to carry, in order.
      multiplets: the multiplets of every sector.
    Returns:
      The leg.
    '''
    return Leg(cat, list(sectors), [multiplets] * len(sectors))


def _identity(cat: FusionCategory, bond: Leg, middle: Optional[Leg] = None) -> Tensor:
    '''
    The tensor that closes a chain at one end: the identity on the bond it meets.
    Args:
      cat: the category of the legs.
      bond: the bond of the state, on both the codomain and the domain.
      middle: the virtual bond of the operator, on the codomain; omit it for a plain state.
    Returns:
      The tensor.
    '''
    def entry(tree_out: FusionTree, copy_out: tuple[int, ...], tree_in: FusionTree,
              copy_in: tuple[int, ...]) -> float:
        same = tree_out.sectors[0] == tree_in.sectors[0] and copy_out[0] == copy_in[0]
        return 1.0 if same and (middle is None or tree_out.sectors[1] == cat.trivial()) else 0.0

    codomain = [bond] if middle is None else [bond, middle]
    return entered(cat, codomain, [bond], entry)


class MPS:
    '''
    A matrix product state over a chain of sites, in the mixed canonical form of `center`: every
    site left of the centre is left isometric, every site right of it right isometric, so the norm
    of the state sits in the centre tensor alone.

    Args:
      tensors: one tensor per site; site `i` has the codomain `(bond i, physical)` and the domain
        `(bond i + 1)`, and every site shares the physical leg.
      physical: the physical leg of one site.
      center: the site that carries the norm, the first one when it is omitted.
    Raises:
      ValueError: when the chain is empty or a site has another layout.
      IndexError: when the centre falls outside the chain.
    '''

    def __init__(self, tensors: Sequence[Tensor], physical: Leg, center: int = 0) -> None:
        if not tensors:
            raise ValueError('A state needs at least one site.')
        for at, tensor in enumerate(tensors):
            if tensor.nlegs() != 3:
                raise ValueError(f'site {at} needs three legs, not {tensor.nlegs()}')
        if not 0 <= center < len(tensors):
            raise IndexError(f'the centre {center} is outside the chain of {len(tensors)} sites')
        self.tensors = list(tensors)
        self.physical = physical
        self.center = center
        self.canonicalize()

    @property
    def L(self) -> int:
        '''The number of sites.'''
        return len(self.tensors)

    @property
    def cat(self) -> FusionCategory:
        '''The category of the state.'''
        return self.tensors[0].cat

    @property
    def dtype(self) -> torch.dtype:
        '''The scalar type of the state.'''
        return self.tensors[0].dtype

    @property
    def device(self) -> torch.device:
        '''The device of the state.'''
        return self.tensors[0].device

    @property
    def bond_dims(self) -> list[int]:
        '''The multiplets of every bond, one entry per bond between two sites.'''
        return [self.tensors[at].domain[0].multiplets() for at in range(self.L - 1)]

    @classmethod
    def product(cls, cat: FusionCategory, physical: Leg, patterns: Sequence[Sector],
                sector: Optional[Sector] = None) -> 'MPS':
        '''
        The product state of one physical sector per site, with bonds of one multiplet.
        Args:
          cat: the category of the state.
          physical: the physical leg of one site.
          patterns: the sector every site occupies, in order.
          sector: the charge of the state, the trivial charge when it is None.
        Returns:
          The state.
        Raises:
          ValueError: when the sectors do not couple along the chain, or the chain as a whole
            couples to a charge other than `sector`.
        '''
        charge = cat.trivial() if sector is None else sector
        bonds = [_leg(cat, [charge])]
        for pattern in patterns:
            coupled = [outcome.sector for outcome in cat.outcomes(bonds[-1].sector(0), pattern)]
            if not coupled:
                raise ValueError(f'the sector {pattern} couples to nothing on {bonds[-1].sector(0)}')
            bonds.append(_leg(cat, [coupled[0]]))
        if bonds[-1].sector(0) != charge:
            raise ValueError(f'the sites couple to {bonds[-1].sector(0)}, not to {charge}')

        tensors = []
        for at in range(len(patterns)):
            def entry(tree_out: FusionTree, copy_out: tuple[int, ...], tree_in: FusionTree,
                      copy_in: tuple[int, ...]) -> float:
                return 1.0 if tree_out.sectors[1] == patterns[at] else 0.0

            tensors.append(entered(cat, [bonds[at], physical], [bonds[at + 1]], entry))
        return cls(tensors, physical)

    @classmethod
    def random(cls, cat: FusionCategory, physical: Leg, length: int,
               sector: Optional[Sector] = None, seed: int = 0, multiplets: int = 1) -> 'MPS':
        '''
        The random state of a charge, over every bond charge that charge admits.
        Args:
          cat: the category of the state.
          physical: the physical leg of one site.
          length: the number of sites.
          sector: the charge of the state, the trivial charge when it is None.
          seed: the seed of the random entries.
          multiplets: the multiplets carried by every bond charge.
        Returns:
          The state.
        Raises:
          ValueError: when the charge admits no state of that length.
        '''
        charge = cat.trivial() if sector is None else sector
        left = [reachable_charges(cat, [physical] * at) for at in range(length + 1)]
        tail = [reachable_charges(cat, [physical] * (length - at)) for at in range(length + 1)]
        bonds = [_leg(cat, [bond for bond in left[at]
                            if any(cat.can_fuse_to(bond, reached, charge) for reached in tail[at])],
                      multiplets)
                 for at in range(length + 1)]
        # the two ends of a finite chain carry one multiplet, so the state is a vector and not a map
        bonds[0], bonds[-1] = _leg(cat, [charge]), _leg(cat, [charge])
        if not bonds[0].nblocks() or not bonds[-1].nblocks():
            raise ValueError(f'the charge {charge} admits no state of {length} sites')

        torch.manual_seed(seed)
        tensors = [Tensor.random(cat, [bonds[at], physical], [bonds[at + 1]], charge)
                   for at in range(length)]
        return cls(tensors, physical)

    def _left_move(self, at: int) -> None:
        '''
        Make site `at` left isometric and push its remainders into site `at + 1`.

        The QR of the site splits it into an isometry and the matrix the next site absorbs; the
        centre therefore lands on `at + 1`.
        Args:
          at: the site to canonicalize, in `[0, L - 1)`.
        Raises:
          IndexError: when the site has no site to its right.
        '''
        if at < 0 or at >= self.L - 1:
            raise IndexError(f'index {at} has no site to its right in a chain of {self.L} sites')
        factor, rest = qr(self.tensors[at], [0, 1])
        self.tensors[at] = factor
        self.tensors[at + 1] = contract(rest, self.tensors[at + 1].repartition([0]), [0],
                                        [0]).repartition([0, 1])
        self.center = at + 1

    def _right_move(self, at: int) -> None:
        '''
        Make site `at` right isometric and push its remainders into site `at - 1`.

        The RQ of the site splits it into the matrix the previous site absorbs and an isometry; the
        centre therefore lands on `at - 1`.
        Args:
          at: the site to canonicalize, in `(0, L - 1]`.
        Raises:
          IndexError: when the site has no site to its left.
        '''
        if at <= 0 or at >= self.L:
            raise IndexError(f'index {at} has no site to its left in a chain of {self.L} sites')
        factor, rest = rq(self.tensors[at], [0])
        self.tensors[at] = rest.repartition([0, 1])
        self.tensors[at - 1] = contract(self.tensors[at - 1], factor, [0], [0])
        self.center = at - 1

    def move_center(self, at: int) -> None:
        '''
        Move the centre to a site, keeping the mixed canonical form.
        Args:
          at: the new centre, in `[0, L)`.
        Raises:
          IndexError: when the site falls outside the chain.
        '''
        if not 0 <= at < self.L:
            raise IndexError(f'the centre {at} is outside the chain of {self.L} sites')
        while self.center < at:
            self._left_move(self.center)
        while self.center > at:
            self._right_move(self.center)

    def canonicalize(self) -> None:
        '''
        Bring the state to the mixed canonical form of the current centre.

        Every site left of the centre becomes left isometric and every site right of it right
        isometric, so the norm of the state ends up in the centre tensor alone.
        '''
        for at in range(self.center):
            self._left_move(at)
        for at in range(self.L - 1, self.center, -1):
            self._right_move(at)

    def norm(self) -> float:
        '''The norm of the state, the Frobenius norm of the centre tensor.'''
        return math.sqrt(self._weight())

    def _weight(self) -> float:
        '''The squared norm, read off the centre tensor of the mixed canonical form.'''
        return sum(float((block.abs() ** 2).sum())
                   for block in self.tensors[self.center].to_tuple()[3].values())

    def normalize(self) -> None:
        '''Scale the state to unit norm.'''
        self.tensors[self.center] /= self.norm()

    def expectation(self, operator) -> complex:
        '''
        The expectation value of an operator in the state.
        Args:
          operator: the matrix product operator over the same physical leg, whose virtual bonds
            meet the bonds of the state.
        Returns:
          The value, complex in general.
        '''
        left = _identity(self.cat, self.tensors[0].codomain[0], operator.tensors[0].codomain[0])
        for at, tensor in enumerate(self.tensors):
            left = einsum('AIe,aPb,Feq,IPqj->BJf', left, tensor, tensor.conj(),
                          operator.tensors[at])
        return inner(
            _identity(self.cat, self.tensors[-1].domain[0], operator.tensors[-1].domain[1]), left)

    def copy(self) -> 'MPS':
        '''A copy of the state, at the same centre.'''
        return MPS([tensor.clone() for tensor in self.tensors], self.physical, self.center)

    def __len__(self) -> int:
        '''The number of sites.'''
        return self.L

    def __repr__(self) -> str:
        '''The shape of the state.'''
        return (f'MPS(L={self.L}, blocks={self.physical.nblocks()}, bond_dim={self.bond_dims}, '
                f'center={self.center})')
