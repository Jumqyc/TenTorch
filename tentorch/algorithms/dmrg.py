from typing import Literal, Optional

import torch

from tentorch._ext import Tensor, contract, einsum, inner
from tentorch.linalg.graded import svd
from tentorch.linalg.lanczos import lanczos
from tentorch.models.heisenberg import heisenberg
from tentorch.models.model import Model
from tentorch.networks.mps import MPS, _identity


class DMRG:
    '''
    The two site density matrix renormalisation group of a chain.

    Args:
      model: the model, whose chain and couplings are read once.
      max_bond_dim: the multiplets kept on every bond.
      svd_tol: the singular values at or below it are dropped.
      n_iter: the Krylov steps of every window.
      seed: the seed of the starting state.
    '''

    def __init__(self, model: Model, max_bond_dim: int = 50, svd_tol: float = 1e-14,
                 n_iter: int = 5, seed: int = 0) -> None:
        self.model = model
        self.max_bond_dim = max_bond_dim
        self.svd_tol = svd_tol
        self.n_iter = n_iter
        self.operator = heisenberg(model.cat, model.length, model.coupling)
        physical = model.physical if model.physical is not None else self.operator.physical
        self.state = MPS.random(model.cat, physical, model.length, model.sector, seed = seed)
        self.left: dict[int, Tensor] = {}
        self.right: dict[int, Tensor] = {}

    @property
    def L(self) -> int:
        '''The number of sites.'''
        return self.model.length

    @property
    def center(self) -> int:
        '''The site that carries the orthogonality centre, which is the centre of the state.'''
        return self.state.center

    @center.setter
    def center(self, at: int) -> None:
        '''Move the orthogonality centre onto a site of the state.'''
        self.state.center = at

    def _advance_left(self, left: Tensor, at: int) -> Tensor:
        '''
        The environment to the right of a site, from the one to its left.

        Args:
          left: the environment of the cut `at`, over the codomain `(bond, virtual)` and the domain
            `(bond,)`.
          at: the site left out between the two cuts.
        Returns:
          The environment of the cut `at + 1`, over the codomain `(bra bond, virtual)` and the
          domain `(ket bond,)`.
        '''
        tensor = self.state.tensors[at]
        return einsum('EIa,Feq,aPb,IqPj->FJb', left, tensor.conj(), tensor,
                      self.operator.tensors[at])

    def _advance_right(self, right: Tensor, at: int) -> Tensor:
        '''
        The environment to the left of a site, from the one to its right.

        Args:
          right: the environment of the cut `at + 1`.
          at: the site left out between the two cuts.
        Returns:
          The environment of the cut `at`, over the codomain `(bra bond, virtual)` and the domain
          `(ket bond,)`, so that it meets the bra and the ket of the next site on their own sides.
        '''
        tensor = self.state.tensors[at]
        return einsum('Gfq,aPb,IqPk,Gkb->FIa', tensor.conj(), tensor,
                      self.operator.tensors[at], right)

    def cache_environments(self) -> None:
        '''Rebuild both environment sets from the current state.'''
        self.left = {
            0: _identity(self.model.cat, self.state.tensors[0].codomain[0],
                         self.operator.tensors[0].codomain[0])
        }
        self.right = {
            self.L: _identity(self.model.cat, self.state.tensors[-1].domain[0],
                              self.operator.tensors[-1].domain[1])
        }
        for at in range(self.L - 1, 1, -1):
            self.right[at] = self._advance_right(self.right[at + 1], at)

    def update(self, direction: Literal['left', 'right']) -> tuple[float, float]:
        '''
        One window: the Krylov ground state of the two sites, then the split that truncates it.

        Args:
          direction: the side the sweep carries the orthogonality centre to.
        Returns:
          The energy of the window, and the weight the split threw away.
        '''
        at = self.center
        first, second = self.state.tensors[at], self.state.tensors[at + 1]
        # the side boundary of the window sits between the two sites, so the contraction that joins
        # them already lands on the split the singular value decomposition makes below
        two = contract(first, second.repartition([0]), [0], [0])
        site, following = self.operator.tensors[at], self.operator.tensors[at + 1]
        # The two halves of the effective Hamiltonian do not depend on the vector, so they are taken
        # once per window instead of once per Krylov step.
        head = einsum('EIa,IPqj->EPaqj', self.left[at], site)
        tail = einsum('JRsk,Fkb->JRSFb', following, self.right[at + 2])

        def action(vector: Tensor) -> Tensor:
            '''The Hamiltonian of the window applied to a two site tensor.'''
            return einsum('EPaqj,JRSFb,AQsb->EPrf', head, tail, vector)

        energy, vector = lanczos(action, two, self.n_iter)
        # the singular values go into the factor the sweep carries, so that the site left behind is
        # isometric and the two site tensor is the centre of the chain
        factor = 2 if direction == 'right' else 1
        left, values, right = svd(vector, self.max_bond_dim, self.svd_tol, factor)
        self.state.tensors[at], self.state.tensors[at + 1] = left, right.repartition([0, 1])
        self.center = at + 1 if direction == 'right' else at
        # the block entries carry the tree basis, not the state norm, so the centre is gauged back
        self.state.normalize()
        return energy, 1.0 - float(torch.sum(values * values))

    def sweep(self) -> tuple[float, float]:
        '''
        One sweep: a right pass that grows the left environments, then a left pass that grows the
        right ones.

        Returns:
          The energy of the last window, which is the energy of the state the sweep leaves, and the
          largest weight the sweep threw away.
        '''
        self.state.move_center(0)
        self.cache_environments()
        energy, discarded = 0.0, 0.0

        for at in range(self.L - 1):
            energy, lost = self.update('right')
            discarded = max(discarded, lost)
            if at < self.L - 2:
                self.left[at + 1] = self._advance_left(self.left[at], at)

        # the last window leaves the singular values on the right factor, which the left pass takes
        # back as the isometric site at the end of the chain
        self.state._right_canonicalize(self.L - 1)
        self.right[self.L - 1] = self._advance_right(self.right[self.L], self.L - 1)

        for at in range(self.L - 3, -1, -1):
            self.center = at
            energy, lost = self.update('left')
            discarded = max(discarded, lost)
            if at > 0:
                self.right[at + 1] = self._advance_right(self.right[at + 2], at + 1)
        return energy, discarded

    def energy(self) -> complex:
        '''
        The expectation of the Hamiltonian in the state the last sweep left, measured on the window
        the orthogonality centre sits on.
        Returns:
          The energy, a complex scalar whose imaginary part is round-off.
        Raises:
          RuntimeError: when no sweep has built the environments yet.
        '''
        if not self.left:
            raise RuntimeError('the energy needs the environments of a sweep, and none has run')
        at = min(self.center, self.L - 2)
        two = contract(self.state.tensors[at], self.state.tensors[at + 1].repartition([0]), [0], [0])
        image = einsum('AIe,IPqj,JRsk,Dkf,EQSf->APrd', self.left[at], self.operator.tensors[at],
                       self.operator.tensors[at + 1], self.right[at + 2], two)
        return complex(inner(two, image) / inner(two, two))

    def run(self, num_sweeps: int = 5, energy_tol: float = 1e-10, patience: int = 2,
            verbose: bool = True) -> Model:
        '''
        Sweep until the energy stops moving, then write the state and the energy onto the model.

        Args:
          num_sweeps: the most sweeps to run.
          energy_tol: the change of energy below which a sweep counts as converged.
          patience: the number of converged sweeps to wait for.
          verbose: whether to report one line per sweep.
        Returns:
          The model, with its `state` and its `energy` filled in.
        Raises:
          ValueError: when the run would take no sweep.
        '''
        if num_sweeps < 1:
            raise ValueError(f'a run needs at least one sweep, not {num_sweeps}')
        energy, best, stale = float('inf'), float('inf'), 0
        for sweep in range(1, num_sweeps + 1):
            energy, discarded = self.sweep()
            if verbose:
                print(f'sweep {sweep}: energy = {energy:.12f}, truncated = {discarded:.2e}, '
                      f'bond = {self.state.bond_dims}')
            if abs(energy - best) <= energy_tol:
                stale += 1
                if stale >= patience:
                    break
            else:
                stale, best = 0, energy
        self.model.state = self.state
        self.model.energy = energy
        return self.model

    def __repr__(self) -> str:
        '''One line naming the chain and the bond.'''
        return (f'DMRG(L={self.L}, max_bond_dim={self.max_bond_dim}, '
                f'bond_dim={self.state.bond_dims})')


def dmrg(model: Model, max_bond_dim: int = 50, svd_tol: float = 1e-14, num_sweeps: int = 5,
         energy_tol: float = 1e-10, patience: int = 2, n_iter: int = 5, seed: int = 0,
         verbose: bool = True) -> Model:
    '''
    The ground state of a model, by the two site density matrix renormalisation group.

    Args:
      model: the model.
      max_bond_dim: the multiplets kept on every bond.
      svd_tol: the singular values at or below it are dropped.
      num_sweeps: the most sweeps to run.
      energy_tol: the change of energy below which a sweep counts as converged.
      patience: the number of converged sweeps to wait for.
      n_iter: the Krylov steps of every window.
      seed: the seed of the starting state.
      verbose: whether to report one line per sweep.
    Returns:
      The model, with its `state` and its `energy` filled in.
    '''
    driver = DMRG(model, max_bond_dim, svd_tol, n_iter, seed)
    return driver.run(num_sweeps, energy_tol, patience, verbose)
