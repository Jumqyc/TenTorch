'''The two-site DMRG sweep: the engine that optimises the state of a model, and the function a
user calls to run it.'''
from dataclasses import dataclass
from typing import Literal

import torch

from tentorch import linalg
from tentorch.model import Model
from tentorch.networks.mps import MPS
from tentorch.tensors import Tensor, cached_einsum, multiplet_leg


@dataclass(frozen = True)
class DMRGResult:
    '''
    What one DMRG run leaves behind.

    Args:
        energy: the variational energy of ``state``.
        state: the optimized MPS; the observables of the run hang off it.
        trunc_err: the largest discarded weight of the two-site splits of the run.
        sweeps: the number of sweeps the run took.
    '''
    energy: float
    state: MPS
    trunc_err: float
    sweeps: int


class DMRG:
    '''
    The two-site DMRG sweep on one model.

    The engine owns the working state, the environments and the bond cap; the chain and the
    Hamiltonian come from the model, so the engine defines no physics of its own.

    Args:
        model: the chain and the Hamiltonian written on it.
        max_bond_dim: largest bond dimension a sweep keeps.
        svd_tol: singular values not greater than this are discarded.
    '''
    def __init__(self,
                 model: Model,
                 max_bond_dim: int = 50,
                 svd_tol: float = 1e-14):
        self.model = model
        self.hamiltonian = model.mpo
        self.L = model.L
        self.physical_dim = model.physical_dim
        self.cat = model.cat
        self.site_sector = model.sector
        self.dtype = model.dtype
        self.device = model.device
        self.max_bond_dim = max_bond_dim
        self.svd_tol = svd_tol

        self.center = 0
        self.lenv: dict[int, Tensor] = {}
        self.renv: dict[int, Tensor] = {}
        self._last_energy: float | None = None
        self.state: MPS
        self.initialize_state()

    def initialize_state(self,
                         bond_dims: list[int] | None = None,
                         init_state: list[Tensor] | None = None) -> None:
        '''
        Replace the state to optimize.

        A multiplet-carrying category builds its own start state inside ``MPS``; the arguments
        below apply to every other category.

        Args:
            bond_dims: bond dimensions of the state, as in ``MPS``; ``[1]``, the physical
                dimension on the inner bonds and ``[1]`` if None.
            init_state: the state tensors, as in ``MPS``; random if None.
        '''
        self.state = MPS(self.L, self.physical_dim, cat=self.cat,
                         site_sector=self.site_sector, max_bond_dim=self.max_bond_dim,
                         bond_dims=bond_dims, init_state=init_state,
                         dtype=self.dtype, device=self.device)
        self.center = 0
        self.lenv = {}
        self.renv = {}
        self._last_energy = None

    def run(self,
            num_sweeps: int = 5,
            energy_tol: float = 1e-12,
            patience: int = 3,
            n_iter: int = 8,
            verbose: bool = True) -> DMRGResult:
        '''
        Sweep until the energy density stops changing, or until ``num_sweeps`` passes are done.

        Args:
            num_sweeps: maximum number of full sweeps (a right pass and a left pass each).
            energy_tol: absolute tolerance on the energy density for early stopping.
            patience: number of consecutive non-improving sweeps before stopping.
            n_iter: maximum number of Lanczos iterations per window.
            verbose: print one line per sweep.
        Returns:
            The energy, the state and the largest discarded weight of the run.
        Raises:
            ValueError: If ``num_sweeps`` is smaller than one.
        '''
        if num_sweeps < 1:
            raise ValueError(f'A sweep needs at least one pass, not {num_sweeps}.')
        self.cache_envs()
        best_E = None
        stale = 0
        E = 0.0
        trunc_err = 0.0

        for sweep in range(num_sweeps):
            E, trunc_err = self.singlesweep(n_iter)
            # the Lanczos eigenvalue of the last window is the variational energy of the state the
            # sweep leaves behind, so the environments are rebuilt from that state for the next one
            self.cache_envs()
            self._last_energy = E
            E_density = E / self.L

            if verbose:
                print(f'Sweep {sweep + 1}/{num_sweeps}:  '
                      f'E/L = {E_density:.12f}  '
                      f'max_trunc_err = {trunc_err:.3e}  '
                      f'bond_dim = {self.state.bond_dim} (max {max(self.state.bond_dim)})')

            if best_E is None or E_density < best_E - energy_tol:
                best_E = E_density
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    if verbose:
                        print(f'Early stopping: energy density unchanged '
                              f'({abs(E_density - best_E):.3e} <= {energy_tol:.0e}) '
                              f'for {patience} sweeps.')
                    break

        return DMRGResult(energy=float(E), state=self.state, trunc_err=float(trunc_err),
                          sweeps=sweep + 1)

    def cache_envs(self) -> None:
        '''Rebuild both environment sets from the current state.'''
        L = self.L
        trivial = multiplet_leg(self.cat, self.cat.trivial_sector)
        self.lenv = {0: Tensor.ones(1, 1, 1, legs=(trivial, trivial.conj(), trivial.conj()),
                                    qtotal=self.cat.trivial_sector,
                                    dtype=self.dtype, device=self.device)}
        self.renv = {L: Tensor.ones(1, 1, 1, legs=(trivial.conj(), trivial, trivial),
                                    qtotal=self.cat.trivial_sector,
                                    dtype=self.dtype, device=self.device)}
        for i in range(L - 1, 1, -1):
            self._cache_renv(i)

    def _cache_lenv(self, i: int) -> None:
        '''
        The left environment of the window that starts at site ``i``, grown from the one of site
        ``i-1``: it meets the bra, the ket and the Hamiltonian tensor there.
        Args:
            i: the site the new environment belongs to, at least 1.
        '''
        state = self.state
        self.lenv[i] = cached_einsum('aic,spa,cqd,ijpq->sjd', self.lenv[i - 1],
                                     state.tensors[i - 1].conj(), state.tensors[i - 1],
                                     self.hamiltonian.tensors[i - 1])

    def _cache_renv(self, i: int) -> None:
        '''
        The right environment of the window that ends at site ``i``, grown from the one of site
        ``i+1``: it meets the bra, the ket and the Hamiltonian tensor there.
        Args:
            i: the site the new environment belongs to, at most ``L-2``.
        '''
        state = self.state
        self.renv[i] = cached_einsum('spa,cqd,sjd,ijpq->aic', state.tensors[i].conj(),
                                     state.tensors[i], self.renv[i + 1],
                                     self.hamiltonian.tensors[i])

    def singlesweep(self, n_iter: int = 8) -> tuple[float, float]:
        '''
        One right pass and one left pass over the chain, window by window.

        The environments are updated incrementally, without a QR canonicalization: the SVD inside
        ``update`` already leaves the tensors isometric as the sweep progresses.

        Args:
            n_iter: maximum number of Lanczos iterations per window.
        Returns:
            The variational energy and the largest discarded weight of the pass.
        '''
        state = self.state
        max_trunc = 0.0
        E = 0.0

        for i in range(self.L - 1):
            E, trunc_err = self.update(direction = 'right', n_iter = n_iter)
            max_trunc = max(max_trunc, trunc_err)
            if i < self.L - 2:
                self._cache_lenv(i + 1)

        # the last window of the right pass leaves its right factor as S Vh, which is the
        # right-canonical tensor at the end of the chain
        state._right_canonicalize_site(self.L - 1)
        self.center = state.center
        self._cache_renv(self.L - 1)
        for i in range(self.L - 3, -1, -1):
            self.center = i
            state.center = i
            E, trunc_err = self.update(direction = 'left', n_iter = n_iter)
            max_trunc = max(max_trunc, trunc_err)
            if i > 0:
                self._cache_renv(i + 1)

        return float(E), max_trunc

    def update(self,
               direction: Literal['left', 'right'],
               n_iter: int = 8) -> tuple[float, float]:
        '''
        Optimize the two-site window at the current centre with a Lanczos step, then split it.

        Args:
            direction: the side the sweep carries the orthogonality centre to.
            n_iter: maximum number of Lanczos iterations.
        Returns:
            The variational energy and the discarded weight of the split.
        '''
        i = self.center
        mps = self.state
        assert self.center == mps.center, 'Center site index mismatch between DMRG and MPS.'

        # the two site tensor and the two environments, in the order the state carries its legs
        two = cached_einsum('bjx,xld->bjld', mps.tensors[i], mps.tensors[i + 1])
        lenv = cached_einsum('apb,pqij->abqij', self.lenv[i], self.hamiltonian.tensors[i])
        renv = cached_einsum('crd,qrkl->cdqkl', self.renv[i + 2], self.hamiltonian.tensors[i + 1])

        # the action closes the bra and the ket physical leg on each other, and the graded
        # contraction pairs them with the Frobenius-Schur phase of the physical sector
        phase = 1.0

        def action(v: Tensor) -> Tensor:
            return phase * cached_einsum('abqij,cdqkl,bjld->aikc', lenv, renv, v)

        E, v = linalg.lanczos(action, two, n_iter)
        # the singular values are folded into the factor that carries the gauge of the sweep, so
        # the two new site tensors are the factors themselves and the caller scales nothing
        u, s, vh = linalg.svd(v, (0, 1), (2, 3), max_bond = self.max_bond_dim, tol = self.svd_tol,
                              absorb = 2 if direction == 'right' else 1)
        trunc_err = 1.0 - float(torch.sum(s * s).real)
        mps.tensors[i], mps.tensors[i + 1] = u, vh
        mps.bond_dim[i + 1] = u.legs[2].multiplets
        mps.center = i + 1 if direction == 'right' else i
        self.center = mps.center
        # the split keeps the block entries, not the tree-basis norm of the state, so the two new
        # sites are gauged back onto the unit sphere here
        mps.normalize()
        return float(E), trunc_err

    @property
    def energy(self) -> float:
        '''The energy of the current state, from the last sweep or measured on the centre window.'''
        if self._last_energy is not None:
            return self._last_energy
        if not self.lenv or not self.renv:
            self.cache_envs()
        return float(self._compute_energy().real)

    def _compute_energy(self) -> complex:
        '''
        The expectation of the Hamiltonian in the current state, measured on the centre window.

        The two-site tensor meets the environments of the current sweep, so the value is
        contracted with the same pairing as the operator the sweep optimises.
        Returns:
            The energy, a complex scalar whose imaginary part is round-off.
        '''
        state = self.state
        i = min(self.center, self.L - 2)
        two = cached_einsum('bjx,xld->bjld', state.tensors[i], state.tensors[i + 1])
        lenv = cached_einsum('apb,pqij->abqij', self.lenv[i], self.hamiltonian.tensors[i])
        renv = cached_einsum('crd,qrkl->cdqkl', self.renv[i + 2], self.hamiltonian.tensors[i + 1])
        image = cached_einsum('abqij,cdqkl,bjld->aikc', lenv, renv, two)
        phase = 1.0
        return complex(phase * linalg.inner(two, image) / linalg.inner(two, two))

    def compute_variance(self) -> float:
        r'''
        :math:`\langle H^2 \rangle - \langle H \rangle^2` in the current state.

        It contracts :math:`\langle \psi | H^2 | \psi \rangle` directly, without an :math:`H^2`
        MPO, so the cost stays :math:`O(B^2 D^2)` rather than :math:`O(D^4)`.
        Returns:
            The variance, clamped at zero against round-off.
        '''
        state = self.state
        W = self.hamiltonian.tensors
        A = state.tensors
        trivial = multiplet_leg(self.cat, self.cat.trivial_sector)
        lenv = Tensor.ones(1, 1, 1, 1, legs=(trivial, trivial.conj(), trivial.conj(),
                                            trivial.conj()),
                           qtotal=self.cat.trivial_sector,
                           dtype=self.dtype, device=self.device)
        for i in range(self.L):
            lenv = cached_einsum('bdxy,apb,xwpr,yzrq,dqc->acwz',
                                 lenv, A[i].conj(), W[i], W[i], A[i])
        # the two right edges of the MPO are trivial legs of one component, so closing both
        # leaves a scalar
        edge = Tensor.ones(1, legs=(lenv.legs[2].conj(),), dtype=self.dtype, device=self.device)
        E2 = float(next(iter(cached_einsum('acwz,w,z->ac', lenv, edge, edge).blocks.values())).reshape(()).real.item())
        E = float(state.expectation(self.hamiltonian).real)
        return max(E2 - E * E, 0.0)

    def __repr__(self) -> str:
        '''The chain, the bond cap, the category and the last energy.'''
        energy = '' if self._last_energy is None else f', E = {self._last_energy:.12f}'
        return (f'DMRG(L = {self.L}, phys_dim = {self.physical_dim}, '
                f'max_bond_dim = {self.max_bond_dim}, cat = {self.cat.name}{energy})')


def dmrg(model: Model,
         max_bond_dim: int = 50,
         svd_tol: float = 1e-14,
         num_sweeps: int = 5,
         energy_tol: float = 1e-12,
         patience: int = 3,
         n_iter: int = 8,
         verbose: bool = True) -> DMRGResult:
    '''
    Find the ground state of ``model`` with the two-site DMRG sweep.

    Args:
        model: the chain and the Hamiltonian written on it.
        max_bond_dim: largest bond dimension a sweep keeps.
        svd_tol: singular values not greater than this are discarded.
        num_sweeps: maximum number of full sweeps.
        energy_tol: absolute tolerance on the energy density for early stopping.
        patience: number of consecutive non-improving sweeps before stopping.
        n_iter: maximum number of Lanczos iterations per window.
        verbose: print one line per sweep.
    Returns:
        The energy, the state and the largest discarded weight of the run.
    '''
    engine = DMRG(model, max_bond_dim = max_bond_dim, svd_tol = svd_tol)
    return engine.run(num_sweeps = num_sweeps, energy_tol = energy_tol, patience = patience,
                      n_iter = n_iter, verbose = verbose)
