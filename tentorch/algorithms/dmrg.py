from __future__ import annotations

import pickle as pkl
from typing import Literal

import torch
from numpy.typing import NDArray

from tentorch.config import COMPLEX, CUDA
from tentorch import linalg
from tentorch.symmetry.cat import Group, no_symmetry, has_content
from tentorch.tensors import Tensor, bond_leg, cached_einsum, multiplet_leg
from tentorch._ext import inner

from tentorch.networks.mps import MPS
from tentorch.networks.mpo import MPO

class Broomstick:
    '''
    A 2-site DMRG engine: the Hamiltonian it minimises, the state it optimises, and the
    sweeps that do it.

    The Hamiltonian is built through the engine and the state is filled in by the first sweep,
    so a caller never constructs an ``MPO`` or an ``MPS``:

        stick = Broomstick(L=8, phys_dim=2)
        for i in range(7):
            stick.add_couplings('ZZ', 1.0, (i, i + 1))
        stick.sweep(num_sweeps=4)

    A state or a Hamiltonian built elsewhere is still taken through ``mps`` and ``mpo``.
    '''
    def __init__(self,
                 mps: MPS | None = None,
                 mpo: MPO | None = None,
                 max_bond_dim: int = 50,
                 svd_tol: float = 1e-14,
                 device = None,
                 dtype = None,
                 L: int | None = None,
                 phys_dim: int | None = None,
                 mapping: dict[str, Tensor] | None = None,
                 group: Group = no_symmetry,
                 site_sector: tuple[int, ...] | None = None):
        '''
        Args:
            mps: the state to optimize; filled in later by ``initialize_state`` if None.
            mpo: the Hamiltonian; built here from ``L``, ``phys_dim`` and ``mapping`` if None.
            max_bond_dim: largest bond dimension a sweep keeps.
            svd_tol: singular values not greater than this are discarded.
            device: device of the engine; from ``mps`` or ``mpo`` if None.
            dtype: data type of the engine; from ``mps`` or ``mpo`` if None.
            L: number of sites; from ``mps`` or ``mpo`` if None.
            phys_dim: physical dimension of one site; from ``mps`` or ``mpo`` if None.
            mapping: operator names to their dense tensors, for the Hamiltonian built here.
            group: symmetry group of the state and of the Hamiltonian; ``no_symmetry`` by default.
            site_sector: the multiplet one site lives in, e.g. ``(1,)`` for a spin-1/2 site of
                SU(2) or ``(1, 0)`` for the fundamental of SU(3); a group whose multiplets are
                not told apart by their dimension needs it.
        Raises:
            ValueError: If no source gives the number of sites or the physical dimension,
                or if the sources disagree.
        '''
        if mps is not None and not isinstance(mps, MPS):
            raise TypeError(f'mps must be an MPS or None, not {type(mps).__name__}.')
        if mpo is not None and not isinstance(mpo, MPO):
            raise TypeError(f'mpo must be an MPO or None, not {type(mpo).__name__}.')
        lengths = [value for value in (
            L, None if mps is None else mps.L, None if mpo is None else mpo.L) if value is not None]
        dims = [value for value in (
            phys_dim, None if mps is None else mps.phys_dim,
            None if mpo is None else mpo.physical_dim) if value is not None]
        if not lengths or not dims:
            raise ValueError('The engine needs L and phys_dim, or an MPS, or an MPO.')
        if len(set(lengths)) != 1:
            raise ValueError(f'The site counts disagree: {lengths}.')
        if len(set(dims)) != 1:
            raise ValueError(f'The physical dimensions disagree: {dims}.')

        self.L = lengths[0]
        self.physical_dim = dims[0]
        self.dtype = dtype or (mps.dtype if mps is not None else
                               (mpo.dtype if mpo is not None else COMPLEX))
        self.device = device or (mps.device if mps is not None else
                                 (mpo.device if mpo is not None else CUDA))
        self.state: MPS | None = mps
        self.group = group
        self.site_sector = None if site_sector is None else tuple(int(entry) for entry in site_sector)
        self.Hamiltonian = mpo if mpo is not None else MPO(
            self.L, self.physical_dim, mapping=mapping, dtype=self.dtype, device=self.device,
            group=group, site_sector=self.site_sector)
        self.max_bond_dim = max_bond_dim
        self.svd_tol = svd_tol

        self.center = 0
        self.renv: dict[int, Tensor] = {}
        self.lenv: dict[int, Tensor] = {}
        self._last_energy = None  # cached from the most recent sweep

        if self.state is None:
            self.initialize_state()

    def add_couplings(self, operators: str | tuple[str, ...], coeff: complex,
                      locations: int | NDArray | tuple[int, ...]) -> None:
        '''
        Add one coupling to the Hamiltonian, as ``MPO.add_couplings`` does.

        Args:
            operators: the operator names, one per location; a single name may be given as a
                plain string, and a name of several letters is the product of those mapping
                entries on one site.
            coeff: the coefficient of the coupling.
            locations: the site indices the operators act on; one integer for a single site.
        '''
        self.Hamiltonian.add_couplings(operators, coeff, locations)
        # the built tensors, the cached environments and the last energy describe the old
        # Hamiltonian; drop them so the next sweep rebuilds
        self._last_energy = None
        self.lenv = {}
        self.renv = {}

    def build(self) -> None:
        '''Build the Hamiltonian tensors from the couplings added so far.'''
        self.Hamiltonian.build()

    def _ensure_built(self) -> None:
        '''Build the Hamiltonian and fill in the state, so a sweep needs no setup calls.'''
        if not self.Hamiltonian.tensors:
            self.build()
        if self.state is None:
            self.initialize_state()

    def initialize_state(self,
                         bond_dims: list[int] | None = None,
                         init_state: list[Tensor] | None = None) -> None:
        '''
        Fill in the state to optimize, once the Hamiltonian is built.

        A multiplet-carrying group builds its own start state inside ``MPS``; the arguments
        below apply to every other group.

        Args:
            bond_dims: bond dimensions of the state, as in ``MPS``; ``[1]``, the physical
                dimension on the inner bonds and ``[1]`` if None.
            init_state: the state tensors, as in ``MPS``; random if None.
        '''
        self.state = MPS(self.L, self.physical_dim, group=self.group,
                                 site_sector=self.site_sector, max_bond_dim=self.max_bond_dim,
                                 bond_dims=bond_dims, init_state=init_state,
                                 dtype=self.dtype, device=self.device)

    def _require_state(self) -> MPS:
        '''The state of the engine, or a ValueError if it has none yet.'''
        if self.state is None:
            raise ValueError('The engine has no state yet; call initialize_state first.')
        return self.state

    def to(self, device):
        '''
        Move the DMRG engine state and Hamiltonian to the specified device.
        '''
        self.device = device
        self.Hamiltonian.to(device)
        if self.state is not None:
            self.state.to(device)

    def copy(self):
        '''
        Create a deep copy of the Broomstick instance.
        '''
        new = Broomstick(None if self.state is None else self.state.copy(),
                          self.Hamiltonian.copy(),
                          max_bond_dim=self.max_bond_dim,
                          svd_tol=self.svd_tol,
                          device=self.device,
                          dtype=self.dtype,
                          group=self.group,
                          site_sector=self.site_sector)
        return new

    def cache_envs(self):
        state = self._require_state()
        L = self.L
        self.renv = {}
        self.lenv = {}

        # A state whose tensor legs carry a group meets the Hamiltonian at the trivial
        # charge; the ungrouped state of a dense or abelian chain meets it at the two edges
        # of the MPO bond.
        if state.tensors[0].group != no_symmetry:
            group = state.tensors[0].group
            triv = multiplet_leg(group, group.trivial_sector)
            self.lenv[0] = Tensor.ones(1, 1, 1, legs=(triv, triv.conj(), triv.conj()),
                                       qtotal=group.trivial_sector,
                                       dtype=self.dtype, device=self.device)
            self.renv[L] = Tensor.ones(1, 1, 1, legs=(triv.conj(), triv, triv),
                                       qtotal=group.trivial_sector,
                                       dtype=self.dtype, device=self.device)
        else:
            group = state.tensors[0].group
            triv = multiplet_leg(group, group.trivial_sector)
            self.lenv[0] = Tensor.ones(1, 1, 1, legs=(triv, triv.conj(), triv.conj()),
                                       qtotal=group.trivial_sector,
                                       dtype=self.dtype, device=self.device)
            self.renv[L] = Tensor.ones(1, 1, 1, legs=(triv.conj(), triv, triv),
                                       qtotal=group.trivial_sector,
                                       dtype=self.dtype, device=self.device)

        for i in range(L - 1, 1, -1):
            self._cache_renv(i)

    def sweep(self, 
              num_sweeps: int = 5, 
              compute_variance: bool = False,
              energy_tol: float = 1e-12, 
              patience: int = 3):
        '''
        Perform the Density Matrix Renormalization Group (DMRG) algorithm to find
        the ground state of the spin chain Hamiltonian.

        After each sweep the energy density, max truncation error, and current
        max bond dimension are printed.  Early stopping triggers when the
        absolute change in energy density is below *energy_tol* for *patience*
        consecutive sweeps.

        Args:
            num_sweeps: Maximum number of full DMRG sweeps (right + left passes).
            compute_variance: If True, compute <H²> - <H>² at the end.
            energy_tol: Absolute tolerance on energy density for early stopping.
            patience: Number of consecutive non-improving sweeps before stopping.
        '''
        if num_sweeps < 1:
            raise ValueError(f'A sweep needs at least one pass, not {num_sweeps}.')
        self._ensure_built()
        self.cache_envs()
        state = self._require_state()
        best_E = None
        stale = 0

        for sweep in range(num_sweeps):
            E, trunc = self.singlesweep()
            # the Lanczos eigenvalue of the last site is the variational energy of the state the
            # sweep leaves behind; the environments are rebuilt from that state for the next sweep
            self.cache_envs()
            self._last_energy = E
            E_density = E / self.L
            max_bond = max(state.bond_dim)

            print(f'Sweep {sweep + 1}/{num_sweeps}:  '
                  f'E/L = {E_density:.12f}  '
                  f'max_trunc_err = {trunc:.3e}  '
                  f'bond_dim = {state.bond_dim} (max {max_bond})')

            # ── early stopping ──
            if best_E is None or E_density < best_E - energy_tol:
                best_E = E_density
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    print(f'Early stopping: energy density unchanged '
                          f'({abs(E_density - best_E):.3e} ≤ {energy_tol:.0e}) '
                          f'for {patience} sweeps.')
                    break

        if compute_variance:
            print(f'DMRG stops at sweep {sweep + 1}/{num_sweeps}: '
                  f'energy density = {E / self.L:.12f}, '
                  f'variance = {self.compute_variance() / self.L:.3e}')
        else:
            print(f'DMRG stops at sweep {sweep + 1}/{num_sweeps}: '
                  f'energy density = {E / self.L:.12f}')


    def _cache_lenv(self,i:int):
        '''
        The left environment of the window that starts at site ``i``, grown from the one of site
        ``i-1``: it meets the bra, the ket and the Hamiltonian tensor there.
        Args:
            i: the site the new environment belongs to, at least 1.
        '''
        state = self._require_state()
        self.lenv[i] = cached_einsum('aic,spa,cqd,ijpq->sjd', self.lenv[i-1],
                                     state.tensors[i-1].conj(), state.tensors[i-1],
                                     self.Hamiltonian.tensors[i-1])

    def _cache_renv(self,i:int):
        '''
        The right environment of the window that ends at site ``i``, grown from the one of site
        ``i+1``: it meets the bra, the ket and the Hamiltonian tensor there.
        Args:
            i: the site the new environment belongs to, at most ``L-2``.
        '''
        state = self._require_state()
        self.renv[i] = cached_einsum('spa,cqd,sjd,ijpq->aic', state.tensors[i].conj(),
                                     state.tensors[i], self.renv[i + 1],
                                     self.Hamiltonian.tensors[i])

    def singlesweep(self):
        '''
        Perform a single sweep of the DMRG algorithm, optimizing the MPS tensors
        at each site.  Returns (energy, max_truncation_error).

        The truncation error is a plain float and the energy is synced to the
        CPU once at the end of the sweep.

        Environment tensors are updated incrementally without QR
        canonicalization: the SVD inside update() already guarantees that the
        MPS tensors are left- / right-isometric as the sweep progresses.

        The sweep order is right (0 → L-1) then left (L-2 → 0).  The center
        site index is updated accordingly, and the MPS is left in a canonical
        form at the end of the sweep.
        '''
        self._ensure_built()
        if not self.lenv or not self.renv:
            self.cache_envs()
        state = self._require_state()
        max_trunc = 0.0

        # ── right sweep ──
        for i in range(self.L - 1):
            E, trunc_err = self.update(direction='right')
            max_trunc = max(max_trunc, trunc_err)
            if i < self.L - 2:
                self._cache_lenv(i + 1)
            self.center = i + 1

        # the last site leaves the right sweep as S Vh, so it is right-canonicalized here
        state._right_canonicalize_site(self.L - 1)
        self.center = state.center
        self._cache_renv(self.L - 1)
        for i in range(self.L - 3, -1, -1):
            self.center = i
            state.center = i
            E, trunc_err = self.update(direction='left')
            max_trunc = max(max_trunc, trunc_err)
            if i > 0:
                self._cache_renv(i + 1)

        return float(E), max_trunc

    def update(self, direction: Literal['left', 'right'],
               n_iter: int = 8) -> tuple[float, float]:
        '''
        Perform 2-site optimization of MPS with effective Hamiltonian.
        Returns (energy, truncation_error) — both plain floats, synced once per site.
        Args:
            direction: 'left' or 'right', indicating the sweep direction.
            n_iter: Maximum number of Lanczos iterations for the effective Hamiltonian diagonalization.
        '''
        i = self.center
        mps = self._require_state()

        assert self.center == mps.center, "Center site index mismatch between Broomstick and MPS."

        # the two site tensor and the two environments, in the order the state carries its legs
        state = cached_einsum('bjx,xld->bjld', mps.tensors[i], mps.tensors[i + 1])
        lenv = cached_einsum('apb,pqij->abqij', self.lenv[i], self.Hamiltonian.tensors[i])
        renv = cached_einsum('crd,qrkl->cdqkl', self.renv[i + 2], self.Hamiltonian.tensors[i + 1])

        # the action closes the bra and the ket physical leg on each other, and the graded
        # contraction pairs them with the Frobenius-Schur phase of the physical sector
        phase = self.group.dual_sign(mps.tensors[i].legs[1].sectors[0])

        def action(v: Tensor) -> Tensor:
            return phase * cached_einsum('abqij,cdqkl,bjld->aikc', lenv, renv, v)

        E, v = linalg.lanczos(action, state, n_iter)
        # the values are folded into the factor that carries the gauge of the sweep, so the two new
        # site tensors are the factors themselves and the caller does no scaling of its own
        u, s, vh = linalg.svd(v, (0, 1), (2, 3), max_bond=self.max_bond_dim, tol=self.svd_tol,
                              absorb=2 if direction == 'right' else 1)
        trunc_err = 1.0 - float(torch.sum(s * s).real)
        mps.tensors[i], mps.tensors[i + 1] = u, vh
        mps.bond_dim[i + 1] = u.legs[2].multiplets
        mps.center = i + 1 if direction == 'right' else i
        self.center = mps.center
        # the split keeps the block entries, not the tree-basis norm of the state, so the two
        # new sites are gauged back onto the unit sphere here
        mps.normalize()
        return float(E), trunc_err

    @property
    def energy(self) -> float:
        '''
        Energy of the current MPS state from the most recent sweep, or
        computed on-the-fly from the environments if no sweep has been run.
        '''
        self._ensure_built()
        if self._last_energy is not None:
            return self._last_energy
        # before any sweep the environments are not cached yet; build them for this one read
        self.cache_envs()
        return float(self._compute_energy().real)

    def _norm2(self) -> complex:
        '''
        Squared norm of the current MPS state, contracted as a transfer matrix of the
        sites with their conjugates: the state is a ket, so every site meets its own
        conjugate at the bond and the physical leg.
        Returns:
          The norm, a scalar whose imaginary part is round-off.
        '''
        return self._require_state().norm()

    def _compute_energy(self) -> complex:
        '''
        The expectation of the Hamiltonian in the current state, measured on the centre window.

        The two-site tensor meets the environments of the current sweep, so the value is contracted
        with the same pairing as the operator the sweep optimises.
        Returns:
          The energy, a complex scalar whose imaginary part is round-off.
        '''
        state = self._require_state()
        i = min(self.center, self.L - 2)
        two = cached_einsum('bjx,xld->bjld', state.tensors[i], state.tensors[i + 1])
        lenv = cached_einsum('apb,pqij->abqij', self.lenv[i], self.Hamiltonian.tensors[i])
        renv = cached_einsum('crd,qrkl->cdqkl', self.renv[i + 2], self.Hamiltonian.tensors[i + 1])
        image = cached_einsum('abqij,cdqkl,bjld->aikc', lenv, renv, two)
        phase = self.group.dual_sign(state.tensors[i].legs[1].sectors[0])
        return complex(phase * inner(two, image) / inner(two, two))

    def __sizeof__(self):
        return ((0 if self.state is None else self.state.__sizeof__()) +
                self.Hamiltonian.__sizeof__() +
                sum(t.numel() * t.element_size() for t in self.lenv.values()) +
                sum(t.numel() * t.element_size() for t in self.renv.values()))

    def compute_variance(self) -> float:
        '''
        Compute var(H) = <H²> - <H>² for the current state.
        Contracts <ψ|H²|ψ> directly without building the H² MPO explicitly,
        keeping memory O(B²·D²) instead of O(D⁴).
        '''
        self._ensure_built()
        state = self._require_state()
        W = self.Hamiltonian.tensors
        A = state.tensors
        trivial = multiplet_leg(self.group, self.group.trivial_sector)
        lenv = Tensor.ones(1, 1, 1, 1, legs=(trivial, trivial.conj(), trivial.conj(),
                                            trivial.conj()),
                           qtotal=self.group.trivial_sector,
                           dtype=self.dtype, device=self.device)
        for i in range(self.L):
            lenv = cached_einsum('bdxy,apb,xwpr,yzrq,dqc->acwz',
                                 lenv, A[i].conj(), W[i], W[i], A[i])
        # the MPO's right edges are trivial legs of one component, so closing both leaves a scalar
        edge = Tensor.ones(1, legs=(lenv.legs[2].conj(),), dtype=self.dtype, device=self.device)
        E2 = float(cached_einsum('acwz,w,z->ac', lenv, edge, edge).blocks[0].reshape(()).real.item())

        E = state.expectation(self.Hamiltonian)
        var = float(E2) - float(E.real) ** 2
        return max(var, 0.0)  # clamp numerical noise below zero
    
    def __repr__(self) -> str:
        '''A one-line summary: the chain, the bond cap, the group and the last energy.'''
        group = self.group.name
        energy = '' if self._last_energy is None else f', E = {self._last_energy:.12f}'
        return (f'Broomstick(L = {self.L}, phys_dim = {self.physical_dim}, '
                f'max_bond_dim = {self.max_bond_dim}, group = {group}{energy})')

    def save_as_pkl(self,filename: str | None = None):
        '''
        Save the current state of the DMRG engine, including the MPS and MPO, to a pickle file.
        The filename is generated based on the current timestamp and the number of sweeps completed.
        '''
        import time

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if filename is None:
            filename = f"dmrg_state_{timestamp}.pkl"
        with open(filename, 'wb') as f:
            pkl.dump(self, f)
        
        print(f"DMRG engine state saved to {filename}")
    
    @staticmethod
    def load_from_pkl(filename: str) -> 'Broomstick':
        '''
        Load a DMRG engine state from a pickle file.
        Args:
            filename: The path to the pickle file containing the saved DMRG engine state.
        Returns:
            An instance of the Broomstick class with the loaded state.
        '''
        with open(filename, 'rb') as f:
            stick = pkl.load(f)
        return stick
