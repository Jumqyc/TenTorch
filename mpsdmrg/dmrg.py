import torch
import numpy as np
import pickle as pkl


from math import sqrt
from typing import Literal
from numpy.typing import NDArray

from base import COMPLEX, CUDA
# ``_expr_cache`` is re-exported by name so a profiling script can clear it.
from mpsdmrg.symmetry import (Group, Leg, Sector, Tensor, _expr_cache, cached_einsum, inner,
                              no_grad, no_symmetry, norm, qr, rq, svd, svdvals, trace)
from mpsdmrg.symmetry.hamiltonian import build_mpo
from mpsdmrg.symmetry.invariant import Bond, bond_channels
from mpsdmrg.symmetry.state import initial_state


from mpsdmrg.operators import SpinOperator


class MPS:
    '''
    A class representing a Matrix Product State (MPS) in quantum physics and tensor networks.
    '''
    def __init__(self,
                 L: int,
                 phys_dim:int,
                 bond_dims:list[int]|None = None,
                 init_state:list[Tensor]|None = None,
                 dtype = COMPLEX,
                 device = CUDA):
        '''
        Args:
            L (int): Length of the MPS (number of sites).
            phys_dim (int): Physical dimension of each site.
            bond_dims (list[int] | None): List of bond dimensions for each bond in the MPS. Including the left and right virtual bonds, the length of this list should be L+1. If None, defaults to [1] + [phys_dim] * (L - 1) + [1].
            init_state (list[Tensor] | None): Optional initial state for the MPS. If None, random tensors will be generated.
            dtype: data type of the MPS tensors (default: COMPLEX).
            device: device that stores the MPS tensors (default: CUDA).
        Raises:
            ValueError: If the length of bond_dims does not match L+1.
        '''
        if bond_dims is None:
            bond_dims = [1] + [phys_dim] * (L - 1) + [1]
        self.L = L
        self.phys_dim = phys_dim
        self.bond_dim = bond_dims
        if len(bond_dims) != L + 1:
            raise ValueError(f"Bond dimension list length {len(bond_dims)} does not match the expected length of {L + 1}.")
        self.dtype = dtype
        self.device = device
        self.center = 0  # Center site index for canonical form

        # Initialize MPS tensors (left_bond_dim, phys_dim, right_bond_dim)
        if init_state is None:
            self.tensors = [
            Tensor.rand((bond_dims[i], phys_dim, bond_dims[i + 1]),
                        dtype=dtype,
                        device=device) for i in range(L)
        ]
        else:
            self.tensors = init_state
            if any(t.shape != (bond_dims[i], phys_dim, bond_dims[i + 1]) for i, t in enumerate(init_state)):
                raise ValueError("Initial state tensors do not match the specified bond and physical dimensions.")

        self.canonicalize()  # Ensure the MPS is in canonical form upon initialization
        self.normalize()  # Normalize the MPS upon initialization

    def __getitem__(self, idx):
        return self.tensors[idx]
    def to(self, device):
        '''
        Move the MPS tensors to the specified device (e.g., 'cuda' or 'cpu').
        '''
        self.device = device
        for i in range(self.L):
            self.tensors[i] = self.tensors[i].to(device)

    def move_center_to(self, new_center:int):
        '''
        Move the center site index to a new position in the MPS.
        Args:
            new_center (int): The new center site index (0 <= new_center < L).
        Raises:
            IndexError: If the new center index is out of bounds for the MPS length.
            '''
        if new_center == self.center:
            return  # No need to move if the new center is the same as the current center
        if new_center < 0 or new_center >= self.L:
            raise IndexError('New center index is out of bounds for the MPS length.')
        if new_center < self.center:
            for i in range(self.center, new_center, -1):  # move to the left
                self._right_canonicalize_site(i, update_center=True)
        elif new_center > self.center:
            for i in range(self.center, new_center):      # move to the right
                self._left_canonicalize_site(i, update_center=True)

    def expectation(self,operators, mode:Literal['direct','env'] = 'env')-> complex:
        '''
        Compute the expectation value of operators on the MPS.
        Accepts either a dict[int, Tensor] of single-site operators or an MPO.

        To compute the norm, just pass {} as the operator list.
        Args:
            operators: dict[int, Tensor] | MPO: A dictionary of operators to compute the expectation value,
                       or an MPO representing a full matrix product operator.
            mode: 'direct' or 'env' — contraction strategy (for dict operators only; MPO always uses direct).
        Returns:
            complex: The expectation value of the operators on the MPS.
        Raises:
            ValueError: If the operator shapes do not match the physical dimension of the MPS tensors.
        '''
        if isinstance(operators, MPO):
            return self._mpo_expectation(operators)
        # Otherwise treat as dict[int, Tensor]
        if any(op.shape != (self.phys_dim, self.phys_dim) for op in operators.values()):
            raise ValueError(f"Operator shapes do not match the physical dimension {self.phys_dim}.")
        elif any(idx < 0 or idx >= self.L for idx in operators.keys()):
            raise ValueError(f"Operator indices must be in the range [0, {self.L - 1}].")
        if mode == 'direct':
            return self._full_dir_contract(operators) # uses brute-force contraction of the MPS with the given operators to compute the expectation value.
        elif mode == 'env':
            return self._full_env_contract(operators) # uses canonical condition.

    def norm(self,mode:Literal['direct','env'] = 'env')->complex:
        if mode == 'env':
            return cached_einsum('abc,abc->', self.tensors[self.center].conj(), self.tensors[self.center]).item()
        elif mode == 'direct':
            return self.expectation({}, mode='direct')

    def normalize(self):
        '''
        Normalize the MPS by dividing each tensor by the square root of the norm.
        '''
        norm:complex = self.norm(mode='env')
        self.tensors[self.center] /= sqrt(norm.real)

    @no_grad()
    def _mpo_expectation(self, mpo) -> complex:
        '''
        Compute <psi|O|psi> by contracting MPS* @ MPO @ MPS from left to right.
        The MPO tensors follow the convention: tensors[0] = left boundary (carries site-0 ops),
        tensors[1..L-1] = site 1..L-1 operators, tensors[L] = right boundary.
        '''
        # Site 0: use MPO tensors[0] (left boundary + site-0 operators)
        lenv = cached_einsum('bpa,dqc,uwpq->acw',
                            self.tensors[0].conj(), self.tensors[0], mpo.tensors[0])
        for i in range(1, self.L):
            lenv = cached_einsum('bdw,bpa,dqc,wxpq->acx',
                                lenv, self.tensors[i].conj(), self.tensors[i], mpo.tensors[i])
        # lenv shape: (1, 1, D); pick the MPO end state (index -1)
        return lenv[0, 0, -1].item()

    @no_grad()
    def _full_dir_contract(self,operators:dict[int,Tensor]):
        '''
        Brute-force contraction of the MPS with the given operators to compute the expectation value.
        '''
        if 0 in operators:
            lenv = cached_einsum('aib,ajc,ij->bc', self.tensors[0].conj(), self.tensors[0],
                                 operators[0])
        else:
            lenv = cached_einsum('aib,aic->bc', self.tensors[0].conj(), self.tensors[0])
        for i in range(1,self.L):
            if i in operators:
                lenv = cached_einsum('ab,aic,bjd,ij->cd', lenv, self.tensors[i].conj(),
                                     self.tensors[i], operators[i])
            else:
                lenv = cached_einsum('ab,aic,bid->cd', lenv, self.tensors[i].conj(),
                                     self.tensors[i])
        return trace(lenv).item()


    @no_grad()
    def _full_env_contract(self, operators: dict[int, Tensor]):
        '''
        Compute the expectation value of a list of operators on the MPS using the canonical form.
        '''
        if not operators: # If the operator list is empty, return the norm of the MPS.
            return self.norm(mode='env')

        leftmost = min(operators.keys())
        rightmost = max(operators.keys())

        self.move_center_to(leftmost)
        lenv = cached_einsum('aic,ajd,ij->cd',
                                    self.tensors[leftmost].conj(),
                                    self.tensors[leftmost],
                                    operators[leftmost])
        for i in range(leftmost + 1, rightmost + 1):
            if i in operators:
                lenv = cached_einsum('ab,aic,bjd,ij->cd', lenv,
                                    self.tensors[i].conj(), self.tensors[i], operators[i])
            else:
                lenv = cached_einsum('ab,aic,bid->cd', lenv,
                                    self.tensors[i].conj(), self.tensors[i])
        return trace(lenv).item()

    @no_grad()
    def entanglement_entropy(self,bond:int,alpha:float|None = None, svd_tol:float = 1e-14)-> float:
        '''
        Compute the entanglement entropy of ith and i+1th site.
        Args:
            bond (int): The bond index between sites i and i+1 (0 <= bond < L-1).
            alpha (float | None): The Renyi entropy order. If None, the von Neumann entropy is computed. (default: None)
            svd_tol (float): Tolerance for singular value truncation. (default: 1e-14)
        '''
        if bond < 0 or bond >= self.L - 1:
            raise IndexError(f"Bond index {bond} is out of bounds for MPS of length {self.L}.")
        self.move_center_to(bond)
        s = svdvals(self.tensors[bond], (0, 1), (2,))
        s = s[s > svd_tol]
        if alpha is None:
            return -torch.sum(s**2 * torch.log(s**2)).item()
        else:
            return 1/(1-alpha) * torch.log(torch.sum(s**(2*alpha))).item()


    @no_grad()
    def _left_canonicalize_site(self, i, update_center=True):
        '''
        Canonicalize the MPS site at index i to the left canonical form using QR decomposition. Optionally update the center site index.
                |          |           |
        --i-1--A[i-1]--i-- A[i]-i+1- A[i+1]-i+2-
        =>
                |        |           |
        -i-1--A[i-1]--i--Q---R-i+1- A[i+1]-i+2-
        Args:
            i (int): Index of the site to canonicalize (0 <= i < L-1).
            update_center (bool): Whether to update the center site index after canonicalization (default: True).
        Raises:
            IndexError: If the index i is out of bounds for the MPS length.
        '''
        if i < 0 or i >= self.L-1:
            raise IndexError(f"Index {i} is out of bounds for MPS of length {self.L}.")
        q, r = qr(self.tensors[i], (0, 1), (2,))
        self.bond_dim[i+1] = q.shape[-1]
        self.tensors[i] = q
        self.tensors[i+1] = cached_einsum('ab,bcd->acd', r, self.tensors[i+1])
        if update_center:
            self.center = i + 1

    @no_grad()
    def _right_canonicalize_site(self, i, update_center=True):
        '''
        Canonicalize the MPS site at index i to the right canonical form using RQ decomposition. Optionally update the center site index.
                |          |           |
        --i-1--A[i-1]--i-- A[i]-i+1- A[i+1]-i+2-
        =>
                |              |          |
        -i-1--A[i-1]--i--R.T---Q.T-i+1- A[i+1]-i+2-
        Args:
            i (int): Index of the site to canonicalize (1 <= i < L).
            update_center (bool): Whether to update the center site index after canonicalization (default: True).
        Raises:
            IndexError: If the index i is out of bounds for the MPS length.
        '''
        if i <= 0 or i >= self.L:
            raise IndexError(f"Index {i} is out of bounds for MPS of length {self.L}.")
        r, q = rq(self.tensors[i], (0,), (1, 2))
        self.bond_dim[i] = q.shape[0]
        self.tensors[i] = q
        self.tensors[i-1] = cached_einsum('lpb,br->lpr', self.tensors[i-1], r)
        if update_center:
            self.center = i - 1

    def canonicalize(self):
        '''
        Canonicalize the MPS in the specified mode.
        '''
        for i in range(self.center): # to center -1
            self._left_canonicalize_site(i, update_center=True)

        for i in range(self.L - 1, self.center, -1):   # i = L-1, L-2,...,self.center+1
            self._right_canonicalize_site(i, update_center=True)
    def copy(self):
        '''
        Create a deep copy of the MPS instance.
        '''
        return MPS(self.L,
                      self.phys_dim,
                      self.bond_dim.copy(),
                      [t.clone().detach() for t in self.tensors],
                      self.dtype,
                      self.device)
    def __repr__(self):
        return f"MPS(L={self.L}, phys_dim={self.phys_dim}, bond_dim={self.bond_dim}, center={self.center})"
    def __sizeof__(self):
        return sum(t.numel() * t.element_size() for t in self.tensors)

class MPO:
    '''
    A class representing a Matrix Product Operator (MPO) in quantum physics and tensor networks.
    '''
    @no_grad()
    def __init__(self,
                 L:int,
                 phys_dim:int,
                 mapping:dict[str,torch.Tensor]|None = None,
                 dtype = COMPLEX,
                 device = CUDA):
        '''
        Args:
            L (int): Length of the MPO (number of sites).
            phys_dim (int): Physical dimension of the MPO.
            mapping (dict[str, torch.Tensor]): A dictionary mapping operator names to their dense torch operator matrices. If None, default mappings will be used.
        '''
        self.dtype = dtype
        self.device = device
        self.L = L
        self.group: Group | None = None  # set by the engine that owns the Hamiltonian
        self.couplings: list[tuple[tuple[str, ...], complex, tuple[int, ...]]] = []
        self.physical_dim = phys_dim
        self.tensors: list[Tensor] = []
        self.bond_dim = 0

        if mapping is None:
            spin_op = SpinOperator(phys_dim=phys_dim, dtype=dtype, device=device).data
            self.mapping = {
                'X': spin_op[0],
                'Y': spin_op[1],
                'Z': spin_op[2],
                'I': torch.eye(phys_dim, dtype=dtype, device=device)
            }
        else:
            self.mapping = mapping

    def to(self, device):
        '''
        Move the MPO tensors to the specified device (e.g., 'cuda' or 'cpu').
        '''
        self.device = device
        for i in range(len(self.tensors)):
            self.tensors[i] = self.tensors[i].to(device)
        for k in self.mapping:
            self.mapping[k] = self.mapping[k].to(device)

    def copy(self):
        '''
        Create a deep copy of the MPO instance.
        '''
        new = MPO(self.L,
                  self.physical_dim,
                  mapping={k: v.clone().detach() for k, v in self.mapping.items()},
                  dtype=self.dtype,
                  device=self.device)
        new.couplings = [(ops, coeff, locs) for ops, coeff, locs in self.couplings]
        new.group = self.group
        if self.tensors:
            new.tensors = [t.clone().detach() for t in self.tensors]
        new.bond_dim = self.bond_dim
        return new

    def __repr__(self):
        return f"MPO(L={self.L}, physical_dim={self.physical_dim}, bond_dim={self.bond_dim})"
    # ── Arithmetic dunders ──────────────────────────────────────────────

    def __add__(self, other):
        '''
        MPO + MPO: merge couplings.   MPO + dict: add single-site operators.
        '''
        if isinstance(other, MPO):
            new = self.copy()
            # Copy any missing operator matrices from other into new.mapping
            for op_name in set(op for ops, _, _ in other.couplings for op in ops):
                new._register_merged_op(op_name)
            # Directly merge couplings (they are already merged by add_couplings)
            new.couplings.extend(other.couplings)
            new.build()
            return new
        elif isinstance(other, dict):
            new = self.copy()
            for site, op_mat in other.items():
                name = f"__dict_{len(new.mapping)}"
                new.mapping[name] = op_mat.to(dtype=new.dtype, device=new.device)
                new.add_couplings((name,), 1.0, (site,))
            new.build()
            return new
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, dict):
            return self.__add__(other)
        return NotImplemented

    def __mul__(self, other):
        '''
        MPO * scalar: scale couplings.
        MPO * MPO:   operator composition (contract physical indices).
        '''
        if isinstance(other, (int, float, complex)):
            new = self.copy()
            if new.tensors:
                # scale the tensors directly (simpler for already-built MPOs)
                for i in range(len(new.tensors)):
                    new.tensors[i] = new.tensors[i] * other
            else:
                new.couplings = [(ops, coeff * other, locs) for ops, coeff, locs in new.couplings]
                new.build()
            return new
        elif isinstance(other, MPO):
            # MPO * MPO: contract physical indices, outer-product the bond dims
            new = MPO(self.L, self.physical_dim,
                      mapping={k: v.clone().detach() for k, v in self.mapping.items()},
                      dtype=self.dtype, device=self.device)
            new.tensors = []
            for i in range(len(self.tensors)):
                t1 = self.tensors[i]
                t2 = other.tensors[i]
                # t1: (wL1, wR1, p, q), t2: (wL2, wR2, q, r)  →  (wL1*wL2, wR1*wR2, p, r)
                new_t = cached_einsum('abpq,cdqr->acbdpr', t1, t2).combine_legs(
                    ((0, 1), (2, 3), (4,), (5,)))
                new.tensors.append(new_t)
            new.bond_dim = self.bond_dim * other.bond_dim
            return new
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            return self.__mul__(other)
        return NotImplemented

    def __neg__(self):
        return self.__mul__(-1.0)

    def __sub__(self, other):
        return self.__add__((-1.0) * other)

    # ── Coupling management ─────────────────────────────────────────────

    def add_couplings(self,operators:tuple[str,...],coeff:complex,locations:NDArray[np.int64]|tuple[int,...]):
        '''
        Add new couplings to the MPO.
        Args:
            operators (tuple[str]): A tuple of operator names (e.g., ('X', 'Y')).  A name of
                several letters is the product of those mapping entries on one site, e.g.
                ``'XY'`` is :math:`S_x S_y`.
            coeffs (complex): A complex coefficient for the coupling.
            locations (tuple[int]): A tuple of site indices where the operators act.
        Raises:
            ValueError: If the length of operators does not match the number of locations,
                        or if any site index is out of bounds for the MPO length,
                        or if any operator letter is not defined in the mapping.
        '''
        if len(operators) != len(locations):
            raise ValueError(f"Length of operators {len(operators)} does not match the number of locations {len(locations)}.")
        if any(loc < 0 or loc >= self.L for loc in locations):
            raise ValueError(f"Site indices {locations} are out of bounds for MPO of length {self.L}.")
        if not all(op in self.mapping or all(letter in self.mapping for letter in op)
                   for op in operators):
            raise ValueError(f"Operators {operators} are not defined in the mapping. Available operators: {list(self.mapping.keys())}.")
        operators = tuple(self._register_merged_op(op) for op in operators)

        # Merge consecutive operators on the same site
        merged_ops = []
        merged_locs = []
        current_op = operators[0]
        current_loc = locations[0]
        for op, loc in zip(operators[1:], locations[1:]):
            if loc == current_loc:
                current_op = current_op + op
            else:
                merged_ops.append(self._register_merged_op(current_op))
                merged_locs.append(current_loc)
                current_op = op
                current_loc = loc
        merged_ops.append(self._register_merged_op(current_op))
        merged_locs.append(current_loc)
        self.couplings.append((tuple(merged_ops), coeff, tuple(merged_locs)))

    def _register_merged_op(self, op_string: str) -> str:
        '''
        Register a merged operator matrix if not already present, return its name.
        '''
        if op_string not in self.mapping:
            base = op_string.split('@')[0]  # e.g. "XY@0" → "XY"
            mat = self.mapping[base[0]]
            for ch in base[1:]:
                mat = mat @ self.mapping[ch]
            self.mapping[op_string] = mat.to(dtype=self.dtype, device=self.device)
        return op_string

    @staticmethod
    def preprocess(group: Group,
                   phys_dim: int,
                   mapping: dict[str, torch.Tensor],
                   couplings: list[tuple[tuple[str, ...], complex, tuple[int, ...]]],
                   dtype = COMPLEX,
                   device = CUDA,
                   tol: float = 1e-10
                   ) -> tuple[dict[int, torch.Tensor], dict[Sector, torch.Tensor],
                              list[Bond]] | None:
        r'''
        Reduce nearest-neighbour couplings to the invariant channels of every bond.

        A coupling names its operators by string; a name of several letters is the product of
        the one-letter entries of the mapping, so a caller can write ``('XY', 'XY')`` as the
        composite :math:`S_x S_y` without extending the mapping himself.  The group theory
        lives in :mod:`mpsdmrg.symmetry.invariant`; this is the entry point of the Hamiltonian
        into it.

        Args:
            group: the symmetry group of the Hamiltonian.
            phys_dim: physical dimension of one site.
            mapping: operator names to their dense matrices.
            couplings: the terms :meth:`add_couplings` recorded.
            dtype, device: data type and device of the operators.
            tol: relative residual that still counts as invariant.
        Returns:
            ``(operators, channels, bonds)`` as :func:`mpsdmrg.symmetry.invariant.bond_channels`
            returns them, or None when the couplings have no invariant realisation.
        Raises:
            ValueError: If a coupling names a letter that is not in the mapping.
        '''
        return bond_channels(group, phys_dim, mapping, couplings, dtype, device, tol)

    def build(self):
        '''
        Build the MPO tensors from the current couplings and mapping.

        The bond automaton and every group question live in
        :mod:`mpsdmrg.symmetry.hamiltonian`; an operator name of several letters is the
        product of those mapping entries on one site.
        '''
        self.tensors, self.bond_dim = build_mpo(self.group, self.L, self.physical_dim,
                                                self.mapping, self.couplings, self.dtype,
                                                self.device)

    def __sizeof__(self):
        return sum(t.numel() * t.element_size() for t in self.tensors)

class Broomstick:
    '''
    A 2-site DMRG engine: the Hamiltonian it minimises, the state it optimises, and the
    sweeps that do it.

    The Hamiltonian is built through the engine and the state is filled in afterwards, so a
    caller never constructs an ``MPO`` or an ``MPS``:

        stick = Broomstick(L=8, phys_dim=2)
        for i in range(7):
            stick.add_couplings(('Z', 'Z'), 1.0, (i, i + 1))
        stick.build()
        stick.initialize_state(bond_dims=[1] + [4] * 7 + [1])
        stick.sweep()

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
                 mapping: dict[str, torch.Tensor] | None = None,
                 group: Group | None = None):
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
            mapping: operator names to their dense torch matrices, for the Hamiltonian built here.
            group: symmetry group of the state and of the Hamiltonian; no symmetry if None.
        Raises:
            ValueError: If no source gives the number of sites or the physical dimension,
                or if the sources disagree.
        '''
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
        self.Hamiltonian = mpo if mpo is not None else MPO(
            self.L, self.physical_dim, mapping=mapping, dtype=self.dtype, device=self.device)
        self.Hamiltonian.group = group
        self.max_bond_dim = max_bond_dim
        self.svd_tol = svd_tol

        self.center = 0
        self.renv: dict[int, Tensor] = {}
        self.lenv: dict[int, Tensor] = {}
        self._last_energy = None  # cached from the most recent sweep

        if self.state is None and group is not None:
            self.initialize_state()

    def add_couplings(self, operators: tuple[str, ...], coeff: complex,
                      locations: NDArray[np.int64] | tuple[int, ...]) -> None:
        '''Add one coupling to the Hamiltonian, as ``MPO.add_couplings`` does.'''
        self.Hamiltonian.add_couplings(operators, coeff, locations)

    def build(self) -> None:
        '''Build the Hamiltonian tensors from the couplings added so far.'''
        self.Hamiltonian.build()

    def initialize_state(self,
                         bond_dims: list[int] | None = None,
                         init_state: list[Tensor] | None = None) -> None:
        '''
        Fill in the state to optimize, once the Hamiltonian is built.

        The symmetry layer provides the state whenever it has a default for the group: one
        singlet tensor per site, whose bond legs carry every sector the physical legs to
        their left fuse to. The arguments below apply to every other group.

        Args:
            bond_dims: bond dimensions of the state, as in ``MPS``; ``[1]``, the physical
                dimension on the inner bonds and ``[1]`` if None.
            init_state: the state tensors, as in ``MPS``; random if None.
        '''
        built = initial_state(self.group, self.L, self.physical_dim, self.max_bond_dim,
                              self.dtype, self.device)
        if built is not None:
            init_state, bond_dims = built
        self.state = MPS(self.L, self.physical_dim, bond_dims=bond_dims, init_state=init_state,
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
                          group=self.group)
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
            triv = Leg.trivial(1, group)
            self.lenv[0] = Tensor.ones(1, 1, 1, legs=(triv, triv.conj(), triv.conj()),
                                       qtotal=group.trivial_sector,
                                       dtype=self.dtype, device=self.device)
            self.renv[L] = Tensor.ones(1, 1, 1, legs=(triv.conj(), triv, triv),
                                       qtotal=group.trivial_sector,
                                       dtype=self.dtype, device=self.device)
        else:
            # Right boundary environment: MPO bond dim = Hamiltonian.bond_dim, final state (D-1) set to 1
            self.renv[L] = Tensor.zeros(1, self.Hamiltonian.bond_dim, 1,
                                        dtype=self.dtype, device=self.device)
            self.renv[L][0, -1, 0] = 1.0
            self.lenv[0] = Tensor.ones(1, 1, 1, dtype=self.dtype, device=self.device)

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
        self.cache_envs()
        state = self._require_state()
        best_E = None
        stale = 0

        for sweep in range(num_sweeps):
            E, trunc = self.singlesweep()
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
        state = self._require_state()
        self.lenv[i] = cached_einsum('aic,apb,cqd,ijpq->bjd',
                                        self.lenv[i-1], 
                                        state.tensors[i-1].conj(), 
                                        state.tensors[i-1],
                                        self.Hamiltonian.tensors[i-1])
    
    def _cache_renv(self,i:int):
        state = self._require_state()
        self.renv[i] = cached_einsum('apb,cqd,bjd,ijpq->aic',
                                        state.tensors[i].conj(), 
                                        state.tensors[i],
                                        self.renv[i + 1],
                                        self.Hamiltonian.tensors[i])

    @no_grad()
    def singlesweep(self):
        '''
        Perform a single sweep of the DMRG algorithm, optimizing the MPS tensors
        at each site.  Returns (energy, max_truncation_error).

        All intermediate values (max_trunc, energy) are kept on the GPU and
        synced to the CPU only once at the end of the sweep.

        Environment tensors are updated incrementally without QR
        canonicalization: the SVD inside update() already guarantees that the
        MPS tensors are left- / right-isometric as the sweep progresses.

        The sweep order is right (0 → L-1) then left (L-2 → 0).  The center
        site index is updated accordingly, and the MPS is left in a canonical
        form at the end of the sweep.
        '''
        if not self.lenv or not self.renv:
            self.cache_envs()
        state = self._require_state()
        max_trunc = torch.tensor(0.0, dtype=torch.float64, device=self.device)

        # ── right sweep ──
        for i in range(self.L - 1):
            E, trunc_err = self.update(direction='right')
            max_trunc = torch.maximum(max_trunc, trunc_err)
            if i < self.L - 2:
                self._cache_lenv(i + 1)
            self.center = i + 1

        # special handling for the last site: renormalize the last tensor to be right-isometric
        s_norm = norm(state.tensors[self.L - 1], dim=(1, 2))
        s_norm = torch.clamp(s_norm, min=1e-30)
        state.tensors[self.L - 1] = state.tensors[self.L - 1] / s_norm[:, None, None]
        state.tensors[self.L - 2] = state.tensors[self.L - 2] * s_norm[None, None, :]
        self.center = self.L - 2
        state.center = self.L - 2
        self._cache_renv(self.L - 1)
        for i in range(self.L - 3, -1, -1):
            self.center = i
            state.center = i
            E, trunc_err = self.update(direction='left')
            max_trunc = torch.maximum(max_trunc, trunc_err)
            if i > 0:
                self._cache_renv(i + 1)

        # ── sync to CPU once at end of sweep ──
        return E.item(), max_trunc.item()

    @no_grad()
    def _lanczos(self,
                 i: int,
                 v0: Tensor,
                 n_iter: int = 8) -> tuple[torch.Tensor, Tensor]:
        v0 = v0 / norm(v0)

        vecs: list[Tensor] = []
        alphas = torch.zeros(n_iter, dtype=torch.float64, device=v0.device)
        betas = torch.zeros(n_iter, dtype=torch.float64, device=v0.device)

        # pre computed contraction
        lenv = cached_einsum('apb,pqij->abqij', 
                            self.lenv[i],
                            self.Hamiltonian.tensors[i])
        renv = cached_einsum('crd,qrkl->cdqkl',
                            self.renv[i + 2],
                            self.Hamiltonian.tensors[i + 1])
        def helper(v: Tensor) -> Tensor:
            return cached_einsum('abqij,cdqkl,bjld->aikc', lenv, renv, v)
    
        vecs.append(v0)

        for k in range(n_iter):
            v_k = vecs[k]
            w = helper(v_k)

            alpha = inner(v_k, w).real
            alphas[k] = alpha

            if k == 0:
                w = w - alpha * v_k
            else:
                w = w - alpha * v_k - betas[k - 1] * vecs[k - 1]

            beta = norm(w).real
            betas[k] = beta

            v_next = w / beta
            v_next[beta == 0] = 0
            vecs.append(v_next)
        # for performance reasons, we will NOT check for convergence and run
        # all n_iter iterations to avoid GPU→CPU syncs.  The caller can check
        # the eigenvalue convergence if desired.

        # A negligible beta closes the Krylov space: the vectors generated after it are
        # round-off, and folding them into the tridiagonal mixes in directions that are not
        # orthogonal to the ones already found, which loses the normalisation of the ground
        # vector.  Keeping only the vectors up to the closing restores it.
        closed = (betas < 1e-12).nonzero()
        n_eff = n_iter if closed.numel() == 0 else min(n_iter, int(closed[0]) + 1)
        T = torch.diag(alphas[:n_eff])
        if n_eff > 1:
            T = T + torch.diag(betas[:n_eff - 1], diagonal=1) \
                  + torch.diag(betas[:n_eff - 1], diagonal=-1)

        eigvals, eigvecs_T = torch.linalg.eigh(T)

        E0 = eigvals[0]
        coeffs = eigvecs_T[:n_eff, 0].to(dtype=v0.dtype)
        v_ground = vecs[0] * coeffs[0]
        for k in range(1, n_eff):
            v_ground = v_ground + coeffs[k] * vecs[k]

        return E0, v_ground

    def update(self, direction: Literal['left', 'right'],
               n_iter: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Perform 2-site optimization of MPS with effective Hamiltonian.
        Returns (energy_tensor, truncation_error_tensor) — both are 0-d GPU
        tensors to avoid GPU→CPU syncs.  The caller is responsible for
        syncing only when needed (e.g. for progress display).
        Args:
            direction: 'left' or 'right', indicating the sweep direction.
            n_iter: Maximum number of Lanczos iterations for the effective Hamiltonian diagonalization.
        '''
        i = self.center
        mps = self._require_state()

        assert self.center == mps.center, "Center site index mismatch between Broomstick and MPS."

        state = cached_einsum('bjx,xld->bjld', mps.tensors[i], mps.tensors[i + 1])
        E, v = self._lanczos(i, state, n_iter=n_iter)
        u, s, vh = svd(v,
                       (0, 1),
                       (2, 3),
                       max_bond=self.max_bond_dim,
                       tol=self.svd_tol,
                       direction=direction)

        s2_kept = torch.sum(s*s).real
        trunc_err =  1.0 - s2_kept

        s /= torch.sqrt(s2_kept)
        new_bond = u.shape[-1]  # kept leg dimension, not the number of kept singular values
        if direction == 'right':
            mps.tensors[i] = u  # M[i] = U
            mps.tensors[i + 1] = s.to(vh.dtype).view(-1, 1, 1) * vh  # M[i+1] = S Vh
            mps.center = i + 1  # singular values absorbed into site i+1

        elif direction == 'left':
            mps.tensors[i] = u * s.to(u.dtype).view(1, 1, -1)  # M[i] = U S
            mps.tensors[i + 1] = vh  # M[i+1] = Vh
            mps.center = i  # singular values stay at site i
        mps.bond_dim[i + 1] = new_bond
        return E.real, trunc_err

    @property
    def energy(self) -> float:
        '''
        Energy of the current MPS state from the most recent sweep, or
        computed on-the-fly from the environments if no sweep has been run.
        '''
        if self._last_energy is not None:
            return self._last_energy
        # Fallback: compute from environments (only used before any sweep)
        return float(self._compute_energy().real)

    @no_grad()
    def _compute_energy(self) -> complex:
        state = self._require_state()
        lenv = cached_einsum('apc,pqij,aib,cjd->bqd',
                             self.lenv[self.center],
                         self.Hamiltonian.tensors[self.center],
                         state.tensors[self.center].conj(),
                         state.tensors[self.center])
        renv = cached_einsum('apc,qpij,bia,djc->bqd',self.renv[self.center + 2],
                         self.Hamiltonian.tensors[self.center + 1],
                         state.tensors[self.center + 1].conj(),
                         state.tensors[self.center + 1])

        return cached_einsum('bqd,bqd->', lenv, renv).item()
    
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
        state = self._require_state()
        W = self.Hamiltonian.tensors
        A = state.tensors
        lenv = cached_einsum('bpa,uwpr,vzrq,dqc->acwz',A[0].conj(), W[0], W[0], A[0])
        for i in range(1, self.L):
            lenv = cached_einsum('bdxy,bpa,xwpr,yzrq,dqc->acwz',
                                 lenv, A[i].conj(), W[i], W[i], A[i])
        # lenv shape: (1, 1, D, D) → pick the final MPO states
        E2 = float(lenv[0, 0, -1, -1].real.item().real)

        E = state.expectation(self.Hamiltonian)
        var = float(E2) - float(E.real) ** 2
        return max(var, 0.0)  # clamp numerical noise below zero
    
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
