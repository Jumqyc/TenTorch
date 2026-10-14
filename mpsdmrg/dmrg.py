import torch
import numpy as np
import pickle as pkl


from math import sqrt
from typing import Literal
from numpy.typing import NDArray

from base import COMPLEX, CUDA
# ``_expr_cache`` is re-exported by name so a profiling script can clear it.
from mpsdmrg.symmetry import (Group, Leg, Tensor, _expr_cache, cached_einsum, inner, no_grad,
                              norm, qr, rq, svd, svdvals, trace)


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

    @property
    def shape(self)-> list[int]:
        '''
        Return the virtual bond dimensions of the MPS.
        '''
        return self.bond_dim

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
        if hasattr(self, 'state_idx'):
            new.state_idx = self.state_idx.copy()
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
            operators (tuple[str]): A tuple of operator names (e.g., ('X', 'Y')).
            coeffs (complex): A complex coefficient for the coupling.
            locations (tuple[int]): A tuple of site indices where the operators act.
        Raises:
            ValueError: If the length of operators does not match the number of locations,
                        or if any site index is out of bounds for the MPO length,
                        or if any operator is not defined in the mapping.
        '''
        if len(operators) != len(locations):
            raise ValueError(f"Length of operators {len(operators)} does not match the number of locations {len(locations)}.")
        if any(loc < 0 or loc >= self.L for loc in locations):
            raise ValueError(f"Site indices {locations} are out of bounds for MPO of length {self.L}.")
        if not all(op in self.mapping for op in operators):
            raise ValueError(f"Operators {operators} are not defined in the mapping. Available operators: {list(self.mapping.keys())}.")

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

    def build(self):
        '''
        Build the MPO tensors from the current couplings and mapping.

        Intermediate states are identified by ``(op_str, rel_pos)`` pairs
        where *op_str* is the operator just placed and *rel_pos* is the
        distance (in sites) to the next operator.  Nearest-neighbour states
        are shared across couplings; long-range states are kept unique per
        coupling to avoid spurious paths.

        A non-abelian group is delegated to :meth:`_build_invariant`, which builds the
        isotropic nearest-neighbour SU(2) Heisenberg chain and refuses every other
        coupling; the automaton below is the general path.
        '''
        group = self.group
        if group is not None and not group.is_abelian:
            self._build_invariant(group)
            return
        first_site_counts: dict[tuple, int] = {}
        for c_idx, (ops, _, locs) in enumerate(self.couplings):
            if len(ops) > 1 and locs[1] - locs[0] == 1:
                key = (ops[0], 1, locs[0])  # (op_str, rel_pos=1, site)
                first_site_counts[key] = first_site_counts.get(key, 0) + 1
        state_ranges: dict[tuple, set[tuple[int, int]]] = {}
        coupling_keys: list[list[tuple]] = [[] for _ in self.couplings]
        for c_idx, (ops, _, locs) in enumerate(self.couplings):
            for n in range(len(ops) - 1):
                rel_pos = locs[n + 1] - locs[n]
                if n == 0 and rel_pos == 1:
                    site_key = (ops[0], 1, locs[0])
                    if first_site_counts.get(site_key, 0) > 1:
                        key = (c_idx, ops[n], rel_pos)  # unique per coupling
                    else:
                        key = (ops[n], rel_pos)          # shared
                else:
                    # long-range states stay unique per coupling, so no shared identity
                    # segment can open a spurious path between two couplings
                    key = (ops[n], rel_pos) if rel_pos == 1 else (c_idx, ops[n], rel_pos)
                coupling_keys[c_idx].append(key)
                state_ranges.setdefault(key, set()).add((locs[n], locs[n + 1]))

        sorted_keys = sorted(state_ranges, key=lambda x: (len(x), x))
        self.state_idx = {k: i + 1 for i, k in enumerate(sorted_keys)}
        self.bond_dim = len(self.state_idx) + 2  # start (0) + end (D-1)

        D = self.bond_dim
        p = self.physical_dim
        self.tensors = [Tensor.zeros((1, D, p, p), dtype=self.dtype, device=self.device)]
        self.tensors += [Tensor.zeros((D, D, p, p), dtype=self.dtype, device=self.device)
                         for _ in range(1, self.L)]
        self.tensors += [Tensor.zeros((D, 1, p, p), dtype=self.dtype, device=self.device)]

        eye = torch.eye(p, dtype=self.dtype, device=self.device)
        self.tensors[0][0, 0] = eye                       # site-0 start
        self.tensors[self.L][D - 1, 0] = eye               # right boundary
        for i in range(1, self.L):
            self.tensors[i][0, 0] = eye                    # bulk start
            self.tensors[i][D - 1, D - 1] = eye            # bulk end

        # identity on skipped sites for long-range couplings
        for key, ranges in state_ranges.items():
            rel_pos = key[-1]
            if rel_pos <= 1:
                continue
            s = self.state_idx[key]
            for from_site, to_site in ranges:
                for k in range(from_site + 1, to_site):
                    if 0 < k < self.L:
                        self.tensors[k][s, s] = eye

        # ── place operator matrices ──
        for c_idx, (ops, coeff, locs) in enumerate(self.couplings):
            keys = coupling_keys[c_idx]
            for n, (op_str, site) in enumerate(zip(ops, locs)):
                mat = self.mapping[op_str]
                if n == 0:
                    mat = coeff * mat

                state_in = 0 if n == 0 else self.state_idx[keys[n - 1]]
                state_out = D - 1 if n == len(ops) - 1 else self.state_idx[keys[n]]

                if site == 0:
                    self.tensors[0][0, state_out] += mat
                elif site == self.L:
                    self.tensors[self.L][state_in, 0] += mat
                else:
                    self.tensors[site][state_in, state_out] += mat

    def _build_invariant(self, group: Group) -> None:
        r'''
        Build the tensors of an isotropic nearest-neighbour chain under SU(2).

        Every site carries one invariant tensor on the legs
        ``(bond_l, bond_r, bra_phys, ket_phys)``.  The bond is the direct sum of a
        ``j = 0`` channel, whose two copies are the start and the end of an interaction,
        and a ``j = 1`` channel carrying the spin operator of an interaction that has
        started, so the only nonzero paths pair one nearest-neighbour bond with
        :math:`\mathbf{S}_i \cdot \mathbf{S}_{i+1}`.

        The gauge is fixed by the dense limit: the ``j = 0`` block is :math:`\sqrt{2}`,
        which expands to the identity, and the opening and closing ``j = 1`` insertions
        are :math:`\sqrt{3/2}` and :math:`-\sqrt{3/2}`, which expand to the spin operator.
        A bond coupling :math:`J` is split as :math:`\sqrt{J}` on its opening and its
        closing, so the pair picks up exactly :math:`J`.

        Args:
            group: the symmetry group of the chain.
        Raises:
            NotImplementedError: If the group is not SU(2), the physical leg is not
                spin-1/2, or the couplings are not an isotropic nearest-neighbour
                ``(X, X)``, ``(Y, Y)``, ``(Z, Z)`` triplet.
        '''
        if group.group_name != 'SU(2)' or self.physical_dim != 2:
            raise NotImplementedError(
                f'An invariant Hamiltonian needs SU(2) spin-1/2 sites, not {group.group_name} '
                f'with physical dimension {self.physical_dim}.')
        components: dict[int, dict[str, complex]] = {}
        for ops, coeff, locs in self.couplings:
            if len(ops) != 2 or ops[0] != ops[1] or ops[0] not in 'XYZ' or locs[1] != locs[0] + 1:
                raise NotImplementedError(
                    f'An invariant Hamiltonian needs nearest-neighbour (X, X), (Y, Y) and '
                    f'(Z, Z) couplings, not {tuple(ops)} at {tuple(locs)}.')
            components.setdefault(int(locs[0]), {})[ops[0]] = coeff
        strength: dict[int, complex] = {}
        for site, terms in components.items():
            if set(terms) != set('XYZ') or max(abs(v - terms['X']) for v in terms.values()) > 1e-10:
                raise NotImplementedError(
                    f'The couplings at site {site} are not the isotropic Heisenberg '
                    f'triplet: {terms}.')
            strength[site] = complex(terms['X']) ** 0.5

        triv = Leg.trivial(1, group)
        phys = Leg(group, ((self.physical_dim - 1,),), (0, self.physical_dim), qconj=1)
        # the bond of the nearest-neighbour automaton: a ``j = 0`` channel whose two copies
        # are the start and the end of an interaction, and a ``j = 1`` channel carrying the
        # spin operator of an interaction that is under way
        link = Leg(group, ((0,), (2,)), (0, 2, 5), qconj=1)
        identity, opening, closing = sqrt(2.0), sqrt(1.5), -sqrt(1.5)
        self.tensors = []
        for i in range(self.L):
            if i == 0:
                wl, wr, role = triv, link.conj(), 'left'
            elif i == self.L - 1:
                wl, wr, role = link, triv.conj(), 'right'
            else:
                wl, wr, role = link, link.conj(), 'bulk'
            tensor = Tensor.zeros(wl.dim, wr.dim, self.physical_dim, self.physical_dim,
                                  legs=(wl, wr, phys, phys.conj()), qtotal=group.trivial_sector,
                                  dtype=self.dtype, device=self.device)
            for block, q in zip(tensor.blocks, tensor.qdata):
                if q[0] == 0 and q[1] == 0:
                    if role == 'right':
                        block[wl.multiplicity(0) - 1, 0, 0, 0] = identity
                    else:
                        block[0, 0, 0, 0] = identity
                        if role == 'bulk':
                            block[wl.multiplicity(0) - 1, wr.multiplicity(0) - 1, 0, 0] = identity
                elif q[0] == 0 and q[1] == 1 and role != 'right':
                    block[0, 0, 0, 0] = torch.tensor(opening * strength[i], dtype=block.dtype,
                                                     device=block.device)
                elif q[0] == 1 and q[1] == 0 and role != 'left':
                    block[0, wr.multiplicity(0) - 1, 0, 0] = torch.tensor(
                        closing * strength[i - 1], dtype=block.dtype, device=block.device)
            self.tensors.append(tensor)
        self.bond_dim = link.dim

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

        With a group the state is built invariant: one singlet tensor per site, whose bond
        legs carry every sector the physical legs to their left fuse to.

        Args:
            bond_dims: bond dimensions of the state, as in ``MPS``; ``[1]``, the physical
                dimension on the inner bonds and ``[1]`` if None. Ignored with a group.
            init_state: the state tensors, as in ``MPS``; random if None. Ignored with a group.
        '''
        if self.group is not None:
            tensors, bond_dims = self._invariant_tensors(self.group)
            init_state = tensors
        self.state = MPS(self.L, self.physical_dim, bond_dims=bond_dims, init_state=init_state,
                         dtype=self.dtype, device=self.device)

    def _invariant_tensors(self, group: Group) -> tuple[list[Tensor], list[int]]:
        '''
        One singlet tensor per site of a chain with a symmetry group.

        The physical leg carries the sector of dimension ``phys_dim``, every tensor fuses
        its two bond legs and the physical leg to the trivial sector, and each bond carries
        every sector the physical legs to its left fuse to, so the state is a singlet.

        Args:
            group: the symmetry group of the chain.
        Returns:
            The tensors and the dimension of every bond, left boundary first.
        Raises:
            NotImplementedError: If the group is not SU(2).
            ValueError: If the chain has an odd number of sites, which has no singlet.
        '''
        if group.group_name != 'SU(2)':
            raise NotImplementedError(
                f'An invariant state needs SU(2) sectors, not {group.group_name}.')
        if self.L % 2:
            raise ValueError(f'An SU(2) singlet chain needs an even number of sites, not {self.L}.')
        phys = Leg(group, ((self.physical_dim - 1,),), (0, self.physical_dim), qconj=1)
        bonds = [Leg.trivial(1, group)]
        for _ in range(self.L - 1):
            bonds.append(bonds[-1].fuse(phys))
        bonds.append(Leg.trivial(1, group))
        tensors = [Tensor.rand((bonds[i].dim, phys.dim, bonds[i + 1].dim),
                               legs=(bonds[i], phys, bonds[i + 1].conj()),
                               qtotal=group.trivial_sector,
                               dtype=self.dtype, device=self.device) for i in range(self.L)]
        return tensors, [bond.dim for bond in bonds]

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
        self._require_state()
        L = self.L
        self.renv = {}
        self.lenv = {}

        if self.group is not None and not self.group.is_abelian:
            triv = Leg.trivial(1, self.group)
            self.lenv[0] = Tensor.ones(1, 1, 1, legs=(triv, triv.conj(), triv.conj()),
                                       qtotal=self.group.trivial_sector,
                                       dtype=self.dtype, device=self.device)
            self.renv[L] = Tensor.ones(1, 1, 1, legs=(triv.conj(), triv, triv),
                                       qtotal=self.group.trivial_sector,
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
        return self._compute_energy()

    @no_grad()
    def _compute_energy(self) -> float:
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
        E2 = lenv[0, 0, -1, -1].real.item()

        E = state.expectation(self.Hamiltonian)
        var = (E2 - (E.real) ** 2)
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
