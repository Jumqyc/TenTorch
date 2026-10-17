"""
The matrix product state and the initial state of a chain.
"""
from __future__ import annotations

from math import sqrt
from typing import Literal, Sequence

import torch

from base import COMPLEX, CUDA
from mpsdmrg import linalg
from mpsdmrg.mpo import MPO
from mpsdmrg.symmetry.group import Group, has_content, no_symmetry
from mpsdmrg.tensor import Tensor, cached_einsum, resolve, capped_leg, fuse_legs, site_leg

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
                 device = CUDA,
                 *,
                 group: Group = no_symmetry,
                 max_bond_dim: int | None = None):
        '''
        Args:
            L (int): Length of the MPS (number of sites).
            phys_dim (int): Physical dimension of each site.
            bond_dims (list[int] | None): List of bond dimensions for each bond in the MPS. Including the left and right virtual bonds, the length of this list should be L+1. If None, defaults to [1] + [phys_dim] * (L - 1) + [1].
            init_state (list[Tensor] | None): Optional initial state for the MPS. If None, random tensors will be generated.
            dtype: data type of the MPS tensors (default: COMPLEX).
            device: device that stores the MPS tensors (default: CUDA).
            group: the symmetry group of the chain, ``no_symmetry`` by default.  A group whose
                operator content is
                implemented builds its own start state -- one singlet tensor per site, whose
                bond legs carry every sector the physical legs to their left fuse to, capped
                at ``max_bond_dim`` -- and ignores ``bond_dims`` and ``init_state``.
            init_state: kept as a copy of the caller's list; the tensors themselves are
                canonicalized in place, so the caller's tensors are left as they were only
                when the constructor does not rebuild them.
            max_bond_dim: largest bond dimension of that invariant start state.
        Raises:
            ValueError: If the length of bond_dims does not match L+1, or if a
                multiplet-carrying group is given without a bond cap.
        '''
        if init_state is None and has_content(group):
            if max_bond_dim is None:
                raise ValueError(f'{group.group_name} builds its own start state and needs a bond cap.')
            # one singlet tensor per site: each bond carries every sector the physical legs to
            # its left fuse to, capped at max_bond_dim, so the state is a singlet and the sweep
            # starts from a bond dimension it can afford
            phys = site_leg(group, phys_dim)
            bonds = [site_leg(group, 1)]
            for _ in range(L - 1):
                bonds.append(capped_leg(fuse_legs(bonds[-1], phys), max_bond_dim))
            if not any(tuple(group.trivial_sector) in group.fusion_outcomes(sector, phys.sectors[0])
                       for sector in bonds[-1].sectors):
                raise ValueError(
                    f'A bond of {bonds[-1].dim} states cannot close {L} sites of dimension '
                    f'{phys_dim} to the trivial sector of {group.group_name}.')
            bonds.append(site_leg(group, 1))
            init_state = [Tensor.rand((bonds[i].dim, phys.dim, bonds[i + 1].dim),
                                      legs=(bonds[i], phys, bonds[i + 1].conj()),
                                      qtotal=group.trivial_sector,
                                      dtype=dtype, device=device) for i in range(L)]
            bond_dims = [bond.dim for bond in bonds]
        if bond_dims is None:
            bond_dims = [1] + [phys_dim] * (L - 1) + [1]
        self.L = L
        self.phys_dim = phys_dim
        self.bond_dim = bond_dims
        if len(bond_dims) != L + 1:
            raise ValueError(f"Bond dimension list length {len(bond_dims)} does not match the expected length of {L + 1}.")
        self.dtype = dtype
        self.device = device
        self.group = group
        self.center = 0  # Center site index for canonical form

        # Initialize MPS tensors (left_bond_dim, phys_dim, right_bond_dim)
        if init_state is None:
            self.tensors = [
            Tensor.rand((bond_dims[i], phys_dim, bond_dims[i + 1]),
                        dtype=dtype,
                        device=device) for i in range(L)
        ]
        else:
            # a copy of the list, so the caller's sequence is left alone
            self.tensors = list(init_state)
            if any(t.shape != (bond_dims[i], phys_dim, bond_dims[i + 1]) for i, t in enumerate(init_state)):
                raise ValueError("Initial state tensors do not match the specified bond and physical dimensions.")

        self._canonical = True  # the tensors below are canonical unless an operator breaks the form
        self.canonicalize()  # Ensure the MPS is in canonical form upon initialization
        self.normalize()  # Normalize the MPS upon initialization

    @staticmethod
    def _from_tensors(tensors: list[Tensor], group: Group, dtype: torch.dtype,
                      device: torch.device) -> MPS:
        '''
        Assemble an MPS from tensors as they are, without canonicalizing or normalizing them.

        The caller guarantees the tensors form a chain.  An operator applied to a state lands in
        another charge sector, whose tensors are not the canonical form of a trivial total charge,
        so the state has to be taken as it is; its left edge carries the total charge.

        Args:
            tensors (list[Tensor]): one tensor per site, legs (left bond, physical, right bond).
            group (Group): the symmetry the tensors carry.
            dtype (torch.dtype): the element type of the state.
            device (torch.device): the device of the state.
        Returns:
            MPS: the state those tensors make.
        '''
        state = MPS.__new__(MPS)
        state.L = len(tensors)
        state.phys_dim = tensors[0].shape[1]
        state.bond_dim = [tensors[0].shape[0]] + [tensor.shape[2] for tensor in tensors]
        state.dtype = dtype
        state.device = device
        state.group = group
        state.center = 0
        state.tensors = list(tensors)
        state._canonical = False
        return state

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
        Accepts either a dict[int, operator] of single-site operators or an MPO.

        To compute the norm, just pass {} as the operator list.
        Args:
            operators: dict[int, Tensor | torch.Tensor | numpy.ndarray] | MPO: a dictionary of the
                       operators to measure, or an MPO representing a full matrix product operator.
                       A value without legs is the matrix of the operator on the physical leg of
                       the state at that site; the dictionary is then a product of one-site
                       operators, measured exactly on the dense state (an invariant state has a
                       dense expansion, so no multiplet structure is needed).
            mode: 'direct' or 'env' — contraction strategy (for dict operators only; MPO always uses direct).
        Returns:
            complex: The expectation value of the operators on the MPS.
        Raises:
            ValueError: If the operator shapes do not match the physical dimension of the MPS tensors.
        '''
        if isinstance(operators, MPO):
            return self._mpo_expectation(operators)
        if any(not isinstance(operator, Tensor) or not operator.legs for operator in operators.values()):
            matrices: dict[int, torch.Tensor] = {}
            for site, operator in operators.items():
                matrix = operator.to_torch() if isinstance(operator, Tensor) else torch.as_tensor(operator)
                if matrix.shape != (self.phys_dim, self.phys_dim):
                    raise ValueError(f"Operator shapes do not match the physical dimension {self.phys_dim}.")
                if site < 0 or site >= self.L:
                    raise ValueError(f"Operator indices must be in the range [0, {self.L - 1}].")
                matrices[site] = matrix.reshape(self.phys_dim, self.phys_dim)
            return self._dense_expectation(matrices)
        if any(op.shape != (self.phys_dim, self.phys_dim) for op in operators.values()):
            raise ValueError(f"Operator shapes do not match the physical dimension {self.phys_dim}.")
        elif any(idx < 0 or idx >= self.L for idx in operators.keys()):
            raise ValueError(f"Operator indices must be in the range [0, {self.L - 1}].")
        if mode == 'direct':
            return self._full_dir_contract(operators) # uses brute-force contraction of the MPS with the given operators to compute the expectation value.
        elif mode == 'env':
            return self._full_env_contract(operators) # uses canonical condition.

    def norm(self,mode:Literal['direct','env'] = 'env')->complex:
        '''
        Return the squared norm <self|self>.
        Args:
            mode (str): 'env' reads the centre tensor, which is the squared norm while the state is
                        canonical and contracts the whole state otherwise; 'direct' always contracts.
        Returns:
            complex: the squared norm of the state.
        Raises:
            ValueError: If the mode is neither 'direct' nor 'env'.
        '''
        if mode == 'direct' or not self._canonical:
            return self._full_dir_contract({})
        if mode == 'env':
            return cached_einsum('abc,abc->', self.tensors[self.center].conj(), self.tensors[self.center]).item()
        raise ValueError(f"A norm is computed in mode 'direct' or 'env', not {mode!r}.")

    def _apply_gate(self, gate: torch.Tensor, targets: tuple[int, int],
                    max_bond_dim: int | None, tol: float) -> MPS:
        '''
        Return the state with a two-site operator applied to one bond.

        Args:
            gate: the matrix of the operator in the product basis of the two sites, in shape
                (dim, dim, dim, dim) read as (bra1, bra2, ket1, ket2).
            targets: the two neighbouring sites the gate acts on.
            max_bond_dim (int | None): largest bond dimension the split keeps; ``None`` keeps all.
            tol (float): singular values at or below this are dropped.
        Returns:
            MPS: the state after the gate, not canonicalized.
        '''
        first, second = targets
        # a bare matrix on a symmetry-adapted leg carries every component, which only a resolved
        # leg can store; an unresolved leg is the plain dense case
        physical = (resolve(self.tensors[first]) if has_content(self.group)
                    else self.tensors[first]).legs[1]
        dim = physical.dim
        matrix = Tensor.from_torch(
            torch.as_tensor(gate, dtype=self.dtype, device=self.device).reshape(dim, dim, dim, dim),
            legs=(physical, physical, physical.conj(), physical.conj()))
        # a bare gate matrix carries every magnetic component, so a symmetry-adapted window is
        # resolved for the contraction and the factors are resolved again afterwards
        window = (lambda site: resolve(self.tensors[site])) if has_content(self.group) else \
            (lambda site: self.tensors[site])
        merged = cached_einsum('lsr,rtu->lstu', window(first), window(second))
        merged = cached_einsum('lstu,pqst->lpqu', merged, matrix)
        tensor, singular, rest = linalg.svd(merged, (0, 1), (2, 3), max_bond=max_bond_dim,
                                            tol=tol, direction='right')
        if has_content(self.group):
            tensor, rest = resolve(tensor), resolve(rest)
        tensors = list(self.tensors)
        tensors[first] = tensor
        tensors[second] = linalg.absorb(rest, singular)
        return self._from_tensors(tensors, self.group, self.dtype, self.device)

    def resolve(self) -> MPS:
        '''
        Return the same state with every leg resolved.

        Returns:
            MPS: a state of the same amplitudes whose blocks store the dense data of their
            charge windows, which is what a bare operator matrix acts on.
        '''
        return self._from_tensors([resolve(tensor) for tensor in self.tensors],
                                  self.group, self.dtype, self.device)

    def apply(self,
              operator: "MPO | torch.Tensor",
              sites: "int | Sequence[int] | None" = None,
              max_bond_dim: int | None = None,
              tol: float = 0.0) -> MPS:
        '''
        Return the state ``operator |self>`` as a new MPS.

        Each site contracts the state tensor with the operator tensor on the physical leg, and the
        pair of virtual legs the contraction leaves open - the state's and the operator's - is split
        off again, so the bonds of the result carry whatever charge the operator adds.  The result
        is not canonicalized.

        Args:
            operator (MPO): the operator to apply; same length and physical dimension as the state.
            max_bond_dim (int | None): largest bond dimension the splits keep; ``None`` keeps all.
            tol (float): singular values at or below this are dropped.
        Returns:
            MPS: the state after the operator.
        Raises:
            ValueError: If the operator does not act on this state.
        '''
        if not isinstance(operator, MPO):
            if sites is None:
                raise ValueError('Applying a local operator needs the sites it acts on.')
            targets = ((int(sites),) if isinstance(sites, int)
                       else tuple(int(site) for site in sites))
            if len(targets) == 2:
                if targets[1] != targets[0] + 1:
                    raise NotImplementedError(
                        f'A two-site operator acts on two neighbours, not on {targets}.')
                return self._apply_gate(operator, (targets[0], targets[1]), max_bond_dim, tol)
            if len(targets) != 1:
                raise NotImplementedError('A local operator acts on one site or on two neighbours.')
            site = targets[0]
            phys = self.tensors[site].legs[1]
            if has_content(self.group) and not phys.resolved:
                # a bare matrix has no fusion path to follow, so the state resolves its
                # magnetic index into its blocks and the matrix acts on the dense data
                return self.resolve().apply(operator, sites, max_bond_dim, tol)
            matrix = Tensor.from_torch(
                torch.as_tensor(operator, dtype=self.dtype, device=self.device),
                legs=(phys, phys.conj()))
            tensors = list(self.tensors)
            tensors[site] = cached_einsum('lsr,ts->ltr', tensors[site], matrix)
            return self._from_tensors(tensors, self.group, self.dtype, self.device)
        if operator.L != self.L or operator.physical_dim != self.phys_dim:
            raise ValueError(
                f'An operator of {operator.L} sites of dimension {operator.physical_dim} does not '
                f'act on a state of {self.L} sites of dimension {self.phys_dim}.')
        tensors: list[Tensor] = []
        # a resolved state contracts against resolved operator tensors only, so every window is
        # dense data and no reduced round trip can drop a multiplet copy
        carry = None
        for site in range(self.L):
            state, matrix = self.tensors[site], operator.tensors[site]
            if carry is None:
                # the state's left edge is the left bond of the chain; the operator's left edge is a
                # trivial leg, so it is closed with its conjugate
                edge = Tensor.ones(1, legs=(matrix.legs[0].conj(),), dtype=self.dtype,
                                   device=self.device)
                merged = cached_einsum('lsr,abps,a->lprb', state, matrix, edge)
            else:
                # carry: the new left bond, the state's previous right leg, the operator's previous
                # right leg
                merged = cached_einsum('mla,lsr,abps->mprb', carry, state, matrix)
            if site == self.L - 1:
                # the chain closes: the operator's right edge is a trivial leg, the state keeps its
                # own right edge as the last bond
                edge = Tensor.ones(1, legs=(merged.legs[3].conj(),), dtype=self.dtype,
                                   device=self.device)
                tensors.append(cached_einsum('mprb,b->mpr', merged, edge))
                break
            # the left group (the new bond and the physical leg) becomes this site's tensor, the
            # right group (the two virtual legs) becomes the carry into the next site
            tensor, singular, rest = linalg.svd(merged, (0, 1), (2, 3), max_bond=max_bond_dim,
                                                tol=tol, direction='right')
            tensors.append(tensor)
            carry = linalg.absorb(rest, singular)
        return MPS._from_tensors(tensors, self.group, self.dtype, self.device)

    def normalize(self):
        '''
        Normalize the MPS by dividing each tensor by the square root of the norm.
        '''
        norm:complex = self.norm(mode='env')
        self.tensors[self.center] /= sqrt(norm.real)

    def _dense_expectation(self, operators: dict[int, torch.Tensor]) -> complex:
        '''
        The exact expectation of a product of one-site operators, contracted over the dense state.

        An invariant state has trivial total charge, so its tensors expand into the dense basis;
        this is the reference an adapted contraction has to reproduce, and it needs no multiplet
        structure at all.

        Args:
            operators: site -> matrix in shape (phys_dim, phys_dim), at most one per site.
        Returns:
            complex: the expectation value, exact up to the floating point error.
        '''
        tensors = [tensor.to_torch() for tensor in self.tensors]
        environment = torch.ones(1, 1, dtype=self.dtype, device=self.device)
        for site in range(self.L):
            operator = operators.get(site)
            if operator is None:
                environment = torch.einsum('ab,aic,bid->cd', environment, tensors[site].conj(),
                                           tensors[site])
            else:
                environment = torch.einsum('ab,aic,bjd,ij->cd', environment, tensors[site].conj(),
                                           tensors[site],
                                           operator.to(dtype=self.dtype, device=self.device))
        return complex(environment.sum())

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
        return linalg.trace(lenv).item()


    def _full_env_contract(self, operators: dict[int, Tensor]):
        '''
        Compute the expectation value of a list of operators on the MPS.

        The contraction is gauge independent, so the center is left where it is.
        '''
        if not operators: # If the operator list is empty, return the norm of the MPS.
            return self.norm(mode='env')

        leftmost = min(operators.keys())
        rightmost = max(operators.keys())

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
        return linalg.trace(lenv).item()

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
        # the Schmidt values live at the center, so the center moves to the bond and back
        center = self.center
        self.move_center_to(bond)
        s = linalg.svdvals(self.tensors[bond], (0, 1), (2,))
        s = s[s > svd_tol]
        self.move_center_to(center)
        if alpha is None:
            return -torch.sum(s**2 * torch.log(s**2)).item()
        else:
            return 1/(1-alpha) * torch.log(torch.sum(s**(2*alpha))).item()


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
        q, r = linalg.qr(self.tensors[i], (0, 1), (2,))
        self.bond_dim[i+1] = q.shape[-1]
        self.tensors[i] = q
        self.tensors[i+1] = cached_einsum('ab,bcd->acd', r, self.tensors[i+1])
        if update_center:
            self.center = i + 1

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
        r, q = linalg.rq(self.tensors[i], (0,), (1, 2))
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
                      self.device,
                      group=self.group)
    def __repr__(self):
        return (f"MPS(L={self.L}, phys_dim={self.phys_dim}, bond_dim={self.bond_dim}, "
            f"center={self.center}, group={self.group.group_name})")
    def __sizeof__(self):
        return sum(t.numel() * t.element_size() for t in self.tensors)

