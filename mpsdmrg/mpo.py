from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from base import COMPLEX, CUDA
from mpsdmrg.symmetry.cat import Group, Sector, content, has_content, no_symmetry
from mpsdmrg.operators import SpinOperator
from mpsdmrg.tensor import Tensor, bond_leg, cached_einsum, multiplet_leg, phys_leg

class MPO:
    '''
    A class representing a Matrix Product Operator (MPO) in quantum physics and tensor networks.
    '''
    def __init__(self,
                 L:int,
                 phys_dim:int,
                 mapping:dict[str,Tensor]|None = None,
                 dtype = COMPLEX,
                 device = CUDA,
                 group:Group = no_symmetry,
                 total:tuple[int,...]|None = None,
                 site_sector:tuple[int,...]|None = None):
        '''
        Args:
            L (int): Length of the MPO (number of sites).
            phys_dim (int): Physical dimension of the MPO.
            mapping (dict[str, Tensor]): A dictionary mapping operator names to their dense operator tensors. If None, the spin components and the identity are used.
            dtype, device: Data type and device of the tensors.
            group (Group): Symmetry group of the Hamiltonian; ``no_symmetry`` by default.
            total (tuple[int, ...] | None): Total charge the operator carries, for an operator
                that is not a Hamiltonian; ``None`` means the trivial sector.
            site_sector (tuple[int, ...] | None): The multiplet one site lives in, e.g. ``(1,)``
                for a spin-1/2 site of SU(2) or ``(1, 0)`` for the fundamental of SU(3); a group
                whose multiplets are not told apart by their dimension needs it.
        '''
        self.dtype = dtype
        self.device = device
        self.L = L
        self.group: Group = group
        self.total: tuple[int, ...] | None = None if total is None else tuple(int(entry) for entry in total)
        self.site_sector = None if site_sector is None else tuple(int(entry) for entry in site_sector)
        self.couplings: list[tuple[tuple[str, ...], complex, tuple[int, ...]]] = []
        self.physical_dim = phys_dim
        self.tensors: list[Tensor] = []

        if mapping is None:
            spin_op = SpinOperator(phys_dim=phys_dim, dtype=dtype, device=device).data
            mapping = {
                'X': spin_op[0],
                'Y': spin_op[1],
                'Z': spin_op[2],
                'I': Tensor.from_torch(torch.eye(phys_dim, dtype=dtype, device=device))
            }
        self.mapping = dict(mapping)
        if not all(isinstance(matrix, Tensor) for matrix in self.mapping.values()):
            raise TypeError('The mapping of an MPO holds mpsdmrg.tensor.Tensor matrices.')

    @property
    def _buffers(self) -> dict[str, torch.Tensor]:
        '''The dense buffers of the mapping, for the contractions that build the tensors.'''
        return {name: matrix.blocks[0] for name, matrix in self.mapping.items()}

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
                  mapping={k: v.clone() for k, v in self.mapping.items()},
                  dtype=self.dtype,
                  device=self.device,
                  group=self.group,
                  total=self.total,
                  site_sector=self.site_sector)
        new.couplings = [(ops, coeff, locs) for ops, coeff, locs in self.couplings]
        new.group = self.group
        if self.tensors:
            new.tensors = [t.clone().detach() for t in self.tensors]
        return new

    @property
    def bond_dim(self) -> int:
        '''The bond dimension: the largest virtual leg of the MPO tensors.'''
        return max(max(tensor.shape[0], tensor.shape[1]) for tensor in self.tensors) \
            if self.tensors else 0

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
            # the couplings carry the factor as well, so a later build() keeps it
            new.couplings = [(ops, coeff * other, locs) for ops, coeff, locs in new.couplings]
            new.tensors = [tensor * other for tensor in new.tensors]
            return new
        elif isinstance(other, MPO):
            if has_content(self.group) or has_content(other.group):
                raise NotImplementedError(
                    'Composing symmetry-adapted MPOs is not supported; merge the couplings of '
                    'the two Hamiltonians instead.')
            # MPO * MPO: contract physical indices, outer-product the bond dims
            new = MPO(self.L, self.physical_dim,
                      mapping={k: v.clone() for k, v in self.mapping.items()},
                      dtype=self.dtype, device=self.device)
            new.tensors = []
            for i in range(len(self.tensors)):
                t1 = self.tensors[i]
                t2 = other.tensors[i]
                # t1: (wL1, wR1, p, q), t2: (wL2, wR2, q, r)  →  (wL1*wL2, wR1*wR2, p, r)
                new_t = cached_einsum('abpq,cdqr->acbdpr', t1, t2).combine_legs(
                    ((0, 1), (2, 3), (4,), (5,)))
                new.tensors.append(new_t)
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

    def add_couplings(self,operators:str|tuple[str,...],coeff:complex,
                      locations:int|NDArray[np.int64]|tuple[int,...]):
        '''
        Add new couplings to the MPO.
        Args:
            operators (str | tuple[str]): The operator names, one per location.  A name of
                several letters is the product of those mapping entries on one site, e.g.
                ``'XY'`` is :math:`S_x S_y`; a single name may be given as a plain string.
            coeff (complex): A complex coefficient for the coupling.
            locations (int | tuple[int]): The site indices the operators act on; one integer
                for a single site.
        Raises:
            ValueError: If the number of operators does not match the number of locations,
                        or if any site index is out of bounds for the MPO length,
                        or if any operator letter is not defined in the mapping.
        '''
        operators = (operators,) if isinstance(operators, str) else tuple(operators)
        locations = (int(locations),) if isinstance(locations, (int, np.integer)) else tuple(locations)
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
        # the built tensors no longer describe the Hamiltonian
        self.tensors = []

    def _register_merged_op(self, op_string: str) -> str:
        '''
        Register a merged operator matrix if not already present, return its name.
        '''
        if op_string not in self.mapping:
            # a several-letter name is the product of those mapping entries on one site
            buffers = self._buffers
            mat = buffers[op_string[0]]
            for ch in op_string[1:]:
                mat = mat @ buffers[ch]
            self.mapping[op_string] = Tensor.from_torch(mat.to(dtype=self.dtype, device=self.device))
        return op_string

    def build(self) -> None:
        '''
        Build the MPO tensors from the couplings and the mapping recorded so far.

        A group with multiplet content places components on the invariant channels of its bonds;
        every other group walks the operator labels of the couplings.  An operator name of several
        letters is the product of those mapping entries on one site.
        '''
        self.tensors = self._build_adapted() if has_content(self.group) else self._build_labels()

    def _build_adapted(self) -> list[Tensor]:
        group, phys_dim = self.group, self.physical_dim
        total = group.trivial_sector if self.total is None else self.total
        phys = phys_leg(group, phys_dim, self.site_sector)
        raw = {name: matrix.to(dtype=self.dtype, device=self.device)
               for name, matrix in self._buffers.items()}
        components = {tuple(sector): tensor.to(dtype=self.dtype, device=self.device)
                      for sector, tensor in content(group).tensor_operators(
                          group, raw, phys, self.dtype, self.device).items()}
        # one unit placement of a sector acts as the channel of its pair, and a projection divides
        # by the norm that member of the multiplet carries
        channels = {sector: content(group).channel(group, sector, tensor)
                    for sector, tensor in components.items()}
        norms = {sector: torch.einsum('mab,mab->m', tensor.conj(), tensor).real
                 for sector, tensor in components.items()}
        # only the operators the couplings name are expanded; the trivial component is the one
        # that survives on the rail, and it has a single member, so its amplitude is a scalar
        amplitudes: dict[str, dict[Sector, complex]] = {}
        for name in {name for operators, _, _ in self.couplings for name in operators}:
            residual, found = raw[name].clone(), {}
            for sector, tensor in components.items():
                member = torch.einsum('mab,ab->m', tensor.conj(), residual) / norms[sector]
                if float(member.abs().max()) > 1e-10:
                    found[sector] = complex(member[0])
                    residual = residual - torch.einsum('m,mab->ab', member, tensor)
            if float(residual.abs().max()) > 1e-10:
                raise ValueError(f'The operator {name!r} is not a combination of the components '
                                 f'of {group.name}.')
            amplitudes[name] = found
        singles: dict[int, complex] = {}
        placed: list[tuple[int, int, list[tuple[Sector, float, complex, complex]],
                           dict[Sector, int]]] = []
        threads: dict[Sector, int] = {}
        for operators, coeff, locations in self.couplings:
            first, last = int(locations[0]), int(locations[-1])
            if first > last:
                first, last = last, first
            if first == last:
                if total not in amplitudes[operators[0]]:
                    raise ValueError(
                        f'The one-site operator {operators[0]} has no component of charge {total} '
                        f'in {group.name}.')
                if total != group.trivial_sector:
                    raise NotImplementedError(
                        f'A one-site component of {group.name} needs its member index on a '
                        f'leg of its own, which a Tensor of a single multiplicity does not carry; '
                        f'write the operator as a two-site term instead.')
                singles[first] = (singles.get(first, 0.0)
                                  + complex(coeff) * amplitudes[operators[0]][total])
                continue
            if len(operators) != 2:
                raise NotImplementedError(
                    f'The symmetry-adapted builder places two operators per term, not '
                    f'{len(operators)}; build such a Hamiltonian with group=no_symmetry.')
            left_op, right_op = operators
            pair = torch.kron(raw[left_op], raw[right_op]).reshape(-1)
            pairs = []
            for sector in amplitudes[left_op]:
                for partner in amplitudes[right_op]:
                    if total not in {tuple(outcome)
                                       for outcome in group.fusion_outcomes(sector, partner)}:
                        continue
                    overlap = complex(coeff) * complex(
                        torch.vdot(channels[sector].reshape(-1), pair))
                    if abs(overlap) < 1e-12:
                        continue
                    # the amplitude sits with its square root on either placement, and the closing
                    # one also carries the Frobenius-Schur phase of the sector
                    pairs.append((sector, abs(overlap), 1.0,
                                  content(group).placement_phase(group, sector)
                                  * np.exp(1j * np.angle(overlap))))
            if not pairs:
                raise ValueError(f'The operators {operators} do not fuse to the sector {total}.')
            base = {sector: threads.get(sector, 0) for sector, _, _, _ in pairs}
            for sector, _, _, _ in pairs:
                threads[sector] = threads.get(sector, 0) + (last - first)
            placed.append((first, last, pairs, base))
        bond = bond_leg(group, [(group.trivial_sector, 2 + threads.get(group.trivial_sector, 0))]
                        + [(sector, threads[sector] * group.sector_dim(sector))
                           for sector in threads if sector != group.trivial_sector], 1)
        triv = multiplet_leg(group, group.trivial_sector)
        tensors = []
        for site in range(self.L):
            if site == 0:
                left, right = triv, bond.conj()
            elif site == self.L - 1:
                left, right = bond, triv.conj()
            else:
                left, right = bond, bond.conj()
            tensor = Tensor.zeros(left.dim, right.dim, phys_dim, phys_dim,
                                  legs=(left, right, phys, phys.conj()), qtotal=group.trivial_sector,
                                  dtype=self.dtype, device=self.device)
            for block, q in zip(tensor.blocks, tensor.qdata):
                left_block, right_block = cast(int, q[0]), cast(int, q[1])
                if left_block == 0 and right_block == 0:
                    # the rail carries the normalization of the site's physical multiplet
                    rail = np.sqrt(phys_dim)
                    if site == self.L - 1:
                        block[0, 0, 0, 0] = cast(Any, singles.get(site, 0.0))
                        block[1, 0, 0, 0] = rail
                    elif site == 0:
                        block[0, 0, 0, 0] = cast(Any, rail + singles.get(site, 0.0))
                    else:
                        block[0, 0, 0, 0] = cast(Any, rail + singles.get(site, 0.0))
                        block[1, 1, 0, 0] = rail
                # a trivial component is placed in this same block, after the two rail slots
                left_sector = tuple(left.sector(left_block, True))
                right_sector = tuple(right.sector(right_block, True))
                for first, last, pairs, base_of in placed:
                    if not first <= site <= last:
                        continue
                    span = last - first
                    for sector, coeff, left_value, right_value in pairs:
                        # the rails hold the first two slots of the trivial block, the components follow
                        base = (2 if sector == group.trivial_sector else 0) + base_of[sector]
                        # the tree carries one Clebsch-Gordan vertex per placement; each vertex is
                        # normalized to the dimension of the sector it couples to, the bond's and the site's
                        if site == first and left_sector == group.trivial_sector and right_sector == sector:
                            block[0, base + max(span - 1, 0), 0, 0] = np.sqrt(coeff) * left_value
                        if site == last and left_sector == sector and right_sector == group.trivial_sector:
                            end = 1 if site != self.L - 1 else 0
                            # the closing placement carries the conjugate component, whose phase is
                            # the Frobenius-Schur sign of the sector
                            block[base, end, 0, 0] = (np.sqrt(coeff) * right_value
                                                      * content(group).placement_phase(group, sector))
                        elif (first < site < last and left_sector == sector
                              and right_sector == sector
                              and (len(q) <= 4 or q[4] == group.trivial_sector)):
                            # only the path that pairs the two bond legs through the trivial
                            # sector carries the operator across the site; the other fusion
                            # paths would add the adjoint of the site to the operator
                            remaining = last - site
                            block[base + remaining, base + remaining - 1, 0, 0] = float(
                                np.sqrt(group.quantum_dim(sector) * phys_dim))
            tensors.append(tensor)
        return tensors

    def _build_labels(self) -> list[Tensor]:
        raw = {name: matrix.to(dtype=self.dtype, device=self.device)
               for name, matrix in self._buffers.items()}
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
                    # long-range states stay unique per coupling, so no shared identity segment
                    # can open a spurious path between two couplings
                    key = (ops[n], rel_pos) if rel_pos == 1 else (c_idx, ops[n], rel_pos)
                coupling_keys[c_idx].append(key)
                state_ranges.setdefault(key, set()).add((locs[n], locs[n + 1]))

        sorted_keys = sorted(state_ranges, key=lambda x: (len(x), x))
        state_idx = {k: i + 1 for i, k in enumerate(sorted_keys)}
        bond_dim = len(state_idx) + 2  # start (0) + end (bond_dim - 1)

        p = self.physical_dim
        group = self.group
        phys = phys_leg(group, p, self.site_sector)
        # the two edges of the chain are trivial, the bond carries the start and the end state
        bonds = [bond_leg(group, [(group.trivial_sector, dim)], 1)
                 for dim in [1] + [bond_dim] * (self.L - 1) + [1]]
        tensors = []
        for site in range(self.L):
            left, right = bonds[site], bonds[site + 1]
            tensors.append(Tensor.zeros(left.dim, right.dim, p, p,
                                        legs=(left, right, phys, phys.conj()),
                                        codomain=2, qtotal=group.trivial_sector,
                                        dtype=self.dtype, device=self.device))

        def place(tensor: Tensor, out_slot: int, in_slot: int, matrix: torch.Tensor) -> None:
            '''
            Add one dense site matrix at a pair of bond slots of a tensor.

            Args:
              tensor: one MPO tensor, its legs (left bond, right bond, physical, physical dual),
                  the two bond legs on the codomain.
              out_slot: the left bond slot the matrix leaves along.
              in_slot: the right bond slot the matrix enters along.
              matrix: the dense (physical output, physical input) matrix.
            '''
            block = tensor.blocks[0]
            block[out_slot * tensor.legs[1].dim + in_slot] += matrix.reshape(-1)

        eye = torch.eye(p, dtype=self.dtype, device=self.device)
        if self.L > 1:
            place(tensors[0], 0, 0, eye)                             # site-0 start
            place(tensors[self.L - 1], bond_dim - 1, 0, eye)         # right boundary
            for i in range(1, self.L - 1):
                place(tensors[i], 0, 0, eye)                         # bulk start
                place(tensors[i], bond_dim - 1, bond_dim - 1, eye)   # bulk end

        # identity on skipped sites for long-range couplings
        for key, ranges in state_ranges.items():
            rel_pos = key[-1]
            if rel_pos <= 1:
                continue
            s = state_idx[key]
            for from_site, to_site in ranges:
                for k in range(from_site + 1, to_site):
                    if 0 < k < self.L:
                        place(tensors[k], s, s, eye)

        # ── place operator matrices ──
        for c_idx, (ops, coeff, locs) in enumerate(self.couplings):
            keys = coupling_keys[c_idx]
            for n, (op_str, site) in enumerate(zip(ops, locs)):
                mat = raw[op_str]
                if n == 0:
                    mat = coeff * mat

                state_in = 0 if n == 0 else state_idx[keys[n - 1]]
                # the closing leg of the last site has a single slot, the bond's end state
                end = 0 if site == self.L - 1 else bond_dim - 1
                state_out = end if n == len(ops) - 1 else state_idx[keys[n]]

                place(tensors[site], state_in if site else 0, state_out, mat)
        return tensors

    def __sizeof__(self):
        return sum(t.numel() * t.element_size() for t in self.tensors)
