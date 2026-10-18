from math import sqrt

import torch

from base import COMPLEX, CUDA
from mpsdmrg.ext import FusionCategory, su2_symmetry
from mpsdmrg.tensor import Tensor


class SpinOperator:
    '''
    The three spin components of one site, as one dense tensor of shape ``(3, d, d)``.

    Entry 0, 1, 2 are :math:`S_x`, :math:`S_y`, :math:`S_z` with the spin normalization
    :math:`S_z = \\mathrm{diag}(S, S - 1, \\ldots, -S)`, so a coupling built from these operators
    is the spin Hamiltonian, not the Pauli one.

    Args:
        phys_dim: physical dimension of the site, :math:`2S + 1`.
        dtype, device: data type and device of the operators.
    '''
    def __init__(self, phys_dim:int, dtype = COMPLEX, device = CUDA):
        buffer = torch.zeros((3, phys_dim, phys_dim), dtype=dtype, device=device)
        # Sz
        s = (phys_dim - 1) / 2
        for i in range(phys_dim):
            buffer[2, i, i] = s - i
            m = s - i
            if i == phys_dim - 1:
                break
            buffer[0, i   ,i+1] = sqrt(s * (s + 1) - m * (m - 1)) # S+
            buffer[1, i + 1, i] = sqrt(s * (s + 1) - m * (m - 1)) # S-
        buffer[0], buffer[1] = 0.5 * (buffer[0] + buffer[1]), -0.5j * (buffer[0] - buffer[1]) # Sx, Sy

        self.data = Tensor.from_torch(buffer)
        self.phys_dim = phys_dim

    def __repr__(self):
        '''Name the spin the operators describe, :math:`S = (d - 1) / 2`.'''
        half = self.phys_dim - 1
        if half % 2 == 0:
            return f"Spin {half // 2} operator"
        return f"Spin {half}/2 operator"


def wigner_eckart_factor(group:FusionCategory, sector:tuple[int, ...], member:int = 0) -> float:
    '''
    The factor between the component and the invariant value of a two-site tensor-operator
    correlator on a state of total charge zero.

    A state of the reduced representation is a multiplet: it carries no definite magnetic member,
    so a single component of an operator is not one of its observables.  Wigner and Eckart relate
    the two, :math:`\\langle T_m(i) T_{-m}(j)\\rangle = f \\sum_{m'} (-1)^{m'}
    \\langle T_{m'}(i) T_{-m'}(j)\\rangle`, and for SU(2) the factor of the member ``member`` is
    :math:`(-1)^m / (2k + 1)`, the reciprocal dimension of the multiplet.

    Args:
        group: symmetry group of the chain.
        sector: the multiplet the operator transforms in, e.g. ``(2,)`` for a spin-1 operator.
        member: the magnetic member ``m`` of the component, 0 for the :math:`S_z`-like one.
    Returns:
        float: the factor f.
    Raises:
        NotImplementedError: for a group other than SU(2), whose Clebsch-Gordan data would have
            to give the ratio of the coupling coefficients.
    '''
    if not group.is_same_symmetry(su2_symmetry):
        raise NotImplementedError(f'The Wigner-Eckart factor of {group.name} needs its '
                                  f'Clebsch-Gordan data; only the closed SU(2) form is written.')
    return (-1.0) ** member / group.quantum_dim(sector)
