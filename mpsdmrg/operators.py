'''The single-site spin operators of a chain, as the ``SpinOperator`` bundle the models use.'''
from math import sqrt

import torch

from base import COMPLEX, CUDA
from mpsdmrg.tensor import Tensor


class SpinOperator:
    '''
    The three spin components of one site, as one dense tensor of shape ``(3, d, d)``.

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
        if self.phys_dim % 2 == 0:
            return f"Spin {self.phys_dim//2} operator"
        else:
            return f"Spin {self.phys_dim}/2 operator"
