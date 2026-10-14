'''The single-site spin operators of a chain, as the ``SpinOperator`` bundle the models use.'''
from math import sqrt

import torch

from base import COMPLEX, CUDA


class SpinOperator:
    '''
    A tensor representing spin operators (Sx, Sy, Sz) for a given physical dimension.
    '''
    def __init__(self, phys_dim:int, dtype = COMPLEX, device = CUDA):
        self.data = torch.zeros((3, phys_dim, phys_dim),
                                 dtype=dtype,
                                 device=device,
                                 requires_grad=False)
        # Sz
        s = (phys_dim - 1) / 2
        for i in range(phys_dim):
            self.data[2, i, i] = s - i
            m = s - i
            if i == phys_dim - 1:
                break
            self.data[0, i   ,i+1] = sqrt(s * (s + 1) - m * (m - 1)) # S+
            self.data[1, i + 1, i] = sqrt(s * (s + 1) - m * (m - 1)) # S-

        self.data[0],self.data[1] = 0.5 * (self.data[0] + self.data[1]), -0.5j * (self.data[0] - self.data[1])# Sx, Sy

        self.phys_dim = phys_dim

    def __repr__(self):
        if self.phys_dim %2 == 0:
            return f"Spin {self.phys_dim//2} operator \n {self.data}"
        else:
            return f"Spin {self.phys_dim}/2 operator \n {self.data}"
