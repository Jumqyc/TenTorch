"""The covariance-matrix algebra of a Gaussian fermion state: the symplectic form, the pure-state
covariance, the Williamson normal form and the expectation values it makes one line.

The CUDA kernels behind ``pfaffian`` live in ``cuda/``; they are built on first use.
"""
from tentorch.linalg.freefermion.covariance import (
    RandPureCov,
    entropy,
    hamiltonian,
    is_pure,
    occupations,
    purity,
    symplectic_form,
    wick,
    williamson,
)
from tentorch.linalg.freefermion.cuda.fermion import pfaffian

__all__ = ['RandPureCov', 'entropy', 'hamiltonian', 'is_pure', 'occupations', 'pfaffian', 'purity',
           'symplectic_form', 'wick', 'williamson']
