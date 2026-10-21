'''Exact diagonalization of a model in its full Hilbert space: the reference a sweep is checked
against on chains small enough to hold every state.'''
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
from numpy.typing import NDArray

from tentorch.model import Model


def _operator(model: Model, name: str) -> NDArray[np.complex128]:
    '''
    The dense matrix of one operator name of a coupling.
    Args:
        model: the model whose site names the operators.
        name: an operator name, or several letters for a product of site operators.
    Returns:
        The matrix, of shape ``(physical_dim, physical_dim)``.
    '''
    site = model.site
    if name in site.operators:
        return site.matrix(name).cpu().numpy()
    product = np.eye(model.physical_dim, dtype = np.complex128)
    for letter in name:
        product = product @ site.matrix(letter).cpu().numpy()
    return product


def exact_energy(model: Model, k: int = 1) -> float | NDArray[np.float64]:
    '''
    The lowest eigenvalues of the Hamiltonian of a model, in the full Hilbert space.

    The Hamiltonian is assembled as a sparse Kronecker sum of the couplings and its lowest
    eigenpairs come from Lanczos on the sparse matrix.
    Args:
        model: the chain and its Hamiltonian.
        k: how many eigenvalues to return, the lowest first.
    Returns:
        The lowest eigenvalue, or all ``k`` of them when ``k > 1``.
    Raises:
        ValueError: If a coupling names an operator the site does not have, or a location
            outside the chain.
    '''
    L, d = model.L, model.physical_dim
    dim = d ** L
    identity = sparse.identity(d, format = 'csr', dtype = np.complex128)
    hamiltonian = sparse.csr_matrix((dim, dim), dtype = np.complex128)

    for operators, coeff, locations in model.couplings:
        if len(operators) != len(locations):
            raise ValueError(f'Coupling {operators} acts on {len(locations)} sites.')
        if any(location < 0 or location >= L for location in locations):
            raise ValueError(f'Coupling {operators} acts outside the chain of {L} sites.')
        term = sparse.identity(1, format = 'csr', dtype = np.complex128)
        for site in range(L):
            if site in locations:
                term = sparse.kron(term, sparse.csr_matrix(_operator(model, operators[locations.index(site)])),
                                   format = 'csr')
            else:
                term = sparse.kron(term, identity, format = 'csr')
        hamiltonian = hamiltonian + coeff * term

    low = sparse_linalg.eigsh(hamiltonian, k = k, which = 'SA', return_eigenvectors = False)
    low = np.sort(low)
    return float(low[0]) if k == 1 else low
