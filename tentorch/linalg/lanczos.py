from typing import Callable

import torch

from tentorch._ext import Tensor, inner, norm


def lanczos(action: Callable[[Tensor], Tensor],
            start: Tensor,
            n_iter: int = 5,
            tol: float = 1e-12) -> tuple[float, Tensor]:
    '''
    The lowest eigenpair of a Hermitian action on the block-sparse tensors, by the Lanczos iteration
    with full reorthogonalisation.

    Args:
      action: the map $v \\mapsto H v$, linear and Hermitian on the legs of `start`, returning the
        same legs.
      start: the starting vector, in any normalisation.
      n_iter: the number of Krylov steps.
      tol: the norm below which a Krylov step ends the iteration.
    Returns:
      The lowest eigenvalue of the Krylov space, and the Ritz vector, normalised.
    Raises:
      ValueError: when `n_iter` is below one, or when `start` has zero norm.
    '''
    if n_iter < 1:
        raise ValueError(f'Lanczos needs at least one step, not {n_iter}.')
    scale = norm(start)
    if scale == 0.0:
        raise ValueError('Lanczos needs a starting vector of non-zero norm.')

    basis = [start * (1.0 / scale)]
    alphas: list[float] = []
    betas: list[float] = []
    for step in range(n_iter):
        image = action(basis[-1])
        alphas.append(inner(basis[-1], image).real)
        image -= basis[-1] * alphas[-1]
        if step > 0:
            image -= basis[-2] * betas[-1]
        # full reorthogonalisation, twice, so that the Krylov basis stays orthonormal to round-off
        for _ in range(2):
            for vector in basis:
                image -= vector * inner(vector, image)
        beta = norm(image)
        if beta < tol:
            break
        betas.append(beta)
        basis.append(image * (1.0 / beta))

    size = len(alphas)
    matrix = torch.zeros((size, size), dtype = torch.float64, device = start.device)
    for step in range(size):
        matrix[step, step] = alphas[step]
    for step, beta in enumerate(betas[:size - 1]):
        matrix[step, step + 1] = beta
        matrix[step + 1, step] = beta
    values, vectors = torch.linalg.eigh(matrix)
    coefficients = vectors[:, 0]
    vector = basis[0] * float(coefficients[0])
    for step in range(1, size):
        vector += basis[step] * float(coefficients[step])
    return float(values[0]), vector * (1.0 / norm(vector))
