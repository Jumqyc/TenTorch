'''
The exact diagonalisation of a short Heisenberg chain, to check a sweep against.

The operator is dense in one magnetisation sector, so only the lowest eigenvalue is asked for.
'''
from __future__ import annotations

import itertools

import torch

from tentorch.models.model import Model


def exact_energy(model: Model) -> float:
    '''
    The lowest eigenvalue of the Heisenberg chain, by dense diagonalisation.

    Args:
      model: the model, of even length and of trivial charge, so that the sector it names is the
        one of zero magnetisation.
    Returns:
      The ground state energy, from `torch.linalg.eigvalsh` of the sector matrix.
    Raises:
      ValueError: when the length is odd, the charge is not the trivial one, or the coupling has an
        imaginary part.
    '''
    length = model.length
    if length % 2:
        raise ValueError(f'a chain of {length} sites has no sector of zero magnetisation')
    if model.sector != model.cat.trivial():
        raise ValueError(f'the exact energy is written for the trivial charge, not {model.sector}')
    coupling = complex(model.coupling)
    if coupling.imag:
        raise ValueError(f'the exact energy is written for a real coupling, not {coupling}')

    states = [bits for bits in itertools.product((0, 1), repeat = length) if sum(bits) == length // 2]
    index = {bits: at for at, bits in enumerate(states)}
    matrix = torch.zeros(
        (len(states), len(states)),
        dtype = torch.float64,
        device = torch.device(model.cat.device))
    for at, bits in enumerate(states):
        for site in range(length - 1):
            # $\vec S_i \cdot \vec S_{i+1}$: the $S^z S^z$ term is diagonal, the flip term moves one
            # unit of spin across the bond whenever the two sites differ
            if bits[site] == bits[site + 1]:
                matrix[at, at] += 0.25 * coupling.real
            else:
                matrix[at, at] -= 0.25 * coupling.real
                flipped = list(bits)
                flipped[site], flipped[site + 1] = flipped[site + 1], flipped[site]
                matrix[at, index[tuple(flipped)]] += 0.5 * coupling.real
    return float(torch.linalg.eigvalsh(matrix)[0])
