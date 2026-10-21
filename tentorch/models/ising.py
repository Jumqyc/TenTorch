'''The transverse-field Ising chain: a ferromagnetic coupling on z and a field on x.'''
from tentorch.config import COMPLEX, CUDA
from tentorch.model import Model
from tentorch.sites import spin
from tentorch.symmetry import FusionCategory, no_symmetry


def ising(L: int,
          J: float = 1.0,
          h: float = 1.0,
          cat: FusionCategory = no_symmetry,
          dtype = COMPLEX,
          device = CUDA) -> Model:
    r'''
    :math:`H = -J \sum_{i=0}^{L-2} Z_i Z_{i+1} - h \sum_{i=0}^{L-1} X_i` on an open chain.

    The chain is critical at ``J == h``, and its ground energy per site there approaches
    :math:`-4 / \pi` from below in the length.

    Args:
        L: number of sites.
        J: coupling of the two neighbouring spins.
        h: the transverse field.
        cat: the fusion category the chain transforms under.
        dtype, device: data type and device of the tensors.
    Returns:
        The model, one coupling per bond and one per field term.
    '''
    model = Model(L, spin(0.5, cat, dtype = dtype, device = device),
                  dtype = dtype, device = device)
    for i in range(L - 1):
        model.add_couplings(('Z', 'Z'), -J, (i, i + 1))
    for i in range(L):
        model.add_couplings('X', -h, i)
    return model
