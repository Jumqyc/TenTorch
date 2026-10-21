'''The SU(3) chain: the permutation of two neighbouring fundamental sites, written as the
Casimir-weighted sum of the Gell-Mann generators.'''
from tentorch.config import COMPLEX, CUDA
from tentorch.model import Model
from tentorch.sites import fundamental_su3


def uls(L: int,
        J: float = 1.0,
        dtype = COMPLEX,
        device = CUDA) -> Model:
    r'''
    :math:`H = J \sum_{i=0}^{L-2} P_{i, i+1}` with
    :math:`P = \frac{1}{3} + \frac{1}{2} \sum_{a=1}^{8} L_a \otimes L_a`, the permutation of two
    neighbouring fundamental representations of SU(3), on an open chain.

    Args:
        L: number of sites, each of dimension 3.
        J: overall coupling; positive is antiferromagnetic.
        dtype, device: data type and device of the tensors.
    Returns:
        The model, with the identity and the eight generator couplings of every bond.
    '''
    model = Model(L, fundamental_su3(dtype = dtype, device = device),
                  dtype = dtype, device = device)
    for i in range(L - 1):
        model.add_couplings(('I', 'I'), J / 3, (i, i + 1))
        for generator in range(1, 9):
            model.add_couplings((f'L{generator}', f'L{generator}'), J / 2, (i, i + 1))
    return model
