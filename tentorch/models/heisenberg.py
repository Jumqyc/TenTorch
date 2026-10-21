'''The Heisenberg chain: the isotropic spin exchange on every bond.'''
from tentorch.config import COMPLEX, CUDA
from tentorch.model import Model
from tentorch.sites import spin
from tentorch.symmetry import FusionCategory, su2_symmetry


def heisenberg(L: int,
               J: float = 1.0,
               S: float = 0.5,
               cat: FusionCategory = su2_symmetry,
               dtype = COMPLEX,
               device = CUDA) -> Model:
    r'''
    :math:`H = J \sum_{i=0}^{L-2} \vec{S}_i \cdot \vec{S}_{i+1}` on an open chain.

    Args:
        L: number of sites.
        J: exchange coupling; positive is antiferromagnetic.
        S: the spin of one site.
        cat: the fusion category the chain transforms under.
        dtype, device: data type and device of the tensors.
    Returns:
        The model, with one coupling per component and bond.
    '''
    model = Model(L, spin(S, cat, dtype = dtype, device = device),
                  dtype = dtype, device = device)
    for i in range(L - 1):
        for component in 'XYZ':
            model.add_couplings((component, component), J, (i, i + 1))
    return model
