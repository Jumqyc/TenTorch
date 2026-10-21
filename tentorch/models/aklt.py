'''The Affleck-Kennedy-Lieb-Tasaki chain: the spin-1 bilinear-biquadratic point whose ground
state is the valence-bond solid.'''
from tentorch.config import COMPLEX, CUDA
from tentorch.model import Model
from tentorch.sites import spin
from tentorch.symmetry import FusionCategory, no_symmetry
from tentorch.symmetry.cat import has_content


def aklt(L: int,
         J: float = 1.0,
         cat: FusionCategory = no_symmetry,
         dtype = COMPLEX,
         device = CUDA) -> Model:
    r'''
    :math:`H = J \sum_{i=0}^{L-2} \left[ \vec{S}_i \cdot \vec{S}_{i+1}
    + \frac{1}{3} \left( \vec{S}_i \cdot \vec{S}_{i+1} \right)^2 \right]` on an open chain of
    spin 1.

    The ground state is the valence-bond solid, of energy exactly :math:`-2/3` per bond; the
    Hamiltonian is the parent one of the AKLT point, whose every bond leaves the spin-2 channel
    with the same shift.

    The biquadratic term is a product of two site operators, which the label builder places and
    the symmetry-adapted builder cannot, so the chain runs without a category.

    Args:
        L: number of sites.
        J: overall coupling; positive is antiferromagnetic.
        cat: the fusion category the chain transforms under; only ``no_symmetry`` is built.
        dtype, device: data type and device of the tensors.
    Returns:
        The model, with the bilinear and the biquadratic term of every bond.
    Raises:
        NotImplementedError: If ``cat`` carries a multiplet content, which cannot place a product
            of two site operators.
    '''
    if has_content(cat):
        raise NotImplementedError(
            f'The biquadratic term of AKLT is a product of two site operators, which {cat.name} '
            f'cannot place on its invariant channels; build the chain with cat=no_symmetry.')
    model = Model(L, spin(1, cat, dtype = dtype, device = device), dtype = dtype, device = device)
    for i in range(L - 1):
        for first in 'XYZ':
            model.add_couplings((first, first), J, (i, i + 1))
            for second in 'XYZ':
                model.add_couplings((first + second, first + second), J / 3, (i, i + 1))
    return model
