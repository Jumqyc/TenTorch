'''The local Hilbert space of one site: its dimension, its category and the operators a
Hamiltonian on it is written with.

``Site`` is the table a ``Model`` names its couplings from; ``spin`` and ``fundamental_su3``
build the two the examples use.
'''
import torch

from tentorch.config import COMPLEX, CUDA
from tentorch.operators import SpinOperator
from tentorch.symmetry import FusionCategory, fundamental_operators, no_symmetry, su3_symmetry
from tentorch.symmetry.cat import has_content
from tentorch.symmetry.cat.su3 import FUNDAMENTAL_SECTOR


class Site:
    '''
    One site of a chain: a local dimension, a symmetry and a table of named operators.

    Args:
        phys_dim: dimension of the local Hilbert space.
        cat: the fusion category the site transforms under.
        sector: multiplet the site lives in, e.g. ``(1,)`` for a spin-1/2 site of SU(2);
            ``None`` for a category whose multiplets the dimension tells apart.
        operators: operator name to its dense matrix, e.g. ``'X'`` to :math:`S_x`.
    Raises:
        ValueError: If an operator matrix is not ``phys_dim`` by ``phys_dim``, or there is no
            ``'I'`` entry.
    '''
    def __init__(self,
                 phys_dim: int,
                 cat: FusionCategory,
                 sector: tuple[int, ...] | None,
                 operators: dict[str, torch.Tensor]):
        if 'I' not in operators:
            raise ValueError("A site names its identity operator 'I'.")
        for name, matrix in operators.items():
            if tuple(matrix.shape) != (phys_dim, phys_dim):
                raise ValueError(f'Operator {name} is {matrix.shape}, not '
                                 f'({phys_dim}, {phys_dim}).')
        self.phys_dim = int(phys_dim)
        self.cat = cat
        self.sector = None if sector is None else tuple(int(entry) for entry in sector)
        self.operators = dict(operators)

    def matrix(self, name: str) -> torch.Tensor:
        '''
        The dense matrix of one named operator.
        Args:
            name: an operator name of this site.
        Returns:
            The matrix, of shape ``(phys_dim, phys_dim)``.
        Raises:
            KeyError: If the site has no operator of that name.
        '''
        return self.operators[name]

    def __repr__(self) -> str:
        '''The dimension, the category and the operator names.'''
        return (f'Site(phys_dim = {self.phys_dim}, cat = {self.cat.name}, '
                f'sector = {self.sector}, operators = {sorted(self.operators)})')


def spin(S: float,
         cat: FusionCategory = no_symmetry,
         dtype = COMPLEX,
         device = CUDA) -> Site:
    r'''
    A spin-``S`` site whose ``X``, ``Y`` and ``Z`` are the three spin components, normalized as
    angular momentum (:math:`S_z = \mathrm{diag}(S, S - 1, \ldots, -S)`).
    Args:
        S: the spin, an integer or half-integer; :math:`d = 2S + 1`.
        cat: the category of the chain; an SU(2) site carries the multiplet ``(2S + 1,)``.
        dtype, device: data type and device of the operator matrices.
    Returns:
        The site, with the operators ``'X'``, ``'Y'``, ``'Z'`` and ``'I'``.
    Raises:
        ValueError: If ``S`` is negative.
    '''
    if S < 0:
        raise ValueError(f'A spin cannot be negative, got {S}.')
    phys_dim = int(round(2 * S)) + 1
    components = SpinOperator(phys_dim, dtype = dtype, device = device).data
    operators = {name: component for name, component in zip('XYZ', components)}
    operators['I'] = torch.eye(phys_dim, dtype = dtype, device = device)
    # an SU(2) multiplet is named by twice its spin, which is one less than its dimension
    return Site(phys_dim, cat, (phys_dim - 1,) if has_content(cat) else None, operators)


def fundamental_su3(dtype = COMPLEX, device = CUDA) -> Site:
    '''
    The fundamental representation of SU(3), whose operators are the eight Gell-Mann generators
    ``'L1'`` to ``'L8'`` and the identity.
    Args:
        dtype, device: data type and device of the operator matrices.
    Returns:
        The site, of dimension 3 and multiplet ``(1, 0)``.
    '''
    matrices = fundamental_operators(dtype = dtype, device = device)
    operators = dict(matrices)
    return Site(3, su3_symmetry, FUNDAMENTAL_SECTOR, operators)
