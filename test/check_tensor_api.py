'''
The public API of the engine: everything a user receives is a Tensor, a package class or a
plain value, never a torch object.
'''
import torch

from base import COMPLEX, CUDA
from mpsdmrg import Broomstick, MPO, MPS, SpinOperator
from mpsdmrg.symmetry.cat import su2_symmetry
from mpsdmrg.tensor import Tensor


def main() -> None:
    '''Run a small symmetric DMRG through the public API and audit every value it returns.'''
    spin = SpinOperator(2)
    assert isinstance(spin.data, Tensor), type(spin.data)
    mapping = {name: spin.data[index] for index, name in enumerate('XYZ')}
    mapping['I'] = Tensor.from_torch(torch.eye(2, dtype=COMPLEX, device=CUDA))

    stick = Broomstick(L=4, phys_dim=2, max_bond_dim=4, svd_tol=1e-14,
                       group=su2_symmetry, mapping=mapping, site_sector=(1,))
    for i in range(3):
        for name in 'XYZ':
            stick.add_couplings((name, name), 1.0, (i, i + 1))
    stick.build()
    assert isinstance(stick.Hamiltonian, MPO), type(stick.Hamiltonian)
    assert all(isinstance(matrix, Tensor) for matrix in stick.Hamiltonian.mapping.values())

    stick.initialize_state([1, 2, 2, 2, 1])
    energy, truncation = stick.singlesweep()
    assert isinstance(energy, float) and isinstance(truncation, float), (type(energy), type(truncation))
    assert isinstance(stick.energy, float), type(stick.energy)

    state = stick.state
    assert isinstance(state, MPS), type(state)
    assert all(type(tensor) is Tensor for tensor in state.tensors), [type(t) for t in state.tensors]
    assert not isinstance(state.tensors, torch.Tensor)
    print('energy', energy, 'truncation', truncation, 'bond dims', state.bond_dim)
    print('TENSOR API CHECK OK')


if __name__ == '__main__':
    main()
