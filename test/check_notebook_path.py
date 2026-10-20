"""Scratch check that the engine runs the calculation path the notebooks use."""
from mpsdmrg import Broomstick, MPO, MPS, SpinOperator

# the Heisenberg notebook path: Broomstick(L, phys_dim) + named couplings + build + sweep
stick = Broomstick(L=6, phys_dim=2, max_bond_dim=32, svd_tol=1e-16)
for i in range(5):
    for op in ('X', 'Y', 'Z'):
        stick.add_couplings((op, op), 1.0, (i, i + 1))
stick.build()
stick.initialize_state([1, 2, 4, 8, 4, 2, 1])
stick.sweep(num_sweeps=4)
print('notebook heisenberg energy', stick.energy)

# the (MPS, MPO) form the notebooks also use, on a spin-1 chain
mps = MPS(4, 3)
mpo = MPO(4, phys_dim=3)
stick3 = Broomstick(mps, mpo, max_bond_dim=16, svd_tol=1e-16)
stick3.add_couplings(('Z', 'Z'), 1.0, (0, 1))
stick3.build()
stick3.sweep(num_sweeps=2)
print('notebook spin-1 energy', stick3.energy)
print('NOTEBOOK PATH OK')
