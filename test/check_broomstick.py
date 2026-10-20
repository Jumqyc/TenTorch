"""Scratch check: the Broomstick-first usage — Hamiltonian through the engine, state filled later."""
import torch

from mpsdmrg import Broomstick, MPO, MPS

L, PHYS, BOND = 8, 2, 4
COUPLINGS = [(op, 1.0, (i, i + 1)) for i in range(L - 1) for op in ('X', 'Y', 'Z')]

# old style: the MPO and the MPS built outside the engine
torch.manual_seed(0)
mpo = MPO(L, phys_dim=PHYS)
for op, coeff, sites in COUPLINGS:
    mpo.add_couplings((op, op), coeff, sites)
mpo.build()
mps = MPS(L, phys_dim=PHYS, bond_dims=[1] + [BOND] * (L - 1) + [1])
reference = Broomstick(mps, mpo, max_bond_dim=BOND, svd_tol=1e-14)

# new style: the engine first, the state later
torch.manual_seed(0)
stick = Broomstick(L=L, phys_dim=PHYS, max_bond_dim=BOND, svd_tol=1e-14)
for op, coeff, sites in COUPLINGS:
    stick.add_couplings((op, op), coeff, sites)
stick.build()
print('state before fill', stick.state)
try:
    stick.sweep(num_sweeps=1)
except ValueError as exc:
    print('sweep without state ->', exc)
stick.initialize_state(bond_dims=[1] + [BOND] * (L - 1) + [1])

print('hamiltonian equal', bool(all(torch.equal(a.blocks[0], b.blocks[0]) for a, b in
                                    zip(stick.Hamiltonian.tensors, reference.Hamiltonian.tensors))))
print('state equal      ', bool(all(torch.equal(a.blocks[0], b.blocks[0]) for a, b in
                                    zip(stick.state.tensors, reference.state.tensors))))
reference.sweep(num_sweeps=1)
stick.sweep(num_sweeps=1)
print('energy reference ', f'{reference.energy:.12f}')
print('energy engine    ', f'{stick.energy:.12f}')
print('energies equal   ', bool(abs(reference.energy - stick.energy) < 1e-12))
print('copy ok          ', type(stick.copy()).__name__, stick.copy().state is not None)
print('BROOMSTICK CHECK DONE')
