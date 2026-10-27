'''
Time the two site DMRG sweep on the Heisenberg chain, at the size the caller asks for.

One line per sweep: the wall clock of the sweep, the energy it leaves, the weight the split threw
away and the bond it reached.  The device is the one the category carries, which is CUDA when torch
sees a device and the CPU otherwise; `TENTORCH_DEVICE=cpu` forces the CPU, which is faster than the
GPU while the bond stays small.

    .venv/bin/python -u bench/bench_dmrg.py 16 32 3
    TENTORCH_DEVICE=cpu .venv/bin/python -u bench/bench_dmrg.py 32 64 4 --cat nosym
'''

import argparse
import time

import torch

from tentorch import Model, no_symmetry, u1
from tentorch.algorithms import DMRG

CATEGORIES = {'u1': u1, 'nosym': no_symmetry}


def main() -> None:
    '''Sweep the chain the caller asks for and print one line per sweep.'''
    parser = argparse.ArgumentParser(description = __doc__,
                                     formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('length', type = int, nargs = '?', default = 16, help = 'the number of sites')
    parser.add_argument('bond', type = int, nargs = '?', default = 32, help = 'the multiplets kept')
    parser.add_argument('sweeps', type = int, nargs = '?', default = 3, help = 'the sweeps to run')
    parser.add_argument('--cat', choices = sorted(CATEGORIES), default = 'u1',
                        help = 'the symmetry of the chain')
    parser.add_argument('--iterations', type = int, default = 5,
                        help = 'the Krylov steps of every window')
    arguments = parser.parse_args()

    model = Model(arguments.length, cat = CATEGORIES[arguments.cat])
    driver = DMRG(model, max_bond_dim = arguments.bond, n_iter = arguments.iterations)
    print(f'L={arguments.length} bond={arguments.bond} cat={arguments.cat} '
          f'device={model.cat.device} threads={torch.get_num_threads()}')
    total = 0.0
    for sweep in range(1, arguments.sweeps + 1):
        start = time.perf_counter()
        energy, discarded = driver.sweep()
        spend = time.perf_counter() - start
        total += spend
        print(f'sweep {sweep}: {spend:6.2f} s energy = {energy:.12f} '
              f'energy/site = {energy / arguments.length:.12f} truncated = {discarded:.2e} '
              f'bond = {max(driver.state.bond_dims)}')
    print(f'total {total:.2f} s, {total / arguments.sweeps:.2f} s per sweep')


if __name__ == '__main__':
    main()
