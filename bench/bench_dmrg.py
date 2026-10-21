'''Time the DMRG sweep against the bond dimension: one line per run.'''
from __future__ import annotations

import argparse
import time

import torch

from tentorch import dmrg, models
from tentorch.algorithms.exact import exact_energy


def main() -> None:
    '''Run the sweeps of one model at a range of bond caps and print the measurements.'''
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--model', default = 'heisenberg', choices = ['heisenberg', 'ising', 'aklt', 'uls'])
    parser.add_argument('--L', type = int, default = 20)
    parser.add_argument('--max-bond', type = int, nargs = '+', default = [16, 32, 64])
    parser.add_argument('--sweeps', type = int, default = 5)
    parser.add_argument('--exact', action = 'store_true', help = 'compare with exact diagonalization')
    arguments = parser.parse_args()

    maker = getattr(models, arguments.model)
    for max_bond in arguments.max_bond:
        model = maker(arguments.L)
        start = time.perf_counter()
        result = dmrg(model, max_bond_dim = max_bond, num_sweeps = arguments.sweeps, verbose = False)
        wall = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
        line = (f'{arguments.model:<12} L={arguments.L:<4} bond={max_bond:<5} '
                f'sweeps={result.sweeps:<3} E/L={result.energy / arguments.L:.12f} '
                f'time={wall:7.2f}s ({wall / result.sweeps:5.2f}s/sweep) '
                f'peak={peak:.2f}GiB')
        if arguments.exact:
            line += f' exact={exact_energy(model) / arguments.L:.12f}'
        print(line)
        torch.cuda.reset_peak_memory_stats()


if __name__ == '__main__':
    main()
