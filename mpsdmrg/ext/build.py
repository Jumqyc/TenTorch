'''
Build the C++ symmetry engine of ``mpsdmrg``.

    PYTHONPATH=. .venv/bin/python -u mpsdmrg/ext/build.py [--jobs 1] [--force]

The compiled module is written to ``mpsdmrg/ext/build/symmetry_ext.so``, which
``mpsdmrg/ext/__init__.py`` imports directly.  Ninja drives the build
(``python -m pip install ninja``).  ``pch.h`` is precompiled first and every C++ unit includes
it, which cuts a unit from ~22 s / 1.9 GiB to ~16 s / 1.8 GiB; the units are then compiled
``--jobs`` at a time (one by default) and linked once.  A parallel build of these
template-heavy units peaks near 13 GiB and has frozen this 7.6 GiB box, so the script refuses
to start when the host memory is short and never exceeds two jobs.
'''
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(HERE, 'build')
LIBRARY = os.path.join(BUILD_DIR, 'symmetry_ext.so')
SOURCES = (os.path.join(HERE, 'bindings.cpp'),
           os.path.join(HERE, '..', 'symmetry', 'group', 'group.cpp'),
           os.path.join(HERE, '..', 'symmetry', 'group', 'abelian.cpp'),
           os.path.join(HERE, '..', 'symmetry', 'group', 'su2.cpp'),
           os.path.join(HERE, '..', 'symmetry', 'group', 'su3.cpp'),
           os.path.join(HERE, '..', 'symmetry', 'tree', 'tree.cpp'),
           os.path.join(HERE, '..', 'tensor', 'tensor.cpp'),
           os.path.join(HERE, '..', 'linalg', 'linalg.cpp'))
HEADERS = tuple(sorted(glob.glob(os.path.join(HERE, '..', '**', '*.h'), recursive=True)))
PCH = os.path.join(HERE, 'pch.h')
PCH_OUTPUT = PCH + '.gch'
GIB_PER_JOB = 2.5


def compiler_flags() -> list[str]:
    '''The include and ABI flags torch uses for an extension, so the precompiled header matches.'''
    import sysconfig

    from torch.utils.cpp_extension import include_paths

    includes = include_paths(device_type='cuda') + [sysconfig.get_paths()['include']]
    return ['-DTORCH_EXTENSION_NAME=symmetry_ext', '-DTORCH_API_INCLUDE_EXTENSION_H',
            *[f'-isystem{path}' for path in includes], '-fPIC', '-std=c++20', '-O3']


def build_pch(verbose: bool) -> None:
    '''Precompile ``pch.h`` unless it is newer than the headers it pulls in; every .cpp includes it.'''
    newest = max(os.path.getmtime(path) for path in (PCH, *HEADERS))
    if os.path.exists(PCH_OUTPUT) and os.path.getmtime(PCH_OUTPUT) > newest:
        return
    command = ['c++', *compiler_flags(), '-x', 'c++-header', PCH, '-o', PCH_OUTPUT]
    if verbose:
        print(' '.join(command), flush=True)
    subprocess.run(command, check=True)
    print(f'precompiled {os.path.basename(PCH)}', flush=True)


def available_gib() -> float:
    '''Host memory a new process may claim, in GiB, from ``MemAvailable``.'''
    with open('/proc/meminfo') as meminfo:
        for line in meminfo:
            if line.startswith('MemAvailable:'):
                return float(line.split()[1]) / 1024 ** 2
    return float('inf')


def clear_stale_lock(verbose: bool) -> None:
    '''
    Drop torch's build lock when the build that held it is gone.

    ``torch.utils.cpp_extension.load`` waits forever on ``build/lock``, so one interrupted build
    blocks every later one.  The lock is stale when it is older than half a minute and no
    compiler is working on a source of the engine, which lives in four directories.

    Args:
      verbose: print when a stale lock is removed.
    '''
    path = os.path.join(BUILD_DIR, 'lock')
    if not os.path.exists(path) or time.time() - os.path.getmtime(path) < 30:
        return
    pattern = rf'{os.path.dirname(HERE)}/\S+\.(cpp|cu)'
    running = subprocess.run(['pgrep', '-f', pattern], capture_output=True)
    if not running.stdout.strip():
        if verbose:
            print(f'removing the stale build lock {path}', flush=True)
        os.remove(path)


def main() -> int:
    '''Precompile the shared header, compile the units with at most ``--jobs`` at once, link.'''
    parser = argparse.ArgumentParser(description='build the mpsdmrg symmetry engine')
    parser.add_argument('--verbose', action='store_true', help='show the compiler output')
    parser.add_argument('--force', action='store_true', help='rebuild every unit')
    parser.add_argument('--jobs', type=int, default=1, help='units compiled at once, 1 or 2')
    arguments = parser.parse_args()

    free = available_gib()
    needed = GIB_PER_JOB * arguments.jobs
    if free < needed:
        print(f'{free:.1f} GiB of host memory available but {arguments.jobs} job(s) need '
              f'{needed:.1f} GiB; free some memory or lower --jobs', file=sys.stderr)
        return 1
    if arguments.jobs > 2:
        print('at most 2 jobs: a unit peaks near 2.5 GiB', file=sys.stderr)
        return 1

    os.makedirs(BUILD_DIR, exist_ok=True)
    # a bare .venv/bin/python does not put the environment's ninja on PATH
    os.environ['PATH'] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get('PATH', '')
    os.environ['MAX_JOBS'] = str(arguments.jobs)
    os.environ.setdefault('TORCH_EXTENSIONS_DIR', BUILD_DIR)
    if arguments.force:
        for path in (*glob.glob(os.path.join(BUILD_DIR, '*.o')), LIBRARY):
            if os.path.exists(path):
                os.remove(path)
    build_pch(arguments.verbose)
    if not os.path.exists(LIBRARY) and shutil.which('ninja') is None:
        print('ninja is required for the first build: python -m pip install ninja', file=sys.stderr)
        return 1

    from torch.utils.cpp_extension import load

    module: Any = load(
        name = 'symmetry_ext',
        sources = list(SOURCES),
        extra_cflags = ['-O3', '-include', PCH],
        build_directory = BUILD_DIR,
        verbose = arguments.verbose)
    print('built:', module.__file__, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
