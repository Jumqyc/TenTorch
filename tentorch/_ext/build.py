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
           os.path.join(HERE, '..', 'symmetry', 'cat', 'cat.cpp'),
           os.path.join(HERE, '..', 'symmetry', 'cat', 'group.cpp'),
           os.path.join(HERE, '..', 'symmetry', 'cat', 'abelian', 'abelian.cpp'),
           os.path.join(HERE, '..', 'symmetry', 'cat', 'fibonacci', 'fibonacci.cpp'),
           os.path.join(HERE, '..', 'symmetry', 'cat', 'su2', 'su2.cpp'),
           os.path.join(HERE, '..', 'symmetry', 'cat', 'su3', 'su3.cpp'),
           os.path.join(HERE, '..', 'symmetry', 'cat', 'symbol.cpp'),
           os.path.join(HERE, '..', 'symmetry', 'tree', 'tree.cpp'),
           os.path.join(HERE, '..', 'tensors', 'diagram.cpp'),
           os.path.join(HERE, '..', 'tensors', 'tensor.cpp'),
           os.path.join(HERE, '..', 'linalg', 'linalg.cpp'))
HEADERS = tuple(sorted(glob.glob(os.path.join(HERE, '..', '**', '*.h'), recursive = True)))
PCH = os.path.join(HERE, 'pch.h')
PCH_OUTPUT = PCH + '.gch'
GIB_PER_JOB = 2.5
# the programs a build of the engine runs: the host driver and its compiler proper, and nvcc
COMPILERS = ('c++', 'cc1plus', 'g++', 'gcc', 'clang++', 'nvcc')


def compiler_flags() -> list[str]:
    '''The include and ABI flags torch uses for an extension, so the precompiled header matches.'''
    import sysconfig

    from torch.utils.cpp_extension import include_paths

    includes = include_paths(device_type = 'cuda') + [sysconfig.get_paths()['include']]
    return ['-DTORCH_EXTENSION_NAME=symmetry_ext', '-DTORCH_API_INCLUDE_EXTENSION_H',
            *[f'-isystem{path}' for path in includes], '-fPIC', '-std=c++20', '-O3']


def build_pch(verbose: bool) -> None:
    '''Precompile ``pch.h`` unless it is newer than the headers it pulls in; every .cpp includes it.'''
    newest = max(os.path.getmtime(path) for path in (PCH, *HEADERS))
    if os.path.exists(PCH_OUTPUT) and os.path.getmtime(PCH_OUTPUT) > newest:
        return
    command = ['c++', *compiler_flags(), '-x', 'c++-header', PCH, '-o', PCH_OUTPUT]
    if verbose:
        print(' '.join(command), flush = True)
    subprocess.run(command, check = True)
    print(f'precompiled {os.path.basename(PCH)}', flush = True)


def available_gib() -> float:
    '''Host memory a new process may claim, in GiB, from ``MemAvailable``.'''
    with open('/proc/meminfo') as meminfo:
        for line in meminfo:
            if line.startswith('MemAvailable:'):
                return float(line.split()[1]) / 1024 ** 2
    return float('inf')


def add_venv_to_path() -> None:
    '''Put the interpreter's own bin directory first on ``PATH``, where ninja and the compiler are.'''
    directory = os.path.dirname(sys.executable)
    entries = os.environ.get('PATH', '').split(os.pathsep)
    if not entries or entries[0] != directory:
        os.environ['PATH'] = os.pathsep.join([directory, *entries])


def compiler_is_running() -> bool:
    '''
    Whether a compiler is working on a source of the engine right now.

    Returns:
      True when a compiler process names one of the engine's sources; an editor or a search
      that merely mentions such a path does not count, or a dead build's lock would survive.
    '''
    pattern = rf'{os.path.dirname(HERE)}/\S+\.(cpp|cu)'
    listing = subprocess.run(['pgrep', '-a', '-f', pattern], capture_output = True, text = True)
    for line in listing.stdout.splitlines():
        fields = line.split(maxsplit = 1)
        if len(fields) == 2 and os.path.basename(fields[1].split()[0]) in COMPILERS:
            return True
    return False


def clear_stale_lock(verbose: bool) -> None:
    '''
    Drop torch's build lock when the build that held it is gone.

    ``torch.utils.cpp_extension.load`` waits forever on ``build/lock``, so one interrupted build
    blocks every later one.  The lock is stale when it is older than half a minute and no
    compiler is working on a source of the engine.

    Args:
      verbose: print when a stale lock is removed.
    '''
    path = os.path.join(BUILD_DIR, 'lock')
    if (not os.path.exists(path) or time.time() - os.path.getmtime(path) < 30
            or compiler_is_running()):
        return
    if verbose:
        print(f'removing the stale build lock {path}', flush = True)
    os.remove(path)


def build_extension(verbose: bool, jobs: int) -> Any:
    '''
    Compile the stale units, link the engine and load it.

    The one build path: ``main`` and the import of ``tentorch._ext`` both come through here, so
    the lock handling, the precompiled header and the job cap cannot drift apart.

    Args:
      verbose: show the compiler output.
      jobs: units compiled at once, which caps the host memory the build claims.

    Returns:
      The loaded ``symmetry_ext`` module.
    '''
    os.makedirs(BUILD_DIR, exist_ok = True)
    add_venv_to_path()
    os.environ['MAX_JOBS'] = str(jobs)
    os.environ.setdefault('TORCH_EXTENSIONS_DIR', BUILD_DIR)
    clear_stale_lock(verbose)
    build_pch(verbose)

    # ``DENSEPATH=0`` compiles the dense route out, so a contraction the categorical route
    # declines raises instead of falling back; the default build keeps it as the reference.
    dense = os.environ.get('DENSEPATH', '1') != '0'

    from torch.utils.cpp_extension import load

    module: Any = load(
        name = 'symmetry_ext',
        sources = list(SOURCES),
        extra_cflags = ['-O3', '-include', PCH] + (['-DDENSEPATH'] if dense else []),
        build_directory = BUILD_DIR,
        verbose = verbose)
    # the loader treats this script as an input, so a successful build has to stamp the library
    os.utime(LIBRARY)
    return module


def main() -> int:
    '''Precompile the shared header, compile the units with at most ``--jobs`` at once, link.'''
    parser = argparse.ArgumentParser(description = 'build the tentorch symmetry engine')
    parser.add_argument('--verbose', action = 'store_true', help = 'show the compiler output')
    parser.add_argument('--force', action = 'store_true', help = 'rebuild every unit')
    parser.add_argument('--jobs', type = int, default = 1, help = 'units compiled at once, 1 or 2')
    parser.add_argument('--gib-per-job', type = float, default = GIB_PER_JOB,
                        help = f'host memory one unit may need, in GiB (default {GIB_PER_JOB})')
    arguments = parser.parse_args()

    free = available_gib()
    needed = arguments.gib_per_job * arguments.jobs
    if free < needed:
        print(f'{free:.1f} GiB of host memory available but {arguments.jobs} job(s) need '
              f'{needed:.1f} GiB; free some memory, lower --jobs or lower --gib-per-job',
              file = sys.stderr)
        return 1
    if arguments.jobs > 2:
        print('at most 2 jobs: a unit peaks near 2.5 GiB', file = sys.stderr)
        return 1

    add_venv_to_path()
    if not os.path.exists(LIBRARY) and shutil.which('ninja') is None:
        print('ninja is required for the first build: python -m pip install ninja', file = sys.stderr)
        return 1
    if arguments.force:
        for path in (*glob.glob(os.path.join(BUILD_DIR, '*.o')), LIBRARY):
            if os.path.exists(path):
                os.remove(path)
    module: Any = build_extension(arguments.verbose, arguments.jobs)
    print('built:', module.__file__, flush = True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
