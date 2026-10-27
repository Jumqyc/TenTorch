'''Build the engine benchmarks and run them.

The engine objects are compiled into ``bench/build`` and rebuilt only when a source is newer than
its object.  ``--profile`` adds debug symbols and frame pointers, which is what ``perf record``
needs to attribute a sample to a function.

    .venv/bin/python bench/build.py --run
    .venv/bin/python bench/build.py --run --only contraction --profile
'''

import argparse
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
BUILD = HERE / 'build'

SOURCES = (
    'symmetry/cat/cat',
    'symmetry/cat/group',
    'symmetry/cat/abelian/abelian',
    'symmetry/cat/su2/su2',
    'symmetry/cat/su3/su3',
    'symmetry/cat/fibonacci/fibonacci',
    'symmetry/tree/tree',
    'tensors/tensor',
    'linalg/graded/contract',
    'linalg/graded/svd',
    'linalg/graded/compare',
    'linalg/graded/einsum',
)
BENCHES = (
    'symbols',
    'tensor_ops',
    'contraction',
    'paths',
    'su3_fusion',
)


def torch_paths() -> tuple[list[str], list[str]]:
    '''
    The include and library directories the engine needs from torch.
    Returns:
      The include directories, with the repository first, and the library directories.
    '''
    from torch.utils.cpp_extension import include_paths, library_paths

    return [str(ROOT), *include_paths()], library_paths()


def newest_header() -> float:
    '''
    The modification time of the newest header the objects depend on.
    Returns:
      The time, so that an object older than it is rebuilt: a header change recompiles every object
      that includes it, and the sources alone do not say so.
    '''
    headers = [*(ROOT / 'tentorch').rglob('*.h'), *HERE.glob('*.h')]
    return max(header.stat().st_mtime for header in headers)


def compile_file(source: Path, target: Path, includes: list[str], flags: list[str], stamp: float) -> None:
    '''
    Compiles one translation unit, unless its object is already newer than the source and the headers.
    Args:
      source: The C++ source.
      target: The object to write.
      includes: The include directories.
      flags: Extra compiler flags, such as the ones a profile needs.
      stamp: The newest header time, see :func:`newest_header`.
    Raises:
      subprocess.CalledProcessError: When the compiler fails.
    '''
    if target.exists() and target.stat().st_mtime >= max(source.stat().st_mtime, stamp):
        return
    target.parent.mkdir(parents = True, exist_ok = True)
    command = [
        'g++', '-std=c++20', '-O2', '-Wno-attributes', *flags,
        *[f'-I{directory}' for directory in includes],
        '-c', str(source), '-o', str(target),
    ]
    print(' '.join(command), flush = True)
    subprocess.run(command, check = True)


def main() -> None:
    '''Builds every requested bench and, with ``--run``, runs it.'''
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--profile', action = 'store_true', help = 'compile with -g -fno-omit-frame-pointer')
    parser.add_argument('--run', action = 'store_true', help = 'run each bench after linking it')
    parser.add_argument('--only', nargs = '+', choices = BENCHES, help = 'only the named benches')
    parser.add_argument('--seconds', default = '1.0', help = 'the time budget a bench gives each case')
    arguments = parser.parse_args()

    includes, libraries = torch_paths()
    python_library = sysconfig.get_config_var('LIBDIR')
    python_name = f'python{sys.version_info.major}.{sys.version_info.minor}'
    flags = ['-g', '-fno-omit-frame-pointer'] if arguments.profile else []
    # A profile build differs in its flags, not only in its sources, so it lives beside the release
    # objects instead of replacing them.
    variant = 'profile' if arguments.profile else 'release'
    objects = BUILD / variant / 'obj'

    stamp = newest_header()
    engine = []
    for name in SOURCES:
        target = objects / f'{Path(name).name}.o'
        compile_file(ROOT / 'tentorch' / f'{name}.cpp', target, includes, flags, stamp)
        engine.append(str(target))

    links = [
        *[f'-L{directory}' for directory in libraries],
        '-ltorch', '-ltorch_cpu', '-ltorch_python', '-lc10',
        f'-L{python_library}', f'-l{python_name}',
        *[f'-Wl,-rpath,{directory}' for directory in (*libraries, python_library)],
    ]
    for name in arguments.only or BENCHES:
        source = HERE / f'{name}.cpp'
        target = objects / f'bench_{name}.o'
        compile_file(source, target, includes, flags, stamp)
        binary = BUILD / variant / name
        command = ['g++', str(target), *engine, *links, '-o', str(binary)]
        print(' '.join(command), flush = True)
        subprocess.run(command, check = True)
        if arguments.run:
            print(f'== {name}', flush = True)
            subprocess.run([str(binary), '--seconds', arguments.seconds], check = True)


if __name__ == '__main__':
    main()
