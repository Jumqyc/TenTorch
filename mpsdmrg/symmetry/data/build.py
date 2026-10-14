"""One-click build of the C++ extension of ``mpsdmrg/symmetry/data/su2data.cpp``.

    python mpsdmrg/symmetry/data/build.py [--verbose]

The compiled module is written to ``mpsdmrg/symmetry/data/build/su2data_ext.so``, which
``mpsdmrg/symmetry/data/su2data.py`` imports directly; ninja is needed for the build
(``python -m pip install ninja``).  Running the script again only recompiles when the
source changed.
"""
import argparse
import os
import shutil
import sys
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(
        description='build the mpsdmrg/symmetry/data/su2data.cpp extension')
    parser.add_argument('--verbose', action='store_true', help='show the compiler output')
    arguments = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    build = os.path.join(here, 'build')
    os.makedirs(build, exist_ok=True)
    # a bare .venv/bin/python does not put the environment's ninja on PATH
    os.environ['PATH'] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get('PATH', '')
    os.environ.setdefault('TORCH_EXTENSIONS_DIR', build)
    if not os.path.exists(os.path.join(build, 'su2data_ext.so')) and shutil.which('ninja') is None:
        print('ninja is required for the first build: python -m pip install ninja',
              file=sys.stderr)
        return 1

    from torch.utils.cpp_extension import load

    module: Any = load(name='su2data_ext',
                       sources=[os.path.join(here, 'su2data.cpp')],
                       extra_cflags=['-O3'],
                       build_directory=build,
                       verbose=arguments.verbose)
    print('built:', module.__file__)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
