"""Python side of the CUDA extension of ``CGO/gaussian.py``.  The kernels live in
``CGO/cuda/project.cu`` and are built just in time with
``torch.utils.cpp_extension.load``; the compiled module is cached in
``CGO/cuda/build``, so only the first import pays for the compilation.  The first
build needs ninja (``python -m pip install ninja``); later imports load the cached
``CGO/cuda/build/project_ext.so`` directly.
"""
import importlib.util
import os
import sys

import torch
from torch import Tensor
from base import COMPLEX

_HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(_HERE, 'build')
LIBRARY = os.path.join(BUILD_DIR, 'project_ext.so')

if os.environ.get('TORCH_EXTENSIONS_DIR') is None:
    os.environ['TORCH_EXTENSIONS_DIR'] = BUILD_DIR
os.makedirs(BUILD_DIR, exist_ok=True)

if os.path.exists(LIBRARY):
    _spec = importlib.util.spec_from_file_location('project_ext', LIBRARY)
    _extension = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_extension)
else:
    # the build needs ninja on PATH, which a bare .venv/bin/python does not add
    os.environ['PATH'] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get('PATH', '')
    from torch.utils.cpp_extension import load
    _extension = load(name='project_ext',
                      sources=[os.path.join(_HERE, 'project.cu')],
                      extra_cuda_cflags=['-O3'],
                      build_directory=BUILD_DIR,
                      verbose=False)


def project(rotation: Tensor,
            internal: list[int],
            mixed: list[int],
            pure: list[int],
            *terms: tuple[complex, list[int]]) -> Tensor:
    '''
    Projected Majorana strings on the mixed register of a patch, P_V gamma_A P_V
    (CGO/cuda/project.cu): every label rotates to the canonical basis of the patch, the
    pure modes contract in their ground state by Wick's theorem, and the mixed Majoranas
    stay an operator on their register.
    Args:
        rotation: Williamson rotation of the patch, shape (2m, 2m) float64 on CUDA.
        internal: the Majorana labels of the patch in ascending order, the local basis
            that `rotation` is expressed in.
        mixed: the mixed modes of the patch, in register order.
        pure: the pure modes of the patch.
        term: a term of the Majorana string, its coefficient and its patch labels in
            order; a term holds at most 8 labels.
    Returns:
        The projected operator, shape (size, size) complex128 with size = 2^q, on CUDA.
    '''
    position = {label: index for index, label in enumerate(internal)}
    positions: list[int] = []
    offsets = [0]
    coefficients: list[complex] = []
    for coefficient, labels in terms:
        coefficients.append(coefficient)
        positions += [position[label] for label in labels]
        offsets.append(len(positions))
    return _extension.project(rotation,
                              torch.tensor(mixed, dtype=torch.int64),
                              torch.tensor(pure, dtype=torch.int64),
                              torch.tensor(positions, dtype=torch.int64),
                              torch.tensor(offsets, dtype=torch.int64),
                              torch.tensor(coefficients, dtype=COMPLEX))
