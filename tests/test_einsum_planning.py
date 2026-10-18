"""Shape-only regression checks for the public tensor contraction wrapper."""
from __future__ import annotations

import importlib.util
from math import prod
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import opt_einsum


class _RawShape:
    def __init__(self, shape: tuple[int, ...], dtype: str = 'complex128') -> None:
        """Store axis lengths and dtype without allocating tensor data."""
        self.shape = shape
        self._dtype = dtype

    @property
    def dtype(self) -> str:
        """Return the dtype metadata of a raw tensor."""
        return self._dtype


class _ShapeTensor(_RawShape):
    def __init__(self, shape: tuple[int, ...], is_dense: bool = True,
                 dtype: str = 'complex128') -> None:
        """Store the shape, dtype, and density flag of an engine tensor."""
        super().__init__(shape, dtype)
        self.is_dense = is_dense
        self._block = _RawShape(shape, dtype)

    @property
    def dtype(self) -> str:
        """Reject the engine dtype accessor, which would allocate a temporary tensor."""
        raise AssertionError('Read dense dtype metadata from the underlying block.')

    @property
    def blocks(self) -> list[_RawShape]:
        """Expose the dense block while requiring the charged route to short-circuit."""
        if not self.is_dense:
            raise AssertionError('Charged contractions retain their route without dtype checks.')
        return [self._block]


class _ShapeEngine:
    def __init__(self) -> None:
        """Record engine calls and the sizes of their intermediate tensors."""
        self.calls: list[tuple[str, int]] = []
        self.intermediate_sizes: list[int] = []
        self.clear_caches = Mock()

    def __call__(self, spec: str, *operands: _RawShape) -> _ShapeTensor:
        """Model ATen's default left-to-right einsum using axis metadata only.

        Args:
            spec: An explicit-output equation without ellipses.
            operands: One shape per term of the equation.
        Returns:
            The output shape; every materialized intermediate size is recorded.
        """
        self.calls.append((spec, len(operands)))
        inputs, output = spec.split('->')
        terms = inputs.split(',')
        if len(terms) != len(operands):
            raise ValueError('The equation must name every operand.')
        dimensions: dict[str, int] = {}
        for term, operand in zip(terms, operands):
            if len(term) != len(operand.shape):
                raise ValueError('Every operand axis must have a label.')
            for label, size in zip(term, operand.shape):
                previous = dimensions.get(label, 1)
                if previous != size and previous != 1 and size != 1:
                    raise ValueError('A shared label has incompatible dimensions.')
                dimensions[label] = max(previous, size)

        remaining = [set(term) for term in terms]
        while len(remaining) > 1:
            needed = set(output).union(*remaining[2:])
            combined = (remaining[0] | remaining[1]) & needed
            self.intermediate_sizes.append(prod(dimensions[label] for label in combined))
            remaining = [combined, *remaining[2:]]
        result_shape = tuple(dimensions[label] for label in output)
        if len(terms) == 1:
            self.intermediate_sizes.append(prod(result_shape))
        return _ShapeTensor(result_shape)


def _load_wrapper(engine: _ShapeEngine) -> ModuleType:
    """Load the real public wrapper with metadata-only Torch and extension stand-ins."""
    package = ModuleType('mpsdmrg')
    package.__path__ = []
    extension = ModuleType('mpsdmrg.ext')
    extension._engine = SimpleNamespace(clear_caches=engine.clear_caches)
    extension.Group = object
    extension.Tensor = _ShapeTensor
    extension.cached_einsum = engine
    for name in ('bond_leg', 'capped_leg', 'fuse_legs', 'multiplet_leg', 'site_leg'):
        setattr(extension, name, object())
    path = Path(__file__).resolve().parents[1] / 'mpsdmrg' / 'tensor' / '__init__.py'
    spec = importlib.util.spec_from_file_location('_einsum_wrapper_under_test', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load the tensor wrapper at {path}.')
    wrapper = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        'mpsdmrg': package,
        'mpsdmrg.ext': extension,
        'torch': ModuleType('torch'),
    }):
        spec.loader.exec_module(wrapper)
    return wrapper


class EinsumPlanningTests(unittest.TestCase):
    def setUp(self) -> None:
        """Load a fresh wrapper and shape engine for each regression check."""
        self.engine = _ShapeEngine()
        self.wrapper = _load_wrapper(self.engine)

    def test_lanczos_contraction_bounds_intermediates_for_all_representations(self) -> None:
        """Apply the effective Hamiltonian without materializing its dense matrix."""
        bond, physical, mpo = 64, 2, 5
        shapes = (
            (bond, bond, mpo, physical, physical),
            (bond, bond, mpo, physical, physical),
            (bond, physical, physical, bond),
        )
        equation = 'abqij,cdqkl,bjld->aikc'
        limit = bond**2 * mpo * physical**2
        for representation in ('raw', 'dense', 'charged'):
            with self.subTest(representation=representation):
                self.engine.intermediate_sizes.clear()
                if representation == 'raw':
                    operands = tuple(_RawShape(shape) for shape in shapes)
                else:
                    operands = tuple(_ShapeTensor(shape, representation == 'dense')
                                     for shape in shapes)
                result = self.wrapper.cached_einsum(equation, *operands)
                self.assertEqual(result.shape, (bond, physical, physical, bond))
                self.assertLessEqual(max(self.engine.intermediate_sizes), limit)

    def test_identical_shapes_reuse_the_plan_and_clear_rebuilds_it(self) -> None:
        """Reuse planning work across new operands until the cache is cleared."""
        equation = 'ab,bc,cd->ad'
        shapes = ((7, 13), (13, 3), (3, 5))
        with patch.object(opt_einsum, 'contract_path', wraps=opt_einsum.contract_path) as plan:
            for _ in range(2):
                result = self.wrapper.cached_einsum(
                    equation, *(_ShapeTensor(shape) for shape in shapes))
                self.assertEqual(result.shape, (7, 5))
            self.assertEqual(plan.call_count, 1)
            self.wrapper._expr_cache.clear()
            self.engine.clear_caches.assert_called_once_with()
            self.wrapper.cached_einsum(equation, *(_ShapeTensor(shape) for shape in shapes))
            self.assertEqual(plan.call_count, 2)

    def test_mixed_dense_dtypes_keep_original_engine_routing(self) -> None:
        """Preserve the multiplication order that permits mixed-dtype outer products."""
        equation = 'i,j,ij->'
        metadata = (((7,), 'float32'), ((5,), 'float64'), ((7, 5), 'float64'))
        raw = tuple(_RawShape(shape, dtype) for shape, dtype in metadata)
        wrapped = tuple(_ShapeTensor(shape, dtype=dtype) for shape, dtype in metadata)
        cases = {'raw': raw, 'wrapped': wrapped, 'mixed': (wrapped[0], raw[1], raw[2])}
        with patch.object(opt_einsum, 'contract_path',
                          side_effect=AssertionError('Mixed dense dtypes keep the original order.')):
            for name, operands in cases.items():
                with self.subTest(representation=name):
                    self.engine.calls.clear()
                    result = self.wrapper.cached_einsum(equation, *operands)
                    self.assertEqual(result.shape, ())
                    self.assertEqual(self.engine.calls, [(equation, 3)])

    def test_charged_mixed_dtypes_retain_existing_pairwise_route(self) -> None:
        """Keep charged routing independent of the dense dtype compatibility check."""
        metadata = (((7, 13), 'float32'), ((13, 3), 'float64'), ((3, 5), 'float64'))
        operands = tuple(_ShapeTensor(shape, is_dense=False, dtype=dtype)
                         for shape, dtype in metadata)
        with patch.object(opt_einsum, 'contract_path', wraps=opt_einsum.contract_path) as plan:
            result = self.wrapper.cached_einsum('ab,bc,cd->ad', *operands)
        self.assertEqual(result.shape, (7, 5))
        plan.assert_called_once()
        self.assertEqual([count for _, count in self.engine.calls], [2, 2])

    def test_one_and_two_operands_keep_direct_engine_routing(self) -> None:
        """Preserve unary and binary contraction shapes without requesting a plan."""
        with patch.object(opt_einsum, 'contract_path',
                          side_effect=AssertionError('A unary or binary call needs no plan.')):
            unary = self.wrapper.cached_einsum('abc->cba', _ShapeTensor((2, 3, 4)))
            binary = self.wrapper.cached_einsum('ij,jk->ik', _RawShape((2, 3)),
                                                _ShapeTensor((3, 5)))
            scalar = self.wrapper.cached_einsum('ab,ab->', _RawShape((2, 3)),
                                                _RawShape((2, 3)))
        self.assertEqual(unary.shape, (4, 3, 2))
        self.assertEqual(binary.shape, (2, 5))
        self.assertEqual(scalar.shape, ())
        self.assertEqual([count for _, count in self.engine.calls], [1, 2, 2])


if __name__ == '__main__':
    unittest.main()
