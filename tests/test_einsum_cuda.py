"""CUDA numerical regression checks for planned dense tensor contractions."""
from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != 'torch':
        raise
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is not installed; CUDA numerical checks require it.')
class EinsumCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Require CUDA before importing the compiled symmetry engine."""
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA is unavailable; no CPU matrix fallback is used.')
        from base import CUDA
        from mpsdmrg.tensor import Tensor, cached_einsum

        cls.device = CUDA
        cls.tensor_type = Tensor
        cls.contract = staticmethod(cached_einsum)

    def _noncontiguous(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Create a complex CUDA tensor with the requested shape and transposed storage."""
        raw = torch.randn(
            (*shape[:-2], shape[-1], shape[-2]),
            dtype = torch.complex128,
            device = self.device)
        return raw.transpose(-1, -2)

    def test_lanczos_matches_two_stage_reference_on_noncontiguous_inputs(self) -> None:
        """Check raw, wrapped, and mixed dense operands against a CUDA reference."""
        bond, physical, mpo = 4, 2, 3
        left = self._noncontiguous((bond, bond, mpo, physical, physical))
        right = self._noncontiguous((bond, bond, mpo, physical, physical))
        state = self._noncontiguous((bond, physical, physical, bond))
        for tensor in (left, right, state):
            self.assertFalse(tensor.is_contiguous())
        intermediate = torch.einsum('abqij,bjld->aqild', left, state)
        expected = torch.einsum('aqild,cdqkl->aikc', intermediate, right)
        raw = (left, right, state)
        wrapped = tuple(self.tensor_type.from_torch(tensor) for tensor in raw)
        cases = {'raw': raw, 'wrapped': wrapped, 'mixed': (wrapped[0], right, state)}
        for name, operands in cases.items():
            with self.subTest(representation=name):
                actual = self.contract('abqij,cdqkl,bjld->aikc', *operands)
                torch.testing.assert_close(actual.blocks[0], expected, rtol=1e-12, atol=1e-12)

    def test_mixed_dtypes_preserve_outer_product_promotion(self) -> None:
        """Retain a valid float32/float64 contraction whose first outer product promotes."""
        left = torch.randn(
            (7,),
            dtype = torch.float32,
            device = self.device)
        right = torch.randn(
            (5,),
            dtype = torch.float64,
            device = self.device)
        weights = torch.randn(
            (7, 5),
            dtype = torch.float64,
            device = self.device)
        expected = (left[:, None] * right[None, :] * weights).sum()
        raw = (left, right, weights)
        wrapped = tuple(self.tensor_type.from_torch(tensor) for tensor in raw)
        cases = {'raw': raw, 'wrapped': wrapped, 'mixed': (wrapped[0], right, weights)}
        for name, operands in cases.items():
            with self.subTest(representation=name):
                actual = self.contract('i,j,ij->', *operands)
                torch.testing.assert_close(actual.blocks[0], expected, rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
