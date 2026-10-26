'''
The linear algebra of the block-sparse tensor: the splittings and the Krylov eigensolver.
'''
from tentorch.linalg.graded import qr, qr_split, rq, rq_split, split, svd, svdvals
from tentorch.linalg.lanczos import lanczos

__all__ = ['lanczos', 'qr', 'qr_split', 'rq', 'rq_split', 'split', 'svd', 'svdvals']
