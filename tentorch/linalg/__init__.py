'''
The linear algebra of the block-sparse tensor: the splittings and the Krylov eigensolver.
'''
from tentorch.linalg.graded import qr, rq, svd, svdvals
from tentorch.linalg.lanczos import lanczos

__all__ = ['lanczos', 'qr', 'rq', 'svd', 'svdvals']
