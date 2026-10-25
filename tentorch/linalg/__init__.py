'''
The linear algebra of the block-sparse tensor: the splittings and the Krylov eigensolver.
'''
from tentorch.linalg.graded import Decomposition, Split, qr, qr_split, rq, rq_split, split, svd, svdvals

__all__ = ['Decomposition', 'Split', 'qr', 'qr_split', 'rq', 'rq_split', 'split', 'svd', 'svdvals']
