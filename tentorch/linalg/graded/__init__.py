'''
The splittings of the block-sparse tensor: ``svd``, ``svdvals``, ``qr`` and ``rq``, as the engine
returns them.

Every splitting takes the legs of the first factor as `left`: the legs named there are brought to
the codomain of the first factor, and the rest go to the second.  A bond that is already where the
split wants it costs nothing.  Every split shares one value, the new bond: the domain of the first
factor and the codomain of the second.
'''

from tentorch._ext import qr, rq, svd, svdvals

__all__ = ['qr', 'rq', 'svd', 'svdvals']
