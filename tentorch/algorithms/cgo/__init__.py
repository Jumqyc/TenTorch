"""The constrained Gaussian optimisation that prepares the environment of a DMRG sweep, and the
Gaussian variational state it acts on.

``mps`` holds the DMRG side (``EnvMPS`` and the ``Broomstick`` extension in ``ext_register``),
``gaussian`` the Gaussian side, ``sdp`` the convex programmes both call, and ``cuda`` the kernels
behind them.
"""
