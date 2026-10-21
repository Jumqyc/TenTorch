from tentorch import DMRG

# a decorator to extend the DMRG engine with a new method
def extends_DMRG(func):
    setattr(DMRG, func.__name__, func)
    return func

