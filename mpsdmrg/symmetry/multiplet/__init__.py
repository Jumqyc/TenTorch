'''
The multiplet content of a group.

One module per group that carries multiplets, each declaring the ``GROUP`` it answers for; the
builder of an invariant Hamiltonian asks here for the content of its group, so a further group
adds a module instead of a branch.
'''
from __future__ import annotations

from types import ModuleType

from mpsdmrg.symmetry import Group
from mpsdmrg.symmetry.multiplet import su2, su3

# the modules that carry a multiplet content, keyed by the type name of their group
_MODULES = {module.GROUP: module for module in (su2, su3)}


def content(group: Group) -> ModuleType:
    '''
    The module that answers the multiplet questions of a group.

    Args:
        group: the symmetry group.
    Returns:
        The module holding ``tensor_operators`` and ``channel_operators`` for that group.
    Raises:
        NotImplementedError: If no module claims the type of the group.
    '''
    module = _MODULES.get(group.type_name)
    if module is None:
        raise NotImplementedError(f'The multiplets of {group.group_name} have no module yet.')
    return module
