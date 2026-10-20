"""The coherence axioms of a fusion category, checked on the symbols the categorical route reads.

A fusion tree is a nested tuple of leg numbers, ``(0, (1, 2))`` coupling legs 1 and 2 first.  A
state is a tree together with the channel of every internal node, and a step is either one
reassociation of a node (an associator entry) or one braiding of the two children of a node (a
braiding symbol).  Each axiom then states that two sequences of steps between one pair of states
have to sum to the same amplitude over every intermediate channel:

* the pentagon, for two ways of moving the brackets of four legs into one shape;
* the hexagon, for braiding one leg with a pair of legs on the way round the other two.
"""
import torch


def leaves(tree: int | tuple) -> tuple[int, ...]:
    '''The leg numbers a subtree carries.'''
    if isinstance(tree, int):
        return (tree,)
    return tuple(sorted(leaves(tree[0]) + leaves(tree[1])))


def nodes(tree: int | tuple) -> list[tuple[tuple[int, ...], object, object]]:
    '''Every internal node of a tree as (leg numbers, left subtree, right subtree).'''
    if isinstance(tree, int):
        return []
    return nodes(tree[0]) + nodes(tree[1]) + [(leaves(tree), tree[0], tree[1])]


def replace(tree: int | tuple, target: tuple[int, ...], fresh: int | tuple) -> int | tuple:
    '''The tree with the subtree on the leg numbers ``target`` swapped for ``fresh``.'''
    if leaves(tree) == target:
        return fresh
    if isinstance(tree, int):
        return tree
    return (replace(tree[0], target, fresh), replace(tree[1], target, fresh))


class Axioms:
    '''The associator and the braiding of one category, as steps between fusion trees.'''

    def __init__(self, group, device: torch.device):
        '''
        Hold the category and the device; the matrices a step reads are cached per triple.
        Args:
          group: the fusion category; its associator and braiding are the symbols under test.
          device: the device the symbols are built on.
        '''
        self.group = group
        self.device = device
        self._f: dict[tuple, tuple[list, list, torch.Tensor]] = {}
        self._r: dict[tuple, complex] = {}

    def sector(self, part: int | tuple, config: dict, sectors: list) -> tuple:
        '''The coupled sector of a subtree; a leaf carries the sector of its leg.'''
        return config[leaves(part)] if isinstance(part, tuple) else sectors[part]

    def f_matrix(self, a: tuple, b: tuple, c: tuple,
                 total: tuple) -> tuple[list, list, torch.Tensor]:
        '''
        The associator F^{abc}_total with rows on the (ab) channel and columns on the (bc) one.
        Args:
          a, b, c: the sectors of the three legs, in leg order.
          total: the sector the three of them fuse to.
        Returns:
          The rows, the columns and the matrix; every entry is a symbol the route reads.
        '''
        key = (a, b, c, total)
        if key not in self._f:
            outcomes = self.group.fusion_outcomes
            rows = [one for one in outcomes(a, b) if total in outcomes(one, c)]
            columns = [two for two in outcomes(b, c) if total in outcomes(a, two)]
            assert len(rows) == len(columns), f'the associator of {key} is not square: {rows} {columns}'
            matrix = torch.zeros((len(rows), len(columns)), dtype = torch.complex128)
            for row, one in enumerate(rows):
                for column, two in enumerate(columns):
                    symbol = self.group.associator(a, b, c, total, one, two, self.device)
                    assert symbol.numel() == 1, f'the associator of {key} has {symbol.numel()} entries'
                    matrix[row][column] = complex(symbol.reshape(-1)[0].cpu())
            self._f[key] = (rows, columns, matrix)
        return self._f[key]

    def r_entry(self, a: tuple, b: tuple, channel: tuple) -> complex:
        '''The braiding symbol R^{ab}_channel.'''
        key = (a, b, channel)
        if key not in self._r:
            symbol = self.group.braiding(a, b, channel, self.device)
            assert symbol.numel() == 1, f'the braiding of {key} has {symbol.numel()} entries'
            self._r[key] = complex(symbol.reshape(-1)[0].cpu())
        return self._r[key]

    def rightward(self, tree: int | tuple, config: dict, target: tuple[int, ...],
                  sectors: list) -> list[tuple[object, dict, complex]]:
        '''Every reassociation of one node to the right, with its amplitude and its state.'''
        for key, left, right in nodes(tree):
            if key != target:
                continue
            assert isinstance(left, tuple), f'the node on {target} of {tree} is not left nested'
            one = self.sector(left[0], config, sectors)
            two = self.sector(left[1], config, sectors)
            three = self.sector(right, config, sectors)
            total, old = config[key], config[leaves(left)]
            rows, columns, matrix = self.f_matrix(one, two, three, total)
            found = []
            for column, fresh in enumerate(columns):
                moved = dict(config)
                del moved[leaves(left)]
                moved[leaves((left[1], right))] = fresh
                found.append((replace(tree, key, (left[0], (left[1], right))), moved,
                              complex(matrix[rows.index(old)][column])))
            return found
        raise AssertionError(f'the tree {tree} has no node on the legs {target}')

    def leftward(self, tree: int | tuple, config: dict, target: tuple[int, ...],
                 sectors: list) -> list[tuple[object, dict, complex]]:
        '''Every reassociation of one node to the left, with its amplitude and its state.'''
        for key, left, right in nodes(tree):
            if key != target:
                continue
            assert isinstance(right, tuple), f'the node on {target} of {tree} is not right nested'
            one = self.sector(left, config, sectors)
            two = self.sector(right[0], config, sectors)
            three = self.sector(right[1], config, sectors)
            total, old = config[key], config[leaves(right)]
            rows, columns, matrix = self.f_matrix(one, two, three, total)
            inverse = torch.linalg.inv(matrix)
            found = []
            for row, fresh in enumerate(rows):
                moved = dict(config)
                del moved[leaves(right)]
                moved[leaves((left, right[0]))] = fresh
                found.append((replace(tree, key, ((left, right[0]), right[1])), moved,
                              complex(inverse[row][columns.index(old)])))
            return found
        raise AssertionError(f'the tree {tree} has no node on the legs {target}')

    def braid(self, tree: int | tuple, config: dict, target: tuple[int, ...],
              sectors: list) -> list[tuple[object, dict, complex]]:
        '''Every braiding of the two children of one node, with its amplitude and its state.'''
        for key, left, right in nodes(tree):
            if key != target:
                continue
            one = self.sector(left, config, sectors)
            two = self.sector(right, config, sectors)
            return [(replace(tree, key, (right, left)), dict(config),
                     self.r_entry(one, two, config[key]))]
        raise AssertionError(f'the tree {tree} has no node on the legs {target}')

    def route(self, tree: int | tuple, config: dict, steps: list[tuple[str, tuple[int, ...]]],
              sectors: list) -> dict[tuple, complex]:
        '''
        The amplitude of one fixed sequence of steps, one entry per final state.
        Args:
          tree: the tree of the starting state.
          config: the channel of every internal node of that tree.
          steps: ('right' | 'left' | 'braid', leg numbers of the node) in the order they are taken.
          sectors: the sector of every leg, by leg number.
        Returns:
          The summed amplitude of every state the sequence reaches.
        '''
        state = {(tree, tuple(sorted(config.items()))): (tree, config, 1.0 + 0j)}
        for kind, target in steps:
            move = {'right': self.rightward, 'left': self.leftward, 'braid': self.braid}[kind]
            following: dict = {}
            for current_tree, current_config, factor in state.values():
                for fresh_tree, moved, amplitude in move(current_tree, current_config, target,
                                                         sectors):
                    key = (fresh_tree, tuple(sorted(moved.items())))
                    if key in following:
                        following[key] = (fresh_tree, moved, following[key][2] + factor * amplitude)
                    else:
                        following[key] = (fresh_tree, moved, factor * amplitude)
            state = following
        return {key: entry[2] for key, entry in state.items()}

    def channels(self, tree: int | tuple, sectors: list) -> list[tuple[tuple, dict]]:
        '''Every (total sector, channels) a tree reaches over the fusion rules of its legs.'''
        if isinstance(tree, int):
            return [(sectors[tree], {})]
        found = []
        for one, one_config in self.channels(tree[0], sectors):
            for two, two_config in self.channels(tree[1], sectors):
                for coupled in self.group.fusion_outcomes(one, two):
                    config = dict(one_config)
                    config.update(two_config)
                    config[leaves(tree)] = coupled
                    found.append((coupled, config))
        return found

    def compare(self, name: str, steps: list[tuple[str, tuple[int, ...]]],
                other: list[tuple[str, tuple[int, ...]]], start: int | tuple,
                sectors: list, total: tuple) -> float:
        '''
        The worst gap between two routes over every state of one total sector.
        Args:
          name: the axiom, for the message of a failure.
          steps, other: the two sequences; they have to end in the same states.
          start: the tree both sequences start from.
          sectors: the sector of every leg, by leg number.
          total: the sector the legs fuse to.
        Returns:
          The largest gap between the amplitudes the two sequences give one state.
        '''
        worst = 0.0
        compared = 0
        for coupled, config in self.channels(start, sectors):
            if coupled != total:
                continue
            one = self.route(start, config, steps, sectors)
            two = self.route(start, config, other, sectors)
            compared += len(set(one) | set(two))
            worst = max(worst, max(abs(one.get(key, 0.0) - two.get(key, 0.0))
                                   for key in set(one) | set(two)))
        assert compared > 0, f'{name}: no state of {start} fuses to {total}'
        return worst

    def pentagon(self, sectors: list, total: tuple) -> float:
        '''
        The gap between two ways of moving the brackets of four legs into one shape.
        Args:
          sectors: the sector of every leg, by leg number.
          total: the sector the four legs fuse to.
        Returns:
          The largest gap between the two routes.
        '''
        start = (((0, 1), 2), 3)
        return self.compare('pentagon', [('right', (0, 1, 2)), ('right', (0, 1, 2, 3)),
                                         ('right', (1, 2, 3))],
                            [('right', (0, 1, 2, 3)), ('right', (0, 1, 2, 3))], start, sectors,
                            total)

    def hexagon(self, sectors: list, total: tuple) -> float:
        '''
        The gap between braiding one leg with a pair and braiding it past the two legs in turn.
        Args:
          sectors: the sector of every leg, by leg number.
          total: the sector the three legs fuse to.
        Returns:
          The largest gap between the two routes.
        '''
        start = (0, (1, 2))
        return self.compare('hexagon', [('braid', (0, 1, 2))],
                            [('left', (0, 1, 2)), ('braid', (0, 1)), ('right', (0, 1, 2)),
                             ('braid', (0, 2)), ('left', (0, 1, 2))], start, sectors, total)
