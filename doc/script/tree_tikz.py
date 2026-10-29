#!/usr/bin/env python3
r'''
Layered pictures of fusion and braiding, and the two axiom figures.

A picture is a stack of layers, one above another.  A layer is one of

  * a fusion, which joins two neighbouring strands into one, so that it has
    $n$ strands at its lower edge and $n-1$ at its upper edge;
  * a braid, in which neighbouring strands cross, one passing over the other,
    so that it has $n$ strands at both edges, permuted.

The lower edge of a layer is the upper edge of the layer below it, and every
strand leaves or enters an edge vertically, so consecutive layers join without
a kink.  A ``Diagram`` is such a stack: its legs, then a list of layers, built
with ``fuse`` and ``braid``.  ``render`` draws one as one TikZ scope per layer,
and ``pentagon`` and ``yang_baxter`` return the axiom figures.

Usage: ``python3 tree_tikz.py pentagon`` prints the figure as a
``center``/``tikzpicture`` block, to be pasted into the document.
'''

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Sequence

Point = tuple[float, float]

PITCH = 1.15     # horizontal distance between neighbouring strands
LAYER = 0.60     # height of a fusion layer
CROSSING = 0.95  # height of one crossing of a braid layer
STUB = 0.11      # the straight piece a braid layer keeps at each of its edges
RISE = 0.5 * CROSSING * (1.0 - 2.0 * STUB)  # height of a fusion's join, half a crossing's
                                            # rise, so its arms have the slope of a braid
HALO = 3.4       # width of the white stroke that cuts the strand under a crossing
CLEAR = 0.26     # radius of the disc in which a crossing cuts the strand below it
CLEARANCE = 0.20  # how far an arrow stops short of the charges at a diagram's edge
FOOT = 0.12      # length of the straight piece below a leg


@dataclass(frozen=True)
class Fusion:
    '''
    A layer that joins two neighbouring strands into one.
    Args:
      left: the position of the left of the two strands, counting from 1.
      out: the charge the two fuse to, as LaTeX source without the dollars.
      height: the height of the layer.
    '''
    left: int
    out: str
    height: float = LAYER


@dataclass(frozen=True)
class Braid:
    '''
    A layer in which two neighbouring strands cross.
    Args:
      crossing: which two strands cross; ``k`` passes the strand at position
        ``k`` (counting from 1) over the one at ``k + 1``, and ``-k`` passes it
        under.
      height: the height of the layer.
    '''
    crossing: int
    height: float = CROSSING


class Diagram:
    '''
    A stack of layers, built from the lower edge up.
    Args:
      legs: the charges at the lower edge, left to right.
    '''

    def __init__(self, legs: Sequence[str]) -> None:
        self.legs = list(legs)
        self.charges = list(legs)
        self.layers: list[Fusion | Split | Braid] = []

    def fuse(self, left: int, out: str, height: float = LAYER) -> "Diagram":
        '''
        Append a layer that fuses two neighbouring strands.
        Args:
          left: the position of the left of the two strands, counting from 1.
          out: the charge the two fuse to.
          height: the height of the layer.
        Returns:
          This diagram.
        Raises:
          ValueError: if there is no pair of strands at that position.
        '''
        if not 1 <= left < len(self.charges):
            raise ValueError(f"no pair at position {left} of {len(self.charges)} strands")
        self.layers.append(Fusion(left, out, height))
        self.charges[left - 1:left + 1] = [out]
        return self

    def split(self, at: int, first: str, second: str,
              height: float = LAYER) -> "Diagram":
        '''
        Append a layer that splits one strand into two neighbouring ones, the
        inverse of a fusion.
        Args:
          at: the position of the strand that splits, counting from 1.
          first: the charge of the left of the two strands it leaves.
          second: the charge of the right of them.
          height: the height of the layer.
        Returns:
          This diagram.
        Raises:
          ValueError: if there is no strand at that position.
        '''
        if not 1 <= at <= len(self.charges):
            raise ValueError(f"no strand {at} of {len(self.charges)}")
        self.layers.append(Split(at, (first, second), height))
        self.charges[at - 1:at] = [first, second]
        return self

    def braid(self, crossing: int, height: float = CROSSING) -> "Diagram":
        '''
        Append a layer in which two neighbouring strands cross.
        Args:
          crossing: which two strands cross; ``+k`` passes the strand at
            position ``k`` over the next, ``-k`` passes it under.
          height: the height of the layer.
        Returns:
          This diagram.
        Raises:
          ValueError: if the crossing names a position without a next strand.
        '''
        if not 0 < abs(crossing) < len(self.charges):
            raise ValueError(f"no crossing {crossing} of {len(self.charges)} strands")
        self.layers.append(Braid(crossing, height))
        i = abs(crossing) - 1
        self.charges[i], self.charges[i + 1] = self.charges[i + 1], self.charges[i]
        return self


@dataclass
class Split:
    '''
    One layer that splits a strand into its two parts.
    Args:
      at: the position of the strand that splits, counting from 1.
      out: the charges of the two strands it leaves behind.
      height: the height of the layer.
    '''
    at: int
    out: tuple[str, str]
    height: float = LAYER


@dataclass
class Stroke:
    '''
    One strand of one layer.
    Args:
      start: where it enters the lower edge.
      bends: ``(control, control, end)`` for every cubic that carries it up.
    '''
    start: Point
    bends: list[tuple[Point, Point, Point]] = field(default_factory=list)


@dataclass
class Tag:
    '''
    A charge written in the picture.
    Args:
      text: the charge, as LaTeX source without the dollars.
      at: where its anchor sits.
      anchor: which corner of the text sits at ``at``.
      fill: whether to white out the line under the text.
      math: whether the text is a formula.
    '''
    text: str
    at: Point
    anchor: str = "north"
    fill: bool = False
    math: bool = True


@dataclass
class Painted:
    '''
    One layer as it is drawn, in coordinates of its own lower edge.
    Args:
      y0: the height of that edge above the bottom of the stack.
      strokes: its strands.
      crossings: where two strands cross, and the one that passes over.
      '''
    y0: float
    strokes: list[Stroke] = field(default_factory=list)
    crossings: list[tuple[Point, Stroke]] = field(default_factory=list)


@dataclass
class Drawing:
    '''
    A diagram as it is drawn.
    Args:
      layers: one paint per layer, bottom to top.
      tags: every charge written around it.
      dots: the points drawn as filled dots.
    '''
    layers: list[Painted]
    tags: list[Tag]
    dots: list[Point] = field(default_factory=list)


def _height(layer: Fusion | Split | Braid) -> float:
    '''
    Return the height a layer takes in the stack.
    Args:
      layer: the layer.
    Returns:
      Its height, a crossing's worth of it for every crossing of a braid that
      does not fix one.
    '''
    return layer.height


def _swell(start: Point, end: Point) -> tuple[Point, Point, Point]:
    '''
    Return a cubic from one point to another above it that leaves and enters
    vertically and has no curvature at either end.
    Args:
      start: the lower end.
      end: the upper end, above ``start``.
    Returns:
      The two controls and the upper end.
    '''
    third = (end[1] - start[1]) / 3.0
    return (start[0], start[1] + third), (end[0], end[1] - third), end


def _fork(start: Point, join: Point) -> tuple[Point, Point, Point]:
    '''
    Return a cubic from one leg of a fusion to its join, vertical at both ends.
    Args:
      start: the lower end, at the edge of the layer.
      join: the vertex the two fused strands meet at.
    Returns:
      The two controls and the join.
    '''
    return _swell(start, join)


def _fusion_layer(xs: list[float], layer: Fusion, y: float) -> tuple[list[float], Painted]:
    '''
    Lay out one fusion layer.
    Args:
      xs: the positions of the strands at its lower edge.
      layer: the layer.
      y: the height of the lower edge.
    Returns:
      The positions of the strands at its upper edge, and the painted layer.
    '''
    paint = Painted(y)
    i = layer.left - 1
    join = ((xs[i] + xs[i + 1]) / 2.0, RISE)
    paint.strokes += [Stroke((xs[i], 0.0), [_fork((xs[i], 0.0), join)]),
                      Stroke((xs[i + 1], 0.0), [_fork((xs[i + 1], 0.0), join)]),
                      Stroke(join, [_swell(join, (join[0], layer.height))])]
    paint.strokes += [Stroke((x, 0.0), [_swell((x, 0.0), (x, layer.height))])
                      for j, x in enumerate(xs) if j not in (i, i + 1)]
    return xs[:i] + [join[0]] + xs[i + 2:], paint


def _spread(vertex: Point, end: Point) -> tuple[Point, Point, Point]:
    '''
    Return a cubic from the vertex of a split to one of the strands it leaves.
    Args:
      vertex: the point the strand leaves.
      end: its upper end, at the edge of the layer.
    Returns:
      The two controls and ``end``.
    '''
    return _swell(vertex, end)


def _split_layer(xs: list[float], layer: Split,
                 y: float) -> tuple[list[float], Painted, tuple[Point, float, float]]:
    '''
    Lay out one split layer.
    Args:
      xs: the positions of the strands at its lower edge.
      layer: the layer.
      y: the height of the lower edge.
    Returns:
      The positions of the strands at its upper edge, the painted layer, and
      the vertex together with the positions of the pair it leaves.
    '''
    paint = Painted(y)
    i = layer.at - 1
    vertex = (xs[i], layer.height - RISE)
    left, right = xs[i] - PITCH / 2.0, xs[i] + PITCH / 2.0
    paint.strokes += [Stroke((xs[i], 0.0), [_swell((xs[i], 0.0), vertex)]),
                      Stroke(vertex, [_spread(vertex, (left, layer.height))]),
                      Stroke(vertex, [_spread(vertex, (right, layer.height))])]
    paint.strokes += [Stroke((x, 0.0), [_swell((x, 0.0), (x, layer.height))])
                      for j, x in enumerate(xs) if j != i]
    return xs[:i] + [left, right] + xs[i + 1:], paint, (vertex, left, right)


def _sample(stroke: Stroke, steps: int = 40) -> list[Point]:
    '''
    Return points along a strand.
    Args:
      stroke: the strand.
      steps: how many points to place on every cubic.
    Returns:
      The points, from the lower end to the upper one.
    '''
    out = [stroke.start]
    for first, second, end in stroke.bends:
        start = out[-1]
        for i in range(1, steps + 1):
            t = i / steps
            u = 1.0 - t
            out.append((u**3 * start[0] + 3 * u * u * t * first[0]
                        + 3 * u * t * t * second[0] + t**3 * end[0],
                        u**3 * start[1] + 3 * u * u * t * first[1]
                        + 3 * u * t * t * second[1] + t**3 * end[1]))
    return out


def _meet(over: Stroke, under: Stroke, low: float, high: float) -> Point:
    '''
    Return where two strands cross, between two heights.
    Args:
      over: the strand that passes over.
      under: the strand that passes under.
      low: the height the crossing starts at.
      high: the height it ends at.
    Returns:
      The point of the upper strand that comes closest to the lower one.
    '''
    below = [point for point in _sample(under) if low <= point[1] <= high]
    return min((point for point in _sample(over) if low <= point[1] <= high),
               key=lambda point: min(math.dist(point, other) for other in below))


def _braid_layer(xs: list[float], layer: Braid, y: float) -> tuple[list[float], Painted]:
    '''
    Lay out one braid layer, each strand as a single smooth curve from the
    lower edge to the upper one.
    Args:
      xs: the positions of the strands at its lower edge.
      layer: the layer.
      y: the height of the lower edge.
    Returns:
      The positions of the strands at its upper edge, the painted layer, and
      the vertex together with the positions of the pair it leaves.
    '''
    height = _height(layer)
    stub = STUB * height
    low, high = stub, height - stub
    k = abs(layer.crossing) - 1
    paint = Painted(y)
    strokes = [Stroke((x, 0.0)) for x in xs]
    points = [[(x, 0.0), (x, stub)] for x in xs]
    for j, x in enumerate(xs):
        points[j].append((xs[k + 1] if j == k else xs[k] if j == k + 1 else x, high))
    over, under = (k, k + 1) if layer.crossing > 0 else (k + 1, k)
    over_stroke, under_stroke = strokes[over], strokes[under]
    xs[k], xs[k + 1] = xs[k + 1], xs[k]
    strokes[k], strokes[k + 1] = strokes[k + 1], strokes[k]
    points[k], points[k + 1] = points[k + 1], points[k]
    points = [path + [(path[-1][0], height)] for path in points]
    for stroke, path in zip(strokes, points):
        stroke.bends = [_swell(start, end) for start, end in zip(path, path[1:])]
    paint.crossings = [(_meet(over_stroke, under_stroke, low, high), over_stroke)]
    paint.strokes += strokes
    return [path[-1][0] for path in points], paint


@dataclass(frozen=True)
class Strand:
    '''
    A piece of a slot, drawn as a vertical line.
    Args:
      x: the position of the slot.
      low: the height of its lower end.
      high: the height of its upper end.
      charge: what is written on it, or None.
    '''
    x: float
    low: float
    high: float
    charge: str | None = None


@dataclass(frozen=True)
class Instanton:
    '''
    A connector from the lower end of one slot to the upper end of another.
    Args:
      start: where it leaves.
      end: where it arrives.
      charge: what is written on it, or None.
      over: whether it passes over the instantons it crosses.
    '''
    start: Point
    end: Point
    charge: str | None = None
    over: bool = True


def _panel(strands: Sequence[Strand], instantons: Sequence[Instanton],
           dots: Sequence[Point] = (), math: bool = True) -> Drawing:
    '''
    Return the drawing of one panel of slots and the instantons between them.
    Args:
      strands: the vertical pieces.
      instantons: the connectors.
      dots: the points drawn as filled dots.
      math: whether every charge is a formula.
    Returns:
      The drawing: every strand and instanton as one stroke, the charges
      written beside the vertical pieces and beside the connectors, and the
      dots.
    '''
    painted = Painted(0.0)
    painted.strokes += [Stroke((strand.x, strand.low),
                               [_swell((strand.x, strand.low), (strand.x, strand.high))])
                        for strand in strands]
    painted.strokes += [Stroke(instanton.start, [_swell(instanton.start, instanton.end)])
                        for instanton in instantons]
    tags: list[Tag] = []
    for strand in strands:
        if strand.charge is not None:
            tags.append(Tag(strand.charge, (strand.x + 0.12, 0.5 * (strand.low + strand.high)),
                            anchor="west", fill=True, math=math))
    for index, instanton in enumerate(instantons):
        if instanton.charge is not None:
            tags.append(Tag(instanton.charge, _beside(painted.strokes[len(strands) + index]),
                            anchor="west", fill=True, math=math))
    for first in range(len(instantons)):
        for second in range(first + 1, len(instantons)):
            over, under = (instantons[first], instantons[second])
            if any(abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9
                   for a in (over.start, over.end) for b in (under.start, under.end)):
                continue
            low = max(min(over.start[1], over.end[1]), min(under.start[1], under.end[1]))
            high = min(max(over.start[1], over.end[1]), max(under.start[1], under.end[1]))
            if high <= low:
                continue
            upper = painted.strokes[len(strands) + (first if over.over else second)]
            lower = painted.strokes[len(strands) + (second if over.over else first)]
            painted.crossings.append((_meet(upper, lower, low, high), upper))
    return Drawing([painted], tags, list(dots))


def _beside(stroke: Stroke) -> Point:
    '''
    Return a point just to the right of the middle of a connector.
    Args:
      stroke: the connector.
    Returns:
      The point its charge is anchored at.
    '''
    start = stroke.start
    first, second, end = stroke.bends[0]
    return (0.125 * start[0] + 0.375 * first[0] + 0.375 * second[0] + 0.125 * end[0] + 0.12,
            0.125 * start[1] + 0.375 * first[1] + 0.375 * second[1] + 0.125 * end[1])


def draft() -> str:
    r'''
    Return the TikZ of the panel in the draft: five slots, four instantons, one
    crossing and one connector that ends on a slot.
    '''
    slots = [i * PITCH for i in range(5)]
    strands = [Strand(slots[0], 2.76, 5.29),
               Strand(slots[2], 2.68, 3.40),
               Strand(slots[2], 3.40, 5.72)]
    instantons = [Instanton((slots[0], 5.29), (slots[2], 3.40), over=True),
                  Instanton((slots[0], 2.76), (slots[2], 5.72), over=False),
                  Instanton((slots[1], 1.42), (slots[2], 2.68)),
                  Instanton((slots[2], 2.68), (slots[3], 1.10))]
    drawing = _panel(strands, instantons, [(slots[2], 2.68), (slots[3], 1.10)])
    return _frame(render(drawing), 1.0)


def theta() -> str:
    r'''
    Return the TikZ of the theta graph: two nodes joined by one strand, each of
    them with two legs.
    '''
    nodes = [(0.0, 0.0), (0.0, 0.90)]
    strands = [Strand(0.0, nodes[0][1], nodes[1][1], charge="c")]
    instantons: list[Instanton] = []
    tags: list[Tag] = []
    for index, (x, y) in enumerate(nodes):
        way = 1.0 if index else -1.0
        for side, charge in ((-1.0, "a"), (1.0, "b")):
            end = (side * 0.55, y + way * 0.45)
            instantons.append(Instanton((x, y), end))
            strands.append(Strand(end[0], end[1], end[1] + way * FOOT))
            tags.append(Tag(charge, (end[0], end[1] + way * (FOOT + 0.08)),
                            anchor="south" if way > 0 else "north"))
    drawing = _panel(strands, instantons, nodes)
    drawing.tags += tags
    return _inline(render(drawing), 1.0)


def foverlap() -> str:
    r'''
    Return the TikZ of the overlap of a tree with the flipped copy of another:
    the three legs are shared, and the two roots are joined by one curve on the
    left.
    '''
    half = 0.20
    legs = [i * PITCH for i in (-1, 0, 1)]
    strands = [Strand(x, -half, half) for x in legs]
    low = (-0.5 * PITCH, -half - RISE)
    down = (0.5 * PITCH, -half - LAYER - RISE)
    up = (0.5 * PITCH, half + RISE)
    top = (-0.5 * PITCH, half + LAYER + RISE)
    strands += [Strand(low[0], -half - LAYER, low[1], charge="b"),
                Strand(up[0], up[1], half + LAYER, charge="b'"),
                Strand(legs[2], -half - LAYER, -half),
                Strand(legs[0], half, half + LAYER)]
    instantons = [Instanton((legs[0], -half), low), Instanton((legs[1], -half), low),
                  Instanton((low[0], -half - LAYER), down),
                  Instanton((legs[2], -half - LAYER), down),
                  Instanton((legs[1], half), up), Instanton((legs[2], half), up),
                  Instanton((up[0], half + LAYER), top),
                  Instanton((legs[0], half + LAYER), top)]
    dots = [low, down, up, top]
    drawing = _panel(strands, instantons, dots)
    xc, r = -1.9, 0.38                                # the straight side, corner radius
    yA, yB = top[1], down[1]

    def flat(p: Point, q: Point) -> tuple[Point, Point, Point]:
        r'''
        Return the control points of the straight cubic from one point to another.
        Args:
          p: the start.
          q: the end.
        Returns:
          The two controls and the end.
        '''
        return ((p[0] + (q[0] - p[0]) / 3, p[1] + (q[1] - p[1]) / 3),
                (q[0] - (q[0] - p[0]) / 3, q[1] - (q[1] - p[1]) / 3), q)

    root = Stroke(top, [
        flat(top, (xc + r, yA)),
        ((xc + 0.45 * r, yA), (xc, yA - 0.55 * r), (xc, yA - r)),
        _swell((xc, yA - r), (xc, yB + r)),
        ((xc, yB + 0.55 * r), (xc + 0.45 * r, yB), (xc + r, yB)),
        flat((xc + r, yB), down)])
    drawing.layers[0].strokes.insert(0, root)
    drawing.tags += [Tag("a_1", (legs[0] - 0.12, 0.0), anchor="east"),
                     Tag("a_2", (legs[1] + 0.12, 0.0), anchor="west"),
                     Tag("a_3", (legs[2] + 0.12, 0.0), anchor="west"),
                     Tag("c", (xc - 0.12, 0.0), anchor="east")]
    return _inline(render(drawing), 1.0)


def rank3() -> str:
    r'''
    Return the TikZ of the rank-3 tree with its four charges left to the caller.
    The caller defines ``\ftreelegA``, ``\ftreelegB``, ``\ftreelegC`` and
    ``\ftreeinner``.
    '''
    return tree([r"\ftreelegA", r"\ftreelegB", r"\ftreelegC"],
                [(1, 0, r"\ftreeinner"), (1, 0, "0")], scale=0.85, math=False)


def layout(diagram: Diagram) -> Drawing:
    '''
    Lay a diagram out with the middle of its lower edge at the origin.
    Args:
      diagram: the stack to lay out.
    Returns:
      Its layers and the charges written on it.
    '''
    xs = [(i - (len(diagram.legs) - 1) / 2.0) * PITCH for i in range(len(diagram.legs))]
    tags = [Tag(leg, (x, -0.14)) for leg, x in zip(diagram.legs, xs)]
    painted: list[Painted] = []
    forks: dict[int, tuple[float, float]] = {}
    splits: dict[int, tuple[Point, float, float]] = {}
    y = 0.0
    for index, layer in enumerate(diagram.layers):
        if isinstance(layer, Fusion):
            xs, paint = _fusion_layer(xs, layer, y)
            forks[index] = (paint.strokes[2].start[0], y + RISE, y + layer.height)
        elif isinstance(layer, Split):
            xs, paint, split = _split_layer(xs, layer, y)
            splits[index] = split
        else:
            xs, paint = _braid_layer(xs, layer, y)
        painted.append(paint)
        y += _height(layer)
    tags += _node_tags(diagram, forks, splits, y)
    tags += [Tag(charge, (x, y + 0.14), anchor="south")
             for charge, x in zip(diagram.charges, xs)]
    return Drawing(painted, tags)


def _node_tags(diagram: Diagram, forks: dict[int, tuple[float, float, float]],
               splits: dict[int, tuple[Point, float, float]], top: float) -> list[Tag]:
    '''
    Return the charges written at the nodes of a laid-out diagram.
    Args:
      diagram: the diagram that was laid out.
      forks: for every fusion layer, the x of its vertex, its height and the
        height of its upper edge.
      splits: for every split layer, its vertex and the two positions it
        leaves.
      top: the height of the top of the stack.
    Returns:
      One tag per fusion below the top layer, written beside the stem above its
      vertex, and two per split below the top, written at the widest point of
      the pair it leaves.
    '''
    tags: list[Tag] = []
    last = len(diagram.layers) - 1
    for index, (x, vertex, upper) in forks.items():
        layer = diagram.layers[index]
        assert isinstance(layer, Fusion)
        if index < last:
            tags.append(Tag(layer.out, (x + 0.10, vertex + 0.6 * (upper - vertex)),
                            anchor="west", fill=True))
    for index, (vertex, left, right) in splits.items():
        layer = diagram.layers[index]
        assert isinstance(layer, Split)
        if index < last:
            middle = 0.5 * (vertex[1] + layer.height)
            tags += [Tag(layer.out[0], (0.5 * (vertex[0] + left) - 0.12, middle),
                         anchor="east", fill=True),
                     Tag(layer.out[1], (0.5 * (vertex[0] + right) + 0.12, middle),
                         anchor="west", fill=True)]
    return tags


def extent(drawing: Drawing) -> tuple[float, float, float, float]:
    '''
    Return the box a drawing occupies, charges included.
    Args:
      drawing: the laid-out diagram.
    Returns:
      ``(x0, y0, x1, y1)``.
    '''
    points = [tag.at for tag in drawing.tags]
    for paint in drawing.layers:
        for stroke in paint.strokes:
            points.append(stroke.start)
            points += [bend[2] for bend in stroke.bends]
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return min(x) - 0.22, min(y) - 0.30, max(x) + 0.22, max(y) + 0.38


def _path(stroke: Stroke, style: str = "") -> str:
    r'''
    Return the TikZ of one strand.
    Args:
      stroke: the strand.
      style: the options of the drawing command, with the brackets.
    Returns:
      A ``\draw`` path through its bends.
    '''
    out = [f"\\draw{style} ({stroke.start[0]:.3f},{stroke.start[1]:.3f})"]
    for control, other, end in stroke.bends:
        out.append(f" .. controls ({control[0]:.3f},{control[1]:.3f}) and "
                   f"({other[0]:.3f},{other[1]:.3f}) .. ({end[0]:.3f},{end[1]:.3f})")
    return "".join(out) + ";"


def render(drawing: Drawing, origin: Point = (0.0, 0.0), scale: float = 1.0,
           font: str = r"\footnotesize") -> list[str]:
    r'''
    Return the TikZ of one drawing, one scope per layer, and the strand under
    each crossing cut open around it.
    Args:
      drawing: the laid-out diagram.
      origin: where the point between the lowest charges goes.
      scale: the scale of the picture.
      font: the size of the charges.
    Returns:
      The lines of TikZ source.
    '''
    # TikZ does not scale a shift, but it does scale the coordinates inside the
    # scope, so the shift has to be pre-multiplied to keep the layers joined.
    ox, oy = origin[0] * scale, origin[1] * scale
    out: list[str] = []
    for paint in drawing.layers:
        out.append(f"\\begin{{scope}}[shift={{({ox:.3f},{oy + paint.y0 * scale:.3f})}}, "
                   f"scale={scale:.3f}]")
        out += ["  " + _path(stroke) for stroke in paint.strokes]
        for (x, y), stroke in paint.crossings:
            out += [f"  \\begin{{scope}}",
                    f"    \\clip ({x:.3f},{y:.3f}) circle ({CLEAR:.2f});",
                    f"    {_path(stroke, f'[white, line width={HALO:.1f}pt]')}",
                    f"    {_path(stroke)}",
                    "  \\end{scope}"]
        out.append("\\end{scope}")
    out.append(f"\\begin{{scope}}[shift={{({ox:.3f},{oy:.3f})}}, scale={scale:.3f}]")
    out += [f"  \\fill ({x:.3f},{y:.3f}) circle (0.035);" for x, y in drawing.dots]
    out.append("\\end{scope}")
    out.append(f"\\begin{{scope}}[shift={{({ox:.3f},{oy:.3f})}}, scale={scale:.3f}, "
               f"every node/.style={{inner sep=1pt, font={font}}}]")
    for tag in drawing.tags:
        style = f"anchor={tag.anchor}" + (", fill=white, inner sep=0.8pt" if tag.fill else "")
        text = f"${tag.text}$" if tag.math else tag.text
        out.append(f"  \\node[{style}] at ({tag.at[0]:.3f},{tag.at[1]:.3f}) {{{text}}};")
    out.append("\\end{scope}")
    return out


def _place(drawing: Drawing, centre: Point) -> tuple[Point, tuple[float, float, float, float]]:
    '''
    Return where to draw one diagram so that its box is centred on a point.
    Args:
      drawing: the laid-out diagram.
      centre: where the middle of its box goes.
    Returns:
      The origin for ``render``, and the box it then occupies.
    '''
    x0, y0, x1, y1 = extent(drawing)
    origin = (centre[0] - (x0 + x1) / 2.0, centre[1] - (y0 + y1) / 2.0)
    return origin, (origin[0] + x0, origin[1] + y0, origin[0] + x1, origin[1] + y1)


def _frame(body: list[str], scale: float) -> str:
    r'''
    Wrap TikZ lines in a centred picture.
    Args:
      body: the lines of the picture.
      scale: the scale of the picture.
    Returns:
      A complete ``center``/``tikzpicture`` block.
    '''
    lines = ["\\begin{center}",
             f"\\begin{{tikzpicture}}[thick, >=stealth, scale={scale:.3f}]"]
    lines += body
    lines += ["\\end{tikzpicture}", "\\end{center}"]
    return "\n".join(lines)


def _inline(body: list[str], scale: float) -> str:
    r'''
    Wrap TikZ lines in a picture that sits in a formula.
    Args:
      body: the lines of the picture.
      scale: the scale of the picture.
    Returns:
      A bare ``tikzpicture``, centred on its own bounding box.
    '''
    lines = [f"\\begin{{tikzpicture}}[thick, >=stealth, "
             f"baseline={{(current bounding box.center)}}, scale={scale:.3f}]"]
    lines += body
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


def _arrow(start: Point, end: Point, label: str, font: str = r"\normalsize") -> str:
    '''
    Return one arrow between two diagrams, with its symbol on the way.
    Args:
      start: where the arrow begins.
      end: where it ends.
      label: the symbol, as LaTeX source without the dollars.
      font: the size of the symbol.
    Returns:
      The line of TikZ source.
    '''
    return (f"  \\draw[->, >=stealth] ({start[0]:.3f},{start[1]:.3f}) -- "
            f"({end[0]:.3f},{end[1]:.3f}) node[pos=0.50, above, fill=white, "
            f"inner sep=1.5pt, font={font}] {{${label}$}};")


def _grow(box: tuple[float, float, float, float], margin: float
          ) -> tuple[float, float, float, float]:
    '''
    Return a box with every side pushed out, so that an arrow stops short of the
    charges written at the edge of the diagram.
    Args:
      box: ``(x0, y0, x1, y1)``.
      margin: how far to push each side out.
    Returns:
      The larger box.
    '''
    return (box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin)


def _clip(box: tuple[float, float, float, float], centre: Point,
          toward: Point) -> Point:
    '''
    Return where the segment between two centres leaves a box.
    Args:
      box: the box around one diagram.
      centre: the centre of that diagram.
      toward: the centre of the other.
    Returns:
      The point of the box boundary in the direction of ``toward``.
    '''
    dx, dy = toward[0] - centre[0], toward[1] - centre[1]
    scale = min((box[2] - centre[0]) / dx if dx > 0 else
                (box[0] - centre[0]) / dx if dx < 0 else math.inf,
                (box[3] - centre[1]) / dy if dy > 0 else
                (box[1] - centre[1]) / dy if dy < 0 else math.inf)
    return (centre[0] + scale * dx, centre[1] + scale * dy)


def yang_baxter(scale: float = 1.00) -> str:
    r'''
    Return the Yang--Baxter equation: the two orders of three crossings, one
    layer each.
    Args:
      scale: the scale of the picture.
    Returns:
      A complete ``center``/``tikzpicture`` block.
    '''
    legs = ["a", "b", "c"]
    first = _stack(legs, [(1, 2, 1), (2, 2, 1), (1, 2, 1)])
    second = _stack(legs, [(2, 2, 1), (1, 2, 1), (2, 2, 1)])
    left, right = extent(first), extent(second)
    gap = 2.0
    shift = left[2] + gap - right[0]
    body = render(first, (0.0, 0.0)) + render(second, (shift, 0.0))
    body.append(f"  \\node[font=\\Large] at ({(left[2] + shift + right[0]) / 2.0:.3f},"
                f"{(left[3] + left[1]) / 2.0:.3f}) {{$=$}};")
    return _frame(body, scale)


def pentagon(radius: float = 6.8, scale: float = 0.66) -> str:
    r'''
    Return the pentagon: five bracketings of four legs, joined along two routes.
    Args:
      radius: the radius of the circle the five diagrams sit on.
      scale: the scale of the picture.
    Returns:
      A complete ``center``/``tikzpicture`` block.
    '''
    legs = ["a", "b", "c", "d"]
    trees = [_stack(legs, [(1, 0, "u"), (1, 0, "v"), (1, 0, "e")]),      # ((ab)c)d
             _stack(legs, [(2, 0, "w"), (1, 0, "v'"), (1, 0, "e")]),     # (a(bc))d
             _stack(legs, [(2, 0, "w"), (2, 0, "y"), (1, 0, "e")]),      # a((bc)d)
             _stack(legs, [(3, 0, "z"), (2, 0, "t"), (1, 0, "e")]),      # a(b(cd))
             _stack(legs, [(1, 0, "u"), (2, 0, "z"), (1, 0, "e")])]      # (ab)(cd)
    edges = [(0, 1, r"F^{v}_{abc}"), (1, 2, r"F^{e}_{awd}"), (2, 3, r"F^{y}_{bcd}"),
             (0, 4, r"F^{e}_{ucd}"), (4, 3, r"F^{e}_{abz}")]
    return _ring(trees, edges, radius, scale, (90, 162, 234, 306, 18))


def hexagon(radius: float = 8.8, scale: float = 0.60, inverse: bool = False) -> str:
    r"""
    Return the hexagon: six diagrams of the same three legs $a$, $b$, $c$ at
    the bottom, the braid exchanged with the rebracketing along the two routes
    round the ring.
    Args:
      radius: the radius of the circle the six diagrams sit on.
      scale: the scale of the picture.
      inverse: whether the braids are the inverse ones, which is the hexagon of
        the reversed exchange.
    Returns:
      A complete ``center``/``tikzpicture`` block.
    """
    legs = ["a", "b", "c"]
    sign = -1 if inverse else 1
    trees = [_stack(legs, [(1, 0, "u"), (1, 0, "d")]),
             _stack(legs, [(2, 0, "v"), (1, 0, "d")]),
             _stack(legs, [(2, 2, sign), (2, 0, "v"), (1, 0, "d")]),
             _stack(legs, [(2, 2, sign), (1, 0, "w"), (1, 0, "d")]),
             _stack(legs, [(2, 2, sign), (1, 2, sign), (1, 0, "w"), (1, 0, "d")]),
             _stack(legs, [(2, 2, sign), (1, 2, sign), (2, 0, "u"), (1, 0, "d")])]
    def exchange(label: str) -> str:
        '''Return the label of an exchange, inverted when the crossings are.'''
        return f"({label})^{{-1}}" if inverse else label

    edges = [(0, 1, r"F^{d}_{abc}"), (1, 2, exchange(r"R_{bc}^d")),
             (2, 3, r"F^{d}_{acb}"), (4, 3, exchange(r"R_{ac}^w")),
             (5, 4, r"F^{d}_{cab}"), (0, 5, exchange(r"R_{uc}^d"))]
    return _ring(trees, edges, radius, scale, (180, 120, 60, 0, -60, -120))


def _ring(trees: Sequence[Drawing], edges: Sequence[tuple[int, int, str]],
          radius: float, scale: float, angles: Sequence[float]) -> str:
    r'''
    Return the TikZ of a ring of diagrams joined by labelled arrows.
    Args:
      trees: the diagrams, in the order of the angles.
      edges: the arrows, as the indices of the two diagrams and the label.
      radius: the radius of the circle the diagrams sit on.
      scale: the scale of the picture.
      angles: the angle of every diagram, in degrees.
    Returns:
      A complete ``center``/``tikzpicture`` block.
    '''
    centres = [(radius * math.cos(math.radians(t)), radius * math.sin(math.radians(t)))
               for t in angles]
    body: list[str] = []
    boxes = []
    for drawing, centre in zip(trees, centres):
        origin, box = _place(drawing, centre)
        body += render(drawing, origin)
        boxes.append(box)
    body += [_arrow(_clip(_grow(boxes[i], CLEARANCE), centres[i], centres[j]),
                    _clip(_grow(boxes[j], CLEARANCE), centres[j], centres[i]), label)
             for i, j, label in edges]
    return _frame(body, scale)


def _stack(legs: Sequence[str], layers: Sequence[tuple[int, ...]],
           math: bool = True) -> Drawing:
    r'''
    Return the drawing of a stack of layers over the given legs.
    Args:
      legs: the charges at the bottom, left to right.
      layers: the layers from the bottom up; ``(left, 0, charge)`` fuses the
        strands at the 1-based positions ``left`` and ``left + 1``,
        ``(at, 1, first, second)`` splits the strand at ``at``, and
        ``(at, 2, sign)`` braids the strands at ``at`` and ``at + 1``, the left
        one passing over when ``sign`` is positive.
      math: whether every charge is a formula.
    Returns:
      The drawing of the whole stack, the legs' charges below it and the
      charges of the top strands above it.
    '''
    def charge(text: str) -> str:
        '''Return a charge as it is written in TeX.'''
        return text.replace("@", "'")

    slots = [(i - (len(legs) - 1) / 2.0) * PITCH for i in range(len(legs))]
    charges = [charge(text) for text in legs]
    strands = [Strand(x, -FOOT, 0.0) for x in slots]
    instantons: list[Instanton] = []
    dots: list[Point] = []
    tags = [Tag(text, (x, -FOOT - 0.08), anchor="north", math=math)
            for text, x in zip(charges, slots)]
    y = 0.0
    for index, step in enumerate(layers):
        last = index == len(layers) - 1
        if step[1] == 0:
            left = step[0] - 1
            join = (0.5 * (slots[left] + slots[left + 1]), y + RISE)
            strands += [Strand(x, y, y + LAYER) for i, x in enumerate(slots)
                        if i not in {left, left + 1}]
            instantons += [Instanton((slots[left], y), join),
                           Instanton((slots[left + 1], y), join)]
            strands.append(Strand(join[0], join[1], y + LAYER,
                                  None if last else charge(step[2])))
            slots[left:left + 2] = [join[0]]
            charges[left:left + 2] = [charge(step[2])]
            y += LAYER
        elif step[1] == 1:
            at = step[0] - 1
            vertex = (slots[at], y + LAYER - RISE)
            strands += [Strand(x, y, y + LAYER) for i, x in enumerate(slots) if i != at]
            strands.append(Strand(slots[at], y, vertex[1]))
            instantons += [Instanton(vertex, (slots[at] - PITCH / 2.0, y + LAYER)),
                           Instanton(vertex, (slots[at] + PITCH / 2.0, y + LAYER))]
            slots[at:at + 1] = [slots[at] - PITCH / 2.0, slots[at] + PITCH / 2.0]
            charges[at:at + 1] = [charge(step[2]), charge(step[3])]
            if not last:
                middle = y + LAYER - 0.5 * RISE
                tags += [Tag(charges[at], (slots[at] - 0.12, middle),
                             anchor="east", fill=True, math=math),
                         Tag(charges[at + 1], (slots[at + 1] + 0.12, middle),
                             anchor="west", fill=True, math=math)]
            dots.append(vertex)
            y += LAYER
        else:
            at = step[0] - 1
            left, right = slots[at], slots[at + 1]
            stub = STUB * CROSSING
            strands += [Strand(x, y, y + CROSSING) for i, x in enumerate(slots)
                        if i not in {at, at + 1}]
            strands += [Strand(left, y, y + stub), Strand(right, y, y + stub),
                        Strand(right, y + CROSSING - stub, y + CROSSING),
                        Strand(left, y + CROSSING - stub, y + CROSSING)]
            instantons += [Instanton((left, y + stub), (right, y + CROSSING - stub),
                                     over=step[2] > 0),
                           Instanton((right, y + stub), (left, y + CROSSING - stub),
                                     over=step[2] < 0)]
            charges[at:at + 2] = [charges[at + 1], charges[at]]
            y += CROSSING
    tags += [Tag(text, (x, y + 0.14), anchor="south", math=math)
             for text, x in zip(charges, slots)]
    drawing = _panel(strands, instantons, dots, math=math)
    drawing.tags += tags
    return drawing


def tree(legs: Sequence[str], steps: Sequence[tuple[int, ...]], scale: float = 1.0,
         font: str = r"\footnotesize", math: bool = True) -> str:
    r'''
    Return the TikZ of a fusion tree as a bare ``tikzpicture``.
    Args:
      legs: the charges at the bottom, left to right.
      steps: the layers from the bottom up, in the form ``_stack`` reads.
      scale: the scale of the picture.
      font: the size of the charges.
      math: whether every charge is a formula.
    Returns:
      The picture, centred on its bounding box so that it can sit in a formula.
    '''
    return _inline(render(_stack(legs, steps, math=math), font=font), scale)


def _tree_args(argv: list[str]) -> tuple[list[str], list[tuple[int, ...]], float, str]:
    '''
    Read the arguments of the ``inline`` command; the layers come in the order
    they are given, a fusion as ``--fuse left:charge`` and a split as
    ``--split at:first,second``.
    A charge is written with ``@`` where a prime goes, because a literal
    apostrophe does not survive the shell that TeX runs the pipe in.
    Args:
      argv: the flags, ``--legs``, ``--fuse``, ``--split``, ``--braid``,
        ``--scale`` and ``--font``.
    Returns:
      The legs, the layers, the scale and the font.
    Raises:
      ValueError: if a flag is unknown or a value is malformed.
    '''
    legs: list[str] = []
    fusions: list[tuple[int, ...]] = []
    scale, font = 1.0, r"\footnotesize"
    i = 0
    while i < len(argv):
        flag, value = argv[i], argv[i + 1] if i + 1 < len(argv) else ""
        if flag == "--legs":
            legs = [charge.replace("@", "'") for charge in value.split(",")]
        elif flag == "--fuse":
            left, charge = value.split(":", 1)
            fusions.append((int(left), 0, charge.replace("@", "'")))
        elif flag == "--braid":
            at, sign = value.split(":", 1)
            fusions.append((int(at), 2, int(sign)))
        elif flag == "--split":
            at, pair = value.split(":", 1)
            first, second = pair.split(",")
            fusions.append((int(at), 1, first.replace("@", "'"), second.replace("@", "'")))
        elif flag == "--scale":
            scale = float(value)
        elif flag == "--font":
            font = value
        else:
            raise ValueError(f"unknown flag {flag!r}")
        i += 2
    if not legs:
        raise ValueError("a picture needs its legs")
    return legs, fusions, scale, font


FIGURES = {
    "draft": draft,
    "rank3": rank3,
    "foverlap": foverlap,
    "theta": theta,
    "yang_baxter": yang_baxter,
    "pentagon": pentagon,
    "hexagon": hexagon,
    "hexagon2": lambda: hexagon(inverse = True),
}


def write(directory: str) -> list[str]:
    r'''
    Write every figure as a ``.tex`` file, so that the document can ``\input``
    it instead of holding a copy of the TikZ source.
    Args:
      directory: where the files go.
    Returns:
      The paths written.
    '''
    os.makedirs(directory, exist_ok = True)
    paths = []
    for name, figure in FIGURES.items():
        path = os.path.join(directory, f"{name}.tex")
        with open(path, "w") as f:
            f.write(figure() + "\n")
        paths.append(path)
    return paths


def main(argv: list[str]) -> int:
    r'''
    Print one of the axiom figures, or write them all as files.
    Args:
      argv: the command line; ``argv[1]`` names one of ``FIGURES``,
        ``yang_baxter`` by default, or is ``--write`` to write every figure into
        ``figures`` beside the script.
    Returns:
      The process exit code.
    '''
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    if len(argv) > 1 and argv[1] == "inline":
        print(tree(*_tree_args(argv[2:])))
        return 0
    if len(argv) > 1 and argv[1] == "--write":
        print("\n".join(write(directory)))
        return 0
    figure = argv[1] if len(argv) > 1 else "yang_baxter"
    if figure not in FIGURES:
        print(f"unknown figure {figure!r}; use {', '.join(FIGURES)}", file=sys.stderr)
        return 2
    print(FIGURES[figure]())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
