#!/usr/bin/env python3
r'''
Fusion trees, and the two moves of section 1.4, as TikZ pictures.

The module is laid out in four parts:

  * ``Point``, ``Wire``, ``Picture`` — what a drawing is made of;
  * ``place``, ``connect`` — the layout that turns a ``Leaf``/``Node`` tree into
    wires and vertices;
  * ``render``, ``arrow``, ``clip`` — the layout turned into TikZ source;
  * ``pentagon``, ``hexagon`` — the two axiom figures that ``doc/tensor.tex`` shows.

Every branch leaves its vertex at 45 degrees, so the vertex of two subtrees sits
where the two rays meet and the drawing needs no absolute coordinates.  A move of
the $F$ symbol re-brackets a tree, and is drawn as an arrow between two trees.  A
move of the $R$ symbol exchanges the two legs of one vertex, and is drawn as an
arrow between two trees whose leaves stand in the exchanged order.

Usage: ``python3 tree_tikz.py pentagon`` or ``python3 tree_tikz.py hexagon``
prints the figure; the output is pasted into an open ``tikzpicture`` in the
document.
'''
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Union

Point = tuple[float, float]

ROOT = 0.55  # length of the line the root of a tree leaves on, in units of ``unit``


@dataclass(frozen=True)
class Leaf:
    '''
    A leg of the tensor, carrying one charge.
    Args:
      label: the charge, as LaTeX source without the dollars.
    '''
    label: str


@dataclass(frozen=True)
class Node:
    '''
    A fusion vertex, together with the two subtrees that feed it.
    Args:
      charge: the charge the two legs fuse to, as LaTeX source without the dollars.
      left: the left subtree; its legs enter the vertex first.
      right: the right subtree.
    '''
    charge: str
    left: "Tree"
    right: "Tree"


Tree = Union[Leaf, Node]


@dataclass
class Wire:
    '''
    One line of the picture, between two vertices.
    Args:
      start: the first end, in tree coordinates.
      end: the second end.
    '''
    start: Point
    end: Point


@dataclass
class Picture:
    '''
    One tree as it is drawn: the lines, the vertices they meet at, and the legs.
    Args:
      wires: every line, in drawing order.
      nodes: ``(charge, x, y)`` for every inner vertex.
      leaves: ``(charge, x)``; every leaf sits at ``y = 0``.
    '''
    wires: list[Wire] = field(default_factory=list)
    nodes: list[tuple[str, float, float]] = field(default_factory=list)
    leaves: list[tuple[str, float]] = field(default_factory=list)


def leaves(tree: Tree) -> list[Leaf]:
    '''
    Return the leaves of a tree, left to right.
    Args:
      tree: the tree to walk.
    Returns:
      Its leaves.
    '''
    if isinstance(tree, Leaf):
        return [tree]
    return leaves(tree.left) + leaves(tree.right)


def connect(picture: Picture, vertex: Point, left: Point, right: Point) -> None:
    '''
    Wire two subtree roots into the vertex above them.
    Args:
      picture: the picture the wires are appended to.
      vertex: where the two feeds meet; above both of them.
      left: the root of the left subtree.
      right: the root of the right subtree.
    '''
    picture.wires += [Wire(left, vertex), Wire(right, vertex)]


def place(tree: Tree, x: float, unit: float, picture: Picture) -> Point:
    '''
    Lay a tree out left to right and record its wires, vertices and leaves.
    Args:
      tree: the tree to lay out; its leftmost leaf sits at ``x``.
      x: the position of the leftmost leaf of this subtree.
      unit: the distance between neighbouring leaves.
      picture: the picture the drawing is appended to.
    Returns:
      The position of this subtree's root.
    '''
    if isinstance(tree, Leaf):
        picture.leaves.append((tree.label, x))
        return (x, 0.0)
    left = place(tree.left, x, unit, picture)
    right = place(tree.right, left[0] + unit * len(leaves(tree.left)), unit, picture)
    # both feeds leave at 45 degrees, so the vertex is where the two rays meet
    vertex = ((left[0] + right[0] + right[1] - left[1]) / 2.0,
              (left[1] + right[1] + right[0] - left[0]) / 2.0)
    connect(picture, vertex, left, right)
    picture.nodes.append((tree.charge, vertex[0], vertex[1]))
    return vertex


def render(tree: Tree, origin: Point = (0.0, 0.0), unit: float = 1.2, scale: float = 1.0,
           label_leaves: bool = True, label_inner: bool = True,
           labels: dict[str, Point] | None = None, font: str = "\\small") -> str:
    '''
    Return one tree as a TikZ scope.
    Args:
      tree: the tree to draw, its leaves left to right.
      origin: where the leftmost leaf sits.
      unit: the distance between neighbouring leaves.
      scale: the scale of the scope.
      label_leaves: write every leg charge under its leaf.
      label_inner: write every inner charge beside its vertex.
      labels: extra text and position pairs, used for the R symbols.
      font: the size of the charge labels.
    Returns:
      TikZ source, to be pasted into an open tikzpicture.
    '''
    picture = Picture()
    root = place(tree, 0.0, unit, picture)
    top = (root[0], root[1] + ROOT * unit)
    # the root of a tree is a leg too, so it leaves the picture on a line of its own
    picture.wires.append(Wire(root, top))
    ox, oy = origin
    out = [f"\\begin{{scope}}[thick, scale={scale}, shift={{({ox:.3f},{oy:.3f})}},",
           f"                 every node/.style={{inner sep=1pt, font={font}}}]"]
    for wire in picture.wires:
        out.append(f"  \\draw ({wire.start[0]:.3f},{wire.start[1]:.3f}) -- "
                   f"({wire.end[0]:.3f},{wire.end[1]:.3f});")
    for _, x, y in picture.nodes:
        out.append(f"  \\fill ({x:.3f},{y:.3f}) circle (1.3pt);")
    if label_leaves:
        for name, x in picture.leaves:
            out.append(f"  \\node[anchor=north] at ({x:.3f},-0.10) {{${name}$}};")
    if label_inner:
        for charge, x, y in picture.nodes:
            if abs(x - root[0]) < 1e-9 and abs(y - root[1]) < 1e-9:
                out.append(f"  \\node[anchor=south, fill=white, inner sep=0.6pt] "
                           f"at ({top[0]:.3f},{top[1] + 0.06:.3f}) {{${charge}$}};")
            else:
                out.append(f"  \\node[anchor=south east, fill=white, inner sep=0.6pt] "
                           f"at ({x - 0.05:.3f},{y + 0.06:.3f}) {{${charge}$}};")
    for name, (x, y) in (labels or {}).items():
        out.append(f"  \\node[anchor=west, font=\\footnotesize] at ({x:.3f},{y:.3f}) {{${name}$}};")
    out.append("\\end{scope}")
    return "\n".join(out)


def clip(bounds: tuple[float, float, float, float], centre: Point, toward: Point) -> Point:
    '''
    Return where the segment from one centre toward another leaves a box.
    Args:
      bounds: the box ``(x0, y0, x1, y1)`` around one tree.
      centre: the centre of that tree.
      toward: the centre of the tree the arrow points at.
    Returns:
      The point of the box boundary in the direction of ``toward``.
    '''
    dx, dy = toward[0] - centre[0], toward[1] - centre[1]
    scale = min((bounds[2] - centre[0]) / dx if dx > 0 else
                (bounds[0] - centre[0]) / dx if dx < 0 else math.inf,
                (bounds[3] - centre[1]) / dy if dy > 0 else
                (bounds[1] - centre[1]) / dy if dy < 0 else math.inf)
    return (centre[0] + scale * dx, centre[1] + scale * dy)


def arrow(start: Point, end: Point, label: str, font: str = "\\scriptsize") -> str:
    '''
    Return one arrow between two trees, with its symbol on the way.
    Args:
      start: where the arrow begins.
      end: where it ends.
      label: the symbol, as LaTeX source without the dollars.
      font: the size of the symbol.
    Returns:
      TikZ source for the arrow.
    '''
    return (f"  \\draw[->, >=stealth] ({start[0]:.3f},{start[1]:.3f}) -- "
            f"({end[0]:.3f},{end[1]:.3f}) "
            f"node[pos=0.50, above, fill=white, inner sep=1.5pt, font={font}] {{${label}$}};")


def pentagon(radius: float = 7.8, unit: float = 0.84) -> str:
    '''
    Return the pentagon: five bracketings of four legs, joined along two routes.
    Args:
      radius: the radius of the circle the five trees sit on.
      unit: the leaf spacing inside each tree.
    Returns:
      A complete ``center``/``tikzpicture`` block.
    '''
    a, b, c, d = Leaf("a"), Leaf("b"), Leaf("c"), Leaf("d")
    start = Node("e", Node("v", Node("u", a, b), c), d)                # ((ab)c)d
    two = Node("e", Node("v'", a, Node("w", b, c)), d)                # (a(bc))d
    three = Node("e", a, Node("y", Node("w", b, c), d))                # a((bc)d)
    end = Node("e", a, Node("t", b, Node("z", c, d)))                  # a(b(cd))
    five = Node("e", Node("u", a, b), Node("z", c, d))                 # (ab)(cd)
    trees = [start, two, three, end, five]
    edges = [(0, 1, r"F^{v}_{abc}"), (1, 2, r"F^{e}_{awd}"), (2, 3, r"F^{y}_{bcd}"),
             (0, 4, r"F^{e}_{ucd}"), (4, 3, r"F^{e}_{abz}")]
    return _ring(trees, edges, (90, 160, 232, 308, 20), radius, unit, scale=0.46, squash=0.92,
                 font=r"\scriptsize", arrow_font=r"\tiny")


def hexagon(radius: float = 7.6, unit: float = 0.78) -> str:
    '''
    Return the hexagon: two routes of three moves from one tree to another.
    Args:
      radius: the radius of the ellipse the six trees sit on.
      unit: the leaf spacing inside each tree.
    Returns:
      A complete ``center``/``tikzpicture`` block.
    '''
    a, b, c = Leaf("a"), Leaf("b"), Leaf("c")
    start = Node("d", Node("u", a, b), c)                              # ((ab)c)
    two = Node("d", a, Node("v", b, c))                                # (a(bc))
    three = Node("d", a, Node("v", c, b))                              # (a(cb))  legs of v exchanged
    end = Node("d", Node("w", a, c), b)                                # ((ac)b)
    five = Node("d", Node("w", c, a), b)                               # ((ca)b)  legs of w exchanged
    six = Node("d", c, Node("u", a, b))                                # (c(ab))  legs of the root exchanged
    trees = [start, two, three, end, five, six]
    edges = [(0, 1, r"F^{d}_{abc}"), (1, 2, r"R_{bc}"), (2, 3, r"F^{d}_{acb}"),
             (0, 5, r"R_{ab,c}"), (5, 4, r"F^{d}_{cab}"), (4, 3, r"R_{ac}")]
    return _ring(trees, edges, (180, 120, 60, 0, 300, 240), radius, unit, scale=0.62, squash=0.95)


def _ring(trees: list[Tree], edges: list[tuple[int, int, str]], angles: tuple[float, ...],
          radius: float, unit: float, scale: float, squash: float,
          font: str = "\\small", arrow_font: str = "\\scriptsize") -> str:
    '''
    Return trees placed on an ellipse and joined by labelled arrows.
    Args:
      trees: one tree per site.
      edges: ``(from, to, label)`` for every arrow.
      angles: the polar angle of every site, in degrees.
      radius: the radius of the ellipse.
      unit: the leaf spacing inside each tree.
      scale: the scale of the whole picture.
      squash: the vertical radius, as a fraction of ``radius``.
      font: the size of the charge labels.
      arrow_font: the size of the symbols on the arrows.
    Returns:
      A complete ``center``/``tikzpicture`` block.
    '''
    centres = [(radius * math.cos(math.radians(t)), squash * radius * math.sin(math.radians(t)))
               for t in angles]
    origins = [(cx - 1.5 * unit, cy - 0.5 * unit) for cx, cy in centres]
    boxes = [(-0.35 + ox, -0.65 + oy, 3 * unit + 0.35 + ox, (1.6 + ROOT) * unit + 0.55 + oy)
             for ox, oy in origins]
    out = ["\\begin{center}",
           f"\\begin{{tikzpicture}}[thick, >=stealth, scale={scale}]"]
    out += [render(tree, origin, unit, font=font) for tree, origin in zip(trees, origins)]
    out += [arrow(clip(boxes[i], centres[i], centres[j]), clip(boxes[j], centres[j], centres[i]),
                  label, arrow_font)
            for i, j, label in edges]
    out += ["\\end{tikzpicture}", "\\end{center}"]
    return "\n".join(out)


def vertices(tree: Tree, unit: float) -> dict[str, Point]:
    '''
    Return the position of every inner vertex of one tree.
    Args:
      tree: the tree to lay out.
      unit: the leaf spacing inside the tree.
    Returns:
      The vertex position of every inner charge.
    '''
    picture = Picture()
    place(tree, 0.0, unit, picture)
    return {charge: (x, y) for charge, x, y in picture.nodes}


def main(argv: list[str]) -> int:
    '''
    Print one of the two axiom figures.
    Args:
      argv: the command line; ``argv[1]`` names the figure, ``hexagon`` by default.
    Returns:
      The process exit code.
    '''
    figure = argv[1] if len(argv) > 1 else "hexagon"
    if figure not in ("pentagon", "hexagon"):
        print(f"unknown figure {figure!r}; use pentagon or hexagon", file=sys.stderr)
        return 2
    print(pentagon() if figure == "pentagon" else hexagon())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
