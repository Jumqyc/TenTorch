r'''Tensor diagrams for the categorical chapters, as TikZ.

Writes one `fig-<name>.tex` per figure next to this file; `tensor.tex` pulls each
one in with `\input{fig-...}` inside its own `figure` environment, so the labels
and captions stay in the chapter.  Run `.venv/bin/python doc/figures.py` after
editing a figure, then rebuild the document with pdflatex.
'''
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent

TREE = r'''\begin{tikzpicture}[thick, every node/.style={font=\small}]
  % ((a b) c) -> d, coupled through e
  \node (a) at (0,0) {$a$};
  \node (b) at (0.9,0) {$b$};
  \node[circle, fill, inner sep=1.6pt] (v1) at (0.45,0.75) {};
  \node (c) at (1.8,0) {$c$};
  \node[circle, fill, inner sep=1.6pt] (v2) at (1.125,1.5) {};
  \node (d) at (0.9,2.5) {$d$};
  \draw (a) -- (v1); \draw (b) -- (v1);
  \draw (v1) -- (v2) node[midway, left] {$e$};
  \draw (c) -- (1.8,0.75) -- (v2);
  \draw (v2) -- (d);
  \node at (2.6,1.1) {\Large $\xrightarrow{\;F\;}$};
  % a (b c) -> d, coupled through f
  \node (a2) at (4.0,0) {$a$};
  \node (b2) at (4.9,0) {$b$};
  \node[circle, fill, inner sep=1.6pt] (w1) at (5.35,0.75) {};
  \node (c2) at (5.8,0) {$c$};
  \node[circle, fill, inner sep=1.6pt] (w2) at (4.675,1.5) {};
  \node (d2) at (4.9,2.5) {$d$};
  \draw (b2) -- (w1); \draw (c2) -- (w1);
  \draw (w1) -- (w2) node[midway, right] {$f$};
  \draw (a2) -- (4.0,0.75) -- (w2);
  \draw (w2) -- (d2);
  \node[align=center] at (2.9,-0.85)
    {\footnotesize $F^{abc}_{d}(e,f)=\langle (ab)c{\to}d\,|\,a(bc){\to}d\rangle$};
\end{tikzpicture}'''

EXACT = r'''\begin{tikzpicture}[thick, every node/.style={font=\small}]
  % SU(2): 1 (x) 1 = 0 + 1 + 2
  \node (l1) at (0,0) {$1$};
  \node (l2) at (1.1,0) {$1$};
  \node[circle, fill, inner sep=1.6pt] (v) at (0.55,0.8) {};
  \draw (l1) -- (v); \draw (l2) -- (v);
  \draw (v) -- (0.55,1.7);
  \node[align=left] at (0.55,2.2) {\footnotesize $0,1,2$};
  \node[align=center] at (0.55,-0.75)
    {\footnotesize $\SU{2}$\\ \footnotesize $d_s=s+1$};
  \node at (2.0,0.8) {$\quad$};
  % Fibonacci: tau (x) tau = 1 + tau
  \node (m1) at (3.3,0) {$\tau$};
  \node (m2) at (4.4,0) {$\tau$};
  \node[circle, fill, inner sep=1.6pt] (w) at (3.85,0.8) {};
  \draw (m1) -- (w); \draw (m2) -- (w);
  \draw (w) -- (3.85,1.7);
  \node[align=left] at (3.85,2.2) {\footnotesize $1,\tau$};
  \node[align=center] at (3.85,-0.75)
    {\footnotesize Fibonacci\\ \footnotesize $d_\tau=\varphi$};
  \node[align=left] at (5.6,0.5)
    {\footnotesize no $C^{ab}_{c}$\\ \footnotesize only $F$, $R$};
\end{tikzpicture}'''

DIAGRAM = r'''\begin{tikzpicture}[thick, every node/.style={font=\small}]
  % the left chain
  \foreach \i/\x in {1/0,2/1.1,3/2.2,4/3.3} {
    \node[circle, fill, inner sep=1.6pt] (L\i) at (\x,1.4) {};
  }
  \node (lA) at (0,0) {$u_1$};
  \node (lB) at (1.1,0) {$u_2$};
  \node (lC) at (2.2,0) {$u_3$};
  \node (lD) at (3.3,0) {$u_4$};
  \draw (lA) -- (L1); \draw (lB) -- (L2); \draw (lC) -- (L3); \draw (lD) -- (L4);
  \draw (L1) -- (L2); \draw (L2) -- (L3); \draw (L3) -- (L4);
  % the right chain, glued on the first two legs
  \foreach \i/\x in {1/0.0,2/1.1,3/2.2} {
    \node[circle, fill, inner sep=1.6pt] (R\i) at (\x,3.0) {};
  }
  \node (rA) at (0.0,4.4) {$v_1$};
  \node (rB) at (1.1,4.4) {$v_2$};
  \node (rC) at (2.2,4.4) {$v_3$};
  \draw (rA) -- (R1); \draw (rB) -- (R2); \draw (rC) -- (R3);
  \draw (R1) -- (R2); \draw (R2) -- (R3);
  \draw (L1) -- (R1); \draw (L2) -- (R2);
  \node[align=center] at (5.0,1.4) {\footnotesize left chain\\ \footnotesize $u_1u_2$ glued};
  \node[align=center] at (5.0,3.0) {\footnotesize right chain};
  \node[align=left] at (0.4,5.2)
    {\footnotesize $W=\langle Y_{\mathrm{out}}\,|\,Y_L\bowtie Y_R\rangle$};
\end{tikzpicture}'''

JUNCTION = r'''\begin{tikzpicture}[thick, every node/.style={font=\small}]
  % a closed loop carrying the shared block, with one associator on its left
  \node[circle, fill, inner sep=1.6pt] (p) at (0,0) {};
  \node[circle, fill, inner sep=1.6pt] (q) at (1.5,0) {};
  \draw (p) to[out=150,in=30,looseness=1.5] node[above] {$d_{\mathrm{blk}}$} (q);
  \draw (q) to[out=-150,in=-30,looseness=1.5] node[below] {$d_{\mathrm{blk}}$} (p);
  \draw (p) -- (-1.0,0.6) node[left] {$f_0$};
  \draw (p) -- (-1.0,-0.6) node[left] {$f_1$};
  \draw (q) -- (2.5,0.6) node[right] {$g_0$};
  \draw (q) -- (2.5,-0.6) node[right] {$g_1$};
  \node[align=center] at (0.75,-1.9)
    {\footnotesize one $F$ recouples the left free part,\\
     \footnotesize the loop carries $1/\sqrt{d_{\mathrm{blk}}}$};
  \node[align=left] at (2.4,-1.9)
    {\footnotesize $+$ self-duality sign\\ \footnotesize when the left leg is dual};
\end{tikzpicture}'''

SWAP = r'''\begin{tikzpicture}[thick, every node/.style={font=\small}]
  \node (x1) at (0,0) {$x_1$};
  \node (x2) at (1.0,0) {$x_2$};
  \node (x3) at (2.0,0) {$x_3$};
  \node[circle, fill, inner sep=1.6pt] (a) at (0.5,1.0) {};
  \node[circle, fill, inner sep=1.6pt] (b) at (1.5,1.0) {};
  \draw (x1) -- (a); \draw (x2) -- (a);
  \draw (a) -- (b) node[midway, above] {$c_1$};
  \draw (x3) -- (b);
  \draw (b) -- (1.5,2.0) node[above] {$c_2$};
  \node at (2.7,1.0) {$=$};
  \node (y1) at (3.8,0) {$x_1$};
  \node (y2) at (4.6,0) {$x_2$};
  \node (y3) at (5.4,0) {$x_3$};
  \node[circle, fill, inner sep=1.6pt] (c) at (4.2,1.0) {};
  \node[circle, fill, inner sep=1.6pt] (d) at (5.0,1.0) {};
  \draw (y1) -- (c); \draw (y2) -- (c);
  \draw (c) -- (d);
  \draw (y3) -- (d);
  \draw (d) -- (5.0,2.0) node[above] {$c_2$};
  \node at (6.5,1.0) {$\sum_{f}$};
  \node[align=left] at (7.6,1.0)
    {\footnotesize $F(x_0,x_1,x_2;c_2;c_1,f)\,R(x_1,x_2;f)$\\[-2pt]
     \footnotesize $\times\,F(x_0,x_2,x_1;c_2;c_1',f)$};
\end{tikzpicture}'''

FIGURES = {
    'tree': TREE,
    'exact': EXACT,
    'diagram': DIAGRAM,
    'junction': JUNCTION,
    'swap': SWAP,
}


def main() -> None:
    '''Write every tikzpicture to `doc/fig-<name>.tex`.'''
    for name, body in FIGURES.items():
        path = HERE / f'fig-{name}.tex'
        path.write_text(f'% generated by doc/figures.py -- edit the script, not this file\n{body}\n')
        print(f'wrote {path.name} ({len(body)} chars)')


if __name__ == '__main__':
    main()
