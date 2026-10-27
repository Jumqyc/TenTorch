# bench

| File | What it is |
| --- | --- |
| `bench_dmrg.py` | The Python driver end to end: a two site DMRG run on the Heisenberg chain, one line per sweep. |
| `build.py` | Compiles the engine sources and the five C++ benches below into `bench/build/<flavour>/<name>`. |
| `timing.h` | The two timing loops the C++ benches share (`repeat` for a fixed number of calls, `run_for` for a time budget) and `describe`, which prints how many codomain legs, domain legs and blocks an operand has. |
| `symbols.cpp` | The recoupling coefficients on their own: `F`, `R` and `CG` for SU(2) and SU(3), plus SU(3) `n_symbol` and `outcomes`. |
| `tensor_ops.cpp` | Moving legs on one tensor with `place`: one exchange, a cyclic shift, one leg crossing sides, and three legs crossing, on single copy and on two copy legs. |
| `contraction.cpp` | The two einsums a DMRG window spends its time in, and a plain bond `contract`, at bond multiplets 2, 4 and 8. |
| `paths.cpp` | Three graded einsums of the sizes the driver hits: two three leg, one four operand. |
| `su3_fusion.cpp` | Checks the SU(3) fusion against a weight based reference, then times it on large labels. |

## Running

```bash
.venv/bin/python bench/build.py --run                        # build everything, then run it
.venv/bin/python bench/build.py --run --only contraction paths --seconds 0.5
.venv/bin/python bench/build.py --profile --run              # -g -fno-omit-frame-pointer, for perf
.venv/bin/python -u bench/bench_dmrg.py 16 32 3              # L=16, bond 32, 3 sweeps
TENTORCH_DEVICE=cpu .venv/bin/python -u bench/bench_dmrg.py 32 64 3 --cat nosym
```

`build.py` compiles the eleven engine translation units plus the bench, then links against the
torch and Python that the `.venv` has. It uses `--jobs 1` by default: a single compile peaks at
about 2.5 GiB, and this box has 7.6 GiB. Nothing else heavy should run while it does.

`--seconds` is the budget each case gets; a case that is too slow to repeat often is timed for
that long and reports the mean.

## Numbers

Measured on this box (WSL, RTX 4060 present but unused here, single threaded BLAS) at
`--seconds 0.2`, `-O2`. Treat them as relative: run to run the same case moves by about ±10%, so
compare two implementations inside one run, never across runs.

`symbols`, warm cache:

| Call | µs |
| --- | --- |
| SU(2) `F(half, half, half -> 0)` / `-> one` | 0.029 / 0.035 |
| SU(2) `R(half, half)` | 0.015 |
| SU(2) `CG(half, half -> 0, copy 0)` | 0.024 |
| SU(3) `F(8, 8, 8 -> 1)` / `-> 8` | 0.043 / 0.040 |
| SU(3) `R(8, 8)` | 0.023 |
| SU(3) `CG(8, 8 -> 0, copy 0)` | 0.056 |
| SU(3) `n_symbol((6, 6), (1, 1), (0, 0))` | 0.039 |
| SU(3) `outcomes((6, 6), (6, 6))` | 0.023 |

`tensor_ops`, six half legs in one block:

| Move | µs |
| --- | --- |
| `place(1,0,2,3,4,5)` / `place(0,1,3,2,4,5)`, one exchange | 2.66 / 2.40 |
| `place(1,2,3,4,5,0)` / `place(3,4,5,0,1,2)`, a cyclic shift | 4.58 / 8.56 |
| `place({1..5}, {0})` / `place({0..4}, {5})`, one leg crosses | 5.04 / 2.35 |
| `place(0,1,2)`, three legs cross | 6.28 |
| `multi place(1,0,2)` / `multi place(2,1,0)` | 1.71 / 2.49 |

An exchange costs about 1 µs and a fold about 2 µs, which is what the rows above are: 1 exchange;
5 and 9 exchanges for the cyclic shifts; 5 exchanges and a fold, and a fold alone, for the two
single crossings; 3 exchanges and 3 folds for the three that cross together.

`contraction`, bond multiplets 2 / 4 / 8:

| Call | µs |
| --- | --- |
| `einsum("LSr,RUv->lsuv")` | 39.6 / 57.6 / 176.6 |
| `einsum("ASb,Bc,CdS->ad")` | 35.9 / 37.2 / 40.2 |
| `contract(two bond legs)` | 4.58 / 5.12 / 7.69 |

The contraction itself is only 4.6 µs at bond 2 and 7.7 at bond 8, so nearly all of the first row
is the turnover: `->lsuv` asks for the product with all four legs below, which append-only moves
cannot reach in fewer than eleven steps (two legs down, five exchanges, four folds). Ask for the
same result with the legs above and the row shrinks to the contraction — the letters are the whole
difference, so write the output on the side it is wanted.

`paths`: `"AB,CD,AC,BD->"` 37.1 µs, `"ASb,BTc,CUd,DVe->astuve"` 158.4 µs, `"AB,BC,CA->"` 27.4 µs.

`su3_fusion` first checks, then times: 225 sector pairs (1037 outcome triples) against the weight
based reference, 0 differ; the multiplicity sums agree (`(6,6) × (6,6)`: 85 outcomes,
Σ m·d = 117649, 643 µs). A cold `n_symbol` is 10–41 µs, and building a fresh `Tensor` is 178 µs —
both are why a first sweep costs more than the ones after it.

`bench_dmrg.py`, u1, two site DMRG, `n_iter = 5`, CPU:

| Chain | bonds | CPU (s), five sweeps | GPU (s) |
| --- | --- | --- | --- |
| L=16, bond 32 | 32 | 0.30 / 0.32 / 0.28 / 0.23 / 0.23 | 2.90 / 2.63 / 2.39 |
| L=32, bond 64 | 64 | 1.42 / 2.17 / 2.49 / 1.97 / 1.67 | 11.16 / 13.15 / 9.47 |

The CPU rows are 0.6–0.9× what the same runs cost before the carrying chain was reworked
(0.46/0.46/0.39/0.31/0.35 and 2.42/3.21/3.11/2.18/2.15): the chain now keys each block plan by an
interned leg structure instead of a rebuilt vector, reuses its intermediate buffers, and reads a
block's shape from the plan rather than from the layout, so a sweep spends a third less time
planning. The GPU column is from the earlier engine and has not been re-measured.

The GPU is 6–9× slower at these sizes: a block is a few hundred entries, so each `matmul` is a
~19 µs kernel launch plus allocator churn on the GPU against a few µs of BLAS on the CPU. For
reference, TeNPy 1.1.1 with the same Hamiltonian, `chi_max` and mixer off does L=16 in
0.08 / 0.22 / 0.29 s and L=32 in 0.15 / 0.39 / 0.52 s, so this engine is level at L=16 and 2–4×
slower at L=32.

## Reading the numbers

- **The cost is the bytes moved, not the FLOPs.** The same window action as one einsum is 1 ms,
  as a pre contracted eight leg operator it is 110 ms: the operator holds 10⁶ entries and every leg
  it must move is a gather over all of them. Do not pre contract the environments.
- **A new shape pays for a plan.** The first call of a leg structure builds the plan of every block
  and step, at a few tens of microseconds each; the same shape afterwards only looks them up. That
  is the whole difference between the first sweep and the rest.
- **The case of a letter is a layout instruction.** For a free leg the case says which side that
  operand should keep it on, and a mismatch costs the turnover above; for a summed leg it is
  ignored. So write the output in the layout the caller wants and let the einsum do the move it has
  to do anyway: as a separate `place` afterwards the same move copies the whole result one more
  time, which is what the driver used to do on the window tensor.
- **Memory settles, it does not creep.** The resident set is 690 MiB before a single tensor exists
  (torch and the extension), 865 MiB at L=16/bond 32 and 1.4 GiB at L=32/bond 64, and each of those
  is reached by the third or fourth sweep and then flat. What a run accumulates is the plan of every
  block it has walked through, which the bond stops changing once it is saturated; the plan cache
  keeps at most 256 MB of it and drops the lot past that, so a run that keeps meeting new shapes
  rebuilds plans instead of growing.
- **`describe` prints the layout** of every operand (`2 cod, 1 dom, 2 blocks`), which is the thing
  the letters have to agree with.
