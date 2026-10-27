# Tests

```shell
.venv/bin/python -m pytest tests -q                    # 26 tests, about two seconds
TENTORCH_SLOW=1 .venv/bin/python -m pytest tests -q    # adds the long chain, about 45 s
```

`conftest.py` seeds the global torch generator before every test, so a random state is reproducible.
`exact.py` holds the dense diagonalisation the sweeps are checked against, which is a test helper
and not part of the package.

## What is covered

`test_mps.py` — the mixed canonical form and the expectation value.

- the norm is the same wherever the centre sits, for no symmetry and for $U(1)$;
- the expectation value of the Heisenberg operator is the same wherever the centre sits;
- a product state has the energy its spins give, one $-\tfrac14$ per aligned pair.

`test_dmrg.py` — the sweep, and the numbers it produces.

- the energy reaches exact diagonalisation for $L = 4, 6, 8$, with no symmetry and with $U(1)$;
- a run leaves a normalised state, and writes the state and the energy onto the model;
- `DMRG.energy()` agrees with `MPS.expectation` on the converged state;
- a larger bond reaches a lower energy, and a bond of two throws weight away where the full bond
  throws none;
- the bond never exceeds the cap it was given;
- `energy()` without a sweep and `run()` without a sweep raise instead of returning nonsense.

With `TENTORCH_SLOW=1`, one system test checks the bulk energy density of the spin-$\tfrac12$
Heisenberg chain, $\epsilon_0 = \tfrac14 - \ln 2 = -0.4431471805599453$, on $L = 64$ with a bond of
64 multiplets to within $5 \times 10^{-3}$. The finite chain is open, so its energy density sits
above the bulk value.

## What is not covered

- **The engine itself.** The `Tensor`, `einsum`, `svd`, `qr` and `rq` checks are C++ programs, and
  they are not in this repository: the Python tests only exercise them through the networks. A bug
  inside a leg move that still returns a consistent tensor would not be caught here.
- **Non-abelian operators.** There is no $SU(2)$, $SU(3)$ or Fibonacci `MPO` yet, so the end-to-end
  sweep cannot be tested for them; the categories are only exercised at the tensor level, through
  the dense checks in the engine.
- **The GPU path.** Every test runs on whatever `TENTORCH_DEVICE` names, which defaults to CUDA when
  torch sees a device; nothing asserts that the two devices agree.
- **Truncation edge cases.** An SVD that drops a whole charge sector, a bond that hits the cap
  exactly, or a state whose total charge admits no bond at all are not tested.
- **The einsum semantics on their own.** The upper case / lower case reading of the equation — which
  side a leg is asked to take, and that a summed leg keeps its side — is only tested through the
  DMRG specs, never equation by equation.
