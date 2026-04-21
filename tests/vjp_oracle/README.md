# VJP oracle

Differential tests using JAX's `value_and_grad` as an oracle for the
hand-derived VJPs in `LeanMlir/Proofs/`.

## How it works

Each test case is a minimal NetSpec that exercises one axiom in
isolation. The runner:

1. Trains one batch in **phase 3** (Lean → MLIR → IREE) with
   `LEAN_MLIR_INIT_DUMP` and `LEAN_MLIR_NO_SHUFFLE=1`, writing a trace.
2. Trains the same batch in **phase 2** (Lean → JAX → XLA) starting
   from the identical init (`LEAN_MLIR_INIT_LOAD` points at the phase-3
   dump) with `LEAN_MLIR_NO_SHUFFLE=1`, writing a trace.
3. Diffs step-2 loss. Step 1 is forward-only; step 2 is the first
   step whose value depends on the backward pass + optimizer. A small
   `|step2-loss-delta|` means the hand-derived VJP matches JAX autodiff
   at float32 precision.

## Running

```bash
./tests/vjp_oracle/run.sh             # default: all cases
./tests/vjp_oracle/run.sh dense       # one case
```

Before running, build the binaries:

```bash
lake build vjp-oracle-dense
(cd jax && lake build vjp-oracle-dense)
```

On mars, phase 2 needs `JAX_PLATFORMS=cpu` (see
`upstream-issues/2026-04-rocm-miopen-conv-segv/`). The runner picks
it up from the environment.

## Test cases

| case | axiom probed | expected step-2 Δ |
|---|---|---|
| `dense` | `dense_has_vjp` + `softmaxCE_grad` | ~1e-5 (f32 dense floor) |

More to add: `dense_relu`, `conv_only`, `convbn_only`, `residual`, ...
one per axiom in `LeanMlir/Proofs/`.

## Adding a case

1. Create `Main<CaseName>.lean` at repo root (phase 3) and
   `jax/Main<CaseName>.lean` (phase 2) with the same NetSpec + cfg.
   Keep the cfg minimal — no cosine, no warmup, no wd, no augment,
   `batchSize := 4`, `epochs := 1`.
2. Add `lean_exe` entries for each binary to the two lakefiles.
3. Add the case name to the default loop in `run.sh`.
4. Document expected tolerance in the table above.
