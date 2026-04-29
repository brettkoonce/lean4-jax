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

Phase 2 runs on whatever JAX targets by default — ROCm on AMD hosts,
CUDA on NVIDIA hosts. The earlier `JAX_PLATFORMS=cpu` requirement on
mars (ROCm/MIOpen#3955) is fixed in jax 0.10.0+; the runner no longer
auto-pins JAX to CPU. If you need to force CPU for any reason, pass
`JAX_PLATFORMS=cpu` in your environment.

## Test cases

| case | axiom probed | tolerance | observed step-2 Δ |
|---|---|---|---|
| `dense` | `dense_has_vjp` + `softmaxCE_grad` | 1e-5 | 2.73e-07 |
| `dense-relu` | `relu_has_vjp` + `vjp_comp` | 1e-5 | 4.77e-07 |
| `conv` | `conv2d_has_vjp` + `flatten_has_vjp` | 1e-5 | 2.24e-07 |
| `convbn` | `convBn_has_vjp` (conv + BN + ReLU) | 1e-4 | 2.18e-06 |
| `conv-pool` | `maxPool_has_vjp` | 1e-3 | 1.18e-04 |
| `residual` | `biPath_has_vjp` (additive fan-in VJP) | 1e-5 | 3.06e-07 |
| `depthwise` | depthwise-conv VJP via `.invertedResidual` | 1e-4 | 1.11e-05 |
| `attention` | ViT block: `patchEmbed` + `transformerBlock_has_vjp_mat` + classifier | 1e-5 | 1.81e-07 |
| `mbconv` | `elemwiseProduct_has_vjp` (SE gate) + MBConv composition | 1e-5 | 1.56e-06 |
| `global-avg-pool` | `globalAvgPool` spatial-mean VJP | 1e-5 | 1.35e-06 |
| `bottleneck` | `.bottleneckBlock` (ResNet-50 1×1-3×3-1×1 + skip) | 1e-4 | 8.78e-06 |
| `mbconv-v3` | `.mbConvV3` with h-swish + h-sigmoid SE | 1e-4 | 5.81e-06 |
| `fused-mbconv` | `.fusedMbConv` (EfficientNet V2 block) | 1e-4 | 8.09e-07 |
| `uib` | `.uib` (MobileNet V4 universal inverted bottleneck) | 1e-4 | 1.13e-05 |

Why tolerances differ:
- Dense / relu / conv: step-2 Δ sits at or below 1 ULP. 1e-5 is comfortable headroom.
- BN: variance reductions over ~100k-element tensors between XLA and
  IREE diverge at the f32-reduction-tree level, pushing step-2 Δ to
  ~1e-6. 1e-4 keeps headroom.
- MaxPool: argmax tiebreaks disagree between XLA and IREE when two
  input elements are equal (common on MNIST's sparse zero pixels),
  routing gradient mass to different input positions. Mathematically
  valid either way but produces ~1e-4 step-2 Δ. 1e-3 keeps headroom.

More to add: `se_block` (mbConv with SE on), `layernorm` in isolation,
other transformer variants — each requires a matching
`init_params_from_file` extension in `jax/Jax/Codegen.lean` for its
param layout.

## Init heuristic — fixed

Earlier versions of `LeanMlir.SpecHelpers.heInitParams` used a peek-
ahead heuristic to decide whether a rank-1 tensor following a rank-≥2
tensor was a bias (0.0) or the γ of a BN/LN pair (1.0 with next rank-1
as β=0.0). That heuristic misfired on `.patchEmbed` (bias and cls_token
look like a γ/β pair) and at transformer-block boundaries (`Output_b`
and `LN2_γ`, `fc2_b` and `FinalLN_γ`), leading to biases = 1.0 and LN
γ's = 0.0 which collapsed several layers to zero. Both phases saw the
same bug so the oracle still passed, but attention's numerical signal
was degenerate.

`heInitParams` now dispatches per Layer constructor — no shape-peek
heuristic, so the semantics (bias vs γ vs cls_token) are unambiguous.
`attention` step-2 Δ tightened from 6.85e-07 to 1.81e-07 after the fix:
we're now verifying a non-degenerate transformer forward pass, and it
still agrees with JAX autodiff at 1 ULP.

All non-ViT cases produce byte-identical init as before the fix — the
peek-ahead happened to work correctly for conv+BN layouts and was only
wrong at patchEmbed / transformer boundaries.

## Adding a case

1. Create `Main<CaseName>.lean` at repo root (phase 3) and
   `jax/Main<CaseName>.lean` (phase 2) with the same NetSpec + cfg.
   Keep the cfg minimal — no cosine, no warmup, no wd, no augment,
   `batchSize := 4`, `epochs := 1`.
2. Add `lean_exe` entries for each binary to the two lakefiles.
3. Add the case name to the default loop in `run.sh`.
4. Document expected tolerance in the table above.
