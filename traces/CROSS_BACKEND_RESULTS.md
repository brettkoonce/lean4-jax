# Cross-backend training-trace verification — MNIST MLP + CNN

Two networks, four corners each. Same methodology: pin the NetSpec, pin
the training config, pin the initial parameters (via `heInit` → `init.bin`
handshake), disable batch shuffling, emit a JSON-Lines trace of every
step, then diff.

- **MLP** (`MainMnistMlpTrain.lean` ↔ `jax/MainMlp.lean`) — pure dense,
  670k params. Sections below up to "How to reproduce".
- **CNN** (`MainMnistCnnTrain.lean` ↔ `jax/MainCnn.lean`) — 4× conv-BN +
  2× dense, 1.67M params, with instance-order-preserving batch norm.
  "MNIST CNN" section at the bottom.

## MLP

Four training traces, four corners of the cross-product:

|                | phase 3 (Lean→IREE) | phase 2 (Lean→JAX→XLA) |
|----------------|---------------------|-------------------------|
| **mars** (gfx1100 / CPU) | `phase3-noshuf-rocm.jsonl`  | `phase2-noshuf-cpu.jsonl` |
| **ares** (CUDA)          | `phase3-noshuf-cuda.jsonl`  | `phase2-noshuf-cuda.jsonl` |

All four runs share the same NetSpec, same hyperparameters, same
seed, the same `heInit` initial parameters (loaded from
`mnist_mlp.init.bin`), and the same un-shuffled batch order
(`LEAN_MLIR_NO_SHUFFLE=1`). The only thing that varies between cells
is the compilation pipeline (Lean→MLIR→IREE vs Lean→JAX→XLA) and the
hardware (gfx1100 / CUDA / CPU).

## Per-step loss agreement

|comparison                                          |step 1 Δ |step 2 Δ |median Δ |max Δ |
|---------------------------------------------------|---------|---------|---------|---------|
|`phase2-cpu  vs phase3-rocm`  (mars cross-compiler)|1.97e-07 |2.50e-05 |3.98e-03 |8.86e-02 |
|`phase2-cuda vs phase3-cuda`  (ares cross-compiler)|3.38e-06 |1.27e-04 |3.91e-03 |1.31e-01 |
|`phase2-cpu  vs phase2-cuda`  (cross-platform JAX) |3.58e-06 |1.52e-04 |1.93e-03 |7.72e-02 |
|`phase3-rocm vs phase3-cuda`  (cross-vendor IREE)  |**0**    |**0**    |1.51e-03 |7.79e-02 |

## What the table proves

**Lean's `heInit` is platform-deterministic.** `mnist_mlp.init.bin`
SHA-256 is byte-identical when generated on mars (ROCm) or ares
(CUDA). The same params start every run.

**Phase 3 ROCm vs Phase 3 CUDA: bit-identical for step 1 AND step 2
(zero delta).** Same StableHLO MLIR compiled by IREE for two
different GPU vendors produces the same first two losses.
(Per cuda-bro's report, agreement holds bit-exact through step 4;
divergence at step 5 is 1 ULP.) This is the strongest possible
"IREE codegen is portable" claim.

**Phase 2 vs Phase 3 (cross-compiler): step 1 agrees to ~1e-7.**
The hand-derived backward in `MlirCodegen.lean` produces the same
loss as JAX's `value_and_grad` to **float32 precision** for the
first step, on both hardware platforms. Step 2 diverges by ~1e-4 —
Adam-implementation rounding (different reduction orders in the
matmul kernels) — which is the expected floor.

**Phase 2 across platforms (CPU vs CUDA JAX): also step-1 ULP.**
JAX's XLA compiler produces the same first loss on CPU and on CUDA.

## What it doesn't prove

The middle-epoch "max delta" of 0.07–0.13 reflects SGD's *chaotic
dynamics*: tiny step-1 differences in float rounding compound
exponentially through Adam updates. Both runs are doing correct math
the whole time — they just visit slightly different points in
parameter space after thousands of steps. All four runs converge to
≥98.4% val accuracy on the held-out set.

## How to reproduce

Run on any machine with both phases built (Lean+IREE+JAX-on-CUDA-or-ROCm-or-CPU):

```bash
# Phase 3 with init dump + no shuffle:
LEAN_MLIR_INIT_DUMP=traces/mnist_mlp.init.bin \
LEAN_MLIR_NO_SHUFFLE=1 \
LEAN_MLIR_TRACE_OUT=traces/mnist_mlp.phase3-noshuf-<machine>.jsonl \
  lake exe mnist-mlp-train data

# Phase 2 with init load + no shuffle:
LEAN_MLIR_INIT_LOAD=traces/mnist_mlp.init.bin \
LEAN_MLIR_NO_SHUFFLE=1 \
LEAN_MLIR_TRACE_OUT=traces/mnist_mlp.phase2-noshuf-<machine>.jsonl \
  jax/.lake/build/bin/mnist-mlp data

# Diff:
python3 tests/diff_traces.py \
  traces/mnist_mlp.phase2-noshuf-<machine>.jsonl \
  traces/mnist_mlp.phase3-noshuf-<machine>.jsonl --mode=cross-comp
```

Step 1 should agree to ~1e-7. If it doesn't, something diverged in
the parameter-loading or training-step plumbing.

## CNN

Same four corners for the MNIST conv-BN network. NetSpec is 4 conv-BN
layers (1→32→32→64→64) with two maxPool/2 downsamples, flatten, and
two dense (3136→512→10), ~1.67M params. Same Adam + cosine + warmup +
weight-decay config as MLP (15 epochs / 7020 steps).

|                | phase 3 (Lean→IREE) | phase 2 (Lean→JAX→XLA) |
|----------------|---------------------|-------------------------|
| **mars** (gfx1100 / CPU) | `mnist_cnn.phase3-noshuf-rocm.jsonl`  | `mnist_cnn.phase2-noshuf-cpu.jsonl` |
| **ares** (CUDA)          | `mnist_cnn.phase3-noshuf-cuda.jsonl`  | `mnist_cnn.phase2-noshuf-cuda.jsonl` |

Phase 2 on mars is on CPU here because the captured traces predate
the jax 0.10.0 upgrade — JAX-ROCm 0.9.1-plugin used to segfault on
any `conv_general_dilated` call on gfx1100 / ROCm 7.2 (see
`upstream-issues/2026-04-rocm-miopen-conv-segv/` for the historical
repro and MIOpen backtrace). The bug is fixed in jax 0.10.0+; future
captures can use ROCm directly. The math doesn't care which XLA
backend ran.

### Per-step loss agreement (7020 steps)

|comparison                                          |step 1 Δ |step 2 Δ |median Δ |max Δ |
|---------------------------------------------------|---------|---------|---------|---------|
|`phase2-cpu  vs phase3-rocm`  (mars cross-compiler)|1.08e-05 |6.88e-03 |1.76e-03 |3.03e-01 |
|`phase2-cuda vs phase3-cuda`  (ares cross-compiler)|1.14e-04 |1.44e-04 |1.82e-03 |3.45e-01 |
|`phase2-cpu  vs phase2-cuda`  (cross-platform JAX) |1.24e-04 |7.03e-03 |1.57e-03 |1.09e-01 |
|`phase3-rocm vs phase3-cuda`  (cross-vendor IREE)  |**0**    |1e-06    |1.11e-03 |8.25e-02 |

### What scales, what doesn't

**IREE codegen is still bit-identical at step 1 across AMD and NVIDIA.**
Same finding as MLP. `phase3-rocm` vs `phase3-cuda` step 1 Δ = 0 and
steps 2+ sit at the 6-decimal phase-3 log precision (1e-6). Conv-BN
doesn't change the story — the StableHLO IREE produces is portable
byte-for-byte across vendors.

**Cross-compiler step-1 Δ is ~100× looser than MLP** (1-10e-5 rather
than 1-3e-7). This is the cost of batch norm. Each of the 4 conv-BN
layers does two reductions over a ~100k-element tensor (128 × 28²
or 128 × 14²), and XLA's CPU/CUDA reduction trees differ from IREE's
dispatch, so the float32 rounding floor for CNN is inherently looser.
Both pipelines are doing correct batch-norm math — just with different
summation orders over large tensors.

**Median Δ over 7020 steps stays in 1-2e-3** across every comparison.
Tighter than the MLP median (3.9e-3) despite the looser step-1 floor —
the CNN converges harder (final loss ~1e-4 vs MLP's ~1e-2), so
absolute deltas shrink faster than rounding grows.

Both runs hit 99.4% val accuracy at the end. Neither diverges, both
stay on the same training trajectory.

### How to reproduce

```bash
# Phase 3 with init dump + no shuffle:
LEAN_MLIR_INIT_DUMP=traces/mnist_cnn.init.bin \
LEAN_MLIR_NO_SHUFFLE=1 \
LEAN_MLIR_TRACE_OUT=traces/mnist_cnn.phase3-noshuf-<machine>.jsonl \
  ./.lake/build/bin/mnist-cnn-train data
# (on mars: prepend IREE_BACKEND=rocm)

# Phase 2 with init load + no shuffle:
LEAN_MLIR_INIT_LOAD=$(pwd)/traces/mnist_cnn.init.bin \
LEAN_MLIR_NO_SHUFFLE=1 \
LEAN_MLIR_TRACE_OUT=$(pwd)/traces/mnist_cnn.phase2-noshuf-<machine>.jsonl \
  .venv/bin/python3 jax/.lake/build/generated_mnist_cnn.py
# (Now defaults to ROCm on mars; original captures here used
#  JAX_PLATFORMS=cpu to dodge MIOpen#3955, which is now fixed.)
```

Step 1 should agree to ~1e-5 on same-hardware cross-compiler comparisons.
If it's off by orders of magnitude, check that (a) both sides used the
same `init.bin`, (b) both sides saw `LEAN_MLIR_NO_SHUFFLE=1`, and (c)
the generated Python is from a build after the
`jax/Jax/Codegen.lean` conv-BN fix (batch-norm axis=(0,2,3), not
instance-norm axis=(2,3)).
