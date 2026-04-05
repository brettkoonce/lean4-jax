# Lean → MLIR → IREE: What We Built

Notes on the path from "Lean emits JAX Python" to "Lean trains MNIST end-to-end
via native IREE runtime calls, no Python at runtime." Picks up where
`Lean_MLIR.md` left off.

## Result

**MNIST MLP trained in Lean, 97.87% accuracy, 12 epochs, ~200s on RTX 4060 Ti.**
Matches the JAX baseline (97.75%) within training noise. Lean orchestrates,
CUDA executes, zero Python processes at runtime.

```
$ .lake/build/bin/mnist-mlp-train
Epoch  1: loss=0.364  acc=92.63% (16s train + 0.6s eval)
Epoch  6: loss=0.068  acc=97.40%
Epoch 12: loss=0.026  acc=97.87%
```

## Architecture

```
Lean NetSpec
     │
     ▼ MlirCodegen.generate (Lean, ~80 LOC)
forward.mlir (StableHLO)
     │
     ▼ iree-compile (pip)                  JAX export.export (bootstrap)
forward.vmfb                           train_step.mlir
                                              │
                                              ▼ iree-compile
                                          train_step.vmfb
     │                                         │
     ▼                                         ▼
Lean training loop (MainMlpTrain.lean)
     │   - load MNIST (IDX → FloatArray)
     │   - He-init packed params
     │   - for each batch: IreeSession.mlpTrainStep  ────► FFI ──► GPU
     │   - eval: IreeSession.mlpForward              ────► FFI ──► GPU
     ▼
accuracy
```

Two `.vmfb` modules, one Lean binary. All GPU calls flow through
`libiree_ffi.so` (our 1.4 MB runtime wrapper, statically-linked IREE + flatcc).

## How we got here

### Step 1: Toolchain smoke test

Installed `iree-base-compiler` + `iree-base-runtime` via pip. Hand-wrote a
tiny `dense→relu` StableHLO module (`mlir_poc/tiny_mlp.mlir`). Compiled with
`iree-compile`, ran with `iree-run-module`. CPU backend worked first try.
CUDA backend errored with "missing GPU target in #hal.executable.target."

**Gotcha #1: sm_89 is broken in IREE 3.11.** Known upstream issues
[iree-org/iree#21122](https://github.com/iree-org/iree/issues/21122) and
[#22147](https://github.com/iree-org/iree/issues/22147). The compiler lacks
GPU target metadata for Ada and newer architectures. Workaround: use
`--iree-cuda-target=sm_86` (Ampere). PTX is forward-compatible, so the CUDA
driver JITs sm_86 PTX to sm_89 at load time. Verified correct numerical
output on 4060 Ti.

### Step 2: Lean codegen emits StableHLO

Wrote `LeanJax/MlirCodegen.lean` (~80 LOC) mirroring the JAX codegen pattern.
Walks `NetSpec.layers`, emits `stablehlo.dot_general` + `broadcast_in_dim`
+ `add` + `maximum` per dense-ReLU pair. Scope is MLP-only for this phase.

Generated MLIR diffs cleanly against the hand-written version from Step 1.
Accuracy validated end-to-end: Lean-generated `.vmfb` predicts **identically**
(0 diffs / 9984 samples) vs JAX on a trained MLP. Fp32-noise agreement
(2.4e-6 max diff) with numpy reference.

### Step 3: FFI via subprocess (and why it's dead on arrival)

First attempt at orchestrating inference from Lean: shell out to
`iree-run-module` per batch.

**Measured: 770ms per subprocess call.** Of which ~250 µs is actual GPU
compute, and 769.75 ms is IREE runtime init + CUDA device init + module load,
paid every single time. Training MNIST at 12 epochs × 469 batches = 5628
calls → **72 minutes** of subprocess launch overhead alone.

Unusable. Needed a persistent runtime session.

### Step 4: IREE from source, runtime-only

Cloned `iree-org/iree`. Naive recursive clone pulled in LLVM via
torch-mlir/stablehlo submodule chains and ballooned to 9 GB+ with no end in
sight. Killed it.

**Gotcha #2: Submodule discipline.** IREE's `build_tools/scripts/git/runtime_submodules.txt`
lists the 10 submodules actually needed for a runtime-only build. Shallow
clone + init those → 470 MB total. Build tree sits at
`/home/skoonce/lean/klawd_max_power/iree-build/`.

CMake flags:
```
-DCMAKE_BUILD_TYPE=Release
-DIREE_BUILD_COMPILER=OFF              # we use pip's iree-compile
-DIREE_BUILD_TESTS=OFF
-DIREE_BUILD_SAMPLES=OFF
-DIREE_HAL_DRIVER_DEFAULTS=OFF
-DIREE_HAL_DRIVER_CUDA=ON
-DIREE_HAL_DRIVER_LOCAL_SYNC=ON
-DIREE_HAL_DRIVER_LOCAL_TASK=ON
-DBUILD_SHARED_LIBS=OFF                # static, link into our own .so
```

Runtime-only ninja build took **~30 seconds** on the box. Produces
`libiree_runtime_unified.a` (2.3 MB static) containing everything we need.

### Step 5: C FFI wrapper

Wrote `ffi/iree_ffi.c` — a ~150 LOC thin wrapper over IREE's high-level
`iree_runtime_*` API. Exposes three functions:

```c
iree_ffi_session_t* iree_ffi_session_create(const char* vmfb_path);
void                iree_ffi_session_release(iree_ffi_session_t* sess);
int                 iree_ffi_invoke_f32(sess, fn_name,
                                        n_inputs, ranks, dims_flat, input_data,
                                        n_outputs, output_totals, output_data);
int                 iree_ffi_train_step_mlp(...);  // int32 labels + scalar lr
```

**Gotcha #3: `IREE_ALLOCATOR_SYSTEM_CTL`.** The `iree_allocator_system()`
function is gated behind a compile-time macro. Compiler invocation needs
`-DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl`.

**Gotcha #4: Flatcc split.** `flatcc_verify_*` symbols live in
`libflatcc_parsing.a`, not `libflatcc_runtime.a`. Both needed under
`--start-group/--end-group` for mutual symbol resolution.

**Gotcha #5: Function namespacing.** MLIR `module @mnist_mlp { func.func @forward }`
is invoked as `mnist_mlp.forward`, not `module.forward`.

**Gotcha #6: One-shot driver registration.** `iree_hal_cuda_driver_module_register`
is global; calling it twice (e.g. for two sessions) returns `ALREADY_EXISTS`.
Guard with a static flag inside session_create.

Packaged as `ffi/libiree_ffi.so` (1.4 MB). The IREE runtime + flatcc are
`--whole-archive`'d in, so consumers just link `-liree_ffi`.

**Measured FFI performance: 7.0 ms per call.**
**110× faster than subprocess** (770ms → 7ms). Pure GPU compute is still
~250 µs, so the remaining 6.7ms is buffer alloc + host↔device transfer
per call.

### Step 6: Lean FFI bindings

`ffi/iree_lean_ffi.c` bridges Lean's `FloatArray` (Float64) and `ByteArray`
to the C wrapper. Converts f64↔f32 at the boundary, handles packed int32
labels, wraps opaque session pointers in Lean external classes for GC.

`LeanJax/IreeRuntime.lean` declares three `@[extern]` functions:

```lean
opaque IreeSession : Type
def IreeSession.create     (path : @& String) : IO IreeSession
def IreeSession.mlpForward (sess, x, W0, b0, W1, b1, W2, b2, batch) : IO FloatArray
def IreeSession.mlpTrainStep (sess, params, x, y, lr, batch)        : IO FloatArray
```

`mlpTrainStep` uses a **packed-params** convention: all 669,706 MLP weights
flow as a single `FloatArray` in (6 concatenated tensors) and the same flat
layout out (plus loss appended at index 669706). Keeps the FFI surface
narrow.

Lakefile wiring uses a custom `target ireeLeanFfiO` that compiles the shim
.c file with Lean headers, wraps it in an `extern_lib`, and adds
`-liree_ffi` + rpath to `moreLinkArgs`.

**Gotcha #7: `--no-allow-shlib-undefined`.** Lean's bundled clang/lld is
strict about symbols referenced by shared libraries. Our `libiree_ffi.so`
references glibc symbols (`log2f`, `dlopen`, etc.) that ld.lld refuses to
resolve transitively. Pass `-Wl,--allow-shlib-undefined` to override.

**Measured Lean→FFI→GPU: 7.8 ms per call** (vs 7.0 ms direct C). 0.8 ms
of Lean overhead is the Float64→Float32 staging.

### Step 7: JAX-bootstrap train_step

Per the `Lean_MLIR.md` plan, Option B (bootstrap via `jax.export.export`)
gives us a known-correct training module while deferring hand-written
VJPs (Option A) to a pure refactor phase.

`mlir_poc/export_train_step.py` uses JAX to define forward + softmax-CE +
`value_and_grad` + SGD update, then exports via:

```python
exported = export.export(jax.jit(train_step))(
    spec_W0, spec_b0, ..., spec_x, spec_y, spec_lr)
open("train_step.mlir", "w").write(exported.mlir_module())
```

Produces 20 KB of StableHLO. The exported function is `jit_train_step.main`,
taking 9 inputs (6 params + x + y labels + lr scalar) and returning 7 outputs
(6 updated params + scalar loss).

Verified numerically: same random inputs → IREE output matches JAX to
fp32 noise (1.5e-8 on weights, 0.0 on loss).

### Step 8: Training loop in Lean

`LeanJax/MnistData.lean` parses IDX format (big-endian header + u8 pixels,
u8 labels) into `FloatArray` (images normalized to [0,1]) and `ByteArray`
(labels packed as int32 LE for the FFI). ~50 LOC.

`MainMlpTrain.lean`:
- Loads MNIST train (60k) + test (10k)
- Creates two IREE sessions: one for `mlpTrainStep`, one for `mlpForward`
- He-initializes packed params (pseudo-Gaussian via 3-sum of uniforms)
- For each epoch: 468 batches, calls `mlpTrainStep`, tracks mean loss
- After each epoch: unpacks params, runs 78 test batches, computes accuracy

**Result after 12 epochs: 97.87%.** No shuffle, no weight decay, no fancy
init — plain SGD with `lr=0.1`, matching the S4TF book's MLP recipe.

## Performance picture

| Stage | Time per call | Use case |
|---|---|---|
| `iree-run-module` subprocess | 770 ms | forbidden for training |
| Direct C FFI | 7.0 ms | C clients |
| Lean → FFI → GPU | 7.8 ms | current training loop |
| Pure GPU compute (iree-benchmark-module) | 250 µs | theoretical ceiling |

**Per-epoch wall clock: 16s.** This is ~20× slower than JAX-CPU, but the
bottleneck is NOT compute:

- 669,706 f64→f32 conversions on every step (~5 MB of Lean-heap activity)
- `sliceImages` does 100,352 `FloatArray.push` calls per batch (→ 47M/epoch)
- Params shipped host→device every step; nothing persists across calls

The GPU is idle most of the time. Closing the gap:

- **Persistent on-device params** (~3× win) — ship weights once, update
  in-place on GPU. Requires IREE output-buffer reuse semantics.
- **ByteArray FFI variant** (~2× win) — store params as raw float32 bytes
  in Lean, skip the f64 conversion entirely.
- **Pre-sliced batch views** — compute the 468 batch offsets at load time,
  reuse buffers.

Together these should bring us under 2 ms/step (~1 s/epoch), competitive
with JAX.

## CNN extension (partial)

`MlirCodegen.lean` now handles `.conv2d`, `.maxPool`, and `.flatten` in
addition to `.dense`. Walks layer list tracking current tensor shape (flat
or NCHW), emits an input reshape if the first layer is conv, advances a
param index on each conv/dense layer.

**Smoke test (`mnist_cnn.mlir`, hand-written):** 2 conv + max pool + flatten
+ 3 dense. Compiles for both CPU and CUDA. Matches JAX reference to 1e-6
(batch 4) and 1e-6 (batch 128) via `mlir_poc/validate_cnn.py`.

**Lean-generated CNN (`MainCnnMlir.lean`):** `NetSpec` → 4704-char MLIR →
`.vmfb`. Numerical match with JAX at batch 128: **max diff 1.0e-6**.

**CNN forward perf (batch 128, 4060 Ti):** 7.29 ms/call. 29× more compute
than the MLP forward (0.25 ms), as expected — conv FLOPs dominate.

**CNN *training* is blocked** by an IREE bug in StableHLO→linalg lowering:
the backward convolutions that JAX autodiff generates have non-standard
`dim_numbers` like `[f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f]`, which IREE's
pipeline miscompiles to `linalg.conv_2d_nhwc_hwcf` with a malformed
`strides` attribute. Minimal repro: `jax.grad(sum(conv(x,W)**2))` fails
to compile with IREE 3.11. This blocks the JAX-bootstrap path (Option B)
for any convolutional model.

The forced path is **Option A (hand-written VJPs in MLIR)** for CNN.
That's the next real chunk of work: conv backward (dW and dx), pool backward
(argmax scatter), plus reusing the existing dense backward. ~400 LOC of
MLIR-emission pattern, non-trivial but mechanical.

Projected CNN training wall time once backward is written: ~140s for 12
epochs × 468 batches × ~25 ms/step (compute-bound, unlike MLP which was
FFI-overhead-bound).

## What's next

1. **Hand-written VJPs** for dense + conv + pool — needed for CNN training
   and now the critical path, not optional.
2. **Perf optimizations** for MLP — persistent on-device weights,
   ByteArray FFI variant. Would close the 20× gap vs JAX on MNIST MLP.
3. **More architectures.** CIFAR-10 CNN is near-free once the above lands
   (same ops, different dataset loader). ResNet-34 needs `convBn` +
   instance norm + residual skip — real StableHLO variety.

## File map

```
LeanJax/
  MlirCodegen.lean          Lean NetSpec → StableHLO emitter (~80 LOC, MLP-only)
  IreeRuntime.lean          @[extern] bindings to libiree_ffi.so
  MnistData.lean            IDX parser

ffi/
  iree_ffi.c / .h           generic C wrapper (3 fns over iree_runtime_*)
  iree_lean_ffi.c           Lean shim (Float64↔f32, packed params)
  libiree_ffi.so            1.4 MB, static IREE runtime + flatcc inside
  test_ffi.c                C smoke test

mlir_poc/
  export_train_step.py      JAX bootstrap for train_step MLIR
  tiny_mlp.mlir             hand-written smoke test
  validate_mnist_e2e.py     accuracy check: Lean .vmfb vs JAX

MainMlpMlir.lean            codegen → compile → one forward pass
MainMlpTrain.lean           full training loop, He init, eval
TestIreeRuntime.lean        FFI session/invoke smoke test
TestTrainStep.lean          single train-step loop on one batch
```

Upstream dependencies live sibling to this repo at
`/home/skoonce/lean/klawd_max_power/iree/` (source, 470 MB) and
`iree-build/` (build tree). `libiree_ffi.so` links the runtime statically,
so the shipped binary has no IREE build-tree dependency at runtime.
