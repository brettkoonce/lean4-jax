# Lean → MLIR: Plan

Notes and implementation plan for lowering Lean `NetSpec`s to MLIR/StableHLO,
compiled via IREE, orchestrated from Lean with no Python at runtime.

## Context

This project (`lean4-mlir`) has working Lean → JAX and Lean → MLX backends.
Both match JAX baseline accuracy on MNIST, CIFAR-10, and ResNet-34 on Imagenette.
The codegen pattern is established: walk the `NetSpec`, emit framework-specific
text, shell out to Python to execute.

The pitch for MLIR: same codegen pattern, but emit **StableHLO text** instead
of Python. Compile via IREE (`iree-compile`) to a portable `.vmfb` bytecode
module. Lean itself orchestrates training via C FFI to the IREE runtime. No
Python at runtime, no framework dependency when the model actually runs.

This is the "last chapter of the book" architecture that wasn't ready in the
S4TF era but is operational today.

## Architecture

```
Lean NetSpec
     │
     ▼ MlirCodegen.generate (emit StableHLO text)
model.mlir
     │
     ▼ iree-compile --iree-hal-target-backends=cuda
model.vmfb  (IREE bytecode module)
     │
     ▼ Lean orchestrator calls libiree_runtime via FFI
     │   - iree_runtime_session_create
     │   - iree_runtime_session_append_bytecode_module_from_file
     │   - iree_runtime_call_initialize_by_name
     │   - iree_runtime_call_inputs_push_back_buffer_view
     │   - iree_runtime_call_invoke
     │   - iree_runtime_call_outputs_pop_front_buffer_view
     ▼
GPU execution (CUDA via IREE's HAL)
     │
     ▼ results back to Lean as ByteArray buffer
Lean orchestrator (training loop, optimizer state, data loading)
```

## What each layer owns

| Layer | Code owner | Maintenance burden |
|-------|------------|---------------------|
| `NetSpec`, `MlirCodegen`, training orchestrator | us (Lean) | all of it |
| IREE runtime C ABI bindings | us (Lean FFI, ~200 LOC) | thin wrapper |
| `libiree_runtime_bundled.so` | IREE project | zero (prebuilt) |
| `iree-compile` CLI | IREE project | zero (subprocess) |
| CUDA driver / kernels | NVIDIA | zero |

The only code we maintain is Lean. Everything else is a stable C ABI or
a CLI tool.

## Dependencies

```bash
pip install iree-compiler iree-runtime   # for iree-compile CLI
# or build from source: https://github.com/iree-org/iree
```

Runtime needs `libiree_runtime_bundled.so` on the linker path. `iree-runtime`
pip package provides this in its site-packages directory.

---

## Phase 1: MNIST MLP — Inference Only

Goal: take a trained MNIST MLP (weights from existing JAX run or freshly
initialized) and run forward pass through the Lean → MLIR → IREE → CUDA
pipeline. Validate the full pipeline works end-to-end before touching autodiff.

### Scope

**What's in:**
- Emit StableHLO MLIR for: `dense`, `relu`, `identity` (passthrough)
- Emit `@forward` function taking x and weights, returning logits
- IREE C FFI bindings in Lean (~10 extern declarations)
- Lean orchestrator that loads MNIST, loads trained weights, runs forward, checks accuracy
- Validation: match accuracy against JAX-trained MLP on same test set

**What's out (Phase 2+):**
- Training, backward pass, optimizer
- Conv, pool, BN, residual blocks (those come after architecture is validated)

### Implementation steps

1. **Create `LeanMlir/MlirCodegen.lean`** — emit StableHLO for MLP
   - `emitHeader`: module preamble
   - `emitForward`: walk layer list, emit `stablehlo.dot_general` + broadcast_add + activation
   - `generate`: assemble
   - Target: generate valid StableHLO for `mnistMlp` spec

2. **Verify with `iree-compile` CLI**
   - Write the emitted .mlir to file
   - Run `iree-compile model.mlir --iree-hal-target-backends=cuda -o model.vmfb`
   - Also run with `--iree-hal-target-backends=llvm-cpu` as a fallback verification

3. **Verify with `iree-run-module` CLI** (before touching FFI)
   - Run the compiled module with random inputs via CLI
   - Confirms MLIR is valid and the compiled artifact executes

4. **Write IREE C FFI bindings in Lean**
   - `LeanMlir/IreeRuntime.lean`
   - Core types: `Session`, `BufferView` (opaque pointers)
   - Functions needed:
     ```
     iree_runtime_instance_create
     iree_runtime_session_create_with_device
     iree_runtime_session_append_bytecode_module_from_file
     iree_runtime_call_initialize_by_name
     iree_runtime_call_inputs_push_back_buffer_view
     iree_runtime_call_invoke
     iree_runtime_call_outputs_pop_front_buffer_view
     iree_hal_buffer_view_create
     iree_hal_buffer_view_read
     ```
   - Reference: `iree/runtime/api.h`, `iree/samples/runtime/c/` in IREE repo

5. **Transfer trained weights from JAX**
   - One-off Python script: train MLP with existing `MainMlp.lean`, serialize params to a binary file (e.g., `mlp_weights.bin` — concatenated float32 arrays with a simple header)
   - Lean reads this file, passes buffers as inputs to `@forward`

6. **Orchestrator in Lean**
   - `MainMlpMlir.lean` — load MNIST, load weights, loop through test set, accumulate accuracy
   - Compare against JAX run's final test accuracy (~97.9%) — should match within 1e-4

### Open decisions for Phase 1

- **Input layout**: IREE uses row-major. StableHLO tensors are just `tensor<?x784xf32>`. No layout translation needed (unlike NHWC vs NCHW for MLX).
- **Dynamic vs static batch size**: easier to start with static (e.g., `tensor<128x784xf32>`). Dynamic (`tensor<?x784xf32>`) needs `--iree-input-type=stablehlo` and potentially more compilation time. Start static.
- **How to pass params**: either (a) bake weights into the compiled module as constants (`stablehlo.constant` with dense data), or (b) pass as separate function arguments. Option (b) is cleaner — module stays generic, weights passed at call time.

### Success criteria

- `lake build mnist-mlp-mlir` produces an executable
- Running it produces accuracy matching JAX-trained MLP within 1e-4
- No Python processes involved at runtime (verify with `pgrep python` during run)
- Binary runs with only IREE + CUDA libs (no libpython, no libjax, no libmlx)

---

## Phase 2: MNIST MLP — Training

This is where the autodiff decision happens. Three options:

### Option A: Emit backward MLIR by hand (pragmatic middle)

Define VJP rules per layer directly in Lean, emit both forward and backward as
MLIR functions. For MLP, only need VJPs for:
- `dense(x, W, b)`: dx = dy @ W, dW = x.T @ dy, db = sum(dy, axis=0)
- `relu(x)`: dx = dy * (x > 0)
- `softmax_cross_entropy(logits, labels)`: dlogits = softmax(logits) - one_hot(labels)

Emit:
```mlir
func.func @forward(params, x) -> y
func.func @backward(params, x, y, dy) -> (dparams, dx)
func.func @update(params, dparams, lr) -> new_params
func.func @train_step(params, x, y, lr) -> (new_params, loss)
```

Scope: ~1 week for MLP. Clean and self-contained. VJP rules per layer type.

### Option B: Bootstrap via JAX (keep JAX as build-time dep)

Use `jax.export.export` to generate the training-module MLIR once from the JAX
code we already have. JAX computes gradients correctly; we just extract the
resulting StableHLO. Save the `.mlir` artifact, ship the `.vmfb`.

```python
import jax
from jax import export

@jax.jit
def train_step(params, x, y, lr):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    return params, loss

exported = export.export(train_step)(sample_params, sample_x, sample_y, sample_lr)
open("train_step.mlir", "w").write(exported.mlir_module())
```

Python is a build dependency (like gcc), not a runtime dep. Scope: 1-2 days
once the inference pipeline works.

### Option C: Implement autodiff in Lean

Define `Layer.diff : Layer → Diff α β` with `forward` and `vjp` functions,
prove VJP correctness, compose over `NetSpec`. Emit forward and backward MLIR
from the composed derivative tree.

This is the intellectual payoff — provably correct gradients, fully Lean-native
training pipeline. Scope: 2-6 months. Mathlib has differential calculus
infrastructure to lean on, but defining `Tensor` with proper dependent shapes
and handling float arithmetic rigorously is nontrivial.

### Recommended path

Phase 2a (week 1-2): **Option B (JAX bootstrap)**. Gets training working fast,
validates the whole pipeline including optimizer + training loop + CUDA
execution. JAX only runs at build time.

Phase 2b (later): **Option A (hand-written VJPs)**. Once training works end-to-end,
replace the JAX-bootstrapped train_step with hand-written backward MLIR.
Removes JAX as a build dep entirely. Tractable because you now have a working
reference to diff against.

Phase 2c (future research): **Option C (Lean autodiff)**. The theorem-proving
payoff. Defer until the pipeline is proven to work and the motivation is clear.

---

## StableHLO cheat sheet for MLP

```mlir
// Matmul (x @ W.T where W is (out, in), x is (batch, in))
%y = stablehlo.dot_general %x, %W,
       contracting_dims = [1] x [1],
       precision = [DEFAULT, DEFAULT]
     : (tensor<128x784xf32>, tensor<512x784xf32>) -> tensor<128x512xf32>

// Broadcast bias from (512,) to (128, 512)
%b_bcast = stablehlo.broadcast_in_dim %b, dims = [1]
         : (tensor<512xf32>) -> tensor<128x512xf32>

// Element-wise add
%added = stablehlo.add %y, %b_bcast : tensor<128x512xf32>

// ReLU via max with zero constant
%zero = stablehlo.constant dense<0.0> : tensor<128x512xf32>
%relu = stablehlo.maximum %added, %zero : tensor<128x512xf32>

// Identity (no-op, just pass through)
%out = %added

// Softmax cross entropy loss (inline)
%max = stablehlo.reduce(%logits init: %neginf) across dimensions = [1]
%shifted = stablehlo.subtract %logits, %max_bcast
%exp = stablehlo.exponential %shifted
%sum = stablehlo.reduce(%exp init: %zero) across dimensions = [1]
%log_sum = stablehlo.log %sum
%log_probs = stablehlo.subtract %shifted, %log_sum_bcast
// one-hot and gather via iota + select, or use stablehlo.gather
```

## Reference material

- **StableHLO spec**: https://github.com/openxla/stablehlo/blob/main/docs/spec.md
- **StableHLO ops reference**: https://github.com/openxla/stablehlo/blob/main/docs/reference_ops.md
- **IREE runtime API**: https://iree.dev/reference/bindings/c-api/
- **IREE C samples**: https://github.com/iree-org/iree/tree/main/samples/custom_dispatch
- **jax.export**: https://jax.readthedocs.io/en/latest/export/export.html
- **Existing JAX codegen in this repo**: `LeanMlir/Codegen.lean` (reference for emission patterns)
- **Existing MLX codegen**: `LeanMlir/MlxCodegen.lean` (reference for structuring non-JAX backends)

## Concrete next steps (in order)

1. `pip install iree-compiler iree-runtime` and verify `iree-compile --version`
2. Hand-write a tiny MLP in StableHLO MLIR, compile via CLI, run via `iree-run-module` to validate the toolchain
3. Write `LeanMlir/MlirCodegen.lean` with just dense + relu emission, generate MLIR for `mnistMlp`, verify it compiles with `iree-compile`
4. Write `LeanMlir/IreeRuntime.lean` with FFI extern declarations, implement the C bindings in a companion `c/iree_bindings.c` file
5. Serialize JAX-trained weights to binary, write `MainMlpMlir.lean` that loads weights + runs forward, check accuracy matches JAX
6. Only after all that works: move to Phase 2 (training)

## Open questions to resolve during implementation

- Does Lean's FFI handle opaque pointers cleanly, or do we need to wrap in `USize`? (Check Mathlib's FFI examples)
- How to marshal large float32 buffers across FFI without copying? (`ByteArray` should work but verify)
- Does IREE's CUDA backend auto-detect the device, or do we need explicit device enumeration?
- What's the overhead of `iree-compile` per model? If it's slow, cache `.vmfb` files keyed by spec hash.
- Static vs dynamic batch dim — benchmark compilation time difference.

---

Scope total for Phase 1: **~1 week of focused work** once IREE is installed.
Phase 2a adds another week. Phase 2b (replace JAX bootstrap with Lean VJPs) adds
another 1-2 weeks. So ~3-4 weeks to get MNIST MLP fully self-hosted on MLIR.

Once that pipeline works, adding more ops (conv, pool, BN, residual blocks) is
mechanical — each is just more StableHLO emission rules, same pattern.

The interesting research question (Option C, Lean-native autodiff) waits until
the pragmatic pipeline is proven.
