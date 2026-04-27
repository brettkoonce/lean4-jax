# convnext.md — ConvNeXt-Tiny training-loop plan

Goal: a Ch 9 worked example demonstrating LayerNorm + GELU on a CNN
backbone (not a transformer). 1D activation ablation: GELU vs ReLU,
both with LN as the post-DW norm.

BN variant is explicitly **out of scope for this iteration** — it'd
need a standalone "BN with γ/β over [b, c, h, w]" emitter separate
from `convBn`'s built-in BN, and the ablation story is good enough
without it.

## Status

**Stage 1: types + paramShapes** — LANDED (commit `ce697e1`).

- `Activation` gains `.gelu`. `Normalization` enum (`.bn` / `.ln`)
  added in `Types.lean` for future BN/LN ablation work, even though
  this iteration only uses LN.
- `.convNextStage` and `.convNextDownsample` get `norm` and `act`
  fields with paper defaults (`.ln` / `.gelu`).
- `paramShapes` populated for both. `archStr`, `channels`, `inputChannels`
  patterns updated.
- `emitActForward` dispatcher gains `.gelu` case (tanh approximation).
- conv2d / dense forward switches handle `.gelu`.
- All existing Bestiary callers compile via defaults.

**Stages 2–4: codegen + Main + ablation** — TODO (this doc).

## Stage 2 — forward inference codegen

Add four helpers (forward-only) and wire into the main forward body:

### Helper 1: `emitDepthwiseConvRaw` (new)

Just the convolution operation, no BN, no activation. ConvNeXt's DW
goes straight into LN, not into BN. Existing `emitDepthwiseConvBn`
bakes in BN; we need a separate raw emitter.

```lean
private def emitDepthwiseConvRaw (pidx : Nat) (curSSA : String) (curShape : List Nat)
    (channels kSize stride : Nat) : String × String × List Nat
```

Output: just the `stablehlo.convolution` op + bias add. About 15
lines, modeled after lines 299–307 of `emitDepthwiseConvBn`.

### Helper 2: `emitLayerNormForwardNCHW` (new)

Existing `emitLayerNormForward` operates on `[b, n, d]` (transformer
shape). ConvNeXt has `[b, c, h, w]` and applies LN over the channel
axis. Math is identical, indices differ:

- mean over dim 1 (channels) → `[b, h, w]`
- var same
- broadcast `γ`, `β` from `[c]` over dim 1
- output stays `[b, c, h, w]`

```lean
private def emitLayerNormForwardNCHW (tag : String) (xSSA : String) (shape : List Nat)
    (gammaSSA betaSSA : String)
    : String × String  -- code + outSSA
```

About 20 lines. Already partially drafted in this doc's commit
history if needed.

### Helper 3: `emitConvNextStage` (new)

Per-block sequence (×nBlocks):

1. Raw 7×7 depthwise conv (W, b)              → uses helper 1, pidx
2. LN over channels (γ, β)                    → uses helper 2, pidx+1
3. 1×1 conv expand `c → 4c` (W, b)            → existing `emitConv2d` with `.identity`, pidx+2
4. Activation (GELU/ReLU/etc.)                → `emitActForward`
5. 1×1 conv project `4c → c` (W, b)           → existing `emitConv2d` with `.identity`, pidx+3
6. LayerScale: per-channel scalar multiply    → broadcast `%W{p}`: [c] over [b,c,h,w], multiply, pidx+4
7. Residual add                               → `stablehlo.add` of pre-block input and step 6 output

Five pidx slots per block: DW / LN / expand / project / LayerScale.
Match to `paramShapes` order in `SpecHelpers.lean:24` (already correct).

Sketch ~50 lines. Branch on `act` for step 4. Branch on `norm` for
step 2 — but for this iteration, only `.ln` is supported (TODO the
BN branch later).

### Helper 4: `emitConvNextDownsample` (new)

1. LN over channels (γ, β)                    → helper 2, pidx
2. 2×2 conv stride 2 (W, b), valid padding    → inline (don't reuse
   `emitConv2d` since that's SAME-pad / stride-1)

Two pidx slots. ~20 lines.

### Forward inference pattern wiring

`MlirCodegen.lean:1214` (the `emitForwardBody` switch) — add:

```lean
| .convNextStage channels nBlocks _norm act =>
  let (snip, newSSA, newShape, newPidx) :=
    emitConvNextStage pidx curSSA curShape channels nBlocks act
  ...
| .convNextDownsample ic oc _norm =>
  let (snip, newSSA, newShape, newPidx) :=
    emitConvNextDownsample pidx curSSA curShape ic oc
  ...
```

### Forward signature (`@forward` function header)

`MlirCodegen.lean:~1428` — the `@forward` param header construction.
Add patterns for both new layers, mirroring `transformerEncoder` at
1462 onwards. Critical: param-naming convention is **`%W{pidx}` and
optionally `%b{pidx}` per pidx slot**, with the shape coming from
`paramShapes` order. LN reuses W/b as γ/β. LayerScale is a W-only
pidx (no β), like `patchEmbed`'s cls/pos slots.

Per-block convNextStage signature:

```
%W{p+0}: [c, 1, 7, 7], %b{p+0}: [c],     -- DWConv
%W{p+1}: [c],          %b{p+1}: [c],     -- LN γ, β
%W{p+2}: [4c, c, 1, 1], %b{p+2}: [4c],   -- 1×1 expand
%W{p+3}: [c, 4c, 1, 1], %b{p+3}: [c],    -- 1×1 project
%W{p+4}: [c]                              -- LayerScale γ (no β)
```

convNextDownsample:

```
%W{p+0}: [ic],         %b{p+0}: [ic],   -- LN γ, β
%W{p+1}: [oc, ic, 2, 2], %b{p+1}: [oc]  -- 2×2 stride-2 conv
```

### Other forward pattern sites that need entries

(Iteration passes that walk every layer kind for shape/feature queries.
Each takes ~3 lines.)

- `MlirCodegen.lean:~1826` — `emitForwardEvalBody` (eval forward body,
  similar to inference but uses fixed BN — for ConvNeXt this is
  identical to inference since LN doesn't have running stats).
- `MlirCodegen.lean:~4712` — utility pass (exact purpose: shape
  iteration for some intermediate format). Check the existing
  `transformerEncoder` arm.
- `MlirCodegen.lean:~4893` — same pattern.
- `MlirCodegen.lean:~5034` — same.

Forward eval signature (`@forward_eval`) at `MlirCodegen.lean:~1855`
— mirror the same pattern as the `@forward` header.

## Stage 3 — training forward + backward

The risky piece. Each layer in the convNextStage block needs a
`FwdRec` record emitted during training forward, and a backward
switch handling that record kind.

### Training forward emit

`MlirCodegen.lean:~3473` — the training body switch. Add convNextStage
and convNextDownsample arms. Each arm:

1. Emits the forward training MLIR (same shape as inference, but
   captures intermediates needed for backward).
2. Pushes `FwdRec` records for each "layer" in the block: DW raw,
   LN-NCHW, 1×1 expand, GELU/ReLU, 1×1 project, LayerScale, residual
   add. The records carry SSA names of the intermediate tensors that
   backward will consume (the `preActSSA`, `outputSSA`, `inputSSA` etc.
   in `FwdRec`).

The closest existing template: `transformerEncoder`'s training
emitter. Copy the structural pattern, adapt the shapes from
`[b, n, d]` to `[b, c, h, w]`.

### Backward switches

`MlirCodegen.lean:~3913` and surrounding — the backward iterator that
walks `records` in reverse and emits gradients per record kind. Need
new branches for:

- **Raw depthwise conv backward**: the existing `emitDepthwiseConvBnBackward`
  bundles BN backward in. Need a separate `emitDepthwiseConvRawBackward`
  that emits just the conv-input-grad and conv-weight-grad (no BN).
  The math is `grad_W = einsum(input, dy)` (depthwise convolution-
  style) and `grad_input = conv-transpose(W, dy)`. About 50 lines.
  
- **LN-NCHW backward**: the existing LN backward in `MlirCodegen.lean:3990`-ish
  area is for `[b, n, d]`. NCHW version needs analogous derivation
  (the three-term Jacobian cancellation, indexed over channel axis
  instead of feature axis). ~40 lines.

- **GELU backward**: already exists.

- **LayerScale backward**: trivial. `grad_input = γ * dy` (broadcast),
  `grad_γ = sum_{b,h,w} (input * dy)` over batch + spatial. ~10 lines.

- **Residual fan-in backward**: trivial. `grad_a = dy`, `grad_b = dy`.
  Already supported via existing residual block patterns.

Most of the backward code is composing standard primitives. The
biggest unknowns are (a) the LN-NCHW three-term backward (need to
derive carefully for the channel-axis layout) and (b) LayerScale's
parameter grad (sum reduction over the spatial+batch axes).

### Where the trust-but-verify boundary lives

Every theorem we'd appeal to is already proved:

- `depthwise_has_vjp3_correct` — Ch 7
- `layerNorm_has_vjp_correct` — Ch 9 (axis is different but the
  proof template is identical, three-term cancellation)
- `dense_has_vjp` — Ch 3
- `gelu_has_vjp_correct` — Ch 9 (just landed)
- `relu_has_vjp_correct` — Ch 3
- `elemwiseProduct_has_vjp` — Ch 8 (LayerScale's math)
- `residual_has_vjp_correct` — Ch 6

Zero new theorems needed. The work is purely codegen — making the
StableHLO emitter produce gradients that match the proved formulas.

## Stage 4 — MainConvNeXtTrain.lean + ablation configs

Once stages 2 and 3 are clean:

1. **`MainConvNeXtTrain.lean`** — analogous to `MainEfficientNetTrain.lean`.
   ConvNeXt-Tiny spec (compute ratio (3, 3, 9, 3), channels 96/192/384/768),
   r34-style training recipe (Adam, cosine, warmup, augment, label
   smoothing).

2. **Two ablation configs in `MainAblation.lean`**:

   ```lean
   def convNextTinyGeluSpec : NetSpec where
     name := "ConvNeXt-Tiny-GELU"
     ...
     .convNextStage 96 3 .ln .gelu, ...

   def convNextTinyReluSpec : NetSpec where
     name := "ConvNeXt-Tiny-ReLU"
     ...
     .convNextStage 96 3 .ln .relu, ...
   ```

   Plus shared `convNextTinyConfig`. Register as `convnext-tiny-gelu`
   and `convnext-tiny-relu` in the ablation registry.

3. **Smoke test**: `./.lake/build/bin/ablation convnext-tiny-gelu` —
   should compile + emit StableHLO + IREE-compile, mirroring the
   Swish smoke test pattern. Runtime will fail on whichever HAL
   backend isn't configured (CUDA/ROCm), but that's expected.

## Time estimate

- Stage 2 (forward inference, all 4 helpers + wiring): **~1.5 hr**.
  Mostly composing existing helpers. The new ones (raw DW, LN-NCHW)
  are ~20 lines each. Multiple pattern-match touch sites.

- Stage 3 (training + backward): **~2.5 hr**. The risky piece. The
  LN-NCHW backward derivation is the deepest part. Test-as-you-go
  with `lake build LeanMlir` to catch missing-case errors fast.

- Stage 4 (Main + ablation + smoke): **~30 min**.

- Total: **4–5 hr** focused session.

## Blueprint copy (Ch 9) — separate task

Once the ablation runs land, add a
`\section{Example: ConvNeXt-Tiny on Imagenette}` and
`\section{Ablation: GELU vs ReLU}` to Ch 9, mirroring Ch 8's
structure. Numbers go in last.
