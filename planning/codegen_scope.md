# codegen_scope.md — what earns a real emit

## The rule

A `Layer` enum entry is **codegen-backed** (real forward + backward
emit, full param shapes, train-step plumbing) only if it's used by a
network that's **actually trained somewhere in the book**.

Everything else stays **shape-only** — `paramShapes` for accounting,
`archStr` for blueprint string output, and a panic-stub or empty arm
in any per-pass match. No backward, no MLIR.

## Why

Cross-cutting features (loss changes, soft labels, augmentation
plumbing, BN-stats threading, etc.) pay a tax proportional to the
count of codegen-backed kinds. Bestiary entries pay basically zero
because their per-pass arms are stubs.

Keeping the codegen surface narrow is what lets us treat new
cross-cutting features as ~5-7 touchpoints rather than ~30.

## Current scope (as of 2026-04-29, 16 backed kinds)

Every backed kind maps to a network that's actually trained:

| Layer kind | Backing network |
|---|---|
| `conv2d` / `convBn` / `dense` / `flatten` | universal |
| `globalAvgPool` / `maxPool` | universal |
| `residualBlock` | R18 / R34 |
| `bottleneckBlock` | R50 |
| `invertedResidual` | MnV2 |
| `mbConvV3` | MnV3 |
| `mbConv` | EnetB0 / EnetV2 |
| `fusedMbConv` | EnetV2 |
| `uib` | MnV4 |
| `patchEmbed` / `transformerEncoder` | ViT-Tiny |
| `convNextStage` / `convNextDownsample` | ConvNeXt-T |

Bestiary (shape-only): `mambaBlock`, `swinStage`, `patchMerging`,
`unetDown`, `unetUp`, `transformerDecoder`, `detrHeads`,
`shuffleBlock`, `shuffleV2Block`, `evoformerBlock`, `structureModule`,
`mobileVitBlock`, `waveNetBlock`, `positionalEncoding`, `nerfMLP`,
`darknetBlock`, `cspBlock`, `inceptionModule`, `asppModule`,
`fpnModule`, `fireModule`, `separableConv`.

## When adding a new layer

1. **Default to shape-only.** Add the constructor + `paramShapes` +
   `archStr`. That's enough for the bestiary.
2. **Earn codegen-backing** by also adding it to a network in the main
   training pipeline (a `Main*Train.lean` or `MainAblation.lean` cell)
   that actually runs.
3. If step 2 doesn't happen for a release cycle, leave it shape-only.

## Static-graph-only

The MLIR network shape and `train_step` signature are fully determined
at compile time. **No per-step random state is injected through the
MLIR signature.** This rules out:

- dropout
- stochastic depth / DropPath / DropBlock
- anything that needs per-layer random masks generated at runtime

What it leaves available is essentially all of the modern recipe
surface that doesn't touch the graph internals:

- **Data pipeline (tier 1)** — anything that mutates the input tensor
  before the graph sees it (Mixup, CutMix, KNN-Mixup, RandAugment,
  Random Erasing, CutOut, Mosaic, AugMix, etc.).
- **Loss block (tier 2)** — focal loss, label smoothing, distillation,
  contrastive, soft-label routing. Touches one MLIR block, no cross-
  cutting plumbing.
- **Weight-space (tier 1)** — EMA, SWA, SWAG sample, temperature
  scaling. Operates on the param ByteArray outside the graph.
- **Eval-time (tier 1)** — TTA, KNN classifier head, calibration.
  Doesn't change training; runs forward at most.

This excludes ~20% of the modern recipe surface — mostly regularization
techniques that mattered at much larger scales than the book targets.
Worth the trade for keeping the codegen narrow.

## Watch-outs

- **Don't speculatively back-emit** a layer for a network you might
  add later. The forward + backward + per-pass plumbing is real work
  and bit-rots fast if the network never lands.
- **Don't promote bestiary entries** during a cross-cutting refactor
  unless you're also wiring them into the main pipeline. Promotion is
  a +1 to the cross-cutting tax for everything that follows.
- **The boundary is "trained, not just specified."** A spec that
  exists in `LeanMlir/Bestiary/*` for blueprint-string purposes is
  bestiary, regardless of how complete it looks.
