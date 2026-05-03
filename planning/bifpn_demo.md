# bifpn_demo.md — EfficientDet (BiFPN + EfficientNet) detection demo

Goal: a worked detection example for the bestiary that **expands the
verified op set** with geometric resampling primitives. Pairs with
`yolo_demo.md` (cheaper, no new primitives). Targets bestiary entry
status, not a full pedagogical chapter.

## The rule

EfficientDet's contribution beyond YOLOv1 is the **BiFPN**
(bidirectional feature pyramid) — multi-scale feature aggregation
with weighted cross-scale fusion. To support it we need:

1. **Multi-scale feature extraction** from the backbone (expose
   intermediate stage outputs, not just final logits).
2. **Resize primitives** (bilinear + nearest-neighbor up/downsample,
   forward + VJP + codegen).
3. **Weighted-sum fusion cells** (already covered by existing
   primitives — softmax over learned weights times features).

Items 1 and 2 are the new proof / codegen work. Item 3 is reuse.

## Architecture: EfficientNet backbone + BiFPN + per-anchor head

| Piece | Source | New work |
|---|---|---|
| Backbone | EnetB0 (already trained, already proved) | expose stages P3-P7 |
| BiFPN cells | new — N rounds of weighted cross-scale fusion | needs resize ops |
| Per-anchor head | dense + reshape per FPN level | NetSpec edit |
| Anchor decoding | data-side | none |

EfficientDet-D0 (the smallest variant) uses EnetB0 backbone + 3
BiFPN repeats + per-anchor classification (90 classes COCO) + box
regression. For a demo on Pascal VOC: same backbone, simpler head
(20 classes), 1-2 BiFPN repeats.

## New primitive: resize ops

Bilinear and nearest-neighbor up/downsample. Each needs:

| Item | Effort |
|---|---|
| Forward kernel (FFI / MLIR) | 2-3 days each |
| VJP (transpose of resize) | 3-4 days each |
| Lean spec entry + paramShapes | 1 day |
| MLIR codegen | 2-3 days each |
| Numerical FD test in `check_jacobians.py` | 1 day each |
| Bestiary primitive definition + blueprint write-up | 1 day |

**Per resize op: ~1.5 weeks done properly.** Both ops together:
~2-3 weeks.

The VJP for bilinear is the interesting one. It's the transpose of
the bilinear-interpolation matrix — basically scatter-add with the
interpolation weights. Nearest-neighbor is gather/scatter with hard
indices; conceptually simpler but the codegen still needs care.

These primitives unlock more than detection — they're also needed
for UNet (segmentation), diffusion-model upsampling blocks, any
architecture with geometric resampling.

## Multi-scale backbone hooks

Currently the spec/codegen produces final-layer logits. BiFPN needs
intermediate features at multiple scales (typically P3, P4, P5, P6,
P7 — strides 8, 16, 32, 64, 128 from a 224×224 input).

Two approaches:

| Approach | Pros | Cons |
|---|---|---|
| Chop backbone into named stages with explicit handoff | clean, composable | larger NetSpec change, ~1 week |
| Add "save intermediate" mechanism in the spec | small change | leaks abstraction, ~3 days |

Option 1 is cleaner architecturally and pays off later (segmentation,
multi-task heads, etc. all want the same thing). Recommended.

## BiFPN cells

Each BiFPN repeat fuses features across scales with learned weights:

```
P_i_out = conv(softmax(w) * [resize(P_{i-1}), P_i, resize(P_{i+1})])
```

The conv + softmax + weighted sum + element-wise add are all
existing primitives. The resize is the new piece (covered above).
A BiFPN repeat is then a sequence of these fusion cells across
levels. ~1 week to wire up.

## Detection head

Per FPN level, a small per-anchor MLP predicting:
- Class probabilities (20 for VOC, 90 for COCO)
- Box offsets (Δx, Δy, Δw, Δh) relative to anchor

For a demo: shared head across all levels (the EfficientDet paper
also does this). Implementation: per-level dense → reshape →
sigmoid (class probs). All existing primitives. ~1 week including
anchor generation.

## Loss functions

| Loss | New? | Effort |
|---|---|---|
| Focal loss (for class imbalance) | yes | forward + VJP + codegen + FD test, ~3-4 days |
| Smooth-L1 box regression | yes | forward + VJP + codegen + FD test, ~2-3 days |

Both are scalar losses on the leaves of the gradient graph — they
don't propagate any new derivative machinery, just contribute their
own gradient at the loss-step level.

## Data pipeline

Pascal VOC same as YOLO demo, but with FPN-level target encoding:

| Item | Effort |
|---|---|
| VOC XML parser | 1 day (shared with YOLO if we do that first) |
| Anchor generation per FPN level | 2-3 days |
| Anchor matching (which anchor is responsible for each GT box) | ~1 week |
| Target encoding per FPN level | 2-3 days |
| Box-aware augmentation | ~2 weeks (shared with YOLO if we do that first) |

If `yolo_demo.md` lands first, the VOC parser + box-aware aug are
already paid for, dropping ~3 weeks off this estimate.

## Inference + eval

Same as YOLO demo: NMS + mAP@0.5. ~1-2 weeks. Shared infrastructure
if YOLO lands first.

## Sequencing

**Phase 1 — proof / codegen (3-4 weeks):**
- Resize ops: bilinear + nearest-neighbor, forward + VJP + codegen + FD
- Multi-stage backbone hooks (chop EnetB0 into stages)
- Bestiary entries for the new primitives in the blueprint

**Phase 2 — detection scaffolding (2-3 weeks, halved if YOLO landed first):**
- VOC parser + box-aware aug (or reuse from YOLO)
- Anchor generation + matching
- Focal loss + Smooth-L1
- Per-anchor detection head

**Phase 3 — BiFPN + integration (2-3 weeks):**
- BiFPN cells (1-2 repeats for demo)
- End-to-end EfficientDet-D0-on-VOC training
- mAP eval, bestiary write-up

**Total: ~7-10 weeks if YOLO lands first; ~9-12 weeks standalone.**

## Cells to add

```
("efficientdet-d0-voc",  ⟨efficientDetD0Spec, edConfig, .pascalVoc, "data/voc"⟩),
```

Single cell since the architecture is the demo. Could add an ablation
for BiFPN-1-repeat vs BiFPN-3-repeat to show what weighted fusion
buys you, but that's optional polish.

## What this unlocks beyond detection

The resize primitives are the **gateway op for several architecture
families** that currently can't be trained in our framework:

- **UNet** (segmentation) — bilinear upsample in decoder
- **DeepLabv3+** — bilinear upsample for segmentation head
- **Diffusion models** — UNet-style architecture, needs upsample/downsample
- **Super-resolution networks** — pixel shuffle, upsample
- **NeRF** — coarse-to-fine sampling

Even if the EfficientDet demo itself is the smallest payoff, the
resize work amortizes across all of these.

## Why BiFPN over YOLO

| | YOLOv1 | EfficientDet/BiFPN |
|---|---|---|
| Effort | 5-7 weeks | 7-10 weeks (with YOLO done first) |
| New primitives | 0 | 2 (resize ops) + multi-stage hooks |
| Future leverage | none | unlocks UNet, diffusion, super-resolution |
| Architectural novelty | grid prediction | weighted multi-scale fusion |
| "Beat the paper" prospect | low (~50% mAP vs paper's 63%) | low (similar gap) |

YOLO is "demo detection cheap." EfficientDet is "demo detection +
expand the verified op set." Different goals; pick based on whether
the long-term plan includes segmentation or diffusion.

## Out of scope (deferred)

- COCO dataset — 10x VOC, real eval pipeline, real GPU budget
- EfficientDet-D1 through D7 — same architecture scaled up; D0 is
  the demo target
- BiFPN ablation against vanilla FPN — interesting but doubles the
  scope
- Detection-specific augmentation (Mosaic, MixUp-on-boxes) — exists
  in YOLOv4+ but not in EfficientDet's recipe; skip
