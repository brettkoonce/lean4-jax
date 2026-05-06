# unet_demo.md — UNet segmentation demo on Oxford-IIIT Pets

Goal: a worked segmentation example for the bestiary. The simplest
of the three vision-extension demos (`yolo_demo.md`,
`bifpn_demo.md`, this one) and the natural "first new domain"
because per-pixel classification is conceptually closest to the
existing classification chapters.

## The rule

UNet's contribution beyond the classification chapters is
**dense (per-pixel) prediction**. The forward is encoder-decoder
with skip connections; the backward is per-pixel cross-entropy lifted
across spatial dimensions, which our existing row-wise lifting
machinery already handles. The new proof / codegen work is:

1. **Bilinear upsample** primitive (forward + VJP + codegen) —
   needed for the decoder. *Same primitive needed for BiFPN.*
2. **Channel concat** primitive — needed for skip connections.
3. **Per-pixel softmax CE loss** — same VJP as existing
   classification, lifted spatially.

Item 1 is the dominant new work. Items 2 and 3 are small.

## Architecture: standard encoder-decoder + skips

| Stage | What | Source |
|---|---|---|
| Encoder | 4 downsample stages (conv-bn-relu × 2 + maxpool) | existing primitives |
| Bottleneck | conv-bn-relu × 2 at lowest resolution | existing |
| Decoder | 4 upsample stages (bilinear + 1×1 conv + concat skip + conv-bn-relu × 2) | needs upsample + concat |
| Head | 1×1 conv to (n_classes, H, W) | existing |

Standard UNet from the 2015 paper has 5 encoder stages with channel
doubling (64 → 128 → 256 → 512 → 1024) and symmetric decoder. For
a small demo we'd shrink: 4 stages, fewer channels (32 → 64 → 128
→ 256). Total params would be roughly 5-10M — small enough to
train fast, big enough to actually segment.

The bestiary already has `unetDown` and `unetUp` as shape-only
`Layer` enum entries (`Types.lean:56-65`). This demo would promote
them to **codegen-backed** by implementing the actual forward +
backward + MLIR for the new primitives they reference.

## New primitives

### Bilinear upsample

| Item | Effort |
|---|---|
| Forward kernel (FFI / MLIR) | 2-3 days |
| VJP (transpose of bilinear interpolation = scatter-add with weights) | 3-4 days |
| Lean spec entry + paramShapes (zero params, just changes shape) | 1 day |
| MLIR codegen | 2-3 days |
| Numerical FD test in `check_jacobians.py` | 1 day |

**~1.5 weeks done properly.** This is the same primitive BiFPN
needs — paid once, reused twice.

The choice of bilinear over transpose-conv is deliberate: bilinear
is parameter-free and conceptually simpler. Modern UNet variants
(and the BiFPN, FPN, DeepLab line) all use bilinear. The original
2015 UNet used transpose conv, but bilinear + a 1×1 conv after gives
equivalent expressiveness with cleaner math.

### Channel concat

| Item | Effort |
|---|---|
| Forward (stack along channel axis) | 1 day |
| VJP (split along channel axis — distribute gradient back to inputs) | 1 day |
| Lean spec + codegen | 1-2 days |
| FD test | 1 day |

**~3-4 days total.** Conceptually trivial — channel-concat-forward
and split-backward are inverse permutations of channel data. Should
be fast to implement once the bookkeeping is sorted.

## Loss function: per-pixel softmax CE

Per-pixel softmax cross-entropy. Already in the framework — it's
the same primitive as classification, just **lifted across the
spatial dimensions** via existing row-wise lifting machinery.

For a 224×224 image with C classes: forward output is `(C, 224, 224)`
logits, target is `(224, 224)` integer labels (or `(C, 224, 224)`
soft labels), loss is the average over all pixels of standard
softmax CE. VJP lifts the per-pixel CE gradient over the spatial
axes — a one-line application of the existing `rowwise_has_vjp`
pattern.

**No new loss primitive needed.** ~1 day to wire up.

## Data pipeline

Oxford-IIIT Pet Dataset: 37 categories, ~7400 images with per-pixel
trimaps (foreground, background, boundary). For a clean demo,
collapse boundary into one of the other two → binary segmentation
(foreground vs background) or keep as 3-class.

| Item | Effort |
|---|---|
| Download + extraction + format conversion (PNG masks → tensor) | 2-3 days |
| Dataset loader (`pets` variant of `DatasetIO`) | 2-3 days |
| Augmentation (mask-aware: random crop, flip, scale must transform mask alongside image) | ~1 week |

Mask-aware augmentation is the only piece that doesn't trivially
exist. Like box-aware aug for detection, the existing pixel-only
aug pack needs to know about a paired second tensor. But masks are
easier than boxes — no box geometry to track, just resample the
mask the same way as the image.

**Alternative datasets** if Pets is too small:
- **Pascal VOC segmentation** (~3K train, 21 classes including background) —
  same VOC repo as the YOLO demo, paid once if both demos happen
- **ADE20K** (~20K train, 150 classes) — bigger, more interesting,
  but a bigger pipeline lift

Pets is the friendliest demo target: small, fast to train, visually
satisfying ("here's a cat with the cat's outline traced").

## Eval

Mean IoU (intersection-over-union, averaged across classes). Much
simpler than detection's mAP — no precision-recall sweep, no
matching. Per-class IoU is `|pred ∩ truth| / |pred ∪ truth|` over
the spatial dimensions. ~2-3 days to implement cleanly.

## Sequencing

**Phase 1 — primitives (2-3 weeks):**
- Bilinear upsample (forward + VJP + codegen + FD test)
- Channel concat (forward + VJP + codegen + FD test)
- Bestiary entries promoted from shape-only to codegen-backed

**Phase 2 — UNet plumbing (1 week):**
- NetSpec for `unetTinySpec` (4-stage encoder + decoder)
- Per-pixel CE loss + spatial lifting
- mIoU eval
- Per-pixel train-step ABI: classification's `trainStepAdamF32` /
  `trainStepAdamF32Soft` assume vector logits + scalar label per
  record; segmentation needs a `trainStepAdamF32Seg` variant
  (or generalize the existing one) that takes `(C, H, W)` logits
  and `(H, W)` int labels per record.

**Phase 3 — data + training (1-2 weeks):**
- Pets dataset loader ✓ done — see "Phase 3 progress" below
- Mask-aware augmentation
- End-to-end training, mIoU report

**Total: ~4-6 weeks** for a polished bestiary entry.

## Phase 3 progress (2026-05-05)

**Phase 1+2 progress (2026-05-06):**

Layer registration + Spec wiring (commit `c196915`):
- `Layer.bilinearUpsample (scale : Nat)` in `Types.lean`. `archStr`
  arm in `Spec.lean` renders `Upsample(×N)`; all other dispatchers
  handle it via existing wildcards. Reach test passed: 3-layer
  `[conv2d, bilinearUpsample 2, conv2d]` builds, validates, param
  count correct (251 = 224 + 0 + 27).
- `unetPets : NetSpec` in `Bestiary/UNet.lean` — depth 4, base 32,
  224×224 RGB → 3-class trimap, **7.76M params**. Validates cleanly.
  Sits between `unetSmall` (1.9M) and `unet` (31M).

Bilinear upsample codegen (commit `b5b6982`):
- `bilinearWeights1D` precomputes `(outLen × inLen)` matrix using
  half-pixel centers, no align_corners (PyTorch / JAX default).
- Forward emits two `dot_general` + transpose, factorizing
  `Y = Wy · X · Wxᵀ`. Wired into `emitForwardBody`,
  `emitForwardSig`, `emitForwardEvalSig`, and `emitTrainStepBody`.
- Backward emits the transpose pair: `dM = dY · Wx`,
  `dXᵀ = dM · Wy`, `dX = transpose dXᵀ`. Wired into the train-step
  backward dispatcher with a `FwdRec` so shapes thread.
- IREE-verified at three shapes: `1×1×2×2 → 4×4` (11KB vmfb),
  smallest UNet decoder `4×64×14×14 → 28×28` (16KB), largest
  `4×32×112×112 → 224×224` (114KB). Full train step on a
  `[conv → upsample → conv → GAP → dense]` spec compiles to a
  45KB vmfb in ~0.4s with zero warnings.

FD verification — all of these match analytical gradients to
~1e-11 (well below the `1e-4` tolerance):
- `bilinearUpsample_input_grad` (commit `fbad929`)
- `bilinearUpsample_edge_clamp` (commit `fbad929`)
- `channelConcat_input_grad` (commit `51dba99`)
- `perPixelSoftmaxCE_grad` (commit `633e666`)

Suite is now 29/29 PASS.

Channel concat helpers (commit `51dba99`):
- `emitChannelConcat` (forward `stablehlo.concatenate dim=1`)
- `emitChannelSplitGrad` (backward `stablehlo.slice` per branch)
- Binary primitive (two NCHW inputs), so no Layer constructor —
  these are sub-primitives that future `emitUnetUp` codegen will
  call directly.

**Phase 3 done:**
- `download_pets.sh` + `preprocess_pets.py` — fetch, extract, resize,
  trimap remap (1/2/3 → 0/1/2), pack to flat `train.bin` / `val.bin`
  with `<count:u32 LE><record>*` records of `3*224*224 + 224*224`
  bytes each.
- Data on disk verified by python: `train.bin` 3680 records, `val.bin`
  3669 records, file sizes match header exactly, mask values strictly
  in `{0, 1, 2}`, fg/bg/boundary distribution ~25/60/12% as expected.
- `lean_f32_load_pets` FFI in `ffi/f32_helpers.c` — slurps a `.bin`,
  ImageNet-normalizes the image bytes into a flat f32 buffer, copies
  the mask bytes through unchanged. Returns
  `(imgF32 : ByteArray, maskU8 : ByteArray, count : Nat)`.
  Loads val.bin (~700MB → 2.2GB f32) in ~1.2s.
- `F32.loadPets` Lean binding, `F32.sliceLabels` generalized with a
  `bytesPerLabel : Nat := 4` parameter (default keeps classification
  call-sites zero-touch).
- `DatasetKind.pets` + `petsIO : DatasetIO` with
  `labelBytesPerRecord := 224 * 224` and an identity-augment
  placeholder. New `labelBytesPerRecord` field on `DatasetIO`
  threaded through the four `sliceLabels` call sites in
  `trainGeneric`.
- End-to-end batch sanity verified: `loadPets` → `sliceImages` /
  `sliceLabels` for batch=4 produces 2,408,448 / 200,704 byte slices
  at the expected offsets.

**Not yet wired — what's left to actually train Pets:**

1. **Per-pixel softmax-CE in MLIR emission.** Math is FD-verified
   (commit `633e666`) and the Lean helper signatures are clear, but
   `emitTrainStepBody` has classification CE (`B, NC` shapes)
   hardcoded into the loss + backward sections (lines ~4168-4234).
   Lifting to per-pixel `(B, NC, H, W)` is a parallel rewrite of
   that block. ~1-2 days.
2. **`trainStepAdamF32Seg` IREE ABI.** Segmentation labels are
   `(N, H, W)` int32, not `(N,)` int32 — so the train-step
   signature, the FFI call site in `LeanMlir/IreeRuntime.lean`,
   and the Lean-side label slicing all need a parallel path.
   ~3-5 days.
3. **`unetDown` / `unetUp` codegen.** The hard one. Skip
   connections require cross-layer state — encoder layers stash
   features that decoder layers consume. The current dispatcher is
   purely sequential (`for l in spec.layers`); skip threading
   needs an additional pass or a new state field. The bilinear
   upsample + channel concat sub-primitives are ready
   (commits `b5b6982`, `51dba99`), so the work is the wiring +
   skip-state plumbing, not new math. ~1 week.
4. **Mask-aware augmentation.** `petsIO.augmentBatch` is identity.
   Real augmentation applies the same geometric transform (random
   crop, hflip, scale) to image and mask, plus image-only color
   ops. ~1 week.
5. **mIoU eval.** Per-class IoU averaged across classes, evaluated
   on val set. ~3 days.

A faster intermediate target: an **autoencoder** variant of the
above (no skip connections) is trainable end-to-end once steps 1+2
land — items 3-5 can come later. Skip connections are what makes
UNet "U-shaped"; without them it's a CNN autoencoder, which is
worse for segmentation but a useful stepping stone that exercises
the bilinear upsample primitive in a real training loop.

## Cells to add

```
("unet-pets",       ⟨unetPets, unetConfig, .pets, "data/pets"⟩),
("unet-pets-aug",   ⟨unetPets, unetAugConfig, .pets, "data/pets"⟩),
```

`unetPets` lives in `Bestiary/UNet.lean` (registered 2026-05-06).
Two cells: bare baseline + augmented, mirroring the recipe-ablation
pattern from the classification chapters.

Two cells: bare baseline + augmented. Lets us do the same
recipe-ablation pattern as the classification chapters (does aug
help segmentation? does it help by the same delta as it helps
classification?).

## What this unlocks

The bilinear upsample primitive is the **gateway op for several
architecture families** that currently can't be trained in this
framework:

- **EfficientDet / BiFPN** — exact same primitive (`bifpn_demo.md`)
- **FPN, DeepLabv3+** — bilinear upsample + ASPP (in bestiary)
- **Diffusion models** (DDPM, Stable Diffusion) — UNet-style
  architecture with upsample/downsample blocks
- **Super-resolution** — pixel shuffle, bilinear/bicubic upsample
- **NeRF** — coarse-to-fine sampling

Channel concat is also reused widely — DenseNet, ShuffleNet, all
multi-input architectures.

## Sequencing across the three vision-extension demos

If all three demos eventually happen:

1. **UNet first** (this doc, ~4-6 weeks) — adds bilinear upsample +
   concat in the simplest possible setting. Paid once.
2. **BiFPN second** (`bifpn_demo.md`, ~5-7 weeks if upsample done) —
   reuses upsample, adds multi-scale backbone hooks + detection
   infra. Adds focal loss + Smooth-L1.
3. **YOLO third** (`yolo_demo.md`, ~3-4 weeks if box-aware aug
   exists from BiFPN) — reuses box pipeline, no new primitives.

**Combined ~12-17 weeks** for all three. UNet is the cheapest first
step because:

- Smallest new-primitive set (just upsample + concat)
- Simplest loss (per-pixel CE, no responsibility / matching logic)
- Friendliest dataset (Pets is 7K images, no XML annotations)
- Most reusable primitive (bilinear upsample is the gateway op)

## Why UNet over BiFPN/YOLO first

| | UNet | BiFPN/YOLO |
|---|---|---|
| New primitives | upsample + concat | upsample + multi-scale hooks (BiFPN) or none (YOLO) |
| Loss complexity | per-pixel CE (existing, lifted) | focal + Smooth-L1 + responsibility (new) |
| Data pipeline | mask-aware aug (~1 week) | box-aware aug (~2 weeks) + anchor matching (~1 week) |
| Eval | mIoU (~3 days) | mAP (~1 week) |
| Total effort | 4-6 weeks | 5-7 weeks (YOLO) or 8-10 weeks (BiFPN) |
| Pedagogical clarity | extending classification spatially | new domain (detection) |

UNet is the smoothest extension of the existing classification
chapters. Detection is a bigger conceptual jump.

## Out of scope (deferred)

- 5-stage UNet with full 1024-channel bottleneck — paper-faithful,
  but overkill for the demo
- Transpose-conv upsample variant — original 2015 UNet used it, but
  bilinear + 1×1 conv is the modern equivalent and cheaper to
  implement
- Multi-class segmentation with class imbalance handling (e.g.
  weighted CE, dice loss) — useful for medical imaging but adds
  complexity beyond the demo target
- ADE20K or Cityscapes datasets — much bigger, real eval pipelines.
  Pets is the friendly demo target.
- Instance segmentation (Mask R-CNN style) — combines detection +
  segmentation, deferred until both UNet and a detection demo exist
