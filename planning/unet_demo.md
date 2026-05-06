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

**Phase 1+2 also kicked off (2026-05-06):**

- `Layer.bilinearUpsample (scale : Nat)` registered in `Types.lean`.
  Shape-only — same status as `unetDown` / `unetUp`. `archStr` arm
  in `Spec.lean` renders `Upsample(×N)`; all other Spec/MlirCodegen
  dispatchers handle it via existing wildcards (0 params, channels
  passthrough, codegen silently skipped). Verified by reach test:
  a 3-layer NetSpec `[conv2d, bilinearUpsample 2, conv2d]` builds,
  validates, and produces correct param count (251 = 224 + 0 + 27).
- `Bestiary/UNet.lean` gains `unetPets : NetSpec` — depth 4, base 32,
  224×224 RGB → 3-class trimap. **7.76M params**, arch:
  `UNetDown(3→32) → ... → UNetDown(128→256) → Conv(256→512)+BN →
  Conv(512→512)+BN → UNetUp(512→256) → ... → UNetUp(64→32) →
  Conv(32→3,1x1)`. Validates cleanly. Sits between `unetSmall`
  (1.9M) and the original `unet` (31M).

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

**Not yet wired (blocked on Phase 1 + Phase 2):**
- `trainGeneric` will load Pets correctly but cannot train it: the
  forward path is classification-only, the loss path expects
  4-byte int32 labels at the IREE ABI boundary, and `unetDown` /
  `unetUp` are still shape-only enum entries (no codegen). No
  `unet-pets` cell yet (the `unetPets` NetSpec exists but
  cannot compile to MLIR until upsample / concat codegen lands).
- Mask-aware augmentation: `petsIO.augmentBatch` is identity. Real
  augmentation needs to apply the same geometric transform (random
  crop, hflip, scale) to image and mask, plus image-only color ops.

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
