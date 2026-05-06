import LeanMlir

/-! # UNet — Bestiary entry

UNet (Ronneberger, Fischer, Brox, 2015) was originally designed for
biomedical image segmentation and turned out to be one of the most
versatile architectures in the zoo. Same spatial resolution in and out,
channel hierarchy doubling then halving, skip connections bridging
matching encoder/decoder levels. The shape of the network literally
draws a U:

```
    Input (ic, H, W)
        │                                            Output (oc, H, W)
        ▼                                                   ▲
  unetDown ic → 64        ────skip(64)────►  unetUp 128 → 64
        │                                                   │
        ▼                                                   │
  unetDown 64 → 128       ────skip(128)───►  unetUp 256 → 128
        │                                                   │
        ▼                                                   │
  unetDown 128 → 256      ────skip(256)───►  unetUp 512 → 256
        │                                                   │
        ▼                                                   │
  unetDown 256 → 512      ────skip(512)───►  unetUp 1024 → 512
        │                                                   │
        ▼                                                   │
  [bottleneck: 2× conv3x3  at 1024 channels]
                              │
                              └───────────────────┘
```

Each `unetDown ic oc` is `[conv3×3 + BN + ReLU]×2 + maxPool-2`. It
saves the pre-pool feature map as a **skip** for the matching `unetUp`
later.

Each `unetUp ic oc` is `transposed-conv(ic→oc, stride=2)` (upsample 2×)
then `concat with skip(oc)` (now `2·oc` channels) then
`[conv3×3(2oc→oc) + BN + ReLU] + [conv3×3(oc→oc) + BN + ReLU]`.

The crucial thing about the NetSpec: it's **still a linear list**. The
skip connections are implicit — the codegen matches the `i`-th `unetUp`
from the bottom with the `i`-th `unetDown` from the top. The book's
prose explains the pairing; the NetSpec stays readable.

## Variants

- `unet` — original Ronneberger paper: 1-channel medical input,
  2-class segmentation output, depth 4, base 64 (~31M params).
- `unetRgb` — 3-channel RGB input, 10-class segmentation
  (e.g. Pascal VOC), same depth.
- `unetSmall` — depth 3, base 32 — lighter variant used as a
  Stable-Diffusion-style denoiser backbone (without the attention
  blocks; those'd need additional primitives).
- `unetPets` — depth 4, base 32, 224×224 RGB → 3-class trimap.
  The Pets demo target (`planning/unet_demo.md`); sits between
  `unetSmall` and the original `unet` in size.
- `tinyUnet` — depth 2, base 16 — bestiary fixture.
-/

-- ════════════════════════════════════════════════════════════════
-- § Classic UNet (Ronneberger 2015)  —  1-in / 2-out / depth 4
-- ════════════════════════════════════════════════════════════════

/-- Note: the two `.convBn` layers at the bottom form the bottleneck
    at 1024 channels. `imageH`/`imageW` is the input resolution;
    after 4 rounds of maxPool-2 the bottleneck spatial is H/16 × W/16. -/
def unet : NetSpec where
  name := "UNet (original, grayscale → 2-class)"
  imageH := 512
  imageW := 512
  layers := [
    .unetDown 1   64,                  -- encoder stage 1
    .unetDown 64  128,                 -- encoder stage 2
    .unetDown 128 256,                 -- encoder stage 3
    .unetDown 256 512,                 -- encoder stage 4
    .convBn 512 1024 3 1 .same,        -- bottleneck part 1
    .convBn 1024 1024 3 1 .same,       -- bottleneck part 2
    .unetUp 1024 512,                  -- decoder stage 4 (skip: encoder 4)
    .unetUp 512 256,                   -- decoder stage 3 (skip: encoder 3)
    .unetUp 256 128,                   -- decoder stage 2 (skip: encoder 2)
    .unetUp 128 64,                    -- decoder stage 1 (skip: encoder 1)
    .conv2d 64 2 1 .same .identity     -- output projection (1×1 conv to 2 classes)
  ]

-- ════════════════════════════════════════════════════════════════
-- § RGB UNet (3-channel input, 10 classes)
-- ════════════════════════════════════════════════════════════════

def unetRgb : NetSpec where
  name := "UNet (RGB → 10-class)"
  imageH := 512
  imageW := 512
  layers := [
    .unetDown 3   64,
    .unetDown 64  128,
    .unetDown 128 256,
    .unetDown 256 512,
    .convBn 512 1024 3 1 .same,
    .convBn 1024 1024 3 1 .same,
    .unetUp 1024 512,
    .unetUp 512 256,
    .unetUp 256 128,
    .unetUp 128 64,
    .conv2d 64 10 1 .same .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Small UNet — depth 3, base 32 (SD-denoiser shape without attention)
-- ════════════════════════════════════════════════════════════════

def unetSmall : NetSpec where
  name := "UNet (small, depth 3)"
  imageH := 256
  imageW := 256
  layers := [
    .unetDown 3   32,
    .unetDown 32  64,
    .unetDown 64  128,
    .convBn 128 256 3 1 .same,
    .convBn 256 256 3 1 .same,
    .unetUp 256 128,
    .unetUp 128 64,
    .unetUp 64  32,
    .conv2d 32 3 1 .same .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § unetPets — depth 4, base 32, RGB → 3-class trimap (Pets demo)
-- ════════════════════════════════════════════════════════════════

/-- The Pets demo target. 224×224 RGB input, 3-class output
    (foreground / background / boundary, after trimap remap 1/2/3 →
    0/1/2). Depth 4 with base 32 channels — between `unetSmall`
    (depth 3) and `unet` (depth 4 base 64). Bottleneck doubles to
    512. See `planning/unet_demo.md`. Currently shape-only — depends
    on `.unetDown` / `.unetUp` codegen, which depends on the new
    `.bilinearUpsample` primitive (registered, no codegen yet). -/
def unetPets : NetSpec where
  name := "UNet (Pets, 224×224 RGB → 3-class trimap)"
  imageH := 224
  imageW := 224
  layers := [
    .unetDown 3   32,                  -- encoder stage 1: 224 → 112
    .unetDown 32  64,                  -- encoder stage 2: 112 → 56
    .unetDown 64  128,                 -- encoder stage 3: 56  → 28
    .unetDown 128 256,                 -- encoder stage 4: 28  → 14
    .convBn 256 512 3 1 .same,         -- bottleneck part 1
    .convBn 512 512 3 1 .same,         -- bottleneck part 2
    .unetUp 512 256,                   -- decoder stage 4 (skip: encoder 4)
    .unetUp 256 128,                   -- decoder stage 3 (skip: encoder 3)
    .unetUp 128 64,                    -- decoder stage 2 (skip: encoder 2)
    .unetUp 64  32,                    -- decoder stage 1 (skip: encoder 1)
    .conv2d 32 3 1 .same .identity     -- output projection (1×1 conv to 3 classes)
  ]

-- ════════════════════════════════════════════════════════════════
-- § autoencoderPets — no-skips encoder/decoder (UNet demo Phase 1)
-- ════════════════════════════════════════════════════════════════

/-- Skipless encoder/decoder for the Pets segmentation demo. The point
    is to exercise the per-pixel CE codegen + seg train-step ABI on a
    spec built entirely from primitives the codegen already supports
    (`.convBn`, `.maxPool`, `.bilinearUpsample`, `.conv2d`) — i.e.
    without the `.unetDown` / `.unetUp` skip-state plumbing that the
    real `unetPets` will need. Spatial trip is 224 → 14 → 224 via
    four maxPool-2's then four bilinearUpsample-2's. ~5.5M params.
    Once `Train.lean` routes seg correctly through this spec on the
    Pets data, training loss should decrease. -/
def autoencoderPets : NetSpec where
  name := "Autoencoder (Pets, 224×224 RGB → 3-class trimap, skipless)"
  imageH := 224
  imageW := 224
  layers := [
    -- Encoder: 224 → 14 (four 2× downsamples), 3 → 512 channels
    .convBn 3   64  3 1 .same, .maxPool 2 2,   -- 224 → 112
    .convBn 64  128 3 1 .same, .maxPool 2 2,   -- 112 → 56
    .convBn 128 256 3 1 .same, .maxPool 2 2,   -- 56  → 28
    .convBn 256 512 3 1 .same, .maxPool 2 2,   -- 28  → 14
    .convBn 512 512 3 1 .same,                 -- bottleneck @ 14×14
    -- Decoder: 14 → 224 (four 2× bilinear upsamples), 512 → 64 channels
    .bilinearUpsample 2, .convBn 512 256 3 1 .same,   -- 14  → 28
    .bilinearUpsample 2, .convBn 256 128 3 1 .same,   -- 28  → 56
    .bilinearUpsample 2, .convBn 128 64  3 1 .same,   -- 56  → 112
    .bilinearUpsample 2, .convBn 64  64  3 1 .same,   -- 112 → 224
    .conv2d 64 3 1 .same .identity                    -- output projection (1×1, 3 classes)
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyUnet — bestiary fixture
-- ════════════════════════════════════════════════════════════════

def tinyUnet : NetSpec where
  name := "tiny-UNet (depth 2)"
  imageH := 64
  imageW := 64
  layers := [
    .unetDown 3   16,
    .unetDown 16  32,
    .convBn 32 64 3 1 .same,
    .convBn 64 64 3 1 .same,
    .unetUp 64  32,
    .unetUp 32  16,
    .conv2d 16 1 1 .same .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Main: print-only summary
-- ════════════════════════════════════════════════════════════════

private def summarize (spec : NetSpec) : IO Unit := do
  IO.println s!""
  IO.println s!"  ── {spec.name} ──"
  IO.println s!"  input       : {spec.imageH} × {spec.imageW}"
  IO.println s!"  layers      : {spec.layers.length}"
  IO.println s!"  params      : {spec.totalParams} (~{spec.totalParams / 1000000}M)"
  IO.println s!"  architecture:"
  IO.println s!"    {spec.archStr}"
  match spec.validate with
  | none     => IO.println s!"  validate    : OK"
  | some err => IO.println s!"  validate    : FAIL — {err}"

def main : IO Unit := do
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Bestiary — UNet"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Encoder-decoder with skip connections. Same spatial in/out."
  IO.println "  Originally biomedical segmentation; now the backbone of"
  IO.println "  Stable Diffusion, ControlNet, and most diffusion denoisers."

  summarize unet
  summarize unetRgb
  summarize unetSmall
  summarize unetPets
  summarize autoencoderPets
  summarize tinyUnet

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • `.unetDown` and `.unetUp` are shape-only Layer constructors"
  IO.println "    added for this bestiary. Codegen emits UNSUPPORTED; a real"
  IO.println "    UNet trainer needs the new `.bilinearUpsample` primitive"
  IO.println "    (registered, codegen pending) + a concat kernel + skip-"
  IO.println "    connection threading across the encoder/decoder pair."
  IO.println "  • Skip pairing is implicit: the i-th `unetUp` from the bottom"
  IO.println "    receives the skip from the i-th `unetDown` from the top. A"
  IO.println "    NetSpec with mismatched counts would validate (channel"
  IO.println "    chain passes) but fail at codegen."
  IO.println "  • Bottleneck is expressed as two `convBn` layers — no new"
  IO.println "    primitive needed."
  IO.println "  • Stable Diffusion's UNet adds self- and cross-attention at"
  IO.println "    the lower-resolution stages + time/text conditioning. Those"
  IO.println "    would be two more bestiary primitives (`.attnResBlock`,"
  IO.println "    `.crossAttnBlock`) — left for a follow-up entry."
