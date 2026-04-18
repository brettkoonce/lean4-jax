import LeanMlir

/-! # SAM — Bestiary entry

SAM (Kirillov et al., Meta AI, 2023 — "Segment Anything") is the
promptable segmentation model. Give it an image + a prompt (a point,
a box, a rough mask, or — in principle — text), it returns a mask
for the object the prompt points at. Trained on SA-1B, a Meta-built
dataset with 11M images and 1.1B masks; the sheer scale of the data
is what makes the model work out of the box on objects it has never
seen during training.

Architecturally three components, each exposed separately here:

1. **Image encoder.** A ViT backbone (B / L / H) pretrained with MAE
   on ImageNet then fine-tuned during SAM training. Big: this is
   where \emph{almost all} of SAM's parameters live — 91M / 308M /
   636M for the three variants, and the image encoder accounts for
   $\sim$99\% of each. The encoder runs once per image; the
   prompt-conditioned stages are comparatively free.
2. **Prompt encoder.** Maps prompts into 256-dim tokens. Points
   become positional embeddings + polarity (positive / negative click);
   boxes become corner-point pairs; masks downsample-conv to match
   the image feature grid. Tiny — a few thousand parameters total.
3. **Mask decoder.** A two-block lightweight transformer that cross-
   attends between image tokens, prompt tokens, and a small set of
   learned output tokens (3 mask predictions for the ambiguity case
   + 1 IoU score). Bidirectional: each block updates the prompt/output
   tokens AND the image tokens. Under 4M parameters.

## Why the bestiary splits SAM into specs per component

A linear NetSpec can't express "run one backbone once, then dispatch
the same feature map across many prompt-conditioned decoder calls."
That's an orchestration pattern, not a layer pattern. Splitting the
three components into separate NetSpecs matches how the model is
actually checkpointed and served — and how EfficientSAM
(Xiong et al.\ 2023) distills each part independently.

## Variants

- `samEncoderB`    — ViT-B image encoder, dim 768,  12 layers — $\sim$89M
- `samEncoderL`    — ViT-L image encoder, dim 1024, 24 layers — $\sim$304M
- `samEncoderH`    — ViT-H image encoder, dim 1280, 32 layers — $\sim$632M
- `samMaskDecoder` — shared across variants, dim 256, 2 blocks — $\sim$4M
- `tinySam`        — compact encoder fixture for inspection

Paper totals for SAM (encoder + prompt + decoder): 91M / 308M / 636M.
Our encoder-only specs land within 2--3\% of those; the small
prompt-encoder + 4M decoder are accounted for separately.
-/

-- Common patch-embed constants: SAM takes 1024×1024 images with 16×16
-- patches, so there are 64×64 = 4096 patches. This is the nPatches
-- parameter fed to .patchEmbed.
private def SAM_PATCHES : Nat := 4096

-- ════════════════════════════════════════════════════════════════
-- § SAM ViT-B image encoder — paper total: 91M
-- ════════════════════════════════════════════════════════════════

def samEncoderB : NetSpec where
  name := "SAM ViT-B image encoder"
  imageH := 1024
  imageW := 1024
  layers := [
    .patchEmbed 3 768 16 SAM_PATCHES,
    .transformerEncoder 768 12 3072 12
  ]

-- ════════════════════════════════════════════════════════════════
-- § SAM ViT-L image encoder — paper total: 308M
-- ════════════════════════════════════════════════════════════════

def samEncoderL : NetSpec where
  name := "SAM ViT-L image encoder"
  imageH := 1024
  imageW := 1024
  layers := [
    .patchEmbed 3 1024 16 SAM_PATCHES,
    .transformerEncoder 1024 16 4096 24
  ]

-- ════════════════════════════════════════════════════════════════
-- § SAM ViT-H image encoder — paper total: 636M (the canonical model)
-- ════════════════════════════════════════════════════════════════

def samEncoderH : NetSpec where
  name := "SAM ViT-H image encoder"
  imageH := 1024
  imageW := 1024
  layers := [
    .patchEmbed 3 1280 16 SAM_PATCHES,
    .transformerEncoder 1280 16 5120 32
  ]

-- ════════════════════════════════════════════════════════════════
-- § SAM mask decoder — shared across encoder variants
-- ════════════════════════════════════════════════════════════════
-- Two lightweight transformer blocks; dim 256 (features projected from
-- encoder width to 256 via a 1×1 conv, approximated here by treating
-- the decoder input as already-256-dim). nQueries = 4 tokens (3 mask +
-- 1 IoU score). A real SAM decoder does bidirectional updates — each
-- block refines BOTH the queries AND the image tokens — which our
-- .transformerDecoder doesn't express; param count still lands close.

def samMaskDecoder : NetSpec where
  name := "SAM mask decoder (shared)"
  imageH := 64          -- image-feature grid at 1024 / 16 = 64×64
  imageW := 64
  layers := [
    -- Two transformer-decoder blocks with 4 output queries.
    .transformerDecoder 256 8 2048 2 4,
    -- After the decoder the 4 output-token vectors hit small MLP heads
    -- (mask-head: 3 queries × 3-layer MLP to mask embedding; IoU-head:
    -- 1 query × 3-layer MLP to scalar). Approximate with two dense
    -- layers to capture the head param budget.
    .dense 256 256 .identity,
    .dense 256 256 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinySam fixture — compact encoder in the SAM/EfficientSAM style
-- ════════════════════════════════════════════════════════════════

def tinySam : NetSpec where
  name := "tiny-SAM"
  imageH := 256
  imageW := 256
  layers := [
    .patchEmbed 3 128 16 256,   -- 16×16 patches on 256×256 → 16×16 = 256 patches
    .transformerEncoder 128 4 512 4
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
  IO.println "  Bestiary — SAM (Segment Anything)"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Promptable segmentation. Three components: image encoder,"
  IO.println "  prompt encoder, mask decoder. Image encoder is where 99% of"
  IO.println "  the parameters live. EfficientSAM distills each separately."

  summarize samEncoderB
  summarize samEncoderL
  summarize samEncoderH
  summarize samMaskDecoder
  summarize tinySam

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • ZERO new Layer primitives. Image encoder reuses the ViT kit"
  IO.println "    (.patchEmbed + .transformerEncoder); mask decoder reuses"
  IO.println "    .transformerDecoder (from DETR) with 4 output queries."
  IO.println "  • Prompt encoder (~5K params) is prose-only: point prompts"
  IO.println "    become positional embedding + polarity embedding, box"
  IO.println "    prompts become corner-point pairs, mask prompts downsample"
  IO.println "    via small convs. Total budget is negligible vs the encoder."
  IO.println "  • Our SAM encoders land within 2-3% of paper totals. The"
  IO.println "    real SAM ViT also uses windowed self-attention with global"
  IO.println "    attention at layers {2, 5, 8, 11, ...} for 1024×1024"
  IO.println "    inputs — a parameter-free modification of attention that"
  IO.println "    doesn't change param count but changes compute per step."
  IO.println "  • EfficientSAM (Xiong et al. 2023) is the follow-up that"
  IO.println "    distills SAM-H into tiny ViTs via SAMI masked pretraining,"
  IO.println "    landing at ~10M params with SAM-like performance. Same"
  IO.println "    three-component architecture, each component distilled"
  IO.println "    independently."
