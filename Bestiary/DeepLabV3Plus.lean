import LeanMlir

/-! # DeepLab v3+ — Bestiary entry

DeepLab v3+ (Chen, Zhu, Papandreou, Schroff, Adam, ECCV 2018 ---
"Encoder-Decoder with Atrous Separable Convolution for Semantic
Image Segmentation") is the canonical segmentation architecture
for the pre-transformer era. Still a workhorse in production
pipelines, especially remote-sensing and medical-imaging settings
where the input resolution is large and structured multi-scale
context matters more than model size.

Two architectural ideas defined the DeepLab family:

1. **Atrous (dilated) convolution.** Replace the stride-2
   downsampling in the last 1--2 backbone stages with dilated
   3$\times$3 convs. Receptive field stays large, but the output
   spatial resolution doesn't shrink as much. No extra parameters;
   atrous is a forward-pass choice, not a layer-shape choice.
2. **Atrous Spatial Pyramid Pooling (ASPP).** Five parallel
   branches at the feature map's deepest stage: 1$\times$1 conv,
   three 3$\times$3 atrous convs at rates 6 / 12 / 18, and an
   image-level global-pool. Concat the five outputs, fuse with a
   final 1$\times$1 conv. Dense multi-scale context at a single
   resolution, which matches what your eye actually does when
   looking at a scene.

The ``+'' in v3+ adds one more move: a lightweight decoder that
bilinearly upsamples the ASPP output 4$\times$ and concatenates it
with a low-level feature map (from the backbone's stage 2) to
recover fine spatial details. Compare to v3 (no decoder, output
directly upsampled 16$\times$), which produced blocky masks;
v3+ produces crisper boundaries.

## Where DeepLab v3+ shows up

- Semantic-segmentation baselines in papers through $\sim$2020.
- Remote sensing (satellite imagery land-cover classification;
  the multi-scale ASPP context suits small-object-in-big-image
  workloads).
- Medical imaging (where the canonical UNet reigns, but DeepLab is
  a common cross-comparison).
- Autonomous driving perception (cityscapes, KITTI).

SegFormer (2021) and SAM (2023) have eaten most of the
general-purpose segmentation market since, but the ASPP module
itself still shows up in newer hybrid architectures --- it's a
durable idea.

## Variants

- `deeplabv3plusResnet101` --- ResNet-101 backbone (most common
  in papers, $\sim$60M params)
- `deeplabv3plusMobilenet`  --- MobileNet v2 backbone (mobile
  variant, $\sim$6M, the Google TF-model-garden default)
- `tinyDeepLab`             --- compact fixture

## NetSpec simplifications

- The v3+ decoder's skip from backbone stage 2 to the ASPP output
  doesn't linearize cleanly (concat of two tensors from different
  depths). Our spec shows the decoder's convs inline and notes the
  skip in prose --- same honest hack UNet / WaveNet use for
  non-linear topologies.
- Atrous rates in the backbone are forward-pass-only. Our
  \texttt{.bottleneckBlock} takes a stride argument but doesn't
  take a dilation argument, so the backbone spec uses the
  stride-1 configuration at the last stage (matching atrous-v3+'s
  ``output stride 16'' mode) --- param count is correct, the
  dilation itself lives in the comment.
-/

-- ════════════════════════════════════════════════════════════════
-- § DeepLab v3+ with ResNet-101 backbone — ~60M params
-- ════════════════════════════════════════════════════════════════

def deeplabv3plusResnet101 : NetSpec where
  name := "DeepLab v3+ (ResNet-101 backbone)"
  imageH := 513
  imageW := 513
  layers := [
    -- Stem
    .convBn 3 64 7 2 .same,
    .maxPool 3 2,
    -- ResNet-101 body. Stage 4 uses stride 1 (not 2) — the spatial
    -- downsampling is replaced by atrous convs with rate 2, preserving
    -- receptive field. Our .bottleneckBlock doesn't take a dilation
    -- arg; param count is the same either way.
    .bottleneckBlock 64   256  3  1,
    .bottleneckBlock 256  512  4  2,
    .bottleneckBlock 512  1024 23 2,
    .bottleneckBlock 1024 2048 3  1,    -- stride 1 (atrous in real impl)
    -- ASPP: 2048 → 256 via 5 parallel branches + fusion.
    .asppModule 2048 256,
    -- Decoder: two 3×3 convs over the fused ASPP output. The skip from
    -- stage 2 (512-ch low-level features) that gives v3+ its "plus" is
    -- architecturally a concat-with-skip — not expressible linearly
    -- here, so we note it in prose and run the convs in-line.
    .conv2d 256 256 3 .same .relu,
    .conv2d 256 256 3 .same .relu,
    -- Output: per-pixel class logits (Pascal VOC: 21 classes)
    .conv2d 256 21 1 .same .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § DeepLab v3+ with MobileNet v2 backbone — ~6M params
-- ════════════════════════════════════════════════════════════════
-- The mobile-friendly variant. TF Model Garden ships this as the
-- default DeepLab for on-device deployment.

def deeplabv3plusMobilenet : NetSpec where
  name := "DeepLab v3+ (MobileNet v2 backbone)"
  imageH := 513
  imageW := 513
  layers := [
    -- MobileNet v2 stem + inverted-residual body. Approximated with
    -- our .invertedResidual primitive; exact MNv2 stage config.
    .convBn 3 32 3 2 .same,
    .invertedResidual 32  16  1 1 1,
    .invertedResidual 16  24  6 2 2,
    .invertedResidual 24  32  6 2 3,
    .invertedResidual 32  64  6 2 4,
    .invertedResidual 64  96  6 1 3,
    .invertedResidual 96  160 6 1 3,      -- stride 1 (atrous in real)
    .invertedResidual 160 320 6 1 1,      -- stride 1 (atrous in real)
    -- ASPP at a smaller width: 320 → 256 (paper uses 256 everywhere)
    .asppModule 320 256,
    -- Decoder
    .conv2d 256 256 3 .same .relu,
    .conv2d 256 256 3 .same .relu,
    .conv2d 256 21 1 .same .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyDeepLab fixture
-- ════════════════════════════════════════════════════════════════

def tinyDeepLab : NetSpec where
  name := "tiny-DeepLab v3+"
  imageH := 64
  imageW := 64
  layers := [
    .convBn 3 32 3 2 .same,
    .bottleneckBlock 32 64 2 2,
    .bottleneckBlock 64 128 2 2,
    .asppModule 128 64,
    .conv2d 64 32 3 .same .relu,
    .conv2d 32 10 1 .same .identity
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
  IO.println "  Bestiary — DeepLab v3+"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Atrous convolution + ASPP + lightweight decoder. The canonical"
  IO.println "  pre-transformer segmentation architecture, still deployed in"
  IO.println "  remote sensing / medical / autonomous-driving pipelines."

  summarize deeplabv3plusResnet101
  summarize deeplabv3plusMobilenet
  summarize tinyDeepLab

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • New primitive: .asppModule (ic oc). The 5-branch Atrous"
  IO.println "    Spatial Pyramid Pool + fusion is distinctive enough to"
  IO.println "    warrant its own bundled Layer constructor, same pattern as"
  IO.println "    .inceptionModule / .fireModule / .mobileVitBlock."
  IO.println "  • Atrous (dilated) convs add no parameters over regular"
  IO.println "    convs; the dilation is a forward-pass choice. Real DeepLab"
  IO.println "    replaces stride-2 downsamples in the backbone's last stage"
  IO.println "    with atrous convs, keeping output stride at 16 (not 32)."
  IO.println "    Our spec's param count is correct; the dilation itself is"
  IO.println "    a comment."
  IO.println "  • The v3+ 'plus' is the decoder: bilinear upsample the ASPP"
  IO.println "    output 4×, concat with a low-level skip from backbone"
  IO.println "    stage 2, apply 3×3 convs. The skip doesn't linearize"
  IO.println "    cleanly — our spec runs the convs inline and notes the"
  IO.println "    skip in prose (same hack UNet and WaveNet use)."
  IO.println "  • DeepLab's durable export is the ASPP module itself, which"
  IO.println "    shows up in many hybrid architectures even today. The"
  IO.println "    SegFormer paper is essentially 'do ASPP's multi-scale"
  IO.println "    context via a transformer pyramid instead'; different"
  IO.println "    mechanism, same goal."
