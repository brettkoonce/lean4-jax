import LeanMlir

/-! # ConvNeXt — Bestiary entry

ConvNeXt (Liu, Mao, Wu, Feichtenhofer, Darrell, Xie, 2022 — "A ConvNet
for the 2020s") is the "can a pure CNN still compete?" paper. Start
with ResNet-50, apply a checklist of transformer-inspired design
choices one at a time, and the final CNN **matches or beats Swin** on
every benchmark. The paper is a detailed modernization recipe as much
as it is an architecture.

## The modernization recipe

| # | Change                                    | ConvNeXt takes                          |
|---|-------------------------------------------|-----------------------------------------|
| 1 | Stage compute ratio                        | (3, 3, 9, 3) instead of (3, 4, 6, 3)   |
| 2 | Patchify stem                              | 4×4 conv stride 4 (instead of 7×7 + maxPool) |
| 3 | Depthwise conv as "attention analog"       | 7×7 DWConv (large receptive field)     |
| 4 | Inverted bottleneck                        | Expand to 4×C mid-block (like MobileNetV2 / transformer FFN) |
| 5 | Fewer activations                          | One GELU per block (vs many ReLUs)     |
| 6 | Fewer normalizations                       | One LayerNorm per block                |
| 7 | BN → LN                                    | LayerNorm throughout                    |
| 8 | Separate downsampling layers               | LN + 2×2 conv stride 2 between stages  |
| 9 | ReLU → GELU                                | GELU as the activation                 |

## The ConvNeXt block

```
      x
      │
      │ ──────────────────────┐  (skip)
      │                       │
      ▼                       │
   DWConv 7×7 (channels = c)  │  ← depthwise, preserves channels
      │                       │
      ▼                       │
   LayerNorm                  │
      │                       │
      ▼                       │
   1×1 conv  c → 4c           │  ← "inverted bottleneck" expand
      │                       │
      ▼                       │
   GELU                       │
      │                       │
      ▼                       │
   1×1 conv  4c → c           │  ← project back
      │                       │
      ▼                       │
   LayerScale (γ per channel) │  ← small trainable scalar multiplier
      │                       │
      ▼                       │
      + ──────────────────────┘
      │
      ▼
      y
```

No BatchNorm. No ReLU. One normalization, one activation, a depthwise
conv where a transformer would have attention. Somehow this works.

## Inter-stage downsampling

Unlike ResNet (which combines downsampling and the first block's
stride-2 conv), ConvNeXt uses a **dedicated** downsampling layer
between stages: LayerNorm + 2×2 conv stride 2. Channels double, spatial
dims halve. Same philosophy as Swin's patch merging.

## Variants

| Model      | C (stage 1) | Blocks [s1..s4] | Params | ImageNet top-1 |
|------------|-------------|------------------|--------|---------------|
| ConvNeXt-T | 96          | (3, 3, 9, 3)     | 28M    | 82.1          |
| ConvNeXt-S | 96          | (3, 3, 27, 3)    | 50M    | 83.1          |
| ConvNeXt-B | 128         | (3, 3, 27, 3)    | 89M    | 83.8          |
| ConvNeXt-L | 192         | (3, 3, 27, 3)    | 198M   | 84.3          |

Per-stage channels grow 2× across stages: [C, 2C, 4C, 8C].
-/

-- ════════════════════════════════════════════════════════════════
-- § ConvNeXt-T (tiny, canonical)
-- ════════════════════════════════════════════════════════════════

def convNextT : NetSpec where
  name := "ConvNeXt-T"
  imageH := 224
  imageW := 224
  layers := [
    -- Patchify stem: 4×4 conv stride 4 (3 → 96)
    .convBn 3 96 4 4 .same,

    -- Stage 1: 3 blocks at 96 channels
    .convNextStage 96 3,

    -- Downsample to 192 + Stage 2: 3 blocks
    .convNextDownsample 96 192,
    .convNextStage 192 3,

    -- Downsample to 384 + Stage 3: 9 blocks (the deep one)
    .convNextDownsample 192 384,
    .convNextStage 384 9,

    -- Downsample to 768 + Stage 4: 3 blocks
    .convNextDownsample 384 768,
    .convNextStage 768 3,

    -- Head: GAP + classifier
    .globalAvgPool,
    .dense 768 1000 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § ConvNeXt-S (small)
-- ════════════════════════════════════════════════════════════════

def convNextS : NetSpec where
  name := "ConvNeXt-S"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 96 4 4 .same,
    .convNextStage 96 3,
    .convNextDownsample 96 192,
    .convNextStage 192 3,
    .convNextDownsample 192 384,
    .convNextStage 384 27,                      -- 27 blocks in stage 3
    .convNextDownsample 384 768,
    .convNextStage 768 3,
    .globalAvgPool,
    .dense 768 1000 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § ConvNeXt-B (base)
-- ════════════════════════════════════════════════════════════════

def convNextB : NetSpec where
  name := "ConvNeXt-B"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 128 4 4 .same,
    .convNextStage 128 3,
    .convNextDownsample 128 256,
    .convNextStage 256 3,
    .convNextDownsample 256 512,
    .convNextStage 512 27,
    .convNextDownsample 512 1024,
    .convNextStage 1024 3,
    .globalAvgPool,
    .dense 1024 1000 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § ConvNeXt-L (large)
-- ════════════════════════════════════════════════════════════════

def convNextL : NetSpec where
  name := "ConvNeXt-L"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 192 4 4 .same,
    .convNextStage 192 3,
    .convNextDownsample 192 384,
    .convNextStage 384 3,
    .convNextDownsample 384 768,
    .convNextStage 768 27,
    .convNextDownsample 768 1536,
    .convNextStage 1536 3,
    .globalAvgPool,
    .dense 1536 1000 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyConvNext fixture
-- ════════════════════════════════════════════════════════════════

def tinyConvNext : NetSpec where
  name := "tiny-ConvNeXt"
  imageH := 32
  imageW := 32
  layers := [
    .convBn 3 32 2 2 .same,                     -- smaller stem for a 32×32 input
    .convNextStage 32 2,
    .convNextDownsample 32 64,
    .convNextStage 64 2,
    .convNextDownsample 64 128,
    .convNextStage 128 2,
    .globalAvgPool,
    .dense 128 10 .identity
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
  IO.println "  Bestiary — ConvNeXt"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Modernized ResNet-50 with transformer design choices."
  IO.println "  Proof-by-construction that CNNs didn't need to die in 2021."

  summarize convNextT
  summarize convNextS
  summarize convNextB
  summarize convNextL
  summarize tinyConvNext

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • `.convNextStage (channels nBlocks)` bundles nBlocks ConvNeXt"
  IO.println "    residual blocks at a fixed channel count. Each block:"
  IO.println "    7×7 DWConv → LN → 1×1 expand → GELU → 1×1 project + skip."
  IO.println "  • `.convNextDownsample (ic oc)` is the inter-stage \"patch\""
  IO.println "    transformer-style downsampling: LN + 2×2 conv stride 2."
  IO.println "  • Stem uses `.convBn 3 C 4 4 .same` — conv4×4 stride 4 with BN"
  IO.println "    approximating the paper's LN (param count identical)."
  IO.println "  • Param counts match the paper within ~1% (T=28M, S=50M, B=89M,"
  IO.println "    L=198M)."
  IO.println "  • The ConvNeXt paper is unusual: a careful ablation study more"
  IO.println "    than an architecture. Each change individually small; the sum"
  IO.println "    beats Swin on ImageNet. Worth reading for the discipline."
