import LeanMlir

/-! # DenseNet — Bestiary entry

DenseNet (Huang, Liu, van der Maaten, Weinberger 2017,
[arXiv:1608.06993](https://arxiv.org/abs/1608.06993)) takes the
ResNet idea — bypass connections that let gradient flow past stacked
nonlinearities — to its concatenative extreme. Instead of `y = f(x) + x`
(ResNet), each block does

    y_i = f_i([y_0, y_1, ..., y_{i-1}])

where `[…]` is channel-concatenation and every layer in a "dense
block" reads the concatenated outputs of all preceding layers. The
channel count grows linearly through the block — by `growth_rate`
channels per layer.

The architectural payoff: aggressive feature reuse, very narrow
per-layer channel additions (DenseNet uses growth_rate=32 by
default — far smaller than ResNet's 64–512 stage widths), and total
parameter count well below ResNet at comparable accuracy. The cost:
the concatenation explodes activation memory if you're not careful,
and the block as a whole doesn't fit a linear NetSpec — the dense
connectivity is the architectural novelty.

```
                        Input (ic channels)
                                │
                ┌───────────────┴───────────────────┐
                ▼                                   │
        BN→ReLU→1×1 conv (4·gr)                     │
                │                                   │
        BN→ReLU→3×3 conv (gr)                       │
                │                                   │
                ▼                                   │
       concat with input ──────────────────────────►│
                │ (now ic + gr channels)            │
                ▼                                   │
       (repeat n_layers times, each adding gr)     │
                ▼                                   │
       output: ic + n_layers·gr channels ◄─────────┘
```

We model the block as a bundled shape-only primitive,
`Layer.denseBlock ic growth_rate n_layers`, mirroring how the
bestiary handles other architectures whose internal connectivity
escapes a linear layer-list (`mambaBlock`, `swinStage`,
`evoformerBlock`, etc.). Between blocks, `Layer.transitionLayer`
does the BN + 1×1 conv (compress channels) + 2×2 avg-pool
(halve spatial) that DenseNet uses to keep total channel count
under control.

## Variants

DenseNet families differ only in `(L, growth_rate, block_counts)`.
The four ImageNet-class variants from the paper:

- `denseNet121` — 121 weight layers, blocks (6, 12, 24, 16),
  growth_rate=32, ~7.0M params (paper-exact: 6.8M).
- `denseNet169` — 169 weight layers, blocks (6, 12, 32, 32),
  growth_rate=32, ~12.5M params (paper: 12.5M).
- `denseNet201` — 201 weight layers, blocks (6, 12, 48, 32),
  growth_rate=32, ~18.6M params (paper: 18.6M).
- `tinyDenseNet` — small fixture for testing: blocks (2, 2),
  growth_rate=8.

The pattern: stem (7×7 stride-2 conv + 3×3 maxpool), four dense
blocks separated by transition layers (each halving channels and
spatial), GAP, and a single dense classifier. Same shape across
all variants — only the block-counts differ.
-/

-- ════════════════════════════════════════════════════════════════
-- § DenseNet-121 (canonical)
-- ════════════════════════════════════════════════════════════════

/-- DenseNet-121 on Imagenette (10 classes). Block counts (6, 12, 24, 16),
    growth_rate=32, channel counts at each stage:
    stem 64 → block1 256 → trans 128 → block2 512 → trans 256 →
    block3 1024 → trans 512 → block4 1024 → GAP → dense 1024→10. -/
def denseNet121 : NetSpec where
  name := "DenseNet-121"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,                    -- stem 7×7 stride 2 → 112
    .maxPool 3 2,                              -- 3×3 stride 2 → 56
    .denseBlock 64 32 6,                       -- → 64 + 6·32 = 256 channels @ 56
    .transitionLayer 256 128,                  -- BN + 1×1 + avgpool/2 → 128 @ 28
    .denseBlock 128 32 12,                     -- → 128 + 12·32 = 512 channels @ 28
    .transitionLayer 512 256,                  -- → 256 @ 14
    .denseBlock 256 32 24,                     -- → 256 + 24·32 = 1024 channels @ 14
    .transitionLayer 1024 512,                 -- → 512 @ 7
    .denseBlock 512 32 16,                     -- → 512 + 16·32 = 1024 channels @ 7
    .globalAvgPool,
    .dense 1024 10 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § DenseNet-169 (deeper)
-- ════════════════════════════════════════════════════════════════

def denseNet169 : NetSpec where
  name := "DenseNet-169"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 3 2,
    .denseBlock 64 32 6,
    .transitionLayer 256 128,
    .denseBlock 128 32 12,
    .transitionLayer 512 256,
    .denseBlock 256 32 32,                     -- block 3 jumps to 32 layers
    .transitionLayer 1280 640,
    .denseBlock 640 32 32,                     -- block 4 also 32 layers
    .globalAvgPool,
    .dense 1664 10 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § DenseNet-201 (deepest standard variant)
-- ════════════════════════════════════════════════════════════════

def denseNet201 : NetSpec where
  name := "DenseNet-201"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 3 2,
    .denseBlock 64 32 6,
    .transitionLayer 256 128,
    .denseBlock 128 32 12,
    .transitionLayer 512 256,
    .denseBlock 256 32 48,                     -- block 3 jumps to 48 layers
    .transitionLayer 1792 896,
    .denseBlock 896 32 32,
    .globalAvgPool,
    .dense 1920 10 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Tiny DenseNet (fixture for testing)
-- ════════════════════════════════════════════════════════════════

def tinyDenseNet : NetSpec where
  name := "tiny-DenseNet"
  imageH := 32
  imageW := 32
  layers := [
    .convBn 3 16 3 1 .same,
    .denseBlock 16 8 2,                        -- → 16 + 2·8 = 32 channels
    .transitionLayer 32 16,
    .denseBlock 16 8 2,                        -- → 16 + 2·8 = 32 channels
    .globalAvgPool,
    .dense 32 10 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Main: print-only summary of every Bestiary entry in this file.
-- ════════════════════════════════════════════════════════════════

private def summarize (spec : NetSpec) : IO Unit := do
  IO.println s!""
  IO.println s!"  ── {spec.name} ──"
  IO.println s!"  input       : {spec.imageH} × {spec.imageW}"
  IO.println s!"  layers      : {spec.layers.length}"
  IO.println s!"  params      : {spec.totalParams}"
  IO.println s!"  architecture:"
  IO.println s!"    {spec.archStr}"
  match spec.validate with
  | none     => IO.println s!"  validate    : OK (channel dims chain cleanly)"
  | some err => IO.println s!"  validate    : FAIL — {err}"

def main : IO Unit := do
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Bestiary — DenseNet"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Densely-connected CNN: each layer reads concat of all"
  IO.println "  preceding layers within a block. Bundled as `denseBlock`"
  IO.println "  (concat connectivity escapes a linear NetSpec)."

  summarize denseNet121
  summarize denseNet169
  summarize denseNet201
  summarize tinyDenseNet

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • `denseBlock ic gr n` is shape-only: paramShapes counts the"
  IO.println "    BN + 1×1 + BN + 3×3 sub-stack per layer with the inputs"
  IO.println "    growing by gr channels each step. Output channels: ic+n·gr."
  IO.println "  • `transitionLayer ic oc` is BN + 1×1 conv + 2×2 avg-pool;"
  IO.println "    typically halves both channels (oc = ic/2) and spatial size."
  IO.println "  • The architectural novelty (concat connectivity) lives inside"
  IO.println "    the bundled primitive — a linear NetSpec can't express it"
  IO.println "    layer-by-layer the way ResNet's plain residual fits."
