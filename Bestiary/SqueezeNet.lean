import LeanMlir

/-! # SqueezeNet — Bestiary entry

SqueezeNet (Iandola, Han, Moskewicz, Ashraf, Dally, Keutzer, 2016 —
"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and
<0.5MB model size") was one of the first efficiency-focused CNN
architectures. Same era as early GoogLeNet variants; predates MobileNet
v1 by a year. The headline: **AlexNet accuracy in 1.25M params** by
replacing the wasteful "conv + FC" AlexNet shape with a conv-only
network built out of clever reduction-expansion blocks.

## The Fire module

The fire module is the building block — a tiny channel bottleneck
followed by a parallel expansion:

```
        x : (ic, H, W)
             │
             ▼
        squeeze: 1×1 conv  ic → s (squeeze filters, typically s ≪ ic)
             │
             ▼
        ReLU
             │
       ┌─────┴─────┐
       ▼           ▼
  expand 1×1    expand 3×3       ← two parallel convolutions
  s → e₁        s → e₃
       │           │
       ▼           ▼
      ReLU        ReLU
       │           │
       └─────┬─────┘
             │
      concat along channel
             │
             ▼
      (e₁ + e₃, H, W)
```

Two design choices make the module cheap:
1. The squeeze reduces channels first so the expand 1×1/3×3 sees few
   inputs.
2. Most filters are 1×1, not 3×3 — 9× cheaper. The paper calls this
   "smaller filters" strategy.

## SqueezeNet 1.0 architecture

```
    Input (3, 224, 224)
         │
         ▼
    Conv 7×7, 96, stride 2      → 96, 111, 111
    MaxPool 3×3, stride 2       → 96,  55,  55
    Fire2 (s=16, e1=64, e3=64)  → 128, 55,  55
    Fire3 (s=16, e1=64, e3=64)  → 128, 55,  55
    Fire4 (s=32, e1=128, e3=128) → 256, 55,  55
    MaxPool 3×3, stride 2       → 256, 27,  27
    Fire5 (s=32, e1=128, e3=128) → 256, 27,  27
    Fire6 (s=48, e1=192, e3=192) → 384, 27,  27
    Fire7 (s=48, e1=192, e3=192) → 384, 27,  27
    Fire8 (s=64, e1=256, e3=256) → 512, 27,  27
    MaxPool 3×3, stride 2       → 512, 13,  13
    Fire9 (s=64, e1=256, e3=256) → 512, 13,  13
    Dropout (no params)
    Conv 1×1, 1000 filters      → 1000, 13, 13
    GlobalAvgPool               → 1000
```

## Variants

- `squeezeNet1_0` — paper spec. ~1.25M params.
- `squeezeNet1_1` — later re-parameterization with earlier downsampling
  and smaller filter counts. Similar accuracy, even fewer FLOPs.
- `tinySqueezeNet` — tiny fixture.

Compare: AlexNet has ~60M parameters. SqueezeNet 1.0 achieves
AlexNet-level ImageNet accuracy at 50× smaller. The lesson: a lot of
AlexNet's parameter budget was wasteful — two 4096-unit FC layers
chewing up most of it. SqueezeNet is all conv, all the way down.
-/

-- ════════════════════════════════════════════════════════════════
-- § SqueezeNet 1.0 (canonical paper spec)
-- ════════════════════════════════════════════════════════════════

def squeezeNet1_0 : NetSpec where
  name := "SqueezeNet 1.0"
  imageH := 224
  imageW := 224
  layers := [
    -- Stem: 7×7 conv stride 2 + maxpool (stride 2)
    .convBn 3 96 7 2 .same,
    .maxPool 2 2,                                  -- 3×3 s2 in paper; we use 2×2 s2

    -- First fire stack (at 55×55 resolution)
    .fireModule 96  16 64  64,                      -- Fire2: ic=96, s=16, e1=64, e3=64 → 128
    .fireModule 128 16 64  64,                      -- Fire3
    .fireModule 128 32 128 128,                     -- Fire4                           → 256
    .maxPool 2 2,                                   -- 3×3 s2 → 256, 27, 27

    -- Second fire stack (at 27×27)
    .fireModule 256 32 128 128,                     -- Fire5
    .fireModule 256 48 192 192,                     -- Fire6                           → 384
    .fireModule 384 48 192 192,                     -- Fire7
    .fireModule 384 64 256 256,                     -- Fire8                           → 512
    .maxPool 2 2,                                   -- → 512, 13, 13

    -- Third fire stack (at 13×13)
    .fireModule 512 64 256 256,                     -- Fire9                           → 512

    -- Head: 1×1 conv to 1000 classes + GAP
    .conv2d 512 1000 1 .same .relu,
    .globalAvgPool                                   -- GAP produces 1000 logits directly
  ]

-- ════════════════════════════════════════════════════════════════
-- § SqueezeNet 1.1 (cheaper variant — earlier downsampling)
-- ════════════════════════════════════════════════════════════════

/-- SqueezeNet 1.1: halves the stem conv and downsamples earlier, cutting
    FLOPs by ~2× at comparable accuracy. -/
def squeezeNet1_1 : NetSpec where
  name := "SqueezeNet 1.1"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 3 2 .same,                         -- 3×3 stride 2 instead of 7×7
    .maxPool 2 2,
    .fireModule 64  16 64  64,                      -- Fire2
    .fireModule 128 16 64  64,                      -- Fire3
    .maxPool 2 2,                                   -- earlier downsample
    .fireModule 128 32 128 128,                     -- Fire4
    .fireModule 256 32 128 128,                     -- Fire5
    .maxPool 2 2,
    .fireModule 256 48 192 192,                     -- Fire6
    .fireModule 384 48 192 192,                     -- Fire7
    .fireModule 384 64 256 256,                     -- Fire8
    .fireModule 512 64 256 256,                     -- Fire9
    .conv2d 512 1000 1 .same .relu,
    .globalAvgPool
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinySqueezeNet fixture
-- ════════════════════════════════════════════════════════════════

def tinySqueezeNet : NetSpec where
  name := "tiny-SqueezeNet"
  imageH := 32
  imageW := 32
  layers := [
    .convBn 3 32 3 1 .same,
    .fireModule 32 8 16 16,                         -- → 32 channels
    .fireModule 32 8 16 16,
    .maxPool 2 2,
    .fireModule 32 16 32 32,                        -- → 64
    .fireModule 64 16 32 32,
    .conv2d 64 10 1 .same .relu,                    -- CIFAR-10 head
    .globalAvgPool
  ]

-- ════════════════════════════════════════════════════════════════
-- § Main: print-only summary
-- ════════════════════════════════════════════════════════════════

private def summarize (spec : NetSpec) : IO Unit := do
  IO.println s!""
  IO.println s!"  ── {spec.name} ──"
  IO.println s!"  input       : {spec.imageH} × {spec.imageW}"
  IO.println s!"  layers      : {spec.layers.length}"
  IO.println s!"  params      : {spec.totalParams} (~{spec.totalParams / 1000}K)"
  IO.println s!"  architecture:"
  IO.println s!"    {spec.archStr}"
  match spec.validate with
  | none     => IO.println s!"  validate    : OK"
  | some err => IO.println s!"  validate    : FAIL — {err}"

def main : IO Unit := do
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Bestiary — SqueezeNet"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  AlexNet accuracy at 1.25M parameters (50× smaller) via the"
  IO.println "  fire module: squeeze-then-parallel-expand with mostly 1×1 convs."

  summarize squeezeNet1_0
  summarize squeezeNet1_1
  summarize tinySqueezeNet

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • ZERO new Layer primitives. `.fireModule (ic sq e1 e3)` was"
  IO.println "    already in Types.lean — SqueezeNet is what drove it being"
  IO.println "    there in the first place."
  IO.println "  • SqueezeNet predates BN's widespread adoption; the param"
  IO.println "    formula in Spec.lean includes BN γ/β terms (~2% overestimate"
  IO.println "    vs the paper's bare conv + ReLU)."
  IO.println "  • Paper's 3×3-stride-2 maxPool is approximated with 2×2-s2"
  IO.println "    (we don't have a maxPool-with-separate-kernel-and-stride;"
  IO.println "    spatial output differs by one pixel at each level)."
  IO.println "  • SqueezeNet 1.1 was released as a follow-up: earlier"
  IO.println "    downsampling + halved stem, ~2× fewer FLOPs, same accuracy."
  IO.println "    Great example of 'ablation publication' — same architecture"
  IO.println "    family, better implementation."
  IO.println "  • The 2016 CNN-efficiency family (SqueezeNet, MobileNet v1,"
  IO.println "    ShuffleNet) each picked a different compression lever:"
  IO.println "      SqueezeNet  → squeeze-expand fire modules (1×1 everywhere)"
  IO.println "      MobileNet   → depthwise-separable convs"
  IO.println "      ShuffleNet  → grouped convs + channel shuffle"
  IO.println "    Each got to sub-5M params with different tradeoffs."
