import LeanMlir

/-! # AlexNet — Bestiary entry

AlexNet (Krizhevsky, Sutskever, Hinton, 2012 — "ImageNet Classification
with Deep Convolutional Neural Networks") is the paper that restarted
modern deep learning. The ImageNet 2012 winner by a 10-point margin
over the next-best entry; everything people now take for granted
about "just use a deep CNN on a big dataset" came from this paper.

Stylistic moves still visible in every modern architecture:
- **ReLU activations** (vs tanh / sigmoid) — trains much faster.
- **Dropout** between FC layers — regularization that kept the 60M
  params from overfitting on ImageNet.
- **Data augmentation** — random crops, flips, color jitter.
- **Deep** — 8 weight layers, which was big in 2012.

The paper also used **two GPUs** (a GTX 580 couldn't fit the whole
network), splitting the network across them in a specific zigzag
pattern. That GPU split leaked into the architecture diagrams and
confuses people reading the paper today; modern re-implementations
treat AlexNet as a single-path network.

## Architecture (single-GPU, canonical modern form)

```
    Input 227 × 227 × 3
         │
         ▼
    Conv 11×11, 96, stride 4, ReLU       → 55 × 55 × 96
    [LRN + MaxPool 3×3 stride 2]         → 27 × 27 × 96
         │
         ▼
    Conv 5×5, 256, pad=2, ReLU           → 27 × 27 × 256
    [LRN + MaxPool 3×3 stride 2]         → 13 × 13 × 256
         │
         ▼
    Conv 3×3, 384, pad=1, ReLU
    Conv 3×3, 384, pad=1, ReLU
    Conv 3×3, 256, pad=1, ReLU
    MaxPool 3×3 stride 2                  → 6 × 6 × 256
         │
         ▼
    Flatten → 9216
    Dense 9216 → 4096, ReLU [+ Dropout]
    Dense 4096 → 4096, ReLU [+ Dropout]
    Dense 4096 → 1000
```

LRN (Local Response Normalization) is a relic — subsequent architectures
dropped it once BatchNorm (2015) appeared. Dropout is training-time,
not a layer in the shape sense.

The two FC-4096 layers dominate the parameter count — ~58M of the 62M
total are in those three denses. This is *the same reason* YOLOv1's
50176→4096 FC dominated its budget, and it's why every post-2015 CNN
(GoogLeNet, ResNet, etc.) dropped FCs in favor of globalAvgPool →
one final dense.
-/

-- ════════════════════════════════════════════════════════════════
-- § AlexNet (canonical, single-GPU)
-- ════════════════════════════════════════════════════════════════

def alexNet : NetSpec where
  name := "AlexNet (Krizhevsky 2012)"
  imageH := 227
  imageW := 227
  layers := [
    -- Conv1: 11×11 stride 4 (the only stride-4 layer in the zoo).
    -- We use `.convBn` since our `.conv2d` doesn't take a stride param;
    -- BN adds ~200 params (~2× oc), negligible vs the 35K conv weights.
    .convBn 3 96 11 4 .same,
    .maxPool 2 2,                               -- 3×3 stride 2 in paper

    -- Conv2: 5×5, stride 1 but padded.
    .conv2d 96 256 5 .same .relu,
    .maxPool 2 2,

    -- Conv3, Conv4, Conv5: 3×3 stride 1.
    .conv2d 256 384 3 .same .relu,
    .conv2d 384 384 3 .same .relu,
    .conv2d 384 256 3 .same .relu,
    .maxPool 2 2,

    -- Classifier: flatten + 3 dense. The 9216 → 4096 and 4096² dense
    -- layers are where ~95% of the parameter budget lives.
    .flatten,
    .dense (6 * 6 * 256) 4096 .relu,
    .dense 4096 4096 .relu,
    .dense 4096 1000 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyAlexNet fixture (CIFAR-size)
-- ════════════════════════════════════════════════════════════════

def tinyAlexNet : NetSpec where
  name := "tiny-AlexNet"
  imageH := 32
  imageW := 32
  layers := [
    .conv2d 3 64 3 .same .relu,
    .maxPool 2 2,
    .conv2d 64 128 3 .same .relu,
    .maxPool 2 2,
    .conv2d 128 192 3 .same .relu,
    .conv2d 192 192 3 .same .relu,
    .conv2d 192 128 3 .same .relu,
    .maxPool 2 2,
    .flatten,
    .dense (4 * 4 * 128) 512 .relu,
    .dense 512 256 .relu,
    .dense 256 10 .identity
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
  IO.println "  Bestiary — AlexNet"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  The 2012 ImageNet winner. Restarted modern deep learning."
  IO.println "  60M params, two GPUs, a decade of compounding since."

  summarize alexNet
  summarize tinyAlexNet

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • Zero new Layer primitives. AlexNet is just 5 convs + 3 FCs"
  IO.println "    plus pooling — the same kit LeNet had 14 years earlier,"
  IO.println "    scaled up and trained on much more data (1.2M ImageNet"
  IO.println "    images vs MNIST's 60K)."
  IO.println "  • Local Response Normalization (LRN) is omitted — it was"
  IO.println "    dropped from the field once BatchNorm appeared in 2015."
  IO.println "    Dropout is similarly training-time only."
  IO.println "  • ~58M of the ~62M params live in the three FC layers. The"
  IO.println "    direct lesson: dense classifier heads are wasteful. Every"
  IO.println "    post-2015 CNN replaces the FC stack with globalAvgPool +"
  IO.println "    one dense → classes."
  IO.println "  • The two-GPU split in the original paper was a hardware"
  IO.println "    artifact (a GTX 580 had 3GB VRAM, not enough for the full"
  IO.println "    network). Modern implementations ignore the split. Our"
  IO.println "    spec is the single-path canonical form."
  IO.println "  • Successors (VGG, GoogLeNet, ResNet) all trace back to"
  IO.println "    AlexNet's \"go deeper, use ReLU, throw augmentation at it,"
  IO.println "    train on GPU\" recipe. The rest is compounding."
