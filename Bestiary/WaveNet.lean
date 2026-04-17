import LeanMlir

/-! # WaveNet — Bestiary entry

WaveNet (van den Oord, Dieleman, Zen, Simonyan, Vinyals, Graves,
Kalchbrenner, Senior, Kavukcuoglu, 2016 — "WaveNet: A Generative Model
for Raw Audio") was the first neural network to produce naturalistic
human speech. A deep stack of **dilated causal convolutions** models
raw audio at 16 kHz sample-by-sample; it trained on hours of audio and
learned to generate speech, music, and even weird hybrids from
scratch. The techniques underpin modern neural TTS (Tacotron 2's
vocoder, Parallel WaveNet, etc.) and show up in image-pixel models
too (PixelCNN).

## Two tricks make it work

**Causal convolution.** Predict sample t+1 from samples 1..t. The
standard-conv-with-padding trick doesn't work: you'd leak information
from the future. A causal conv is a regular conv with the output
shifted by a carefully-chosen amount so only past samples contribute.

**Dilated convolution.** Insert zeros between kernel taps. A
2-tap conv with dilation $d$ reads samples $t$ and $t-d$. Stack dilated
convs with dilation rates $2^0, 2^1, 2^2, \ldots, 2^{L-1}$ and the
receptive field grows **exponentially** while the parameter count
grows **linearly** with depth. A 10-layer stack with kernel 2 and
doubling dilation sees 1024 samples of history ($2^{10}$), and a
30-layer, 3-stack WaveNet sees ~$2^{30}$ samples — enough for long-range
phonetic / musical structure.

## The WaveNet residual block

```
   input (res_channels)                                      input (res)
        │                                                        │
        │                                               ┌────────┤
        ▼                                               │        │
   Dilated causal conv 2-tap                            │ (residual skip)
   res → 2·res channels                                 │
        │                                               │
        ▼                                               │
   split into (filter, gate)                            │
        │                                               │
   tanh(filter) ⊙ sigmoid(gate)    ← gated activation   │
        │                                               │
        ▼                                               │
   1×1 conv back to res channels                        │
        │                                               │
        ▼                                               │
        + ──────────────────────────────────────────────┘
        │
        ├── 1×1 conv → skip_channels (skip out, summed into final head)
        │
        ▼
    next block (with 2× dilation)
```

All "skip" outputs are summed across blocks, passed through
`ReLU → 1×1 conv → ReLU → 1×1 conv → softmax` to produce a 256-way
categorical distribution over the next mu-law-quantized sample.

## Variants

The paper uses 3 stacks of 10 layers (dilations $1, 2, 4, \ldots, 512$
per stack), so 30 total dilated blocks. Residual channels = 32, skip
channels = 512 for the canonical speech model (~4.5M params). Higher-
quality music models use larger residual channels (256).

- `waveNet` — canonical speech, 3 × 10 layers, res=32, skip=512.
- `waveNetMusic` — music generation, 3 × 10, res=256, skip=256.
- `tinyWaveNet` — 1 × 8 layers, res=16, skip=64. Fixture.
-/

-- ════════════════════════════════════════════════════════════════
-- § WaveNet (canonical speech model, paper Table 2)
-- ════════════════════════════════════════════════════════════════

/-- Speech-synthesis WaveNet, ~4.5M params. Bestiary-single-stack
    simplification: 10 dilated residual blocks (dilations 1, 2, ..., 512),
    32 residual channels, 512 skip channels, 256 mu-law output classes.

    The paper's canonical model has 3 stacks × 10 layers for a wider
    receptive field; our linear NetSpec expresses one stack cleanly.
    Multi-stack is architecturally "concatenate three waveNetBlocks" but
    the residual-vs-skip channel mismatch between stacks breaks the
    validator (see the `waveNet3Stack` variant below for an honest but
    loose approximation). -/
def waveNet : NetSpec where
  name := "WaveNet (single stack, speech)"
  imageH := 16000   -- one second of audio at 16 kHz
  imageW := 1
  layers := [
    -- Input embedding: 256 mu-law bins → 32 residual channels.
    .conv2d 256 32 1 .same .identity,

    -- One stack of 10 dilated residual blocks.
    .waveNetBlock 32 512 10,

    -- Output head: skip-sum → ReLU → 1×1 conv (512 → 256) → 1×1 conv (256 → 256).
    -- Skip-sum + ReLUs are orchestration left out of NetSpec.
    .conv2d 512 256 1 .same .relu,
    .conv2d 256 256 1 .same .identity
  ]

/-- Three-stack WaveNet as in the paper (30 total layers). In our
    NetSpec, stacking three `waveNetBlock`s reads as "residual 32 →
    skip 512 → skip 512 → skip 512", which is architecturally
    simplified: the real network keeps res=32 on the residual path and
    sums skips from every layer. Bestiary-only approximation. -/
def waveNet3Stack : NetSpec where
  name := "WaveNet (3 stacks, architecturally simplified)"
  imageH := 16000
  imageW := 1
  layers := [
    .conv2d 256 32 1 .same .identity,
    .waveNetBlock 32 512 10,
    .waveNetBlock 512 512 10,   -- stack 2 (res=prev skip in our linear spec)
    .waveNetBlock 512 512 10,   -- stack 3
    .conv2d 512 256 1 .same .relu,
    .conv2d 256 256 1 .same .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § WaveNet (music synthesis variant)
-- ════════════════════════════════════════════════════════════════

def waveNetMusic : NetSpec where
  name := "WaveNet (music, single stack)"
  imageH := 16000
  imageW := 1
  layers := [
    .conv2d 256 256 1 .same .identity,
    .waveNetBlock 256 256 10,
    .conv2d 256 256 1 .same .relu,
    .conv2d 256 256 1 .same .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyWaveNet fixture
-- ════════════════════════════════════════════════════════════════

def tinyWaveNet : NetSpec where
  name := "tiny-WaveNet"
  imageH := 512
  imageW := 1
  layers := [
    .conv2d 64 16 1 .same .identity,
    .waveNetBlock 16 64 8,                      -- single stack, 8 layers (receptive field 256)
    .conv2d 64 32 1 .same .relu,
    .conv2d 32 64 1 .same .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Main: print-only summary
-- ════════════════════════════════════════════════════════════════

private def summarize (spec : NetSpec) : IO Unit := do
  IO.println s!""
  IO.println s!"  ── {spec.name} ──"
  IO.println s!"  samples     : {spec.imageH}"
  IO.println s!"  layers      : {spec.layers.length}"
  IO.println s!"  params      : {spec.totalParams} (~{spec.totalParams / 1000000}M)"
  IO.println s!"  architecture:"
  IO.println s!"    {spec.archStr}"
  match spec.validate with
  | none     => IO.println s!"  validate    : OK"
  | some err => IO.println s!"  validate    : FAIL — {err}"

def main : IO Unit := do
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Bestiary — WaveNet"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Dilated causal convs for raw waveform modelling."
  IO.println "  Exponential receptive field, linear parameter growth."

  summarize waveNet
  summarize waveNet3Stack
  summarize waveNetMusic
  summarize tinyWaveNet

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • `.waveNetBlock (residualCh skipCh nLayers)` bundles one"
  IO.println "    stack of `nLayers` dilated residual blocks with doubling"
  IO.println "    dilation (2⁰, 2¹, ..., 2^(nLayers−1)). A full WaveNet is"
  IO.println "    multiple stacks in series — dilations reset between stacks."
  IO.println "  • The gated activation `tanh(filter) ⊙ sigmoid(gate)` is not"
  IO.println "    a NetSpec primitive; it's internal to the block. Same"
  IO.println "    level of abstraction as Mamba's internal SSM scan."
  IO.println "  • Skip connections across blocks (summed into the output"
  IO.println "    head) are also internal to `.waveNetBlock`'s implementation;"
  IO.println "    NetSpec shows the block-level flow, not the wiring."
  IO.println "  • Approximating causal convs with ordinary `.conv2d` is a"
  IO.println "    bestiary simplification — the parameter count is correct,"
  IO.println "    the receptive-field geometry differs in a way that matters"
  IO.println "    for training but not for param accounting."
  IO.println "  • Descendants: PixelCNN (2D causal pixel generation),"
  IO.println "    Parallel WaveNet (fast sampling via IAF), TTS vocoders"
  IO.println "    (WaveRNN / WaveGlow / HiFi-GAN). Same trick in each case."
