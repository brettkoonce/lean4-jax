# ddpm_demo.md — DDPM diffusion-model demo on CIFAR-10

Goal: a worked generative-model example for the bestiary that shows
the framework handles iterative-sampling architectures end-to-end
with verified gradients. The "framework can do generative" demo,
parallel to UNet ("framework can do segmentation") and YOLO/BiFPN
("framework can do detection").

## The rule

DDPM (Ho, Jain, Abbeel 2020) is **a UNet trained to denoise**.
The architecture is UNet with three additions:

1. **Time embedding** (sinusoidal encoding of the diffusion timestep)
2. **Time-conditioning** of UNet feature maps (small dense projection added to each block)
3. **Self-attention** at one or two intermediate resolutions (typically 8×8 and 16×16 in the bottleneck)

All three pieces decompose to existing primitives once `unet_demo.md`
lands. The new work is the **training-time noise scheduling** and
the **sampling-time iterative denoising loop** — both data-side
plumbing, not codegen.

**Prerequisite: UNet demo done first** (this is built on top of UNet).

## Architecture: time-conditioned UNet

| Piece | Source | New work |
|---|---|---|
| UNet backbone | `unet_demo.md` (encoder + decoder + skips) | none |
| Sinusoidal time embedding | timestep → fixed-dim embedding via sin/cos at multiple frequencies | pure data, no codegen |
| Time-conditioning | small dense (embed → channels) added to each block's feature map | dense + add, both existing |
| Self-attention at conv resolutions | `transformerEncoder` applied to (B, HW, C) reshape of conv feature map | existing primitives, novel shape |

For CIFAR-10 32×32: a small UNet (3 stages, channels 64→128→256→256
or similar), self-attention at the 8×8 bottleneck, ~30-40M params.
Trains comfortably on a single 7900 XTX in ~6-12 hours for 100K
iterations.

## New primitives

**Sinusoidal time embedding**: takes integer timestep `t`, computes
`[sin(t/10000^(2i/d)), cos(t/10000^(2i/d))]` for each frequency `i`.
Pure deterministic function, no params, no learnable VJP — just a
data-side computation that's run once per batch and broadcast.

**Time-conditioning add**: existing primitives (dense + broadcast +
elementwise add). No new codegen.

**No genuinely new VJPs needed.** Everything reduces to: UNet
primitives + dense layers + elementwise ops + self-attention, all
already proved.

## Loss function

Standard DDPM loss is MSE between predicted noise and actual noise:

```
L = ||ε_predicted(x_t, t) - ε_actual||²
```

The model takes noised image `x_t` and timestep `t`, predicts the
noise that was added. Already have MSE VJP. ~1 day to wire up.

## Training loop modifications

This is the substantive new work. Per training step:

1. Sample timestep `t ~ Uniform(1, T)` (T=1000 standard)
2. Sample noise `ε ~ N(0, I)`
3. Compute noised image `x_t = √(ᾱ_t) · x_0 + √(1 - ᾱ_t) · ε` (using precomputed cumulative alpha schedule)
4. Predict `ε_θ(x_t, t)` via the UNet
5. Loss = MSE(ε_θ, ε)
6. Backprop and update

| Item | Effort |
|---|---|
| Beta schedule (linear or cosine, precomputed table of α, ᾱ, β) | 1-2 days |
| Noise sampling + scheduled noising forward | 2-3 days |
| Timestep sampling | 1 day |
| Modified train step that does steps 1-6 above | 1 week |

## Sampling loop (inference)

DDPM's "inference" is the iterative denoising process:

```
x_T ~ N(0, I)
for t in T..1:
    ε_θ = model(x_t, t)
    x_{t-1} = (1/√α_t) · (x_t - β_t/√(1-ᾱ_t) · ε_θ) + σ_t · z
return x_0
```

Pure inference (no gradient), so it's a separate exe that loads the
trained checkpoint. ~1 week to implement DDPM sampling correctly.

For faster sampling: **DDIM** (50-100 steps instead of 1000) is the
canonical speedup. Same trained weights, different sampling
schedule. ~3 days to add as a separate exe.

## Data pipeline

CIFAR-10 is already in the codebase as a `DatasetIO`. The DDPM
training loop wraps the existing data loader — each batch goes
through the noise-scheduling step before hitting the model. **Zero
new dataset work.** This is the cheapest demo on the data side
because diffusion is unsupervised (no labels needed beyond the
images themselves, and we're not even using class labels for
unconditional generation).

## Eval

Two paths, easiest first:

| Item | Effort |
|---|---|
| **Visual inspection** — sample 64 images, save as a grid PNG | 2-3 days (just sampling + image writing) |
| **FID** (Fréchet Inception Distance) — paper-standard metric | ~2 weeks (needs an Inception network for feature extraction, statistics computation) |

For a bestiary demo, visual inspection is enough. The output is "a
grid of 64 generated CIFAR-style images" and the demo line is
"these came out of nothing but Gaussian noise + the trained UNet."
That sells the story without paying the FID complexity tax.

FID is the right metric if we want to compare against published
numbers. Paper DDPM on CIFAR-10 reports FID ~3.17. Our demo would
probably hit FID 10-30 with limited training; honest framing is
"the pipeline works, longer training would close the gap."

## Sequencing

**Prerequisite: UNet demo (`unet_demo.md`) complete.**

**Phase 1 — diffusion plumbing (1-2 weeks):**
- Beta schedule (precomputed α, ᾱ tables)
- Sinusoidal time embedding
- Time-conditioning blocks in the UNet variant

**Phase 2 — training loop (1-2 weeks):**
- Noise sampling + timestep sampling per batch
- Modified train step (steps 1-6 above)
- Self-attention at the bottleneck (reuse existing transformer primitives)

**Phase 3 — sampling + eval (1-2 weeks):**
- DDPM sampling exe (1000-step iterative denoising)
- DDIM sampling exe (50-100 step variant)
- Image grid output for visual demo
- (Optional) FID against held-out CIFAR test set

**Total: ~4-6 weeks if UNet done first.** Standalone (without UNet
prerequisite): add another 4-6 weeks for UNet work.

## Cells to add

```
("ddpm-cifar10-unet32",  ⟨ddpmUnet32Spec, ddpmCifarConfig, .cifar10, "data"⟩),
```

Single cell since the architecture is the demo. Could add a
linear-vs-cosine beta-schedule ablation as a second cell if useful.

## Compute budget

- ~50K-200K training iterations standard for CIFAR-10 DDPM
- Per-iter on the 7900 XTX with a 30M-param UNet at 32×32: ~150-200ms
- 100K iter × 200ms = ~5.5 hours
- Sampling: 1000 steps × ~50ms forward = ~50s per generated image (DDPM)
- DDIM: 50 steps × ~50ms = 2.5s per generated image (much friendlier)

Generating a 64-image grid: ~3 minutes with DDIM, ~50 minutes with full DDPM.

## What this unlocks

The diffusion training pattern (noise schedule + denoising loss +
iterative sampling) is the **gateway pattern** for a large modern
generative-model family:

- **Stable Diffusion** — same architecture pattern at scale, latent space
- **Score-based models** (Song & Ermon) — same math, different parameterization
- **Flow-matching** (Lipman et al.) — same training framework, simpler loss
- **Conditional diffusion** (text-to-image, image-to-image) — adds conditioning, same backbone
- **Video diffusion** — extends time dimension, same primitives

Once DDPM works, "Stable Diffusion's training loop" is a
straight-line extension (replace UNet with a bigger one, add
text-conditioning via cross-attention, train on a latent space
instead of pixels).

## Honest tradeoff

DDPM on CIFAR-10 with limited training:
- **Visual quality**: recognizable CIFAR-style images (planes, cars,
  birds, etc. as blurry-but-recognizable shapes)
- **FID**: probably 10-30 vs paper's 3.17 — would need 200K+ iter
  + careful tuning to close
- **Demo value**: "the framework trains a generative model that
  samples coherent images from noise" is the story — beating the
  paper isn't required

## Out of scope (deferred)

- **CIFAR-100 or ImageNet diffusion** — bigger compute, no
  pedagogical gain over CIFAR-10 demo
- **Conditional diffusion** (class-conditional, text-conditional) — adds
  cross-attention; defer until unconditional demo is solid
- **Latent diffusion / Stable Diffusion** — needs a separate
  encoder-decoder (VAE) for the latent space; significantly more
  scaffolding
- **Score-based parameterization** — equivalent math, alternative
  framing; pick one
- **Variational lower bound loss** — paper uses simple MSE
  (`L_simple`), which works fine; the VLB is a more principled
  but harder-to-implement variant
- **Classifier-free guidance** — for conditional models; not
  applicable to unconditional demo
