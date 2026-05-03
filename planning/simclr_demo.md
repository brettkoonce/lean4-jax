# simclr_demo.md — SimCLR self-supervised demo on Imagenette

Goal: a worked self-supervised-learning example for the bestiary
that shows the framework handles **non-supervised training** end-to-end
with verified gradients. The "framework training paradigm isn't
tied to cross-entropy on labels" demo, parallel to the supervised
classification chapters and the generative DDPM demo.

## The rule

SimCLR (Chen et al. 2020) is **the same backbone we already have
for classification, with three modifications**:

1. **Two-views-per-image data pipeline** (each image augmented twice independently)
2. **Projection head** (small MLP that maps backbone features to a contrastive embedding space)
3. **InfoNCE contrastive loss** (pull the two views of the same image together; push views of different images apart)

All three pieces decompose to existing primitives. **No new VJP
machinery needed** — InfoNCE is just cross-entropy on a similarity
matrix, which decomposes into matmul + softmax + CE that we already
have proved.

The demo is cheap on the proof side; the substantive work is the
two-view data pipeline and the InfoNCE wiring.

## Architecture: backbone + projection head

| Piece | Source | New work |
|---|---|---|
| Backbone | ResNet-34 or EfficientNet-B0 (already trained, already proved) | strip classifier head |
| Projection head | 2-3 layer MLP (e.g., `2048 → 512 → 128`) with ReLU between | existing dense + ReLU |
| Output: contrastive embedding | L2-normalized 128-dim vector per view | dense + per-vector L2 norm |

The projection head is discarded after pre-training — only the
backbone is used for downstream tasks (linear probe, fine-tune).
Standard SSL trick: the head exists only to give the contrastive
loss room to operate without distorting the backbone features.

## New primitives

**InfoNCE loss**: given `2N` embeddings (from a batch of `N` images,
each augmented twice), compute pairwise cosine similarities, then
softmax + cross-entropy where the "correct class" for each view is
its other-view partner.

```
sim[i, j] = cos_sim(z_i, z_j) / τ        # τ = temperature, ~0.1-0.5
logits = sim with diagonal masked to -∞    # don't compare view to itself
labels = [partner index for each i]       # 2N ints
loss = cross_entropy(logits, labels)
```

Decomposes cleanly into existing primitives:
- L2 normalization: `z / ||z||` — element-wise division by row norm
- Pairwise cosine sim: matmul `Z @ Z.T` (after normalization)
- Diagonal masking: add `-∞ * I` to the similarity matrix
- Softmax + cross-entropy: existing primitives, lifted per-row

**No new VJP**, just composition. ~3-5 days to implement and verify
the algebra is right.

## Data pipeline

The substantive new work. Per training batch:

1. Sample `N` images
2. For each image, apply two **independent random augmentations** → `2N` augmented images
3. Forward all `2N` through backbone + projection head → `2N` embeddings
4. Compute InfoNCE loss

Augmentation pack (SimCLR's "strong" augmentation):
- Random crop + resize (already have)
- Random horizontal flip (already have)
- Color jitter (brightness, contrast, saturation, hue — partially have via RandAugment)
- Random grayscale (~20% probability — new but trivial)
- Gaussian blur (new — small kernel size)

| Item | Effort |
|---|---|
| Two-view data pipeline (sample once, augment twice) | 3-5 days |
| Color jitter pack (if not already complete) | 2-3 days |
| Random grayscale + Gaussian blur | 2-3 days |
| Updated `DatasetIO` with two-view variant | 2-3 days |

**~1.5-2 weeks for the data pipeline.** This is the dominant cost.

## Eval: linear probe

Standard SSL eval protocol:

1. After pre-training, **freeze the backbone** (no gradients flow into it)
2. Add a single linear classifier head: `dense(backbone_dim → num_classes)`
3. Train **only the linear head** on labeled Imagenette (cheap, 5-10 epochs)
4. Report top-1 accuracy on the val set

The "frozen backbone" framing is key — if the SSL pre-training
learned good features, the linear probe accuracy should approach
supervised-pretrain accuracy.

| Item | Effort |
|---|---|
| Freeze-backbone helper (zero gradients past backbone) | 2-3 days |
| Linear-probe training cell + config | 2-3 days |
| Linear-probe eval reporting | 1-2 days |

**~1 week.** Pretty mechanical; reuses the existing supervised
training loop with `freeze_backbone: true`.

## Sequencing

**No prerequisite — fully standalone.**

**Phase 1 — primitives + augmentation (1-2 weeks):**
- Two-view data pipeline
- Augmentation pack completion (color jitter, grayscale, blur)
- L2-normalize op

**Phase 2 — InfoNCE + pre-training (1 week):**
- InfoNCE loss wiring
- Modified train step (2N forward + InfoNCE backward)
- Pre-training loop (R34 backbone + projection head)

**Phase 3 — linear probe + eval (1 week):**
- Freeze-backbone mechanism
- Linear probe training + eval
- Bestiary entry write-up

**Total: ~3-4 weeks standalone.**

## Cells to add

```
("simclr-r34-imagenette",       ⟨simclrR34Spec, simclrConfig, .imagenette, "data/imagenette"⟩),
("simclr-r34-linear-probe",     ⟨simclrR34LinearSpec, linearProbeConfig, .imagenette, "data/imagenette"⟩),
```

Two cells: one for SSL pre-training, one for linear-probe eval.
The probe cell loads the pre-trained backbone checkpoint and
trains only its added linear head.

## Compute budget

- Pre-training: batch=128 means 256 forward passes per "batch" (two views × 128)
- ~9.5K images / batch=128 = ~75 batches/epoch
- At ~1s/batch (R34 + 2 forwards) → 75s/epoch
- 200 epochs (SSL needs more epochs than supervised) = ~4 hours
- Linear probe: ~10 epochs of standard supervised on frozen backbone = ~30 minutes

**Total: ~5 hours** for the full pipeline (pre-train + probe).

## What this unlocks

Self-supervised pre-training is the **gateway to the modern
foundation-model paradigm**:

- **MoCo, BYOL, SwAV, DINO** — variants of the same idea (contrastive,
  bootstrap, clustering, distillation)
- **MAE / Masked Autoencoders** (He et al. 2021) — self-supervised
  but reconstruction-based instead of contrastive
- **Foundation-model fine-tuning workflow** — pre-train large model
  on unlabeled data, fine-tune on small labeled data
- **CLIP-style contrastive objectives** — same InfoNCE math, but
  pulling image-text pairs together instead of two views of one image
- **Anything where labeled data is scarce** — medical imaging,
  scientific imagery, etc.

Beyond the specific architectures, SimCLR demonstrates the
**non-supervised training paradigm** works in the framework. That
generalizes to anything that doesn't have ground-truth labels.

## Honest tradeoff

SimCLR's headline metric is **linear probe accuracy** vs
supervised baseline.

| | Supervised R34 (Ch 6 baseline) | SimCLR R34 + linear probe |
|---|---|---|
| Imagenette top-1 | 90.29% | probably 75-82% |

SimCLR's strength scales with batch size and data scale. Paper
ImageNet results (batch=4096, 1000 epochs) reach ~70% linear probe.
Our Imagenette demo (batch=128, 200 epochs, 10x less data) lands
in the high 70s%.

**The demo isn't claiming SimCLR beats supervised** — supervised
training on labeled data wins when labels are available. The demo
is showing **the framework can do SSL**, and that the gap between
supervised and SSL closes as you scale.

## Why this is cheap

Almost everything reuses existing infrastructure:

| Component | Status |
|---|---|
| R34 / EnetB0 backbone | proved, trained |
| Augmentation primitives | mostly exist (color jitter partial) |
| MLP projection head | dense + ReLU, existing |
| Softmax CE loss | existing |
| Matmul (for similarity) | existing |
| L2 normalization | new but trivial (one-line elementwise op) |
| Linear probe training loop | reuse existing supervised loop with `freeze_backbone` |

Net delta: **two-view data pipeline + projection head + InfoNCE
wiring + freeze-backbone mode**. No new VJPs.

## Why SimCLR over MoCo/BYOL/DINO

| | SimCLR | MoCo | BYOL | DINO |
|---|---|---|---|---|
| Architecture novelty | minimal | needs momentum encoder | needs momentum encoder + predictor | needs teacher-student + sharpening |
| New primitives needed | none | running EMA of backbone | EMA + predictor MLP | EMA + sharpened softmax |
| Effort | 3-4 weeks | 4-5 weeks | 5-6 weeks | 6-8 weeks |
| Pedagogical clarity | very high (one batch, one loss) | medium | medium | medium-low |

SimCLR is the simplest and the most pedagogically clear. The
others are mostly performance optimizations on the same idea (avoid
needing huge batches, avoid contrastive negatives, etc.). For the
demo, simpler wins.

## Out of scope (deferred)

- **MoCo / BYOL / DINO / MAE** — same paradigm, more scaffolding.
  Worth a follow-on demo if SimCLR lands and the paradigm proves
  popular with readers.
- **CLIP** — needs both an image encoder (this demo gives you one)
  and a text encoder (`tinygpt_demo.md`). Natural follow-on once
  both modalities exist.
- **Larger batch sizes** — SimCLR scales with batch, but
  multi-GPU is broken on gfx1100. Single-GPU demo at batch=128 is
  the honest limit.
- **Imagewoof or harder dataset** — interesting follow-on; tests
  whether SSL learns features that transfer to fine-grained tasks.
- **Supervised fine-tuning** (instead of frozen linear probe) — more
  paper-faithful, slightly more complex eval. Linear probe is the
  cleaner story.
- **Embedding visualization** (t-SNE / UMAP of learned features) —
  visually compelling but adds a dependency on a 2D-projection
  library. Optional polish.
