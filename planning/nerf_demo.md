# nerf_demo.md — NeRF demo on the Lego synthetic scene

Goal: a worked 3D-reconstruction example for the bestiary that
shows the framework handles **differentiable rendering** end-to-end
with verified gradients. The "framework can do 3D" demo, parallel
to UNet (segmentation), DDPM (generative), YOLO/BiFPN (detection).

## The rule

NeRF (Mildenhall et al. 2020) is **a coordinate-based MLP plus
differentiable volume rendering**. Train inputs are 2D images with
known camera poses; the network learns to predict density and
color at any 3D point in the scene; novel views are rendered by
ray-tracing through the learned field.

The architecture is small (~1M params, just a stack of dense layers)
but the pipeline is unusual: every training pixel requires *many*
MLP evaluations (sample N=64-256 points along the ray, run the MLP
at each, integrate). The volume-rendering math is the substantive
new proof / codegen work; everything else reuses existing primitives.

This is the **most algorithmically novel** of the planned demos —
the only one that introduces a non-trivial new differentiable
operation (the volume rendering integral).

## Architecture: coordinate MLP + positional encoding

| Piece | Source | New work |
|---|---|---|
| Input: 5D ray (3D position + 2D viewing direction) | data-side | none |
| Positional encoding (sinusoidal at multiple frequencies) | applied to coords before MLP | small new primitive |
| MLP backbone | 8-layer dense stack (256 width) with one skip at layer 4 | existing dense + ReLU |
| Output heads | density (1 scalar) + RGB color (3 channels) | existing dense |
| Volume rendering | alpha-compositing along ray samples | **new VJP — see below** |

Standard NeRF: 8 hidden layers × 256 width, ~1M params total.
Tiny by neural-net standards but with a unique compute pattern
(many forward evals per ray).

## New primitives

### Positional encoding for 3D coords

Apply `[sin(2^k π x), cos(2^k π x)]` for `k = 0..L-1` to each
coordinate `x` (and similar for the viewing direction with fewer
frequencies). Pure deterministic computation, no params, no learnable
VJP. Similar in spirit to the transformer positional encoding but
applied to spatial coordinates.

| Item | Effort |
|---|---|
| Forward: `(B, 3) → (B, 6L)` per coord | 1-2 days |
| VJP: just chain rule through sin/cos | 1-2 days |
| Codegen + FD test | 1-2 days |

**~1 week total.** Conceptually simple; the codegen work is the
bookkeeping.

### Volume rendering (the substantive new VJP)

This is the meat of the new work. The volume rendering integral
along a ray with samples `(σ_i, c_i)` at distances `t_i` is:

```
α_i = 1 - exp(-σ_i · δ_i)        where δ_i = t_{i+1} - t_i
T_i = ∏_{j<i} (1 - α_j)           transmittance
C = Σ_i T_i · α_i · c_i           rendered color
```

The VJP backpropagates the rendered-pixel gradient to:
- Per-sample density `σ_i`
- Per-sample color `c_i`

**Forward** is a sequential reduction (cumulative product for
transmittance, then weighted sum). **Backward** has cross-sample
dependencies (the gradient at sample `i` affects samples `j > i`
through transmittance) which makes it the trickiest backward in
the planned demo set.

| Item | Effort |
|---|---|
| Forward kernel | 3-5 days |
| Backward (VJP through alpha-compositing) | ~2 weeks |
| Lean spec entry | 2-3 days |
| MLIR codegen | 1 week |
| FD test in `check_jacobians.py` | 2-3 days |

**~5-6 weeks for the volume rendering primitive done properly.**
This is the dominant cost of the demo.

## Loss function

L2 (MSE) between rendered pixel color and ground truth pixel.
Already proved. ~1 day to wire up.

## Data pipeline

Lego synthetic scene (from the original NeRF paper):

| | Lego scene |
|---|---|
| Train images | 100 |
| Val | 200 |
| Test | 200 |
| Resolution | 800×800 (downsample to 400×400 or 200×200 for demo) |
| Disk | ~50MB |
| Per-image metadata | camera pose (4×4 transform matrix), focal length |

Tiny dataset, very small disk footprint. The complexity is in the
**ray generation** — for each pixel in each training image, generate
the 3D ray from camera pose + intrinsics + pixel coordinates.

| Item | Effort |
|---|---|
| Lego scene loader (NeRF synthetic format JSON + PNGs) | 2-3 days |
| Ray generation from camera pose | 3-5 days |
| Ray sampling (stratified along ray, 64-256 samples) | 2-3 days |
| Per-iter ray batching (e.g., 4096 rays sampled across train images) | 3-5 days |

**~2 weeks for the data pipeline.** No augmentation — for NeRF,
the train images are fixed observations of a static scene, so
"augmentation" doesn't apply.

## Training

Per iteration:
1. Sample 4096 rays from random training images (mix of pixels)
2. For each ray, sample 64-256 points along the ray (stratified)
3. Run the MLP at each point (~262K MLP evals per iter)
4. Volume-render the points to a predicted color per ray
5. L2 loss against the actual pixel color
6. Backprop through volume rendering + MLP

Standard NeRF: 100K-300K training iterations, batch of 4096 rays.
~100ms/iter on the 7900 XTX. **~3-8 hours total**, friendly for demo.

## Eval

Two paths:

| Item | Effort |
|---|---|
| **Visual: render a novel view** — pick a held-out camera pose, render the Lego excavator from a new angle, save as PNG | 2-3 days |
| **PSNR** — peak signal-to-noise ratio on test set | 1 week (rendering all test images + computing PSNR) |

For the demo, visual is the killer feature: "here's the Lego
excavator viewed from an angle the model never saw during
training." Renders in ~30s-2min depending on resolution.

## Sequencing

**No prerequisite — fully standalone.** Doesn't depend on UNet
or any other planned demo.

**Phase 1 — primitives (5-6 weeks):**
- Positional encoding for 3D coords
- Volume rendering (forward + VJP + codegen + FD test)
- Bestiary entry promoted from shape-only

**Phase 2 — NeRF plumbing (2-3 weeks):**
- NeRF MLP NetSpec (8-layer dense + skip + dual heads)
- Lego scene loader
- Ray generation from camera poses
- Stratified ray-point sampling

**Phase 3 — training + eval (1-2 weeks):**
- Modified train step (per-iter ray batching, multi-sample MLP eval)
- Novel view rendering exe
- (Optional) PSNR eval on test set

**Total: ~8-11 weeks standalone.** Volume rendering dominates the
schedule.

## Cells to add

```
("nerf-lego-mlp",  ⟨nerfLegoSpec, nerfLegoConfig, .nerfSynthetic, "data/nerf/lego"⟩),
```

Single cell. Could add a second cell for a different scene (Drums,
Hotdog, Materials, Mic, Ship — all in the same NeRF synthetic
benchmark) but pedagogically one is enough.

## Compute budget

- Per-iter: 4096 rays × 64 samples × ~1M-param MLP = ~262K MLP evals
- On 7900 XTX with optimized batching: ~80-150ms per iter
- 100K iter × 100ms = ~3 hours
- 200K iter × 100ms = ~6 hours (closer to paper-quality)
- Inference (single view rendering): 200×200 image × 64 samples per ray = 2.6M MLP evals = ~20-30 seconds

## What this unlocks

Volume rendering is the **gateway differentiable operation** for a
large 3D research family:

- **NeRF variants** — Mip-NeRF, Mip-NeRF 360, Block-NeRF (same primitive)
- **Instant-NGP** — replaces MLP with hash grids, but still uses volume rendering
- **Gaussian Splatting** — different rendering primitive (ellipsoid splatting), but same "differentiable rendering loss" idea
- **Signed distance fields** (DeepSDF, NeuS) — coordinate MLPs without volume rendering
- **3D-aware GANs** (EG3D, GIRAFFE) — combine NeRF with adversarial training

The positional encoding primitive is also reused in lots of
modern coordinate-based models.

## Honest tradeoff

NeRF on Lego with limited training:
- **Visual quality**: recognizable Lego excavator from novel views,
  some blur in fine detail
- **PSNR**: probably ~25-30 dB vs paper's ~32.5 dB — the gap is
  mostly training duration and tuning
- **Demo value**: rendering a 3D object from a novel angle is the
  most visually compelling demo of the bunch — every viewer
  immediately gets what's happening

## Why NeRF is uniquely good for the framework demo

Most of the planned demos extend the framework into a *new
architecture pattern* (segmentation, detection, generative).
NeRF extends it into a **new differentiable operation** (volume
rendering integral) that doesn't exist in any of the other demos.

This makes NeRF the strongest "the framework's gradient machinery
generalizes to non-standard ops" demo. The volume rendering VJP
is a chain of operations (exp, cumulative product, weighted sum)
that all individually exist in the toolkit but compose into a
non-trivial sequential reduction. Proving correctness of that
composition is the real new content.

## Out of scope (deferred)

- **Real-world NeRF scenes** (LLFF, Mip-NeRF 360 datasets) — bigger
  data pipeline, more difficult lighting / occlusion. Lego is the
  friendly demo target.
- **Hierarchical sampling** (paper's "fine network" pass) — the
  paper trains both a coarse and a fine network; for a demo, just
  the coarse network with stratified sampling is enough
- **View-dependent color** (separate small network for color
  conditional on view direction) — present in the paper, can be
  simplified to view-independent for demo
- **Instant-NGP / hash grids** — much faster but introduces a new
  primitive (multiresolution hash encoding); separate demo
- **3D Gaussian Splatting** — entirely different rendering
  primitive (ellipsoid splatting via custom rasterizer); separate demo
