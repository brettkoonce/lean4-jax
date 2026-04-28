# ConvNeXt-Mini CIFAR-10: 2D ablation {LN, BN} × {GELU, ReLU} (2026-04-28)

Ch 9 activation × normalization ablation on a CIFAR-sized ConvNeXt
(102K params, single ConvNeXt stage at 32 then 64 channels, 32×32
input). All four cells share architecture and recipe; only the
norm primitive and the activation differ.

## Setup

- Recipe: Adam @ 0.001, batch=32, cosine decay, 2-epoch warmup, augment, label_smooth=0.1, weight_decay=0.0001
- Schedule: 30 epochs each
- Backend: IREE + ROCm/HIP on RX 7900 XTX (gfx1100), single GPU, sequential runs
- Wall clock: 03:32 UTC → 06:01 UTC (~2h 29m total)

## Results

| variant | val accuracy | train loss (e30) | per-epoch | runtime |
|---|---|---|---|---|
| `convnext-mini-gelu`    (LN + GELU) | 76.32% | 0.812 | 69 s | 35 min |
| `convnext-mini-relu`    (LN + ReLU) | 76.24% | 0.839 | 75 s | 38 min |
| `convnext-mini-bn-gelu` (BN + GELU) | 77.99% | 0.833 | 74 s | 37 min |
| `convnext-mini-bn-relu` (BN + ReLU) | **78.18%** | 0.860 | 78 s | 39 min |

Findings:
- **BN > LN at this scale** by ~1.8% (avg 78.09% vs 76.28%). Opposite to the paper's ImageNet finding (LN edges BN by ~0.1%). At 102K params on CIFAR-10, BN's batch-statistic averaging seems to regularize better than per-spatial LN.
- **GELU vs ReLU is a wash** within each norm row (LN: +0.08% GELU; BN: +0.18% ReLU). Within seed noise.
- **All four converge cleanly**, no NaN / loss spikes, monotonic cosine decay.
- BN backward is ~7% slower per step than LN backward (extra two-stage [2,3]→[0] reductions to dodge IREE's distribute-pass quirk; see `upstream-issues/2026-04-iree-rocm-stacked-reduce-distribute/`).

## Files

- `ln-gelu.log` — `convnext-mini-gelu`
- `ln-relu.log` — `convnext-mini-relu`
- `bn-gelu.log` — `convnext-mini-bn-gelu`
- `bn-relu.log` — `convnext-mini-bn-relu`
