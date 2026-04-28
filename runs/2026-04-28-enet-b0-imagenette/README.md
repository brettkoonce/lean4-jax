# EfficientNet-B0 Imagenette: Swish vs ReLU (2026-04-28)

The Ch 8 activation ablation. Two variants of EfficientNet-B0 trained
on Imagenette (10 classes, 9469 train / 3904 val), differing only in
the activation function used inside every MBConv block.

## Setup

- Recipe: Adam @ 0.001, batch=32, cosine decay, 3-epoch warmup, augment, label_smooth=0.1, weight_decay=0.0001
- Schedule: 80 epochs each
- Backend: IREE + ROCm/HIP on RX 7900 XTX (gfx1100)
- Dual-GPU parallelism: swish on `HIP_VISIBLE_DEVICES=0`, relu on `=1`
- Wall clock: 09:55 UTC start, 16:53 UTC end (~7 h)
- Both seeded identically; same init bytes; same batch order

## Results

| variant | val accuracy | train loss (e80) | per-epoch | total wall |
|---|---|---|---|---|
| `enet-b0-swish` (GPU 0) | **87.58%** (3419/3904) | 0.504 | 4.83 min | 6h 33m |
| `enet-b0-relu`  (GPU 1) | **87.78%** (3427/3904) | 0.509 | 4.96 min | 6h 53m |

Δ = +0.20% favoring ReLU (sign-flipped vs the paper's +0.6 favoring Swish on ImageNet at much larger scale; well within seed noise on this dataset).

The 0.13 min/epoch wall-clock gap is per-GPU (GPU 0 vs GPU 1 on the same node), not per-activation. Swish's `logistic + multiply` vs ReLU's single `maximum` doesn't measurably move the per-step time on a compute-saturated GPU.

## Files

- `swish.log` — full step + epoch + val output for `enet-b0-swish`
- `relu.log` — full step + epoch + val output for `enet-b0-relu`

Per-step timings, per-epoch losses, and the final val accuracy are all in the logs. The Ch 8 blueprint section (`blueprint/src/content.tex`, sec:enet_swish_ablation) cites these numbers.
