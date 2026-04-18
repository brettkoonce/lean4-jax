# Training Logs

Captured stdout from training runs — both the ablation framework
(`MainAblation.lean`, invoked as `./.lake/build/bin/ablation <name>`)
and the named full-trainer scripts (e.g. `./.lake/build/bin/resnet34-train`).

All ablation runs on AMD 7900 XTX (ROCm, gfx1100). Final accuracies
below are the last `val accuracy` line in each log; see the file for
the full per-epoch training curve.

Not a perfect catalog — some numbers are from earlier tuning
iterations and some runs are incomplete or diverged. Use as a rough
point of reference.

---

## MNIST — progressive complexity

Starting from the simplest possible classifier (one matmul) and
scaling up.

### Linear classifier (1 dense layer, ~7.8K params)

| Log | Config | Final val acc |
|-----|--------|---------------|
| `ablation_linear-sgd.log`  | SGD 0.1, 12 ep             | 91.9% |
| `ablation_linear-adam.log` | Adam 0.001, 12 ep          | 92.6% |

### Shallow MLP (one hidden layer, ~102K params)

| Log | Config | Final val acc |
|-----|--------|---------------|
| `ablation_shallow-sgd.log`  | SGD 0.1, 15 ep       | 98.1% |
| `ablation_shallow-adam.log` | Adam 0.001, 15 ep    | 98.0% |

### MLP (784 → 512 → 512 → 10, ~670K params)

| Log | Config | Final val acc |
|-----|--------|---------------|
| `ablation_mlp-sgd.log`      | **S4TF baseline**: SGD 0.1, 12 ep | **98.57%** |
| `ablation_mlp-sgd-low.log`  | SGD 0.01, 15 ep                   | 97.91% |
| `ablation_mlp-adam.log`     | Adam 0.001, 12 ep                 | 97.92% |
| `ablation_mlp-full.log`     | Adam + cosine + wd + aug + ls     | 98.68% |

`mlp-sgd` is the reference config shown in Chapter 3 of the book.

### CNN (full, ~1.68M params)

Two conv blocks + two dense. BatchNorm and optimizer variants:

| Log | Config | Final val acc |
|-----|--------|---------------|
| `ablation_cnn-nobn-sgd.log`  | SGD 0.1, no BN                    | 98.98% |
| `ablation_cnn-nobn-adam.log` | Adam 0.001, no BN                 | 99.00% |
| `ablation_cnn-bn-sgd.log`    | SGD 0.1 + BN                      | 99.06% |
| `ablation_cnn-bn-adam.log`   | Adam 0.001 + BN                   | 99.31% |
| `ablation_cnn-bn-full.log`   | Adam + BN + cosine + wd + aug     | 99.50% |

### CNN-lite (~150K params, shallower)

Variants used for the width/pooling/BN-effect ablation chapters.

| Log | Config | Final val acc |
|-----|--------|---------------|
| `ablation_cnn-lite-bn-sgd.log`         | SGD 0.1 + BN                        | 95.95% |
| `ablation_cnn-lite-bn-sgd002.log`      | SGD 0.02 + BN                       | 78.53% |
| `ablation_cnn-lite-bn-sgd_nopool.log`  | SGD 0.1 + BN, no maxpool            | 94.84% |
| `ablation_cnn-lite-nobn-sgd.log`       | SGD 0.1, no BN                      | 95.70% |
| `ablation_cnn-lite-nobn-sgd002.log`    | SGD 0.02, no BN                     | 55.03% |
| `ablation_cnn-lite-nobn-sgd_nopool.log`| SGD 0.1, no BN, no maxpool          | 95.41% |

---

## CIFAR-10 — where BN starts to really matter

Same architecture as the MNIST CNN but on color 32×32 images. CIFAR
is harder than MNIST — the gap between nobn and bn configs is
dramatic.

| Log | Config | Final val acc |
|-----|--------|---------------|
| `ablation_cifar-nobn-sgd.log`    | SGD 0.1, no BN                   | 10.0% (diverged) |
| `ablation_cifar-nobn-sgd002.log` | SGD 0.02, no BN                  | 72.5% |
| `ablation_cifar-nobn-adam.log`   | Adam 0.001, no BN                | 72.8% |
| `ablation_cifar-bn-sgd.log`      | SGD 0.1 + BN                     | 61.6% |
| `ablation_cifar-bn-sgd002.log`   | SGD 0.02 + BN                    | 72.9% |
| `ablation_cifar-bn-adam.log`     | Adam 0.001 + BN                  | 76.5% |
| `ablation_cifar-bn-cosine.log`   | Adam + BN + cosine               | 79.6% |
| `ablation_cifar-bn-aug.log`      | Adam + BN + aug                  | 82.7% |
| `ablation_cifar-bn-full.log`     | Adam + BN + cosine + wd + aug    | 82.0% |

`cifar-lite-*` — same ablations against the lite-CNN architecture.

### The BN effect, cleanly

The `cifar-nobn-sgd` run at 10.0% is pure divergence (random guess
on 10 classes). Dropping the learning rate 5× (`-sgd002`) brings it
to 72.5%. Adding BN at the original lr gets back to 61.6% from
divergence. BN stabilizes training faster than lr-tuning does.

---

## Width sweeps

Parameter efficiency comparisons — hold the config constant, vary
the hidden size.

### MLP hidden width (MNIST)

| Log | Hidden width | Final val acc |
|-----|--------------|---------------|
| `ablation_width-h32.log`   | 32  | 93.98% |
| `ablation_width-h64.log`   | 64  | 94.34% |
| `ablation_width-h128.log`  | 128 | 94.72% |
| `ablation_width-h256.log`  | 256 | 94.88% |
| `ablation_width-h512.log`  | 512 | 95.12% |

Diminishing returns after h=128. Chapter on "how much width you
actually need" uses this data.

### CNN channel width (MNIST)

| Log | Channels | Final val acc |
|-----|----------|---------------|
| `ablation_width-cnn-f8.log`  | base 8  | 98.34% |
| `ablation_width-cnn-f16.log` | base 16 | 98.35% |
| `ablation_width-cnn-f32.log` | base 32 | 98.72% |
| `ablation_width-cnn-f64.log` | base 64 | 98.90% |

### CNN channel width (CIFAR)

| Log | Channels | Final val acc |
|-----|----------|---------------|
| `ablation_width-cifar-f8.log`  | base 8  | 67.28% |
| `ablation_width-cifar-f16.log` | base 16 | 72.53% |
| `ablation_width-cifar-f32.log` | base 32 | 73.78% |
| `ablation_width-cifar-f64.log` | base 64 | 75.52% |

---

## ResNet-34 (Imagenette) — early/incomplete runs

These ablation-style r34 runs are early tuning iterations; accuracies
are near-random (~10% on 10-class Imagenette) and not representative.
Keep for the record; the production r34 training logs are the named
ones below.

| Log | Config |
|-----|--------|
| `ablation_r34-sgd.log`        | SGD, no tricks     |
| `ablation_r34-sgd-cosine.log` | SGD + cosine       |
| `ablation_r34-adam.log`       | Adam               |
| `ablation_r34-adam-cosine.log`| Adam + cosine      |

---

## Named full-trainer logs (Imagenette, 224×224)

Full training runs from the dedicated `Main*Train*.lean` binaries,
not the ablation framework. These are the ones we cite as the
"trained it, here's what it got" numbers.

### ResNet family
- `r34_train.log` / `r34_train_v2.log` — ResNet-34 baselines
- `r34_bn.log` — ResNet-34 with BN retuning
- `r34_adam.log` — ResNet-34 with Adam
- `r50_train.log` / `r50_train_v2.log` — ResNet-50
- `r50_bn.log`, `r50_adam.log` — ResNet-50 variants

### MobileNet family
- `mnv2_train.log` — MobileNet v2
- `mnv3_train.log` — MobileNet v3
- `mnv4_train.log` — MobileNet v4

### EfficientNet family
- `effnet_train.log` — EfficientNet baseline
- `effnet_se.log` — EfficientNet with SE blocks highlighted
- `effnet_v2.log` — EfficientNet V2

### Other
- `vgg_train.log` — VGG baseline
- `vit_train.log` — ViT-Tiny on Imagenette

---

## How to reproduce

```
# Build the ablation runner
lake build ablation

# Run a single ablation (by name, e.g. mlp-sgd)
IREE_BACKEND=rocm IREE_CHIP=gfx1100 \
  ./.lake/build/bin/ablation mlp-sgd

# Or run a named full-trainer
IREE_BACKEND=rocm IREE_CHIP=gfx1100 \
  ./.lake/build/bin/resnet34-train
```

Available ablation names are listed in `MainAblation.lean`. Training
configs (`s4tfBaseline`, `adamOnly`, `fullRecipe`, `sgdLowLr`, etc.)
are defined inline in that same file.

For CPU fallback (no GPU), set `IREE_BACKEND=llvm-cpu` and drop the
chip argument. CPU training is ~30× slower than GPU but works
everywhere.
