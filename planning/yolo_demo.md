# yolo_demo.md — YOLOv1 detection demo on Pascal VOC

Goal: a worked detection example for the bestiary that runs end-to-end
with verified gradients. Targets **bestiary entry status**, not a full
pedagogical chapter. Pairs with `bifpn_demo.md` (EfficientDet/BiFPN)
as the "two detection demos" scope.

## The rule

YOLO's "innovation" relative to the classification chapters is the
**loss function** and the **target encoding**, not the architecture
or the gradients. Every primitive needed is already in the codegen
scope. This is what makes the demo cheap.

## Architecture: reuse existing backbone + grid head

| Piece | Source | New work |
|---|---|---|
| Backbone | R34 or EnetB0 (already trained, already proved) | none |
| Head | Strip classification head, add `dense + reshape` to (7, 7, B*5 + C) | NetSpec edit |
| Per-cell decoding | (B boxes × (x, y, w, h, conf) + C class probs) | data-side, no codegen |

For Pascal VOC: 7×7 grid, B=2 boxes/cell, C=20 classes →
output `(7, 7, 2*5 + 20) = (7, 7, 30)`. That's a `dense → reshape`
change to NetSpec. No new `Layer` primitives, no new VJPs.

## Loss function: 5-term MSE

YOLOv1's loss is five MSE terms with paper weights:

| Term | What | Weight |
|---|---|---|
| Localization (x, y) | Cell-relative center | λ_coord = 5 |
| Localization (w, h) | sqrt-scaled (the paper's `√w`, `√h` trick) | λ_coord = 5 |
| Objectness positive | Confidence on cells with objects | 1.0 |
| Objectness negative | Confidence on cells without | λ_noobj = 0.5 |
| Classification | Per-cell class probabilities | 1.0 |

All MSE → standard MSE VJP (already proved). The new logic is
**responsibility assignment**: per cell containing an object, pick
whichever predicted box has higher IoU with the GT box. That's a
forward-only computation — the IoU comparison is hard-assignment, no
gradient flows through it.

Implementation: ~1 week to get the loss + responsibility logic right.

## Data pipeline (the dominant cost)

Pascal VOC 2007+2012 train+val: ~16K images, XML bbox annotations.

| Item | Effort |
|---|---|
| VOC XML parser | 1 day |
| Target encoder: image + boxes → (7, 7, 30) tensor | 2-3 days |
| **Box-aware augmentation** | ~2 weeks |
| Training-time data loader | 2-3 days |

Box-aware augmentation is the only piece that genuinely doesn't
exist yet. Random crop / flip / scale / hue must transform the box
coordinates alongside the pixels. Our existing aug pack (Mixup,
CutMix, RandAug) is pixel-only and assumes labels are scalars.

## Inference + eval

| Item | Effort |
|---|---|
| NMS (sort by confidence, greedy drop overlap > τ) | 1 day |
| mAP@0.5 (Pascal VOC standard) | ~1 week |

mAP is finicky — precision-recall sweep per class then average.
Easy to get off-by-ones wrong; borrow logic from a reference
implementation rather than reimplementing from scratch.

## Sequencing

**MVP (3-4 weeks):**
- R34 backbone + YOLOv1 head + VOC 2007 only (5K images)
- Minimal box-aware aug (just flip + crop, no color)
- Greedy NMS, mAP@0.5
- Goal: training runs, mAP goes up, model emits detections

**Polish (+2-3 weeks):**
- VOC 2012 added (16K total)
- Full box-aware aug (HSV jitter, random affine)
- Proper mAP eval (across multiple IoU thresholds)
- Bestiary entry write-up

**Total ~5-7 weeks** for a polished bestiary entry.

## Cells to add

```
("r34-yolov1-voc",        ⟨r34YoloSpec, yoloVocConfig, .pascalVoc, "data/voc"⟩),
("enetb0-yolov1-voc",     ⟨enetb0YoloSpec, yoloVocConfig, .pascalVoc, "data/voc"⟩),
```

The ablation registry generalizes to detection — same harness, just
a different head + loss + dataset. Cross-architecture comparison
(R34 vs EnetB0 backbone for YOLO) is then a 2-cell ablation.

## Honest tradeoff

YOLOv1 paper accuracy on Pascal VOC: ~63% mAP with full Darknet
backbone + extensive training. Our R34-backbone version on partial
VOC would land somewhere in 40-50% mAP range. The demo is showing
"the framework does detection end-to-end with verified gradients,"
not beating the paper.

## Why YOLO over EfficientDet first

| | YOLOv1 | EfficientDet/BiFPN |
|---|---|---|
| New primitives needed | 0 | resize ops (bilinear/nearest VJP) |
| Architectural complexity | grid + 2 boxes | FPN cross-scale fusion + anchors |
| Loss complexity | 5-term MSE | focal loss + Smooth-L1 |
| Effort | 5-7 weeks | 8-10 weeks |
| Pedagogical clarity | very high (one diagram) | moderate (BiFPN is novel) |

YOLO is the cheaper path to "demo detection." EfficientDet is the
path to "expand the proof framework into geometric resampling."
Different goals; do YOLO first if the goal is a bestiary worked
example, do BiFPN if the goal is to grow the verified op set.

## Out of scope (deferred)

- YOLOv2/v3 (anchor boxes, multi-scale): same data pipeline cost,
  but adds anchor matching which is fiddly. Save for a later pass.
- COCO dataset: 10x larger than VOC, real eval pipeline, real GPU
  budget. VOC is the friendlier demo target.
- Training-from-scratch backbone: paper used Darknet, we're reusing
  R34/EnetB0 — fine for a demo, would be honest to note in the entry.
