"""Hand-curated mapping: (bestiary binary, variant name) → timm model name.

Rules of thumb for adding a row:
- Only for image classifier architectures that timm actually ships (timm is
  image-models-only). LLMs, detection, diffusion, speech belong in a
  future HuggingFace-based equivalent script.
- Our bestiary variants labeled `(bestiary approximation)` or similar
  hedges are *expected* to differ from timm by a few %. Rows are still
  useful to pin the magnitude of that divergence.
- `None` as the timm name marks "we know there's no timm counterpart"
  (documented non-coverage, different from "we forgot to add it").

Expand over time; nothing breaks if a row is missing — the oracle simply
covers a smaller subset.
"""

MAPPING: dict[tuple[str, str], str | None] = {
    # AlexNet — not in timm (torchvision ships it; timm skips the pre-ResNet era)
    ("bestiary-alexnet", "AlexNet (Krizhevsky 2012)"): None,

    # Inception family — GoogLeNet/v1 isn't in timm; v3 and v4 are. Don't
    # cross-check v1 against v3 — they're structurally different.
    ("bestiary-inception", "GoogLeNet (Inception v1)"): None,
    ("bestiary-inception", "Inception-v3 (bestiary approximation)"): "inception_v3",
    ("bestiary-inception", "Inception-v4 (bestiary approximation)"): "inception_v4",

    # Xception
    ("bestiary-xception", "Xception"): "xception",

    # ConvNeXt — ImageNet-1k pretrain variants
    ("bestiary-convnext", "ConvNeXt-T"): "convnext_tiny",
    ("bestiary-convnext", "ConvNeXt-S"): "convnext_small",
    ("bestiary-convnext", "ConvNeXt-B"): "convnext_base",
    ("bestiary-convnext", "ConvNeXt-L"): "convnext_large",

    # Swin Transformer v1
    ("bestiary-swin", "Swin-T"): "swin_tiny_patch4_window7_224",
    ("bestiary-swin", "Swin-S"): "swin_small_patch4_window7_224",
    ("bestiary-swin", "Swin-B"): "swin_base_patch4_window7_224",

    # MobileViT
    ("bestiary-mobilevit", "MobileViT-XXS"): "mobilevit_xxs",
    ("bestiary-mobilevit", "MobileViT-XS"): "mobilevit_xs",
    ("bestiary-mobilevit", "MobileViT-S"): "mobilevit_s",

    # Non-matches (explicit so we don't forget — still in timm's scope but
    # either we have a simplified spec or the timm name doesn't line up)
    ("bestiary-squeezenet", "SqueezeNet 1.0"): None,   # in torchvision, not timm
    ("bestiary-squeezenet", "SqueezeNet 1.1"): None,
    ("bestiary-shufflenet", "ShuffleNet 1.0× (g=3)"): None,
    ("bestiary-shufflenetv2", "ShuffleNet v2 1.0×"): None,
    ("bestiary-lenet", "LeNet-5"): None,               # too small / not a timm target
}
