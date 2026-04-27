import LeanMlir

/-! Static smoke test for every phase-3 trainer.

This catches the bug pattern that bit `mnist-cnn-train` (and would
have bitten future trainers as the codegen drifted): the trainer's
FFI call signature didn't match what `MlirCodegen.generateTrainStep`
emits. The IREE VM rejected the call before the first batch with
"input list and function mismatch; expected 52 arguments but passed 35".

Test strategy: for every spec we ship, generate the train step MLIR
and check that:

1. `spec.paramShapes.size * 3 + 4` input tensors == the function signature.
   - 3× paramShapes because the train step takes params + m (1st moment)
     + v (2nd moment) for Adam.
   - +4 for x_flat, y, lr, t.
2. `spec.shapesBA` is consistent with `spec.paramShapes` (round-trip).

If both invariants hold, the unified `LeanMlir.Train.train` function
will work for the spec without any trainer-specific FFI plumbing. -/

-- Specs (full architecture inventory). Ideally these would be imported
-- from each Main*Train.lean but the way Lean modules work, that would
-- conflict with each file having its own `def main`. So we duplicate
-- the spec definitions here. The TestSpecHelpers byte-equality check
-- ensures these match the trainers' specs structurally.

namespace Smoke

def resnet34 : NetSpec where
  name := "ResNet-34"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same, .maxPool 2 2,
    .residualBlock  64  64 3 1, .residualBlock  64 128 4 2,
    .residualBlock 128 256 6 2, .residualBlock 256 512 3 2,
    .globalAvgPool, .dense 512 10 .identity
  ]

def resnet50 : NetSpec where
  name := "ResNet-50"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same, .maxPool 2 2,
    .bottleneckBlock   64  256 3 1, .bottleneckBlock  256  512 4 2,
    .bottleneckBlock  512 1024 6 2, .bottleneckBlock 1024 2048 3 2,
    .globalAvgPool, .dense 2048 10 .identity
  ]

def mobilenetV2 : NetSpec where
  name := "MobileNet-v2"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 32 3 2 .same,
    .invertedResidual  32  16 1 1 1, .invertedResidual  16  24 6 2 2,
    .invertedResidual  24  32 6 2 3, .invertedResidual  32  64 6 2 4,
    .invertedResidual  64  96 6 1 3, .invertedResidual  96 160 6 2 3,
    .invertedResidual 160 320 6 1 1,
    .convBn 320 1280 1 1 .same, .globalAvgPool, .dense 1280 10 .identity
  ]

def mobilenetV3Large : NetSpec where
  name := "MobileNet v3-Large"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 16 3 2 .same,
    .mbConvV3  16  16  16  3 1 false .relu,
    .mbConvV3  16  24  64  3 2 false .relu,
    .mbConvV3  24  24  72  3 1 false .relu,
    .mbConvV3  24  40  72  5 2 true  false,
    .mbConvV3  40  40 120  5 1 true  false,
    .mbConvV3  40  40 120  5 1 true  false,
    .mbConvV3  40  80 240  3 2 false .hSwish,
    .mbConvV3  80  80 200  3 1 false .hSwish,
    .mbConvV3  80  80 184  3 1 false .hSwish,
    .mbConvV3  80  80 184  3 1 false .hSwish,
    .mbConvV3  80 112 480  3 1 true  true,
    .mbConvV3 112 112 672  5 1 true  true,
    .mbConvV3 112 160 672  5 2 true  true,
    .mbConvV3 160 160 960  5 1 true  true,
    .mbConvV3 160 160 960  5 1 true  true,
    .convBn 160 960 1 1 .same, .globalAvgPool, .dense 960 10 .identity
  ]

def mobilenetV4Medium : NetSpec where
  name := "MobileNet V4-Medium"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 32 3 2 .same,
    .fusedMbConv 32 48 4 3 2 1 false,
    .uib  48  80 4 2 3 5, .uib  80  80 2 1 3 3, .uib  80 160 6 2 0 3,
    .uib 160 160 4 1 3 3, .uib 160 160 4 1 3 5, .uib 160 160 4 1 5 0,
    .uib 160 160 4 1 0 3, .uib 160 160 4 1 3 0, .uib 160 160 4 1 0 0,
    .uib 160 160 4 1 3 3, .uib 160 256 6 2 5 5, .uib 256 256 4 1 5 5,
    .uib 256 256 4 1 0 3, .uib 256 256 4 1 3 0,
    .convBn 256 1280 1 1 .same, .globalAvgPool, .dense 1280 10 .identity
  ]

def efficientNetB0 : NetSpec where
  name := "EfficientNet-B0"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 32 3 2 .same,
    .mbConv  32  16 1 3 1 1 true, .mbConv  16  24 6 3 2 2 true,
    .mbConv  24  40 6 5 2 2 true, .mbConv  40  80 6 3 2 3 true,
    .mbConv  80 112 6 5 1 3 true, .mbConv 112 192 6 5 2 4 true,
    .mbConv 192 320 6 3 1 1 true,
    .convBn 320 1280 1 1 .same, .globalAvgPool, .dense 1280 10 .identity
  ]

def efficientNetV2S : NetSpec where
  name := "EfficientNet V2-S"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 24 3 2 .same,
    .fusedMbConv  24  24 1 3 1 2 false,
    .fusedMbConv  24  48 4 3 2 4 false,
    .fusedMbConv  48  64 4 3 2 4 false,
    .mbConv  64 128 4 3 2 6 true,
    .mbConv 128 160 6 3 1 9 true,
    .mbConv 160 256 6 3 2 15 true,
    .convBn 256 1280 1 1 .same, .globalAvgPool, .dense 1280 10 .identity
  ]

def vitTiny : NetSpec where
  name := "ViT-Tiny"
  imageH := 224
  imageW := 224
  layers := [
    .patchEmbed 3 192 16 196,
    .transformerEncoder 192 3 768 12,
    .dense 192 10 .identity
  ]

def vgg16bn : NetSpec where
  name := "VGG-16-BN"
  imageH := 224
  imageW := 224
  layers := [
    .convBn   3  64 3 1 .same, .convBn  64  64 3 1 .same, .maxPool 2 2,
    .convBn  64 128 3 1 .same, .convBn 128 128 3 1 .same, .maxPool 2 2,
    .convBn 128 256 3 1 .same, .convBn 256 256 3 1 .same, .convBn 256 256 3 1 .same, .maxPool 2 2,
    .convBn 256 512 3 1 .same, .convBn 512 512 3 1 .same, .convBn 512 512 3 1 .same, .maxPool 2 2,
    .convBn 512 512 3 1 .same, .convBn 512 512 3 1 .same, .convBn 512 512 3 1 .same, .maxPool 2 2,
    .globalAvgPool, .dense 512 10 .identity
  ]

-- Smaller specs (the previously-broken trainers).

def mnistMlp : NetSpec where
  name := "MNIST-MLP"
  imageH := 28
  imageW := 28
  layers := [
    .dense 784 512 .relu,
    .dense 512 512 .relu,
    .dense 512  10 .identity
  ]

def mnistCnn : NetSpec where
  name := "MNIST-CNN"
  imageH := 28
  imageW := 28
  layers := [
    .convBn 1 32 3 1 .same,
    .convBn 32 32 3 1 .same,
    .maxPool 2 2,
    .convBn 32 64 3 1 .same,
    .convBn 64 64 3 1 .same,
    .maxPool 2 2,
    .flatten,
    .dense 3136 512 .relu,
    .dense 512 10 .identity
  ]

def cifarCnnBn : NetSpec where
  name := "CIFAR-10-BN"
  imageH := 32
  imageW := 32
  layers := [
    .convBn  3 32 3 1 .same,
    .convBn 32 32 3 1 .same,
    .maxPool 2 2,
    .convBn 32 64 3 1 .same,
    .convBn 64 64 3 1 .same,
    .maxPool 2 2,
    .flatten,
    .dense 4096 512 .relu,
    .dense 512 512 .relu,
    .dense 512 10 .identity
  ]

def cifarCnn : NetSpec where
  name := "CIFAR-10-CNN"
  imageH := 32
  imageW := 32
  layers := [
    .conv2d  3 32 3 .same .relu,
    .conv2d 32 32 3 .same .relu,
    .maxPool 2 2,
    .conv2d 32 64 3 .same .relu,
    .conv2d 64 64 3 .same .relu,
    .maxPool 2 2,
    .flatten,
    .dense 4096 512 .relu,
    .dense 512 512 .relu,
    .dense 512 10 .identity
  ]

end Smoke

/-- Count occurrences of `needle` in `haystack`. -/
private def countSubstr (haystack needle : String) : Nat :=
  (haystack.splitOn needle).length - 1

/-- Given the generated train-step MLIR, count tensor inputs to `@main`.
    The generated signature is canonical — every parameter is a single
    `tensor<...>` declaration — so counting `tensor<` between `@main(`
    and the next `) ->` returns the input count. -/
def countMainArgs (mlir : String) : Nat :=
  match mlir.splitOn "func.func @main(" with
  | _ :: rest :: _ =>
    match rest.splitOn ") ->" with
    | sig :: _ => countSubstr sig "tensor<"
    | [] => 0
  | _ => 0

def main : IO Unit := do
  let specs : Array (String × NetSpec) := #[
    ("ResNet-34",          Smoke.resnet34),
    ("ResNet-50",          Smoke.resnet50),
    ("MobileNetV2",        Smoke.mobilenetV2),
    ("MobileNetV3-Large",  Smoke.mobilenetV3Large),
    ("MobileNetV4-Medium", Smoke.mobilenetV4Medium),
    ("EfficientNet-B0",    Smoke.efficientNetB0),
    ("EfficientNetV2-S",   Smoke.efficientNetV2S),
    ("ViT-Tiny",           Smoke.vitTiny),
    ("VGG-16-BN",          Smoke.vgg16bn),
    ("MNIST-MLP",          Smoke.mnistMlp),
    ("MNIST-CNN",          Smoke.mnistCnn),
    ("CIFAR-10-BN",        Smoke.cifarCnnBn),
    ("CIFAR-10-CNN",       Smoke.cifarCnn)
  ]

  let mut ok := true
  for (name, spec) in specs do
    let nP := spec.paramShapes.size
    -- Adam train step has 3*nP param tensors + x + y + lr + t = 3*nP + 4
    let expected := 3 * nP + 4
    let mlir := MlirCodegen.generateTrainStep spec 32 ("jit_" ++ spec.sanitizedName ++ "_train_step")
    let actual := countMainArgs mlir
    if actual == expected then
      IO.println s!"  ✓ {name}: train step has {actual} input tensors (3×{nP} params + 4)"
    else
      IO.println s!"  ✗ {name}: codegen emits {actual} inputs, FFI expects {expected} (3×{nP} + 4)"
      ok := false
    -- Also verify shapesBA round-trips: first u32 LE = 3 * paramShapes.size
    let ba := spec.shapesBA
    let b0 := (ba.get! 0).toNat
    let b1 := (ba.get! 1).toNat
    let b2 := (ba.get! 2).toNat
    let b3 := (ba.get! 3).toNat
    let n := b0 + b1 * 256 + b2 * 65536 + b3 * 16777216
    if n == 3 * nP then
      pure ()
    else
      IO.println s!"  ✗ {name}: shapesBA has {n} entries, expected {3 * nP}"
      ok := false

  if ok then
    IO.println "All trainers structurally consistent. No drift detected."
  else
    IO.eprintln "Drift detected — at least one trainer's codegen and FFI signature disagree."
    IO.Process.exit 1
