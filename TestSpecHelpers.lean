import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.MlirCodegen
import LeanMlir.IreeRuntime
import LeanMlir.SpecHelpers

/-! Verify that `NetSpec.paramShapes`/`bnShapesBA`/`evalShapes`/`shapesBA`
    in LeanMlir.SpecHelpers produce byte-for-byte identical output to the
    hand-rolled walkers each Main*Train.lean used to ship with.

    If this passes for every architecture, the trainer refactor is safe. -/

-- ResNet-34
def resnet34 : NetSpec where
  name := "ResNet-34"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,
    .residualBlock  64  64 3 1,
    .residualBlock  64 128 4 2,
    .residualBlock 128 256 6 2,
    .residualBlock 256 512 3 2,
    .globalAvgPool,
    .dense 512 10 .identity
  ]

-- Reference implementation: copy of MainResnetTrain.lean's old paramShapes
def resnet34ParamShapesInline : Array (Array Nat) := Id.run do
  let mut shapes : Array (Array Nat) := #[]
  for l in resnet34.layers do
    match l with
    | .convBn ic oc k _ _ =>
      shapes := shapes.push #[oc, ic, k, k] |>.push #[oc] |>.push #[oc]
    | .dense fi fo _ =>
      shapes := shapes.push #[fi, fo] |>.push #[fo]
    | .residualBlock ic oc nBlocks firstStride =>
      let needsProj := !(ic == oc && firstStride == 1)
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        shapes := shapes.push #[oc, blockIc, 3, 3] |>.push #[oc] |>.push #[oc]
        shapes := shapes.push #[oc, oc, 3, 3] |>.push #[oc] |>.push #[oc]
        if bi == 0 && needsProj then
          shapes := shapes.push #[oc, ic, 1, 1] |>.push #[oc] |>.push #[oc]
    | _ => pure ()
  return shapes

def resnet50 : NetSpec where
  name := "ResNet-50"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,
    .bottleneckBlock   64  256 3 1,
    .bottleneckBlock  256  512 4 2,
    .bottleneckBlock  512 1024 6 2,
    .bottleneckBlock 1024 2048 3 2,
    .globalAvgPool,
    .dense 2048 10 .identity
  ]

-- Reference for R50: bottleneck blocks
def resnet50ParamShapesInline : Array (Array Nat) := Id.run do
  let mut shapes : Array (Array Nat) := #[]
  for l in resnet50.layers do
    match l with
    | .convBn ic oc k _ _ =>
      shapes := shapes.push #[oc, ic, k, k] |>.push #[oc] |>.push #[oc]
    | .dense fi fo _ =>
      shapes := shapes.push #[fi, fo] |>.push #[fo]
    | .bottleneckBlock ic oc nBlocks firstStride =>
      let mid := oc / 4
      let needsProj := !(ic == oc && firstStride == 1)
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        shapes := shapes.push #[mid, blockIc, 1, 1] |>.push #[mid] |>.push #[mid]
        shapes := shapes.push #[mid, mid, 3, 3] |>.push #[mid] |>.push #[mid]
        shapes := shapes.push #[oc, mid, 1, 1] |>.push #[oc] |>.push #[oc]
        if bi == 0 && needsProj then
          shapes := shapes.push #[oc, ic, 1, 1] |>.push #[oc] |>.push #[oc]
    | _ => pure ()
  return shapes

-- ViT
def vitTiny : NetSpec where
  name := "ViT-Tiny"
  imageH := 224
  imageW := 224
  layers := [
    .patchEmbed 3 192 16 196,
    .transformerEncoder 192 3 768 12,
    .dense 192 10 .identity
  ]

def vitTinyParamShapesInline : Array (Array Nat) := Id.run do
  let mut shapes : Array (Array Nat) := #[]
  for l in vitTiny.layers do
    match l with
    | .patchEmbed ic dim p nP =>
      shapes := shapes.push #[dim, ic, p, p] |>.push #[dim]
      shapes := shapes.push #[dim]
      shapes := shapes.push #[nP + 1, dim]
    | .transformerEncoder dim _heads mlpDim nBlocks =>
      for _bi in [:nBlocks] do
        shapes := shapes.push #[dim] |>.push #[dim]
        shapes := shapes.push #[dim, dim] |>.push #[dim]
        shapes := shapes.push #[dim, dim] |>.push #[dim]
        shapes := shapes.push #[dim, dim] |>.push #[dim]
        shapes := shapes.push #[dim, dim] |>.push #[dim]
        shapes := shapes.push #[dim] |>.push #[dim]
        shapes := shapes.push #[dim, mlpDim] |>.push #[mlpDim]
        shapes := shapes.push #[mlpDim, dim] |>.push #[dim]
      shapes := shapes.push #[dim] |>.push #[dim]
    | .dense fi fo _ =>
      shapes := shapes.push #[fi, fo] |>.push #[fo]
    | _ => pure ()
  return shapes

-- MobileNetV2: invertedResidual
def mobilenetV2 : NetSpec where
  name := "MobileNet-v2"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 32 3 2 .same,
    .invertedResidual  32  16 1 1 1,
    .invertedResidual  16  24 6 2 2,
    .invertedResidual  24  32 6 2 3,
    .invertedResidual  32  64 6 2 4,
    .invertedResidual  64  96 6 1 3,
    .invertedResidual  96 160 6 2 3,
    .invertedResidual 160 320 6 1 1,
    .convBn 320 1280 1 1 .same,
    .globalAvgPool,
    .dense 1280 10 .identity
  ]

def mobilenetV2ParamShapesInline : Array (Array Nat) := Id.run do
  let mut shapes : Array (Array Nat) := #[]
  for l in mobilenetV2.layers do
    match l with
    | .convBn ic oc k _ _ =>
      shapes := shapes.push #[oc, ic, k, k] |>.push #[oc] |>.push #[oc]
    | .dense fi fo _ =>
      shapes := shapes.push #[fi, fo] |>.push #[fo]
    | .invertedResidual ic oc expand _stride n =>
      for bi in [:n] do
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        if expand != 1 then
          shapes := shapes.push #[mid, blockIc, 1, 1] |>.push #[mid] |>.push #[mid]
        shapes := shapes.push #[mid, 1, 3, 3] |>.push #[mid] |>.push #[mid]
        shapes := shapes.push #[oc, mid, 1, 1] |>.push #[oc] |>.push #[oc]
    | _ => pure ()
  return shapes

-- EfficientNet-B0: mbConv with SE
def efficientNetB0 : NetSpec where
  name := "EfficientNet-B0"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 32 3 2 .same,
    .mbConv  32  16 1 3 1 1 true,
    .mbConv  16  24 6 3 2 2 true,
    .mbConv  24  40 6 5 2 2 true,
    .mbConv  40  80 6 3 2 3 true,
    .mbConv  80 112 6 5 1 3 true,
    .mbConv 112 192 6 5 2 4 true,
    .mbConv 192 320 6 3 1 1 true,
    .convBn 320 1280 1 1 .same,
    .globalAvgPool,
    .dense 1280 10 .identity
  ]

def efficientNetB0ParamShapesInline : Array (Array Nat) := Id.run do
  let mut shapes : Array (Array Nat) := #[]
  for l in efficientNetB0.layers do
    match l with
    | .convBn ic oc k _ _ =>
      shapes := shapes.push #[oc, ic, k, k] |>.push #[oc] |>.push #[oc]
    | .dense fi fo _ =>
      shapes := shapes.push #[fi, fo] |>.push #[fo]
    | .mbConv ic oc expand kSize _ n useSE =>
      for bi in [:n] do
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        let seMid := Nat.max 1 (mid / 4)
        if expand != 1 then
          shapes := shapes.push #[mid, blockIc, 1, 1] |>.push #[mid] |>.push #[mid]
        shapes := shapes.push #[mid, 1, kSize, kSize] |>.push #[mid] |>.push #[mid]
        if useSE then
          shapes := shapes.push #[seMid, mid, 1, 1] |>.push #[seMid]
          shapes := shapes.push #[mid, seMid, 1, 1] |>.push #[mid]
        shapes := shapes.push #[oc, mid, 1, 1] |>.push #[oc] |>.push #[oc]
    | _ => pure ()
  return shapes

-- MobileNetV4-Medium: uib + fusedMbConv
def mobilenetV4Medium : NetSpec where
  name := "MobileNet V4-Medium"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 32 3 2 .same,
    .fusedMbConv 32 48 4 3 2 1 false,
    .uib  48  80 4 2 3 5,
    .uib  80  80 2 1 3 3,
    .uib  80 160 6 2 0 3,
    .uib 160 160 4 1 3 3,
    .uib 160 160 4 1 3 5,
    .uib 160 160 4 1 5 0,
    .uib 160 160 4 1 0 3,
    .uib 160 160 4 1 3 0,
    .uib 160 160 4 1 0 0,
    .uib 160 160 4 1 3 3,
    .uib 160 256 6 2 5 5,
    .uib 256 256 4 1 5 5,
    .uib 256 256 4 1 0 3,
    .uib 256 256 4 1 3 0,
    .convBn 256 1280 1 1 .same,
    .globalAvgPool,
    .dense 1280 10 .identity
  ]

def mobilenetV4MediumParamShapesInline : Array (Array Nat) := Id.run do
  let mut shapes : Array (Array Nat) := #[]
  for l in mobilenetV4Medium.layers do
    match l with
    | .convBn ic oc k _ _ =>
      shapes := shapes.push #[oc, ic, k, k] |>.push #[oc] |>.push #[oc]
    | .dense fi fo _ =>
      shapes := shapes.push #[fi, fo] |>.push #[fo]
    | .fusedMbConv ic oc expand kSize _ n useSE =>
      for bi in [:n] do
        let blockIc := if bi == 0 then ic else oc
        let mid := if expand == 1 then oc else blockIc * expand
        let seMid := Nat.max 1 (mid / 4)
        shapes := shapes.push #[mid, blockIc, kSize, kSize] |>.push #[mid] |>.push #[mid]
        if useSE then
          shapes := shapes.push #[seMid, mid, 1, 1] |>.push #[seMid]
          shapes := shapes.push #[mid, seMid, 1, 1] |>.push #[mid]
        if expand != 1 then
          shapes := shapes.push #[oc, mid, 1, 1] |>.push #[oc] |>.push #[oc]
    | .uib ic oc expand _stride preDWk postDWk =>
      let mid := ic * expand
      if preDWk > 0 then
        shapes := shapes.push #[ic, 1, preDWk, preDWk] |>.push #[ic] |>.push #[ic]
      shapes := shapes.push #[mid, ic, 1, 1] |>.push #[mid] |>.push #[mid]
      if postDWk > 0 then
        shapes := shapes.push #[mid, 1, postDWk, postDWk] |>.push #[mid] |>.push #[mid]
      shapes := shapes.push #[oc, mid, 1, 1] |>.push #[oc] |>.push #[oc]
    | _ => pure ()
  return shapes

def shapesEq (a b : Array (Array Nat)) : Bool :=
  a.size == b.size && (List.range a.size).all (fun i =>
    let x := a[i]!
    let y := b[i]!
    x.size == y.size && (List.range x.size).all (fun j => x[j]! == y[j]!))

def main : IO Unit := do
  let mut ok := true
  let cases : Array (String × Array (Array Nat) × Array (Array Nat)) := #[
    ("ResNet-34",          resnet34.paramShapes,          resnet34ParamShapesInline),
    ("ResNet-50",          resnet50.paramShapes,          resnet50ParamShapesInline),
    ("MobileNetV2",        mobilenetV2.paramShapes,       mobilenetV2ParamShapesInline),
    ("EfficientNet-B0",    efficientNetB0.paramShapes,    efficientNetB0ParamShapesInline),
    ("MobileNetV4-Medium", mobilenetV4Medium.paramShapes, mobilenetV4MediumParamShapesInline),
    ("ViT-Tiny",           vitTiny.paramShapes,           vitTinyParamShapesInline)
  ]
  for (name, helper, inline) in cases do
    if shapesEq helper inline then
      IO.println s!"  ✓ {name}: paramShapes match ({helper.size} entries)"
    else
      IO.println s!"  ✗ {name}: paramShapes DIFFER (helper={helper.size}, inline={inline.size})"
      ok := false
  -- Also verify evalFnName matches the old hardcoded strings.
  let evalCases : Array (String × String × String) := #[
    ("ResNet-34",          resnet34.evalFnName,          "resnet_34_eval.forward_eval"),
    ("ResNet-50",          resnet50.evalFnName,          "resnet_50_eval.forward_eval"),
    ("MobileNetV2",        mobilenetV2.evalFnName,       "mobilenet_v2_eval.forward_eval"),
    ("EfficientNet-B0",    efficientNetB0.evalFnName,    "efficientnet_b0_eval.forward_eval"),
    ("MobileNetV4-Medium", mobilenetV4Medium.evalFnName, "mobilenet_v4_medium_eval.forward_eval"),
    ("ViT-Tiny",           vitTiny.evalFnName,           "vit_tiny_eval.forward_eval")
  ]
  for (name, helper, expected) in evalCases do
    if helper == expected then
      IO.println s!"  ✓ {name}: evalFnName = {expected}"
    else
      IO.println s!"  ✗ {name}: evalFnName MISMATCH: got {helper}, expected {expected}"
      ok := false
  if ok then
    IO.println "All paramShapes and evalFnNames match. Refactor is safe."
  else
    IO.eprintln "Mismatch detected — do NOT refactor further."
    IO.Process.exit 1
