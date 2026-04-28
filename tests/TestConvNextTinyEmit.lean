import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.MlirCodegen
import LeanMlir.SpecHelpers

/-! Emit-only smoke for the full ConvNeXt-Tiny: confirms the codegen
    scales to (3, 3, 9, 3) blocks at (96, 192, 384, 768) channels on
    224×224 input. Skips iree-compile (slow for 28M params) — the
    smaller `convnext-mini` ablation already validates the full pipeline. -/

def convNextTiny : NetSpec where
  name := "ConvNeXt-T"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 96 4 4 .same,
    .convNextStage 96 3 .ln .gelu,
    .convNextDownsample 96 192,
    .convNextStage 192 3 .ln .gelu,
    .convNextDownsample 192 384,
    .convNextStage 384 9 .ln .gelu,
    .convNextDownsample 384 768,
    .convNextStage 768 3 .ln .gelu,
    .globalAvgPool,
    .dense 768 10 .identity
  ]

def main : IO Unit := do
  let spec := convNextTiny
  let batch : Nat := 8
  let moduleName := "jit_" ++ spec.sanitizedName ++ "_train_step"
  let fwdMlir := MlirCodegen.generate spec batch
  let trainMlir := MlirCodegen.generateTrainStep spec batch moduleName
  IO.FS.createDirAll ".lake/build"
  IO.FS.writeFile s!".lake/build/{spec.sanitizedName}_fwd.mlir" fwdMlir
  IO.FS.writeFile s!".lake/build/{spec.sanitizedName}_train_step.mlir" trainMlir
  IO.println s!"  spec       : {spec.name}"
  IO.println s!"  layers     : {spec.layers.length}"
  IO.println s!"  paramShapes: {spec.paramShapes.size} tensors"
  IO.println s!"  totalParams: {spec.totalParams} (~{spec.totalParams / 1000000}M)"
  IO.println s!"  fwd MLIR   : {fwdMlir.length} chars"
  IO.println s!"  train MLIR : {trainMlir.length} chars"
  let unsupFwd := (fwdMlir.splitOn "UNSUPPORTED").length - 1
  let unsupTrain := (trainMlir.splitOn "UNSUPPORTED").length - 1
  if unsupFwd > 0 || unsupTrain > 0 then
    IO.eprintln s!"  ERROR: UNSUPPORTED markers fwd={unsupFwd} train={unsupTrain}"
    IO.Process.exit 1
  IO.println "ConvNeXt-Tiny emit smoke OK."
