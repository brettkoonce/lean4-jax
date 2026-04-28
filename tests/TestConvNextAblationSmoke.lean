import LeanMlir.IreeRuntime
import LeanMlir.F32Array
import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.MlirCodegen
import LeanMlir.SpecHelpers
import LeanMlir.Train

/-! Stage 4 smoke: confirm the GELU vs ReLU ConvNeXt ablation pair
    emits valid forward + train-step MLIR and IREE-compiles cleanly.
    Uses the CIFAR-sized "mini" variant so the full codegen path
    runs in seconds. -/

def convNextMiniGeluSpec : NetSpec where
  name := "ConvNeXt-Mini-GELU"
  imageH := 32
  imageW := 32
  layers := [
    .convBn 3 32 2 2 .same,
    .convNextStage 32 2 .ln .gelu,
    .convNextDownsample 32 64,
    .convNextStage 64 2 .ln .gelu,
    .globalAvgPool,
    .dense 64 10 .identity
  ]

def convNextMiniReluSpec : NetSpec where
  name := "ConvNeXt-Mini-ReLU"
  imageH := 32
  imageW := 32
  layers := [
    .convBn 3 32 2 2 .same,
    .convNextStage 32 2 .ln .relu,
    .convNextDownsample 32 64,
    .convNextStage 64 2 .ln .relu,
    .globalAvgPool,
    .dense 64 10 .identity
  ]

private def findIreeCompile : IO String := do
  if ← System.FilePath.pathExists ".venv/bin/iree-compile" then
    return ".venv/bin/iree-compile"
  return "iree-compile"

private def runOne (spec : NetSpec) (batch : Nat) : IO Unit := do
  IO.println s!"  ── {spec.name} ──"
  let moduleName := "jit_" ++ spec.sanitizedName ++ "_train_step"
  let fwdMlir := MlirCodegen.generate spec batch
  let trainMlir := MlirCodegen.generateTrainStep spec batch moduleName
  IO.FS.createDirAll ".lake/build"
  let pfx := s!".lake/build/{spec.sanitizedName}"
  IO.FS.writeFile s!"{pfx}_fwd.mlir" fwdMlir
  IO.FS.writeFile s!"{pfx}_train_step.mlir" trainMlir
  IO.println s!"     fwd:   {fwdMlir.length} chars"
  IO.println s!"     train: {trainMlir.length} chars"
  let unsup := (trainMlir.splitOn "UNSUPPORTED").length - 1
  if unsup > 0 then
    IO.eprintln s!"     ERROR: {unsup} UNSUPPORTED markers in train MLIR"
    IO.Process.exit 1

  let compiler ← findIreeCompile
  for (label, srcPath, vmfbPath) in [
        ("forward",    s!"{pfx}_fwd.mlir",        s!"{pfx}_fwd.vmfb"),
        ("train-step", s!"{pfx}_train_step.mlir", s!"{pfx}_train_step.vmfb")
      ] do
    let r ← IO.Process.output {
      cmd := compiler,
      args := #[srcPath, "--iree-hal-target-backends=llvm-cpu",
                "--iree-llvmcpu-target-cpu=host", "-o", vmfbPath]
    }
    if r.exitCode != 0 then
      IO.eprintln s!"     iree-compile FAILED ({label}):\n{r.stderr.take 1500}"
      IO.Process.exit 1
    IO.println s!"     iree-compile {label} OK → {vmfbPath}"

  -- One Adam step on synthetic data — confirms loss is finite.
  let p ← spec.heInitParams
  let nP := F32.size p
  let m ← F32.const nP.toUSize 0.0
  let v ← F32.const nP.toUSize 0.0
  let packed := (p.append m).append v
  let pixels := 3 * spec.imageH * spec.imageW
  let xba ← F32.heInit 7 (batch * pixels).toUSize 1.0
  let xSh := packXShape #[batch, pixels]
  let mut yb : ByteArray := .empty
  for i in [:batch] do
    let lbl : UInt32 := (i % 10).toUInt32
    yb := yb.push (lbl &&& 0xFF).toUInt8
    yb := yb.push ((lbl >>> 8) &&& 0xFF).toUInt8
    yb := yb.push ((lbl >>> 16) &&& 0xFF).toUInt8
    yb := yb.push ((lbl >>> 24) &&& 0xFF).toUInt8
  let sess ← IreeSession.create s!"{pfx}_train_step.vmfb"
  let out ← IreeSession.trainStepAdamF32 sess spec.trainFnName
              packed spec.shapesBA xba xSh yb 0.001 1.0 spec.bnShapesBA batch.toUSize
  let loss := F32.extractLoss out (3 * nP)
  IO.println s!"     1-step Adam loss = {loss}"
  if loss.isNaN || loss.isInf then
    IO.eprintln s!"     ERROR: non-finite loss"
    IO.Process.exit 1

def main : IO Unit := do
  IO.println "── Stage 4 ablation smoke ──"
  runOne convNextMiniGeluSpec 2
  runOne convNextMiniReluSpec 2
  IO.println "Stage 4 smoke OK."
