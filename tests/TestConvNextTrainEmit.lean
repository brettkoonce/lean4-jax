import LeanMlir.IreeRuntime
import LeanMlir.F32Array
import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.MlirCodegen
import LeanMlir.SpecHelpers

/-! Diagnostic: try to emit a train_step for a tiny ConvNeXt and see what
    happens. Stage 3 isn't wired yet, so we expect either an UNSUPPORTED
    LAYER marker in the body or a signature/body mismatch when iree-compile
    tries to lower it. This test makes the failure mode explicit. -/

def tinyConvNextSpec : NetSpec where
  name := "tiny-ConvNeXt"
  imageH := 32
  imageW := 32
  layers := [
    .convBn 3 32 2 2 .same,
    .convNextStage 32 2,
    .convNextDownsample 32 64,
    .convNextStage 64 2,
    .globalAvgPool,
    .dense 64 10 .identity
  ]

private def findIreeCompile : IO String := do
  if ← System.FilePath.pathExists ".venv/bin/iree-compile" then
    return ".venv/bin/iree-compile"
  return "iree-compile"

def main : IO Unit := do
  let spec := tinyConvNextSpec
  let mlir := MlirCodegen.generateTrainStep spec 2
  IO.FS.createDirAll ".lake/build"
  let mlirPath := s!".lake/build/{spec.sanitizedName}_train_step.mlir"
  IO.FS.writeFile mlirPath mlir
  IO.println s!"Train-step MLIR: {mlir.length} chars → {mlirPath}"

  -- Count UNSUPPORTED markers — present means Stage 3 hasn't been wired
  -- for this layer kind yet.
  let unsup := (mlir.splitOn "UNSUPPORTED").length - 1
  IO.println s!"  UNSUPPORTED markers: {unsup}"

  -- Try to iree-compile; expect failure if any layer wasn't lowered.
  let compiler ← findIreeCompile
  let r ← IO.Process.output {
    cmd := compiler,
    args := #[mlirPath, "--iree-hal-target-backends=llvm-cpu",
              "--iree-llvmcpu-target-cpu=host",
              "-o", s!".lake/build/{spec.sanitizedName}_train_step.vmfb"]
  }
  if r.exitCode != 0 then
    IO.println s!"  iree-compile FAILED (expected — Stage 3 not done)"
    IO.println s!"  first 600 chars of stderr:"
    IO.println s!"  ----"
    IO.println (r.stderr.take 600)
  else
    IO.println "  iree-compile OK (surprising — Stage 3 may be partially done)"
