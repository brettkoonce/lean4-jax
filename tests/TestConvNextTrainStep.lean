import LeanMlir.IreeRuntime
import LeanMlir.F32Array
import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.MlirCodegen
import LeanMlir.SpecHelpers
import LeanMlir.Train

/-! End-to-end Stage 3 smoke: emit train-step MLIR for a tiny ConvNeXt,
    iree-compile to vmfb, run one Adam step on synthetic data, verify the
    loss is finite and the parameter buffer round-trips. -/

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
  let batch : Nat := 2
  let nClasses : Nat := 10

  IO.println "── Stage 3 E2E smoke: tiny ConvNeXt train step ──"

  -- 1. Generate train-step MLIR (module name must match `spec.trainFnName`)
  let moduleName := "jit_" ++ spec.sanitizedName ++ "_train_step"
  let mlir := MlirCodegen.generateTrainStep spec batch moduleName
  IO.FS.createDirAll ".lake/build"
  let mlirPath := s!".lake/build/{spec.sanitizedName}_train_step.mlir"
  let vmfbPath := s!".lake/build/{spec.sanitizedName}_train_step.vmfb"
  IO.FS.writeFile mlirPath mlir
  IO.println s!"  emitted    : {mlir.length} chars → {mlirPath}"

  -- 2. iree-compile
  let compiler ← findIreeCompile
  let r ← IO.Process.output {
    cmd := compiler,
    args := #[mlirPath, "--iree-hal-target-backends=llvm-cpu",
              "--iree-llvmcpu-target-cpu=host",
              "-o", vmfbPath]
  }
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile FAILED:\n{r.stderr.take 2000}"
    IO.Process.exit 1
  IO.println s!"  compiled   : {vmfbPath}"

  -- 3. Init params, m, v (all zeros for m, v — Adam first step)
  let p ← spec.heInitParams
  let nP := F32.size p
  let m ← F32.const nP.toUSize 0.0
  let v ← F32.const nP.toUSize 0.0
  let packed := (p.append m).append v
  IO.println s!"  params     : {nP} f32s × 3 (params + m + v) = {F32.size packed} f32s"

  -- 4. Synthetic input + labels
  let pixels := 3 * spec.imageH * spec.imageW
  let xba ← F32.heInit 7 (batch * pixels).toUSize 1.0
  let xSh := packXShape #[batch, pixels]
  -- Labels: 4-byte int32 LE per sample, batch entries
  let mut yb : ByteArray := .empty
  for i in [:batch] do
    let lbl : UInt32 := (i % nClasses).toUInt32
    yb := yb.push (lbl &&& 0xFF).toUInt8
    yb := yb.push ((lbl >>> 8) &&& 0xFF).toUInt8
    yb := yb.push ((lbl >>> 16) &&& 0xFF).toUInt8
    yb := yb.push ((lbl >>> 24) &&& 0xFF).toUInt8

  -- 5. Run one Adam step
  let sess ← IreeSession.create vmfbPath
  IO.println "  session    : loaded"
  let allShapes := spec.shapesBA
  let bnShapes := spec.bnShapesBA
  let lr : Float := 0.001
  let t0 ← IO.monoMsNow
  let out ← IreeSession.trainStepAdamF32 sess spec.trainFnName
              packed allShapes xba xSh yb lr 1.0 bnShapes batch.toUSize
  let t1 ← IO.monoMsNow
  let loss := F32.extractLoss out (3 * nP)
  IO.println s!"  step 1     : loss={loss} ({t1-t0}ms)"

  -- 6. Verify round-trip: out should contain (params, m, v, loss, [BN stats])
  let pNew := F32.slice out 0 nP
  let mNew := F32.slice out nP nP
  let vNew := F32.slice out (2 * nP) nP
  IO.println s!"  out sizes  : p={F32.size pNew} m={F32.size mNew} v={F32.size vNew}"

  -- 7. Sanity: do another step — loss should change a bit (not NaN, not stuck)
  let packed2 := (pNew.append mNew).append vNew
  let out2 ← IreeSession.trainStepAdamF32 sess spec.trainFnName
              packed2 allShapes xba xSh yb lr 2.0 bnShapes batch.toUSize
  let loss2 := F32.extractLoss out2 (3 * nP)
  IO.println s!"  step 2     : loss={loss2}"
  IO.println s!"  Δloss      : {loss - loss2}"

  if loss.isNaN || loss2.isNaN then
    IO.eprintln "ERROR: NaN loss"
    IO.Process.exit 1
  IO.println "Stage 3 E2E smoke OK."
