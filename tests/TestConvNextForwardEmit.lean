import LeanMlir.IreeRuntime
import LeanMlir.F32Array
import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.MlirCodegen
import LeanMlir.SpecHelpers

/-! End-to-end smoke for ConvNeXt Stage 2 codegen:
    emit forward MLIR → iree-compile → forwardF32 with random params,
    confirm logits roundtrip through the runtime. -/

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

  IO.println "── Stage 2 E2E smoke: tiny ConvNeXt ──"
  IO.println s!"  spec      : {spec.name}"
  IO.println s!"  layers    : {spec.layers.length}"
  IO.println s!"  inputDim  : {3 * spec.imageH * spec.imageW} (3×{spec.imageH}×{spec.imageW})"

  -- 1. Generate forward MLIR
  let mlir := MlirCodegen.generate spec batch
  IO.FS.createDirAll ".lake/build"
  let mlirPath := s!".lake/build/{spec.sanitizedName}_fwd.mlir"
  let vmfbPath := s!".lake/build/{spec.sanitizedName}_fwd.vmfb"
  IO.FS.writeFile mlirPath mlir
  IO.println s!"  emitted   : {mlir.length} chars → {mlirPath}"

  -- 2. iree-compile (CPU, host)
  let compiler ← findIreeCompile
  let backend ← (IO.getEnv "IREE_BACKEND").map (·.getD "llvm-cpu")
  let mut compileArgs : Array String :=
    #[mlirPath, s!"--iree-hal-target-backends={backend}"]
  if backend == "llvm-cpu" then
    compileArgs := compileArgs.push "--iree-llvmcpu-target-cpu=host"
  compileArgs := compileArgs ++ #["-o", vmfbPath]
  let r ← IO.Process.output { cmd := compiler, args := compileArgs }
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile FAILED:\n{r.stderr.take 2000}"
    IO.Process.exit 1
  IO.println s!"  compiled  : {backend} → {vmfbPath}"

  -- 3. Init params (He for W's, zero biases, γ=1/β=0 for LN, γ=1e-6 for LayerScale)
  let params ← spec.heInitParams
  let paramShapesBA := packShapes spec.paramShapes
  IO.println s!"  params    : {F32.size params} f32s across {spec.paramShapes.size} tensors"

  -- 4. Random input batch
  let pixels := 3 * spec.imageH * spec.imageW
  let xba ← F32.heInit 7 (batch * pixels).toUSize 1.0
  let xSh := packXShape #[batch, pixels]

  -- 5. Forward pass
  let sess ← IreeSession.create vmfbPath
  IO.println "  session   : loaded"
  let nClasses : Nat := 10
  let t0 ← IO.monoMsNow
  let logits ← IreeSession.forwardF32 sess s!"{spec.sanitizedName}.forward"
                  params paramShapesBA xba xSh batch.toUSize nClasses.toUSize
  let t1 ← IO.monoMsNow
  IO.println s!"  forward   : {t1 - t0}ms — logits {F32.size logits} f32s ({batch}×{nClasses})"

  -- 6. Sanity: print argmaxes (untrained)
  for i in [:batch] do
    let p := F32.argmax10 logits (i * nClasses).toUSize
    IO.println s!"    sample {i} argmax = {p.toNat}"

  IO.println "Stage 2 E2E smoke OK."
