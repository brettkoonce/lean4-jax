import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.F32Array
import LeanMlir.IreeRuntime
import LeanMlir.MlirCodegen
import LeanMlir.SpecHelpers

/-! Unified training loop for any spec that uses the Adam codegen path.

Each Main*Train.lean used to inline ~250 lines of identical
code: generate MLIR → compile vmfbs → init params → for-each-epoch
{ shuffle, for-each-batch { trainStepAdamF32, EMA, log }, val eval } →
save. The body was the same for ResNet/MobileNet/EfficientNet/ViT/VGG
modulo a handful of name strings.

`NetSpec.train` is the extracted function. A trainer is now:

    def main (args : List String) : IO Unit :=
      resnet34.train resnet34Config (args.head?.getD "data/imagenette")

The 250 lines collapse to one call. Adding a new architecture means
defining the spec + a `TrainConfig` value — no copy-paste plumbing. -/

namespace NetSpec

/-- File-path prefix for the generated MLIR / vmfb / saved-params files
    associated with this spec. Uses the sanitized spec name so adding
    a new spec automatically gets a unique non-colliding prefix without
    the trainer author having to pick one. -/
def buildPrefix (spec : NetSpec) : String :=
  ".lake/build/" ++ spec.sanitizedName

/-- The fully qualified `<module>.<func>` name the train step's main
    function lives at — what `trainStepAdamF32` wants as its `fnName`.
    Mirrors how `MlirCodegen.generateTrainStep` builds the module
    name from the third argument it's given (`jit_<sanitized>_train_step`). -/
def trainFnName (spec : NetSpec) : String :=
  "jit_" ++ spec.sanitizedName ++ "_train_step.main"

private def runIree (mlirPath outPath : String) : IO Bool := do
  let args ← ireeCompileArgs mlirPath outPath
  let r ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := args }
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile failed for {mlirPath}: {r.stderr.take 3000}"
    return false
  return true

/-- Generate forward / eval-forward / train-step MLIR for `spec`,
    write them to `.lake/build/<sanitized>_*.mlir`, and compile each
    to a `.vmfb`. Returns the path to the train-step vmfb (used to
    create the IreeSession). -/
def compileVmfbs (spec : NetSpec) (cfg : TrainConfig) : IO String := do
  IO.FS.createDirAll ".lake/build"
  let pfx := spec.buildPrefix

  IO.eprintln "Generating train step MLIR..."
  let mlir := MlirCodegen.generateTrainStep spec cfg.batchSize ("jit_" ++ spec.sanitizedName ++ "_train_step")
  IO.FS.writeFile s!"{pfx}_train_step.mlir" mlir
  IO.eprintln s!"  {mlir.length} chars"

  let fwdMlir := MlirCodegen.generate spec cfg.batchSize
  IO.FS.writeFile s!"{pfx}_fwd.mlir" fwdMlir

  let evalFwdMlir := MlirCodegen.generateEval spec cfg.batchSize
  IO.FS.writeFile s!"{pfx}_fwd_eval.mlir" evalFwdMlir

  IO.eprintln "Compiling vmfbs..."
  if ← runIree s!"{pfx}_fwd.mlir" s!"{pfx}_fwd.vmfb" then
    IO.eprintln "  forward compiled"
  if ← runIree s!"{pfx}_fwd_eval.mlir" s!"{pfx}_fwd_eval.vmfb" then
    IO.eprintln "  eval forward compiled"
  if ← runIree s!"{pfx}_train_step.mlir" s!"{pfx}_train_step.vmfb" then
    IO.eprintln "  train step compiled"
  else
    IO.Process.exit 1
  return s!"{pfx}_train_step.vmfb"

/-- Adam + cosine-LR + running-BN-stats training loop. Assumes:
    - Imagenette-style data layout (3-channel, 256×256 train binary
      with random crop to 224×224, 224×224 val binary)
    - 10-class classification
    - The spec has been compiled via `compileVmfbs`

    Loads data, inits params, trains for `cfg.epochs` epochs, runs
    a val eval every 10 epochs (and at the final epoch), and saves
    `params.bin` + `bn_stats.bin`. -/
def runImagenetteTraining (spec : NetSpec) (cfg : TrainConfig) (dataDir : String)
    (sess : IreeSession) : IO Unit := do
  let pfx := spec.buildPrefix
  let batchN : Nat := cfg.batchSize
  let batch  : USize := cfg.batchSize.toUSize

  let (trainImg, trainLbl, nTrain) ← F32.loadImagenetteSized (dataDir ++ "/train.bin") 256
  IO.eprintln s!"  train: {nTrain} images (256×256)"

  let params ← spec.heInitParams
  let adamM ← F32.const (F32.size params).toUSize 0.0
  let adamV ← F32.const (F32.size params).toUSize 0.0
  IO.eprintln s!"  {F32.size params} params + m + v ({(params.size + adamM.size + adamV.size) / 1024 / 1024} MB)"

  let bpE := nTrain / batchN
  let trainPixels := 3 * 256 * 256
  let allShapes := spec.shapesBA
  let xSh := spec.xShape batchN
  let nP := spec.totalParams
  let nT := 3 * nP            -- params + m + v
  let baseLR : Float := cfg.learningRate
  let warmup : Nat := cfg.warmupEpochs
  let epochs : Nat := cfg.epochs
  let nClasses : USize := spec.numClasses.toUSize

  let bnShapes := spec.bnShapesBA
  let nBnStats := spec.nBnStats

  IO.eprintln s!"training: {bpE} batches/epoch, batch={batchN}, Adam, lr={baseLR}, cosine warmup={warmup}, label_smooth=0.1, wd={cfg.weightDecay}"
  IO.eprintln s!"  BN layers: {spec.bnLayers.size}, BN stat floats: {nBnStats}"

  let mut p := params
  let mut m := adamM
  let mut v := adamV
  let mut runningBnStats ← F32.const nBnStats.toUSize 0.0
  let mut curImg := trainImg
  let mut curLbl := trainLbl
  let mut globalStep : Nat := 0

  for epoch in [:epochs] do
    let (sImg, sLbl) ← F32.shuffle curImg curLbl nTrain.toUSize trainPixels.toUSize (epoch + 42).toUSize
    curImg := sImg; curLbl := sLbl

    let lr : Float := if epoch < warmup then
      baseLR * (epoch.toFloat + 1.0) / warmup.toFloat
    else
      baseLR * 0.5 * (1.0 + Float.cos (3.14159265358979 * (epoch.toFloat - warmup.toFloat) / (epochs.toFloat - warmup.toFloat)))

    let mut epochLoss : Float := 0.0
    let t0 ← IO.monoMsNow
    for bi in [:bpE] do
      globalStep := globalStep + 1
      let xba256 := F32.sliceImages curImg (bi * batchN) batchN trainPixels
      let xbaCropped ← F32.randomCrop xba256 batch 3 256 256 224 224 (epoch * 10000 + bi).toUSize
      let xba ← F32.randomHFlip xbaCropped batch 3 224 224 (epoch * 10000 + bi + 7777).toUSize
      let yb := F32.sliceLabels curLbl (bi * batchN) batchN
      let packed := (p.append m).append v
      let ts0 ← IO.monoMsNow
      let out ← IreeSession.trainStepAdamF32 sess spec.trainFnName
                  packed allShapes xba xSh yb lr globalStep.toFloat bnShapes batch
      let ts1 ← IO.monoMsNow
      let loss := F32.extractLoss out nT
      epochLoss := epochLoss + loss
      p := F32.slice out 0 nP
      m := F32.slice out nP nP
      v := F32.slice out (2 * nP) nP
      let batchBnStats := out.extract ((nT + 1) * 4) ((nT + 1 + nBnStats) * 4)
      let bnMom : Float := if globalStep == 1 then 1.0 else 0.1
      runningBnStats ← F32.ema runningBnStats batchBnStats bnMom
      if bi < 3 || bi % 100 == 0 then
        IO.eprintln s!"  step {bi}/{bpE}: loss={loss} ({ts1-ts0}ms)"
    let t1 ← IO.monoMsNow
    let avgLoss := epochLoss / bpE.toFloat
    IO.eprintln s!"Epoch {epoch+1}/{epochs}: loss={avgLoss} lr={lr} ({t1-t0}ms)"

    if (epoch + 1) % 10 == 0 || epoch + 1 == epochs then
      let evalVmfb := s!"{pfx}_fwd_eval.vmfb"
      if ← System.FilePath.pathExists evalVmfb then
        let evalSess ← IreeSession.create evalVmfb
        let (valImg, valLbl, nVal) ← F32.loadImagenette (dataDir ++ "/val.bin")
        let evalBatch := batchN
        let evalSteps := nVal / evalBatch
        let evalXSh := spec.xShape evalBatch
        let evalParams := p.append runningBnStats
        let evalShapesBA := spec.evalShapesBA
        let mut correct : Nat := 0
        let mut total : Nat := 0
        for bi in [:evalSteps] do
          let xba := F32.sliceImages valImg (bi * evalBatch) evalBatch (3 * 224 * 224)
          let logits ← IreeSession.forwardF32 evalSess spec.evalFnName
                          evalParams evalShapesBA xba evalXSh evalBatch.toUSize nClasses
          let lblSlice := F32.sliceLabels valLbl (bi * evalBatch) evalBatch
          for i in [:evalBatch] do
            let pred := F32.argmax10 logits (i * spec.numClasses).toUSize
            let label := lblSlice.data[i * 4]!.toNat
            if pred.toNat == label then correct := correct + 1
            total := total + 1
        let acc := correct.toFloat / total.toFloat * 100.0
        IO.eprintln s!"  val accuracy (running BN): {correct}/{total} = {acc}%"

  IO.FS.writeBinFile s!"{pfx}_params.bin" p
  IO.FS.writeBinFile s!"{pfx}_bn_stats.bin" runningBnStats
  IO.eprintln "Saved params + BN stats."

/-- End-to-end: compile all three vmfbs, load the train-step session,
    and run the Imagenette training loop. The high-level entry point
    that every Main*Train.lean now calls. -/
def train (spec : NetSpec) (cfg : TrainConfig) (dataDir : String) : IO Unit := do
  IO.eprintln s!"{spec.name}: {spec.totalParams} params"
  let trainVmfb ← spec.compileVmfbs cfg
  let sess ← IreeSession.create trainVmfb
  IO.eprintln "  session loaded"
  spec.runImagenetteTraining cfg dataDir sess

end NetSpec
