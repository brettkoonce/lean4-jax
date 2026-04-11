import LeanMlir.IreeRuntime
import LeanMlir.F32Array
import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.MlirCodegen
import LeanMlir.SpecHelpers

/-! ResNet-50 on Imagenette — full training pipeline with bottleneck blocks.
    ~23.5M params, 224×224 input, 10 classes. -/

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

-- Param/shape/init helpers come from `LeanMlir.SpecHelpers`. The local
-- namespace is now thin re-exports.
namespace R50Layout
def nParams      : Nat       := resnet50.totalParams
def shapesBA     : ByteArray := resnet50.shapesBA
def nTotal       : Nat       := 3 * nParams
def bnShapesBA   : ByteArray := resnet50.bnShapesBA
def nBnStats     : Nat       := resnet50.nBnStats
def evalShapesBA : ByteArray := resnet50.evalShapesBA
def xShape (batch : Nat) : ByteArray := resnet50.xShape batch
end R50Layout

def main (args : List String) : IO Unit := do
  let dataDir := args.head? |>.getD "data/imagenette"
  IO.eprintln s!"ResNet-50: {R50Layout.nParams} params"

  -- Generate + compile train step
  IO.FS.createDirAll ".lake/build"
  IO.eprintln "Generating train step MLIR..."
  let batchN : Nat := 32
  let mlir := MlirCodegen.generateTrainStep resnet50 batchN "jit_resnet50_train_step"
  IO.FS.writeFile ".lake/build/resnet50_train_step.mlir" mlir
  IO.eprintln s!"  {mlir.length} chars"

  -- Also generate forward vmfb for eval
  let fwdMlir := MlirCodegen.generate resnet50 batchN
  IO.FS.writeFile ".lake/build/resnet50_fwd.mlir" fwdMlir

  -- Eval forward with fixed BN running stats
  let evalFwdMlir := MlirCodegen.generateEval resnet50 batchN
  IO.FS.writeFile ".lake/build/resnet50_fwd_eval.mlir" evalFwdMlir

  IO.eprintln "Compiling vmfbs..."
  let fwdCompileArgs ← ireeCompileArgs ".lake/build/resnet50_fwd.mlir" ".lake/build/resnet50_fwd.vmfb"
  let rf ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := fwdCompileArgs }
  if rf.exitCode != 0 then
    IO.eprintln s!"forward compile failed: {rf.stderr.take 1000}"
  else
    IO.eprintln "  forward compiled"

  let evalFwdCompileArgs ← ireeCompileArgs ".lake/build/resnet50_fwd_eval.mlir" ".lake/build/resnet50_fwd_eval.vmfb"
  let re ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := evalFwdCompileArgs }
  if re.exitCode != 0 then
    IO.eprintln s!"eval forward (fixed BN) compile failed: {re.stderr.take 1000}"
  else
    IO.eprintln "  eval forward (fixed BN) compiled"

  let compileArgs ← ireeCompileArgs ".lake/build/resnet50_train_step.mlir" ".lake/build/resnet50_train_step.vmfb"
  let r ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := compileArgs }
  if r.exitCode != 0 then
    IO.eprintln s!"compile failed: {r.stderr.take 3000}"
    IO.Process.exit 1
  IO.eprintln "  compiled"

  let sess ← IreeSession.create ".lake/build/resnet50_train_step.vmfb"
  IO.eprintln "  session loaded"

  let (trainImg, trainLbl, nTrain) ← F32.loadImagenetteSized (dataDir ++ "/train.bin") 256
  IO.eprintln s!"  data: {nTrain} images"

  let p ← resnet50.heInitParams
  -- Adam state: m (1st moment) and v (2nd moment), both zero-initialized
  let adamM ← F32.const (F32.size p).toUSize 0.0
  let adamV ← F32.const (F32.size p).toUSize 0.0
  IO.eprintln s!"  {F32.size p} params + m + v ({(p.size + adamM.size + adamV.size) / 1024 / 1024} MB)"

  let batch : USize := 32
  let epochs := 80
  let bpE := nTrain / batchN
  let trainPixels := 3 * 256 * 256
  let allShapes := R50Layout.shapesBA
  let xSh := R50Layout.xShape batchN
  let nP := R50Layout.nParams
  let nT := R50Layout.nTotal
  let baseLR : Float := 0.001

  let bnShapes := R50Layout.bnShapesBA
  let nBnStats := R50Layout.nBnStats

  IO.eprintln s!"training: {bpE} batches/epoch, batch={batchN}, Adam, lr={baseLR}, cosine, label_smooth=0.1, wd=1e-4"
  IO.eprintln s!"  BN layers: {resnet50.bnLayers.size}, BN stat floats: {nBnStats}"
  let mut params := p
  let mut m := adamM
  let mut v := adamV
  -- Running BN stats (EMA, momentum=0.1)
  let mut runningBnStats ← F32.const nBnStats.toUSize 0.0
  let mut curImg := trainImg
  let mut curLbl := trainLbl
  let mut globalStep : Nat := 0
  for epoch in [:epochs] do
    let (sImg, sLbl) ← F32.shuffle curImg curLbl nTrain.toUSize trainPixels.toUSize (epoch + 42).toUSize
    curImg := sImg; curLbl := sLbl
    -- Cosine LR with 3-epoch warmup
    let lr : Float := if epoch < 3 then
      baseLR * (epoch.toFloat + 1.0) / 3.0
    else
      baseLR * 0.5 * (1.0 + Float.cos (3.14159265358979 * (epoch.toFloat - 3.0) / (epochs.toFloat - 3.0)))
    let mut epochLoss : Float := 0.0
    let t0 ← IO.monoMsNow
    for bi in [:bpE] do
      globalStep := globalStep + 1
      let xba256 := F32.sliceImages curImg (bi * batchN) batchN trainPixels
      let xbaCropped ← F32.randomCrop xba256 batch 3 256 256 224 224 (epoch * 10000 + bi).toUSize
      let xba ← F32.randomHFlip xbaCropped batch 3 224 224 (epoch * 10000 + bi + 7777).toUSize
      let yb := F32.sliceLabels curLbl (bi * batchN) batchN
      let packed := (params.append m).append v
      let ts0 ← IO.monoMsNow
      let out ← IreeSession.trainStepAdamF32 sess "jit_resnet50_train_step.main"
                  packed allShapes xba xSh yb lr globalStep.toFloat bnShapes batch
      let ts1 ← IO.monoMsNow
      let loss := F32.extractLoss out nT
      epochLoss := epochLoss + loss
      params := F32.slice out 0 nP
      m := F32.slice out nP nP
      v := F32.slice out (2 * nP) nP
      -- Extract batch BN stats and update running stats via EMA
      let batchBnStats := out.extract ((nT + 1) * 4) ((nT + 1 + nBnStats) * 4)
      -- Use momentum 1.0 for first step (initialize), 0.1 thereafter
      let bnMom : Float := if globalStep == 1 then 1.0 else 0.1
      runningBnStats ← F32.ema runningBnStats batchBnStats bnMom
      if bi < 3 || bi % 100 == 0 then
        IO.eprintln s!"  step {bi}/{bpE}: loss={loss} ({ts1-ts0}ms)"
    let t1 ← IO.monoMsNow
    let avgLoss := epochLoss / bpE.toFloat
    IO.eprintln s!"Epoch {epoch+1}/{epochs}: loss={avgLoss} lr={lr} ({t1-t0}ms)"

    -- Val eval every 10 epochs (using running BN stats)
    if (epoch + 1) % 10 == 0 || epoch + 1 == epochs then
      let evalVmfb := ".lake/build/resnet50_fwd_eval.vmfb"
      if ← System.FilePath.pathExists evalVmfb then
        let evalSess ← IreeSession.create evalVmfb
        let (valImg, valLbl, nVal) ← F32.loadImagenette (dataDir ++ "/val.bin")
        let evalBatch := batchN
        let evalSteps := nVal / evalBatch
        let evalXSh := R50Layout.xShape evalBatch
        -- Pack params + running BN stats for eval forward
        let evalParams := params.append runningBnStats
        let evalShapesBA := R50Layout.evalShapesBA
        let mut correct : Nat := 0
        let mut total : Nat := 0
        for bi in [:evalSteps] do
          let xba := F32.sliceImages valImg (bi * evalBatch) evalBatch (3 * 224 * 224)
          let logits ← IreeSession.forwardF32 evalSess "resnet_50_eval.forward_eval"
                          evalParams evalShapesBA xba evalXSh evalBatch.toUSize 10
          let lblSlice := F32.sliceLabels valLbl (bi * evalBatch) evalBatch
          for i in [:evalBatch] do
            let pred := F32.argmax10 logits (i * 10).toUSize
            let label := lblSlice.data[i * 4]!.toNat
            if pred.toNat == label then correct := correct + 1
            total := total + 1
        let acc := correct.toFloat / total.toFloat * 100.0
        IO.eprintln s!"  val accuracy (running BN): {correct}/{total} = {acc}%"
  IO.FS.writeBinFile ".lake/build/resnet50_params.bin" params
  IO.FS.writeBinFile ".lake/build/resnet50_bn_stats.bin" runningBnStats
  IO.eprintln "Saved params + BN stats."
