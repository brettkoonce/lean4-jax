import LeanMlir.IreeRuntime
import LeanMlir.F32Array
import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.MlirCodegen
import LeanMlir.SpecHelpers

/-! EfficientNet V2-S on Imagenette — Fused-MBConv (early stages) + MBConv + Swish + SE.
    224×224, 10 classes. -/

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
    .convBn 256 1280 1 1 .same,
    .globalAvgPool,
    .dense 1280 10 .identity
  ]

namespace EffNetV2Layout
def nParams      : Nat       := efficientNetV2S.totalParams
def shapesBA     : ByteArray := efficientNetV2S.shapesBA
def nTotal       : Nat       := 3 * nParams
def bnShapesBA   : ByteArray := efficientNetV2S.bnShapesBA
def nBnStats     : Nat       := efficientNetV2S.nBnStats
def evalShapesBA : ByteArray := efficientNetV2S.evalShapesBA
def xShape (batch : Nat) : ByteArray := efficientNetV2S.xShape batch
end EffNetV2Layout

def efficientNetV2SConfig : TrainConfig where
  learningRate := 0.001
  batchSize    := 32
  epochs       := 80
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := true
  warmupEpochs := 3
  augment      := true

def main (args : List String) : IO Unit := do
  let dataDir := args.head? |>.getD "data/imagenette"
  IO.eprintln s!"EfficientNet V2-S: {EffNetV2Layout.nParams} params"

  let cfg := efficientNetV2SConfig
  let batchN : Nat := cfg.batchSize
  let batch : USize := cfg.batchSize.toUSize

  IO.FS.createDirAll ".lake/build"
  IO.eprintln "Generating train step MLIR..."
  let mlir := MlirCodegen.generateTrainStep efficientNetV2S batchN "jit_effnet_v2_train_step"
  IO.FS.writeFile ".lake/build/effnet_v2_train_step.mlir" mlir
  IO.eprintln s!"  {mlir.length} chars"

  let fwdMlir := MlirCodegen.generate efficientNetV2S batchN
  IO.FS.writeFile ".lake/build/effnet_v2_fwd.mlir" fwdMlir

  let evalFwdMlir := MlirCodegen.generateEval efficientNetV2S batchN
  IO.FS.writeFile ".lake/build/effnet_v2_fwd_eval.mlir" evalFwdMlir

  IO.eprintln "Compiling vmfbs..."
  let fwdCompileArgs ← ireeCompileArgs ".lake/build/effnet_v2_fwd.mlir" ".lake/build/effnet_v2_fwd.vmfb"
  let rf ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := fwdCompileArgs }
  if rf.exitCode != 0 then
    IO.eprintln s!"forward compile failed: {rf.stderr.take 2000}"
  else
    IO.eprintln "  forward compiled"

  let evalFwdCompileArgs ← ireeCompileArgs ".lake/build/effnet_v2_fwd_eval.mlir" ".lake/build/effnet_v2_fwd_eval.vmfb"
  let re ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := evalFwdCompileArgs }
  if re.exitCode != 0 then
    IO.eprintln s!"eval forward compile failed: {re.stderr.take 2000}"
  else
    IO.eprintln "  eval forward compiled"

  let compileArgs ← ireeCompileArgs ".lake/build/effnet_v2_train_step.mlir" ".lake/build/effnet_v2_train_step.vmfb"
  let r ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := compileArgs }
  if r.exitCode != 0 then
    IO.eprintln s!"train compile failed: {r.stderr.take 3000}"
    IO.Process.exit 1
  IO.eprintln "  train step compiled"

  let sess ← IreeSession.create ".lake/build/effnet_v2_train_step.vmfb"
  IO.eprintln "  session loaded"

  let (trainImg, trainLbl, nTrain) ← F32.loadImagenetteSized (dataDir ++ "/train.bin") 256
  IO.eprintln s!"  train: {nTrain} images (256×256)"

  let params ← efficientNetV2S.heInitParams
  let adamM ← F32.const (F32.size params).toUSize 0.0
  let adamV ← F32.const (F32.size params).toUSize 0.0
  IO.eprintln s!"  {F32.size params} params + m + v ({(params.size + adamM.size + adamV.size) / 1024 / 1024} MB)"

  let epochs := cfg.epochs
  let bpE := nTrain / batchN
  let trainPixels := 3 * 256 * 256
  let allShapes := EffNetV2Layout.shapesBA
  let xSh := EffNetV2Layout.xShape batchN
  let nP := EffNetV2Layout.nParams
  let nT := EffNetV2Layout.nTotal
  let baseLR : Float := cfg.learningRate

  let bnShapes := EffNetV2Layout.bnShapesBA
  let nBnStats := EffNetV2Layout.nBnStats

  IO.eprintln s!"training: {bpE} batches/epoch, batch={batchN}, Adam, lr={baseLR}, cosine, wd=1e-4"
  IO.eprintln s!"  BN layers: {efficientNetV2S.bnLayers.size}, BN stat floats: {nBnStats}"
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
    let warmup : Nat := cfg.warmupEpochs
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
      let out ← IreeSession.trainStepAdamF32 sess "jit_effnet_v2_train_step.main"
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
      let evalVmfb := ".lake/build/effnet_v2_fwd_eval.vmfb"
      if ← System.FilePath.pathExists evalVmfb then
        let evalSess ← IreeSession.create evalVmfb
        let (valImg, valLbl, nVal) ← F32.loadImagenette (dataDir ++ "/val.bin")
        let evalBatch := batchN
        let evalSteps := nVal / evalBatch
        let evalXSh := EffNetV2Layout.xShape evalBatch
        let evalParams := p.append runningBnStats
        let evalShapesBA := EffNetV2Layout.evalShapesBA
        let mut correct : Nat := 0
        let mut total : Nat := 0
        for bi in [:evalSteps] do
          let xba := F32.sliceImages valImg (bi * evalBatch) evalBatch (3 * 224 * 224)
          let logits ← IreeSession.forwardF32 evalSess "efficientnet_v2_s_eval.forward_eval"
                          evalParams evalShapesBA xba evalXSh evalBatch.toUSize 10
          let lblSlice := F32.sliceLabels valLbl (bi * evalBatch) evalBatch
          for i in [:evalBatch] do
            let pred := F32.argmax10 logits (i * 10).toUSize
            let label := lblSlice.data[i * 4]!.toNat
            if pred.toNat == label then correct := correct + 1
            total := total + 1
        let acc := correct.toFloat / total.toFloat * 100.0
        IO.eprintln s!"  val accuracy (running BN): {correct}/{total} = {acc}%"
  IO.FS.writeBinFile ".lake/build/effnet_v2_params.bin" p
  IO.FS.writeBinFile ".lake/build/effnet_v2_bn_stats.bin" runningBnStats
  IO.eprintln "Saved params + BN stats."
