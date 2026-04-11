import LeanMlir.IreeRuntime
import LeanMlir.F32Array
import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.MlirCodegen
import LeanMlir.SpecHelpers

/-! MobileNet v2 on Imagenette — full IREE training pipeline.
    Inverted residuals with depthwise separable convolutions.
    ~2.2M params, 224×224 input, 10 classes. -/

def mobilenetV2 : NetSpec where
  name := "MobileNet-v2"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 32 3 2 .same,                    -- 224→112
    .invertedResidual  32  16 1 1 1,            -- 112, t=1
    .invertedResidual  16  24 6 2 2,            -- 112→56, t=6
    .invertedResidual  24  32 6 2 3,            -- 56→28, t=6
    .invertedResidual  32  64 6 2 4,            -- 28→14, t=6
    .invertedResidual  64  96 6 1 3,            -- 14, t=6
    .invertedResidual  96 160 6 2 3,            -- 14→7, t=6
    .invertedResidual 160 320 6 1 1,            -- 7, t=6
    .convBn 320 1280 1 1 .same,                 -- 1x1 conv to 1280
    .globalAvgPool,
    .dense 1280 10 .identity
  ]

namespace MNV2Layout
def nParams      : Nat       := mobilenetV2.totalParams
def shapesBA     : ByteArray := mobilenetV2.shapesBA
def nTotal       : Nat       := 3 * nParams
def bnShapesBA   : ByteArray := mobilenetV2.bnShapesBA
def nBnStats     : Nat       := mobilenetV2.nBnStats
def evalShapesBA : ByteArray := mobilenetV2.evalShapesBA
def xShape (batch : Nat) : ByteArray := mobilenetV2.xShape batch
end MNV2Layout

def main (args : List String) : IO Unit := do
  let dataDir := args.head? |>.getD "data/imagenette"
  IO.eprintln s!"MobileNet-v2: {MNV2Layout.nParams} params"

  let batchN : Nat := 32
  let batch : USize := 32

  -- Generate + compile MLIR
  IO.FS.createDirAll ".lake/build"
  IO.eprintln "Generating train step MLIR..."
  let mlir := MlirCodegen.generateTrainStep mobilenetV2 batchN "jit_mnv2_train_step"
  IO.FS.writeFile ".lake/build/mnv2_train_step.mlir" mlir
  IO.eprintln s!"  {mlir.length} chars"

  let fwdMlir := MlirCodegen.generate mobilenetV2 batchN
  IO.FS.writeFile ".lake/build/mnv2_fwd.mlir" fwdMlir

  let evalFwdMlir := MlirCodegen.generateEval mobilenetV2 batchN
  IO.FS.writeFile ".lake/build/mnv2_fwd_eval.mlir" evalFwdMlir

  IO.eprintln "Compiling vmfbs..."
  let fwdCompileArgs ← ireeCompileArgs ".lake/build/mnv2_fwd.mlir" ".lake/build/mnv2_fwd.vmfb"
  let rf ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := fwdCompileArgs }
  if rf.exitCode != 0 then
    IO.eprintln s!"forward compile failed: {rf.stderr.take 1000}"
  else
    IO.eprintln "  forward compiled"

  let evalFwdCompileArgs ← ireeCompileArgs ".lake/build/mnv2_fwd_eval.mlir" ".lake/build/mnv2_fwd_eval.vmfb"
  let re ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := evalFwdCompileArgs }
  if re.exitCode != 0 then
    IO.eprintln s!"eval forward (fixed BN) compile failed: {re.stderr.take 1000}"
  else
    IO.eprintln "  eval forward (fixed BN) compiled"

  let compileArgs ← ireeCompileArgs ".lake/build/mnv2_train_step.mlir" ".lake/build/mnv2_train_step.vmfb"
  let r ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := compileArgs }
  if r.exitCode != 0 then
    IO.eprintln s!"compile failed: {r.stderr.take 3000}"
    IO.Process.exit 1
  IO.eprintln "  compiled"

  let sess ← IreeSession.create ".lake/build/mnv2_train_step.vmfb"
  IO.eprintln "  session loaded"

  -- Load data
  let (trainImg, trainLbl, nTrain) ← F32.loadImagenetteSized (dataDir ++ "/train.bin") 256
  IO.eprintln s!"  train: {nTrain} images (256×256)"

  let params ← mobilenetV2.heInitParams
  let adamM ← F32.const (F32.size params).toUSize 0.0
  let adamV ← F32.const (F32.size params).toUSize 0.0
  IO.eprintln s!"  {F32.size params} params + m + v ({(params.size + adamM.size + adamV.size) / 1024 / 1024} MB)"

  -- Training loop
  let epochs := 80
  let bpE := nTrain / batchN
  let trainPixels := 3 * 256 * 256
  let allShapes := MNV2Layout.shapesBA
  let xSh := MNV2Layout.xShape batchN
  let nP := MNV2Layout.nParams
  let nT := MNV2Layout.nTotal
  let baseLR : Float := 0.001

  let bnShapes := MNV2Layout.bnShapesBA
  let nBnStats := MNV2Layout.nBnStats

  IO.eprintln s!"training: {bpE} batches/epoch, batch={batchN}, Adam, lr={baseLR}, cosine, label_smooth=0.1, wd=1e-4"
  IO.eprintln s!"  BN layers: {mobilenetV2.bnLayers.size}, BN stat floats: {nBnStats}"
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
      let packed := (p.append m).append v
      let ts0 ← IO.monoMsNow
      let out ← IreeSession.trainStepAdamF32 sess "jit_mnv2_train_step.main"
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

    -- Val eval every 10 epochs (using running BN stats)
    if (epoch + 1) % 10 == 0 || epoch + 1 == epochs then
      let evalVmfb := ".lake/build/mnv2_fwd_eval.vmfb"
      if ← System.FilePath.pathExists evalVmfb then
        let evalSess ← IreeSession.create evalVmfb
        let (valImg, valLbl, nVal) ← F32.loadImagenette (dataDir ++ "/val.bin")
        let evalBatch := batchN
        let evalSteps := nVal / evalBatch
        let evalXSh := MNV2Layout.xShape evalBatch
        let evalParams := p.append runningBnStats
        let evalShapesBA := MNV2Layout.evalShapesBA
        let mut correct : Nat := 0
        let mut total : Nat := 0
        for bi in [:evalSteps] do
          let xba := F32.sliceImages valImg (bi * evalBatch) evalBatch (3 * 224 * 224)
          let logits ← IreeSession.forwardF32 evalSess "mobilenet_v2_eval.forward_eval"
                          evalParams evalShapesBA xba evalXSh evalBatch.toUSize 10
          let lblSlice := F32.sliceLabels valLbl (bi * evalBatch) evalBatch
          for i in [:evalBatch] do
            let pred := F32.argmax10 logits (i * 10).toUSize
            let label := lblSlice.data[i * 4]!.toNat
            if pred.toNat == label then correct := correct + 1
            total := total + 1
        let acc := correct.toFloat / total.toFloat * 100.0
        IO.eprintln s!"  val accuracy (running BN): {correct}/{total} = {acc}%"
  IO.FS.writeBinFile ".lake/build/mnv2_params.bin" p
  IO.FS.writeBinFile ".lake/build/mnv2_bn_stats.bin" runningBnStats
  IO.eprintln "Saved params + BN stats."
