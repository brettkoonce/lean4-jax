import LeanJax.IreeRuntime
import LeanJax.F32Array
import LeanJax.Types
import LeanJax.Spec
import LeanJax.MlirCodegen

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

namespace R50Layout

def nParams : Nat := resnet50.totalParams

def paramShapes : Array (Array Nat) := Id.run do
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
        -- 1x1 reduce
        shapes := shapes.push #[mid, blockIc, 1, 1] |>.push #[mid] |>.push #[mid]
        -- 3x3
        shapes := shapes.push #[mid, mid, 3, 3] |>.push #[mid] |>.push #[mid]
        -- 1x1 expand
        shapes := shapes.push #[oc, mid, 1, 1] |>.push #[oc] |>.push #[oc]
        if bi == 0 && needsProj then
          shapes := shapes.push #[oc, ic, 1, 1] |>.push #[oc] |>.push #[oc]
    | _ => pure ()
  return shapes

def allShapes : Array (Array Nat) := paramShapes ++ paramShapes
def shapesBA : ByteArray := packShapes allShapes
def nTotal : Nat := 2 * nParams

def xShape (batch : Nat) : ByteArray :=
  packXShape #[batch, 3 * 224 * 224]

end R50Layout

def main (args : List String) : IO Unit := do
  let dataDir := args.head? |>.getD "data/imagenette"
  IO.eprintln s!"ResNet-50: {R50Layout.nParams} params"

  -- Generate + compile train step
  IO.FS.createDirAll ".lake/build"
  IO.eprintln "Generating train step MLIR..."
  let mlir := MlirCodegen.generateTrainStep resnet50 16 "jit_resnet50_train_step"
  IO.FS.writeFile ".lake/build/resnet50_train_step.mlir" mlir
  IO.eprintln s!"  {mlir.length} chars"

  -- Also generate forward vmfb for eval
  let fwdMlir := MlirCodegen.generate resnet50 16
  IO.FS.writeFile ".lake/build/resnet50_fwd.mlir" fwdMlir

  IO.eprintln "Compiling vmfbs..."
  let fwdCompileArgs ← ireeCompileArgs ".lake/build/resnet50_fwd.mlir" ".lake/build/resnet50_fwd.vmfb"
  let rf ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := fwdCompileArgs }
  if rf.exitCode != 0 then
    IO.eprintln s!"forward compile failed: {rf.stderr.take 1000}"
  else
    IO.eprintln "  forward compiled"

  let compileArgs ← ireeCompileArgs ".lake/build/resnet50_train_step.mlir" ".lake/build/resnet50_train_step.vmfb"
  let r ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := compileArgs }
  if r.exitCode != 0 then
    IO.eprintln s!"compile failed: {r.stderr.take 3000}"
    IO.Process.exit 1
  IO.eprintln "  compiled"

  let sess ← IreeSession.create ".lake/build/resnet50_train_step.vmfb"
  IO.eprintln "  session loaded"

  let (trainImg, trainLbl, nTrain) ← F32.loadImagenette (dataDir ++ "/train.bin")
  IO.eprintln s!"  data: {nTrain} images"

  -- Init params + velocity
  let mut paramParts : Array ByteArray := #[]
  let mut seed : USize := 42
  let shapes := R50Layout.paramShapes
  let mut si : Nat := 0
  while si < shapes.size do
    let shape := shapes[si]!
    let n := shape.foldl (· * ·) 1
    if shape.size >= 2 then
      let fanIn := if shape.size == 4 then shape[1]! * shape[2]! * shape[3]! else shape[0]!
      paramParts := paramParts.push (← F32.heInit seed n.toUSize (Float.sqrt (2.0 / fanIn.toFloat)))
      seed := seed + 1
      si := si + 1
      if si < shapes.size && shapes[si]!.size == 1 then
        let n1 := shapes[si]![0]!
        if si + 1 < shapes.size && shapes[si + 1]!.size == 1 then
          paramParts := paramParts.push (← F32.const n1.toUSize 1.0)
          si := si + 1
          paramParts := paramParts.push (← F32.const (shapes[si]![0]!).toUSize 0.0)
          si := si + 1
        else
          paramParts := paramParts.push (← F32.const n1.toUSize 0.0)
          si := si + 1
    else
      si := si + 1
  let p := F32.concat paramParts
  let v ← F32.const (F32.size p).toUSize 0.0
  IO.eprintln s!"  {F32.size p} params + {F32.size v} velocity ({(p.size + v.size) / 1024 / 1024} MB)"

  let batchN : Nat := 16
  let batch : USize := 16
  let epochs := 80
  let bpE := nTrain / batchN
  let pixelsPerImage := 3 * 224 * 224
  let allShapes := R50Layout.shapesBA
  let xSh := R50Layout.xShape batchN
  let nP := R50Layout.nParams
  let nT := R50Layout.nTotal

  IO.eprintln s!"training: {bpE} batches/epoch, batch={batchN}, momentum=0.9"
  let mut params := p
  let mut vel := v
  let mut curImg := trainImg
  let mut curLbl := trainLbl
  for epoch in [:epochs] do
    let (sImg, sLbl) ← F32.shuffle curImg curLbl nTrain.toUSize pixelsPerImage.toUSize (epoch + 42).toUSize
    curImg := sImg; curLbl := sLbl
    let lr : Float := 0.002 * (1.0 - epoch.toFloat / epochs.toFloat)
    let mut epochLoss : Float := 0.0
    let t0 ← IO.monoMsNow
    for bi in [:bpE] do
      let xba0 := F32.sliceImages curImg (bi * batchN) batchN pixelsPerImage
      let xba ← F32.randomHFlip xba0 batch 3 224 224 (epoch * 10000 + bi).toUSize
      let yb := F32.sliceLabels curLbl (bi * batchN) batchN
      let packed := params.append vel
      let ts0 ← IO.monoMsNow
      let out ← IreeSession.trainStepF32 sess "jit_resnet50_train_step.main"
                  packed allShapes xba xSh yb lr batch
      let ts1 ← IO.monoMsNow
      let loss := F32.extractLoss out nT
      epochLoss := epochLoss + loss
      params := F32.slice out 0 nP
      vel := F32.slice out nP nP
      if bi < 3 || bi % 100 == 0 then
        IO.eprintln s!"  step {bi}/{bpE}: loss={loss} ({ts1-ts0}ms)"
    let t1 ← IO.monoMsNow
    let avgLoss := epochLoss / bpE.toFloat
    IO.eprintln s!"Epoch {epoch+1}/{epochs}: loss={avgLoss} lr={lr} ({t1-t0}ms)"

    -- Val eval every 10 epochs
    if (epoch + 1) % 10 == 0 || epoch + 1 == epochs then
      let fwdVmfb := ".lake/build/resnet50_fwd.vmfb"
      if ← System.FilePath.pathExists fwdVmfb then
        let evalSess ← IreeSession.create fwdVmfb
        let (valImg, valLbl, nVal) ← F32.loadImagenette (dataDir ++ "/val.bin")
        let evalBatch := 16
        let evalSteps := nVal / evalBatch
        let paramShapesBA := packShapes R50Layout.paramShapes
        let evalXSh := R50Layout.xShape evalBatch
        let mut correct : Nat := 0
        let mut total : Nat := 0
        for bi in [:evalSteps] do
          let xba := F32.sliceImages valImg (bi * evalBatch) evalBatch pixelsPerImage
          let logits ← IreeSession.forwardF32 evalSess "resnet_50.forward"
                          params paramShapesBA xba evalXSh evalBatch.toUSize 10
          let lblSlice := F32.sliceLabels valLbl (bi * evalBatch) evalBatch
          for i in [:evalBatch] do
            let pred := F32.argmax10 logits (i * 10).toUSize
            let label := lblSlice.data[i * 4]!.toNat
            if pred.toNat == label then correct := correct + 1
            total := total + 1
        let acc := correct.toFloat / total.toFloat * 100.0
        IO.eprintln s!"  val accuracy: {correct}/{total} = {acc}%"

  IO.FS.writeBinFile ".lake/build/resnet50_params.bin" params
  IO.eprintln "Saved params."
