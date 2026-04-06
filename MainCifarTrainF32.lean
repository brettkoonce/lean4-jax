import LeanJax.IreeRuntime
import LeanJax.F32Array
import LeanJax.Types
import LeanJax.MlirCodegen

/-! CIFAR-10 CNN training — f32 ByteArray version.
    Zero f64↔f32 conversion. Data loading in C. -/

def main : IO Unit := do
  -- Generate + compile forward .vmfb
  let cifarCnn : NetSpec := {
    name := "CIFAR-10 CNN", imageH := 32, imageW := 32,
    layers := [
      .conv2d  3 32 3 .same .relu, .conv2d 32 32 3 .same .relu, .maxPool 2 2,
      .conv2d 32 64 3 .same .relu, .conv2d 64 64 3 .same .relu, .maxPool 2 2,
      .flatten, .dense 4096 512 .relu, .dense 512 512 .relu, .dense 512 10 .identity
    ]
  }
  IO.FS.createDirAll ".lake/build"
  IO.FS.writeFile ".lake/build/cifar_cnn.mlir" (MlirCodegen.generate cifarCnn 128)
  let compileArgs := #[".lake/build/cifar_cnn.mlir",
    "--iree-hal-target-backends=cuda", "--iree-cuda-target=sm_86",
    "-o", ".lake/build/cifar_cnn.vmfb"]
  let r ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := compileArgs }
  if r.exitCode != 0 then
    IO.eprintln s!"forward compile failed: {r.stderr}"
    IO.Process.exit 1
  IO.println "Forward .vmfb compiled."

  IO.println "Loading CIFAR-10 (raw bytes, instant)..."
  -- Load raw batch files as ByteArrays, extract labels
  let mut trainRaw : ByteArray := .empty
  let mut trainLbl : ByteArray := .empty
  let mut nTrain : Nat := 0
  for i in [1:6] do
    let raw ← IO.FS.readBinFile ("data/cifar-10/data_batch_" ++ toString i ++ ".bin")
    let n := raw.size / 3073
    for j in [:n] do
      let off := j * 3073
      trainLbl := trainLbl.push raw[off]!
      trainLbl := trainLbl.push 0
      trainLbl := trainLbl.push 0
      trainLbl := trainLbl.push 0
    trainRaw := trainRaw.append raw
    nTrain := nTrain + n
  IO.println s!"  train: {nTrain} images"

  -- Pre-convert ALL training images to f32 ByteArray at once (in C for speed)
  -- We can't use loadIdxImages since CIFAR is a different format.
  -- Convert raw → f32 per-batch at train time (same as before but into ByteArray)

  IO.println "Loading IREE..."
  let trainSess ← IreeSession.create ".lake/build/cifar_train_step.vmfb"
  IO.println "  ready"

  IO.println "Initializing 2.4M params (f32, instant)..."
  let params := F32.concat #[
    ← F32.heInit 20 (32*3*3*3).toUSize   (Float.sqrt (2.0 / (3.0*3*3))),
    ← F32.const 32 0.0,
    ← F32.heInit 21 (32*32*3*3).toUSize  (Float.sqrt (2.0 / (32.0*3*3))),
    ← F32.const 32 0.0,
    ← F32.heInit 22 (64*32*3*3).toUSize  (Float.sqrt (2.0 / (32.0*3*3))),
    ← F32.const 64 0.0,
    ← F32.heInit 23 (64*64*3*3).toUSize  (Float.sqrt (2.0 / (64.0*3*3))),
    ← F32.const 64 0.0,
    ← F32.heInit 24 (4096*512).toUSize   (Float.sqrt (2.0 / 4096.0)),
    ← F32.const 512 0.0,
    ← F32.heInit 25 (512*512).toUSize    (Float.sqrt (2.0 / 512.0)),
    ← F32.const 512 0.0,
    ← F32.heInit 26 (512*10).toUSize     (Float.sqrt (2.0 / 512.0)),
    ← F32.const 10 0.0
  ]
  IO.println s!"  {F32.size params} f32 params"

  let batch : USize := 128
  let batchN : Nat := 128
  let lr : Float := 0.01
  let epochs := 25
  let bpE := nTrain / batchN
  let shapes := CifarLayout.shapesBA
  let xSh := CifarLayout.xShape batchN

  let mut p := params
  for epoch in [:epochs] do
    let mut epochLoss : Float := 0.0
    let t0 ← IO.monoMsNow
    for bi in [:bpE] do
      let xba ← F32.cifarBatch trainRaw (bi * batchN).toUSize batchN.toUSize
      let yb := F32.sliceLabels trainLbl (bi*batchN) batchN
      let out ← IreeSession.trainStepF32 trainSess "jit_cifar_train_step.main"
                  p shapes xba xSh yb lr batch
      epochLoss := epochLoss + F32.extractLoss out CifarLayout.nParams
      p := F32.dropLoss out CifarLayout.nParams
    let t1 ← IO.monoMsNow
    IO.println s!"Epoch {epoch+1}: loss={epochLoss / bpE.toFloat} ({t1-t0}ms)"
