import LeanMlir

/-- ConvNeXt-T spec — mirrors `convNextTinyGeluSpec` in MainAblation.lean. -/
def specGelu : NetSpec where
  name := "ConvNeXt-T-GELU"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 96 4 4 .same,
    .convNextStage 96 3 .ln .gelu,
    .convNextDownsample 96 192,
    .convNextStage 192 3 .ln .gelu,
    .convNextDownsample 192 384,
    .convNextStage 384 9 .ln .gelu,
    .convNextDownsample 384 768,
    .convNextStage 768 3 .ln .gelu,
    .globalAvgPool,
    .dense 768 10 .identity
  ]

def imagenetteClasses : Array String := #[
  "tench", "English springer", "cassette player", "chain saw", "church",
  "French horn", "garbage truck", "gas pump", "golf ball", "parachute"
]

def main : IO Unit := do
  let pfx := s!".lake/build/convnext_t_gelu_convnext_tiny_gelu"
  let evalVmfb := s!"{pfx}_fwd_eval.vmfb"
  let paramsPath := s!"{pfx}_params.bin"
  let bnPath := s!"{pfx}_bn_stats.bin"

  IO.println s!"loading params from {paramsPath}..."
  let p ← IO.FS.readBinFile paramsPath
  let runningBnStats ← IO.FS.readBinFile bnPath
  let evalParams := p.append runningBnStats

  IO.println s!"loading eval session from {evalVmfb}..."
  let evalSess ← IreeSession.create evalVmfb

  IO.println "loading Imagenette val..."
  let (valImg, valLbl, nVal) ← F32.loadImagenette "data/imagenette/val.bin"
  IO.println s!"  {nVal} val images"

  let evalBatch : Nat := 32
  let evalSteps := nVal / evalBatch
  let valPixels : Nat := 3 * 224 * 224
  let evalXSh := specGelu.xShape evalBatch
  let evalShapesBA := specGelu.evalShapesBA
  let nClasses := specGelu.numClasses.toUSize

  let mut predHist : Array Nat := Array.replicate 10 0
  let mut classCorrect : Array Nat := Array.replicate 10 0
  let mut classTotal : Array Nat := Array.replicate 10 0

  -- Logit statistics on the first batch — see if predictions are
  -- confident-and-wrong or unconfident-spread.
  let mut firstBatchLogits : ByteArray := ByteArray.empty

  IO.println "running eval forward..."
  for bi in [:evalSteps] do
    let xba := F32.sliceImages valImg (bi * evalBatch) evalBatch valPixels
    let logits ← IreeSession.forwardF32 evalSess "convnext_t_gelu_convnext_tiny_gelu_eval.forward_eval"
                    evalParams evalShapesBA xba evalXSh evalBatch.toUSize nClasses
    if bi == 0 then firstBatchLogits := logits
    let lblSlice := F32.sliceLabels valLbl (bi * evalBatch) evalBatch
    for i in [:evalBatch] do
      let pred := (F32.argmax10 logits (i * specGelu.numClasses).toUSize).toNat
      let label := lblSlice.data[i * 4]!.toNat
      predHist := predHist.modify pred (· + 1)
      classTotal := classTotal.modify label (· + 1)
      if pred == label then classCorrect := classCorrect.modify label (· + 1)

  let total : Nat := evalSteps * evalBatch
  let mut correct : Nat := 0
  for c in [:10] do correct := correct + classCorrect[c]!

  IO.println ""
  IO.println s!"=== Overall accuracy: {correct}/{total} = {correct.toFloat / total.toFloat * 100.0}% ==="

  IO.println ""
  IO.println "Prediction histogram (which class is the model picking?):"
  IO.println "  uniform random would be ~10% per class"
  for c in [:10] do
    let pct := predHist[c]!.toFloat / total.toFloat * 100.0
    IO.println s!"  class {c} ({imagenetteClasses[c]!}): {predHist[c]!} preds ({pct}%)"

  IO.println ""
  IO.println "Per-class accuracy (model gets which classes right?):"
  for c in [:10] do
    let cor := classCorrect[c]!
    let tot := classTotal[c]!
    let pct := if tot > 0 then cor.toFloat / tot.toFloat * 100.0 else 0.0
    IO.println s!"  class {c} ({imagenetteClasses[c]!}): {cor}/{tot} = {pct}%"

  IO.println ""
  IO.println "First-batch logit stats (how confident are predictions?):"
  let mut allMin : Float := 1e30
  let mut allMax : Float := -1e30
  for i in [:evalBatch] do
    let mut bMn : Float := F32.read firstBatchLogits (i * 10).toUSize
    let mut bMx : Float := bMn
    for c in [:10] do
      let v := F32.read firstBatchLogits (i * 10 + c).toUSize
      if v < bMn then bMn := v
      if v > bMx then bMx := v
    if bMn < allMin then allMin := bMn
    if bMx > allMax then allMax := bMx
  IO.println s!"  global logit range: [{allMin}, {allMax}]"
  IO.println "  first 4 images, logits per class:"
  for i in [:4] do
    let mut row : String := s!"  img{i}:"
    for c in [:10] do
      let v := F32.read firstBatchLogits (i * 10 + c).toUSize
      row := row ++ s!" {v}"
    IO.println row
