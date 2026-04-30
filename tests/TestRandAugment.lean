import LeanMlir.F32Array

/-! Diagnostic: feed a single Imagenette-normalized synthetic image
    through `F32.randAugment` (with `imagenetNorm := 1`) and print
    before/after statistics. If the de-norm/re-norm round-trip is
    working, the output should still look like ImageNet-normalized
    values (per-channel mean ~0, range roughly [-2, 3]). If it's
    producing garbage (e.g. all zeros, all one value, or wildly
    out-of-range), we'll see it here. -/

def imageNetMean : Array Float := #[0.485, 0.456, 0.406]
def imageNetStd  : Array Float := #[0.229, 0.224, 0.225]

/-- Build a 1×3×4×4 synthetic ImageNet-normalized image: every pixel
    set to the per-channel ImageNet mean (= 0.5 sRGB ish), so the
    normalized value is 0. Lets us measure exactly what the kernel
    does to a "perfectly average" input. -/
def synthImage : IO ByteArray := do
  let plane : USize := 4 * 4
  -- Three planes (R, G, B) — each filled with the normalized value
  -- corresponding to its per-channel ImageNet mean (which is 0).
  let r ← F32.const plane 0.0
  let g ← F32.const plane 0.0
  let b ← F32.const plane 0.0
  return F32.concat #[r, g, b]

def stats (label : String) (ba : ByteArray) (n : Nat) : IO Unit := do
  let mut mn : Float := F32.read ba 0
  let mut mx : Float := mn
  let mut sum : Float := 0.0
  for i in [:n] do
    let v := F32.read ba i.toUSize
    sum := sum + v
    if v < mn then mn := v
    if v > mx then mx := v
  let mean := sum / n.toFloat
  IO.println s!"  {label}: min={mn} max={mx} mean={mean}"

def main : IO Unit := do
  let img ← synthImage
  let pixels := 1 * 3 * 4 * 4
  IO.println "── synthetic uniform-mean 4×4 image ──"
  stats "in " img pixels
  let outNorm ← F32.randAugment img 1 3 4 4 2 9.0 1 12345
  stats "out" outNorm pixels

  -- Real-shape test: 224×224×3 with He-init random values that
  -- simulate the spread of a real ImageNet-normalized image.
  -- Real values land roughly in [-2.5, 2.5] after ImageNet normalization;
  -- He-init at scale 1.0 gives a similar spread (mean 0, std ~1).
  IO.println ""
  IO.println "── real-shape 224×224×3 random image (He-init, sim real Imagenette) ──"
  let pix224 := 1 * 3 * 224 * 224
  let bigImg ← F32.heInit 42 pix224.toUSize 1.0
  stats "in " bigImg pix224
  IO.println "  applying randAugment with imagenetNorm=1, 2 ops, M=9..."
  let outBig ← F32.randAugment bigImg 1 3 224 224 2 9.0 1 67890
  stats "out" outBig pix224

  IO.println "  applying randAugment with imagenetNorm=1, batch=4 (different ops per image expected)..."
  -- Repeat the image 4 times to make a batch.
  let bigBatch := F32.concat #[bigImg, bigImg, bigImg, bigImg]
  let outBatch ← F32.randAugment bigBatch 4 3 224 224 2 9.0 1 67890
  -- Per-image stats — ops chosen per image, so they should differ.
  for i in [:4] do
    let perImage := F32.slice outBatch (i * pix224) pix224
    stats s!"img-{i}" perImage pix224
