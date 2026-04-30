import LeanMlir.F32Array

/-- Read N float32s starting at index 0 from a ByteArray. -/
def readN (ba : ByteArray) (n : Nat) : Array Float := Id.run do
  let mut a : Array Float := Array.mkEmpty n
  for i in [:n] do
    a := a.push (F32.read ba i.toUSize)
  return a

def stats (label : String) (vs : Array Float) : IO Unit := do
  let n := vs.size
  if n == 0 then IO.println s!"  {label}: empty"; return
  let mut mn := vs[0]!
  let mut mx := vs[0]!
  let mut sum : Float := 0.0
  let mut sumSq : Float := 0.0
  for v in vs do
    if v < mn then mn := v
    if v > mx then mx := v
    sum := sum + v
    sumSq := sumSq + v * v
  let mean := sum / n.toFloat
  let var := sumSq / n.toFloat - mean * mean
  let std := if var > 0.0 then var.sqrt else 0.0
  IO.println s!"  {label}: n={n} min={mn} max={mx} mean={mean} std={std}"

def main : IO Unit := do
  -- BN stats layout: per-channel mean (96 floats) then per-channel var (96 floats).
  let path := ".lake/build/convnext_t_gelu_convnext_tiny_gelu_bn_stats.bin"
  let ba ← IO.FS.readBinFile path
  let total := ba.size / 4
  IO.println s!"── ConvNeXt-T saved BN stats ({ba.size} bytes = {total} floats) ──"
  let allFloats := readN ba total

  -- Split into mean and var halves.
  let half := total / 2
  let means := (allFloats.toList.take half).toArray
  let vars := (allFloats.toList.drop half).toArray

  stats "running_mean" means
  stats "running_var " vars

  IO.println ""
  IO.println "Sanity reference: ImageNet input should produce per-channel"
  IO.println "post-conv pre-BN running mean ~0 (BN's job to make it so);"
  IO.println "running var should be O(1) — typical post-conv variance."
  IO.println ""

  IO.println "── first 8 values of each ──"
  IO.println s!"  means[0..7]: {means.toList.take 8}"
  IO.println s!"  vars[0..7]:  {vars.toList.take 8}"

  -- Also peek at the params: are they reasonable, or are some bizarrely large?
  let paramPath := ".lake/build/convnext_t_gelu_convnext_tiny_gelu_params.bin"
  let pba ← IO.FS.readBinFile paramPath
  let nParams := pba.size / 4
  IO.println ""
  IO.println s!"── ConvNeXt-T params ({pba.size} bytes = {nParams} floats) ──"

  -- Just read first 1000 and last 1000 and middle 1000 — sample distribution.
  -- Don't read all 28M — would be slow.
  let mut sample : Array Float := Array.mkEmpty 3000
  for i in [:1000] do
    sample := sample.push (F32.read pba i.toUSize)
  for i in [(nParams / 2 - 500) : (nParams / 2 + 500)] do
    sample := sample.push (F32.read pba i.toUSize)
  for i in [(nParams - 1000) : nParams] do
    sample := sample.push (F32.read pba i.toUSize)
  stats "param sample (3K of 28M)" sample
