import LeanMlir

/-! Render predictions from a trained Pets segmentation checkpoint.

For each of the first N val images, runs the eval forward and writes a
PPM strip showing
    input image | ground-truth mask | predicted mask
side by side. Trimap colors:
    foreground (class 0) → green
    background (class 1) → blue
    boundary   (class 2) → red

Usage:
    lake exe pets-predict [unet|autoencoder] [out.ppm]

Defaults to the UNet checkpoint and `runs/<latest>/pets_pred.ppm`. -/

def unetPets : NetSpec where
  name := "UNet (Pets, 224×224 RGB → 3-class trimap)"
  imageH := 224
  imageW := 224
  layers := [
    .unetDown 3   32,
    .unetDown 32  64,
    .unetDown 64  128,
    .unetDown 128 256,
    .convBn 256 512 3 1 .same,
    .convBn 512 512 3 1 .same,
    .unetUp 512 256,
    .unetUp 256 128,
    .unetUp 128 64,
    .unetUp 64  32,
    .conv2d 32 3 1 .same .identity
  ]

def autoencoderPets : NetSpec where
  name := "Autoencoder (Pets, 224×224 RGB → 3-class trimap, skipless)"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3   64  3 1 .same, .maxPool 2 2,
    .convBn 64  128 3 1 .same, .maxPool 2 2,
    .convBn 128 256 3 1 .same, .maxPool 2 2,
    .convBn 256 512 3 1 .same, .maxPool 2 2,
    .convBn 512 512 3 1 .same,
    .bilinearUpsample 2, .convBn 512 256 3 1 .same,
    .bilinearUpsample 2, .convBn 256 128 3 1 .same,
    .bilinearUpsample 2, .convBn 128 64  3 1 .same,
    .bilinearUpsample 2, .convBn 64  64  3 1 .same,
    .conv2d 64 3 1 .same .identity
  ]

/-- ImageNet de-normalization. Channel `c ∈ {0, 1, 2}`, normalized
    value `v` → uint8 in `[0, 255]`. -/
private def denormToU8 (c : Nat) (v : Float) : UInt8 :=
  let (mean, std) :=
    if c == 0      then (0.485, 0.229)
    else if c == 1 then (0.456, 0.224)
    else                 (0.406, 0.225)
  let p := (v * std + mean) * 255.0
  let pc := if p < 0.0 then 0.0 else if p > 255.0 then 255.0 else p
  pc.toUInt8

/-- Color for a 3-class trimap label. -/
private def maskColor (cls : UInt8) : UInt8 × UInt8 × UInt8 :=
  match cls with
  | 0 => (0, 200, 0)        -- foreground (animal): green
  | 1 => (40, 40, 200)      -- background: blue
  | _ => (200, 0, 0)        -- boundary (or anything else): red

/-- Argmax over 3 channels at a given (b, h, w). The logits buffer is
    laid out NCHW with N first, then C, then H, then W. -/
private def argmaxChannel3 (logits : ByteArray) (b h w : Nat) (H W : Nat)
    : UInt8 :=
  let stride := H * W
  let imgOff := b * 3 * stride
  let pixOff := h * W + w
  let v0 := F32.read logits (imgOff + 0 * stride + pixOff).toUSize
  let v1 := F32.read logits (imgOff + 1 * stride + pixOff).toUSize
  let v2 := F32.read logits (imgOff + 2 * stride + pixOff).toUSize
  if v0 >= v1 && v0 >= v2 then (0 : UInt8)
  else if v1 >= v2 then (1 : UInt8)
  else (2 : UInt8)

def main (args : List String) : IO Unit := do
  let which := args.head?.getD "unet"
  let spec : NetSpec := match which with
    | "autoencoder" => autoencoderPets
    | _             => unetPets
  let outPath := match args with
    | _ :: out :: _ => out
    | _ => "runs/2026-05-06-unet-pets-phase2/pets_pred.ppm"
  IO.FS.createDirAll (System.FilePath.mk outPath).parent.get!.toString
  let pfx := spec.buildPrefix
  let paramsPath := s!"{pfx}_params.bin"
  let bnPath := s!"{pfx}_bn_stats.bin"
  let evalVmfb := s!"{pfx}_fwd_eval.vmfb"
  for p in [paramsPath, bnPath, evalVmfb] do
    if !(← System.FilePath.pathExists p) then
      IO.eprintln s!"missing artifact: {p}"
      IO.eprintln "  (run lake exe unet-pets-train first to produce checkpoint + vmfb)"
      IO.Process.exit 1
  IO.eprintln s!"  loading {paramsPath} ..."
  let params ← IO.FS.readBinFile paramsPath
  let bnStats ← IO.FS.readBinFile bnPath
  let evalParams := params.append bnStats
  IO.eprintln s!"  loading data/pets/val.bin ..."
  let (valImg, valMask, nVal) ← F32.loadPets "data/pets/val.bin"
  -- The eval vmfb was compiled for the training batch size (16).
  -- We render only the first `nVis` images of that batch.
  let evalBatch : Nat := 16
  let nVis : Nat := 4
  if nVal < evalBatch then
    IO.eprintln s!"need ≥ {evalBatch} val records, found {nVal}"; IO.Process.exit 1
  let H := spec.imageH
  let W := spec.imageW
  let imgPixels := 3 * H * W
  let maskPixels := H * W
  let xba := F32.sliceImages valImg 0 evalBatch imgPixels
  let xShape := spec.xShape evalBatch
  let evalShapes := spec.evalShapesBA
  IO.eprintln s!"  running eval forward (batch={evalBatch}, rendering {nVis}) ..."
  let sess ← IreeSession.create evalVmfb
  let outClasses : USize := imgPixels.toUSize
  let logits ← IreeSession.forwardF32 sess spec.evalFnName
                  evalParams evalShapes xba xShape evalBatch.toUSize outClasses
  IO.eprintln s!"  logits {logits.size} bytes; rendering PPM ..."
  -- Layout: 3 panels per image (input | true | pred), each H × W. 4 images
  -- stacked vertically.
  let panelW := W
  let stripW := 3 * panelW
  let stripH := nVis * H
  let mut ppm : ByteArray := ByteArray.empty
  ppm := ppm.append s!"P6\n{stripW} {stripH}\n255\n".toUTF8
  for i in [:nVis] do
    for h in [:H] do
      -- panel 1: input (denormalized RGB)
      for w in [:W] do
        let imgOff := i * imgPixels
        let stride := H * W
        let r := denormToU8 0 (F32.read xba (imgOff + 0 * stride + h * W + w).toUSize)
        let g := denormToU8 1 (F32.read xba (imgOff + 1 * stride + h * W + w).toUSize)
        let b := denormToU8 2 (F32.read xba (imgOff + 2 * stride + h * W + w).toUSize)
        ppm := ppm.push r |>.push g |>.push b
      -- panel 2: ground-truth mask (color-coded)
      for w in [:W] do
        let cls := valMask.get! (i * maskPixels + h * W + w)
        let (r, g, b) := maskColor cls
        ppm := ppm.push r |>.push g |>.push b
      -- panel 3: predicted mask (color-coded)
      for w in [:W] do
        let cls := argmaxChannel3 logits i h w H W
        let (r, g, b) := maskColor cls
        ppm := ppm.push r |>.push g |>.push b
  IO.FS.writeBinFile outPath ppm
  IO.eprintln s!"  wrote {outPath} ({ppm.size} bytes)"
  IO.eprintln s!"  view with e.g. `display {outPath}` or convert:"
  IO.eprintln s!"    convert {outPath} {outPath.dropRight 4}.png"
