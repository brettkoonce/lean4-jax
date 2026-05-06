import LeanMlir

/-! Forward-only smoke test for the new `unetDown` / `unetUp` codegen.

Generates the forward MLIR for `unetPets` (depth 4, base 32, 224×224
RGB → 3-class trimap), writes it to disk, runs iree-compile to produce
a .vmfb, and exits 0 if iree-compile succeeded.

Only validates the **forward** path. Train-step backward (with
skip-grad slots) is the next session.

Usage: `IREE_BACKEND=rocm IREE_CHIP=gfx1100 lake exe test-unet-forward` -/

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

def main : IO Unit := do
  IO.FS.createDirAll ".lake/build"
  let mlir := MlirCodegen.generate unetPets 2
  let mlirPath := ".lake/build/test_unet_forward.mlir"
  let vmfbPath := ".lake/build/test_unet_forward.vmfb"
  IO.FS.writeFile mlirPath mlir
  IO.eprintln s!"  wrote {mlirPath} ({mlir.length} chars)"
  let args ← ireeCompileArgs mlirPath vmfbPath
  IO.eprintln s!"  iree-compile {String.intercalate " " args.toList}"
  let compiler := if (← System.FilePath.pathExists ".venv/bin/iree-compile")
                  then ".venv/bin/iree-compile"
                  else "iree-compile"
  let r ← IO.Process.output { cmd := compiler, args := args }
  if r.exitCode != 0 then
    IO.eprintln s!"FAIL: iree-compile exit {r.exitCode}"
    IO.eprintln (r.stderr.take 4000)
    IO.Process.exit 1
  IO.eprintln s!"OK: {vmfbPath} produced"
