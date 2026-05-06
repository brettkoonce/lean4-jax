import LeanMlir

/-! Skipless autoencoder on Oxford-IIIT Pets — UNet demo Phase 1 smoke test.

    Goal: prove the per-pixel CE codegen + seg train-step ABI work
    end-to-end on real data by training a small encoder/decoder with
    no skip connections. If train loss drops over a handful of epochs,
    the seg pipeline is good and we can move on to the real
    `unetDown` / `unetUp` skip-state plumbing.

    Architecture: 224×224 RGB → 14×14 (4×maxPool) → 224×224 (4×bilinear
    upsample) → 1×1 conv to 3 classes. ~5.5M params. Mirrors
    `Bestiary.UNet.autoencoderPets`.

    Usage:
      lake exe autoencoder-pets-train [data/pets]
-/

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

def autoencoderPetsConfig : TrainConfig where
  learningRate := 0.001
  batchSize    := 16
  epochs       := 3
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := false
  warmupEpochs := 0
  augment      := false

def main (args : List String) : IO Unit :=
  autoencoderPets.train autoencoderPetsConfig
    (args.head?.getD "data/pets") .pets
