import LeanMlir

/-! Real UNet on Oxford-IIIT Pets — UNet demo Phase 2 smoke test.

    Mirrors `MainAutoencoderPetsTrain` but uses `unetPets` (depth 4,
    base 32, with skip connections) — the actual demo target. If
    training loss drops as fast or faster than the skipless
    autoencoder baseline (epoch 1 ≈ 0.800, epoch 2 ≈ 0.770, epoch 3 ≈
    0.766), the new unetDown / unetUp codegen + skip-state plumbing
    is producing usable gradients.

    Architecture: 224×224 RGB → 14×14 (4 × maxPool 2 inside unetDown)
    → 224×224 (4 × bilinearUpsample 2 inside unetUp). 7.85M params.

    Usage:
      lake exe unet-pets-train [data/pets]
-/

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

def unetPetsConfig : TrainConfig where
  learningRate := 0.001
  batchSize    := 16
  epochs       := 3
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := false
  warmupEpochs := 0
  augment      := false

def main (args : List String) : IO Unit :=
  unetPets.train unetPetsConfig
    (args.head?.getD "data/pets") .pets
