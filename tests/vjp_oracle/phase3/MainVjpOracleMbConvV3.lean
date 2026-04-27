import LeanMlir

/-! VJP oracle: mbConvV3 — MobileNet V3 block with h-swish + h-sigmoid SE.
    Tests piecewise-linear activations (h-swish / h-sigmoid) on top of
    the mbConv composition. -/

def mbConvV3Net : NetSpec where
  name   := "vjp-oracle-mbconv-v3"
  imageH := 28
  imageW := 28
  layers := [
    .convBn 1 4 3 1 .same,                -- stem
    .mbConvV3 4 4 8 3 1 true .hSwish,         -- ic=4, oc=4, expandCh=8, k=3, stride=1, SE on, h-swish on
    .flatten,
    .dense 3136 10 .identity
  ]

def vjpCfg : TrainConfig where
  learningRate := 0.001
  batchSize    := 4
  epochs       := 1
  useAdam      := true
  weightDecay  := 0.0
  cosineDecay  := false
  warmupEpochs := 0
  augment      := false

def main (args : List String) : IO Unit :=
  mbConvV3Net.train vjpCfg (args.head?.getD "data") .mnist
