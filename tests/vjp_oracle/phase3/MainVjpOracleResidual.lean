import LeanMlir

/-! VJP oracle: residual — tests `biPath_has_vjp` (additive fan-in VJP) via
    a single residualBlock with no projection. Stem is `.convBn` so both
    phases reshape NCHW correctly. -/

def residualNet : NetSpec where
  name   := "vjp-oracle-residual"
  imageH := 28
  imageW := 28
  layers := [
    .convBn 1 4 3 1 .same,        -- 4×28×28
    .residualBlock 4 4 1 1,        -- 4×28×28, no projection (ic==oc, stride==1)
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
  residualNet.train vjpCfg (args.head?.getD "data") .mnist
