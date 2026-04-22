import LeanMlir

/-! VJP oracle: depthwise — tests the depthwise-conv VJP via one
    `.invertedResidual` block (expand + depthwise + project). The depthwise
    middle step has weight shape (mid, 1, 3, 3) with feature_group_count
    = mid — unusual layout that's easy to get wrong between phases. -/

def depthwiseNet : NetSpec where
  name   := "vjp-oracle-depthwise"
  imageH := 28
  imageW := 28
  layers := [
    .convBn 1 4 3 1 .same,              -- stem: 4×28×28
    .invertedResidual 4 4 2 1 1,         -- expand=2, stride=1, 1 block, 4×28×28
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
  depthwiseNet.train vjpCfg (args.head?.getD "data") .mnist
