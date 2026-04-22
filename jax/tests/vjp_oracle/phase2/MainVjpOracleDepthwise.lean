import Jax

def depthwiseNet : NetSpec where
  name   := "vjp-oracle-depthwise"
  imageH := 28
  imageW := 28
  layers := [
    .convBn 1 4 3 1 .same,
    .invertedResidual 4 4 2 1 1,
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

#eval depthwiseNet.validate!

def main (args : List String) : IO Unit :=
  runJax depthwiseNet vjpCfg .mnist (args.head? |>.getD "data") "generated_vjp_oracle_depthwise.py"
