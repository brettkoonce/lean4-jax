import Jax

/-! VJP oracle mirror: dense_only. See `MainVjpOracleDense.lean` at
    repo root for the phase-3 counterpart. Same NetSpec, same cfg —
    identical math up to XLA vs IREE reduction tree rounding. -/

def denseOnly : NetSpec where
  name   := "vjp-oracle-dense"
  imageH := 28
  imageW := 28
  layers := [
    .dense 784 10 .identity
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

#eval denseOnly.validate!

def main (args : List String) : IO Unit :=
  runJax denseOnly vjpCfg .mnist (args.head? |>.getD "data") "generated_vjp_oracle_dense.py"
