import LeanMlir

/-! VJP oracle: dense_only.

    Minimal NetSpec exercising only `dense_has_vjp`. One dense layer
    784→10, no activation. Trained for one epoch with SGD-flavored
    Adam (no cosine, no warmup, no weight decay, no augment) so the
    only gradient math that runs is what we want to test. Step-2 loss
    is the first step whose value depends on the backward pass; a
    small cross-backend Δ at step 2 means the Lean hand-derived VJP
    matches JAX's `value_and_grad` at f32 precision. -/

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

def main (args : List String) : IO Unit :=
  denseOnly.train vjpCfg (args.head?.getD "data") .mnist
