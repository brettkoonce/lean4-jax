import LeanMlir.Proofs.Attention

/-! # Dischargeability sanity check

    Confirms that every `Differentiable` hypothesis the proofs in
    `LeanMlir/Proofs/` propagate downstream is *actually satisfiable*
    for the architecture functions the book uses. If a hypothesis
    were vacuous (no real use case can discharge it), the theorem
    would be unusable in practice.

    Each `example` here invokes an existing `*_diff` lemma — they all
    typecheck, which means the `Differentiable` evidence is real, not
    a placeholder. -/

open Proofs

-- Foundation pieces (one-arg, no extra hypotheses) ─────────────────

example (m n : Nat) (W : Mat m n) (b : Vec n) :
    Differentiable ℝ (dense W b) :=
  dense_diff W b

example (c : Nat) :
    Differentiable ℝ (softmax c) :=
  softmax_diff c

example (D : Nat) (ε γ β : ℝ) (hε : 0 < ε) :
    Differentiable ℝ (layerNormForward D ε γ β) :=
  layerNorm_diff D ε γ β hε

-- Flattened (Mat-level → Vec) pieces — used inside transformer chains

example (m p q : Nat) (D : Mat p q) :
    Differentiable ℝ (fun v : Vec (m * p) =>
      Mat.flatten (Mat.mul (Mat.unflatten v) D)) :=
  matmul_right_const_flat_diff (m := m) D

example (m n : Nat) (s : ℝ) :
    Differentiable ℝ (fun v : Vec (m * n) =>
      Mat.flatten (fun r c => s * (Mat.unflatten v) r c)) :=
  scalarScale_flat_diff (m := m) (n := n) s

example (m n : Nat) :
    Differentiable ℝ (fun v : Vec (m * n) =>
      Mat.flatten (rowSoftmax (Mat.unflatten v) : Mat m n)) :=
  rowSoftmax_flat_diff m n

-- Per-token lifts (used inside transformer blocks) ─────────────────

example (N inD outD : Nat) (W : Mat inD outD) (b : Vec outD) :
    Differentiable ℝ (fun v : Vec (N * inD) =>
      Mat.flatten ((fun X : Mat N inD => fun n => dense W b (X n))
                   (Mat.unflatten v))) :=
  dense_per_token_flat_diff (N := N) W b

example (N D : Nat) :
    Differentiable ℝ (fun v : Vec (N * D) =>
      Mat.flatten ((fun X : Mat N D => fun n => gelu D (X n))
                   (Mat.unflatten v))) :=
  gelu_per_token_flat_diff N D

example (N D : Nat) (ε γ β : ℝ) (hε : 0 < ε) :
    Differentiable ℝ (fun v : Vec (N * D) =>
      Mat.flatten ((fun X : Mat N D => fun n => layerNormForward D ε γ β (X n))
                   (Mat.unflatten v))) :=
  layerNorm_per_token_flat_diff N D ε γ β hε

-- Multi-head SDPA layer (the chunky bundled one — was an axiom) ────

example (N heads d_head : Nat)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) :
    Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten (mhsa_layer N heads d_head Wq Wk Wv Wo bq bk bv bo
                   (Mat.unflatten v))) :=
  mhsa_layer_flat_diff N heads d_head Wq Wk Wv Wo bq bk bv bo

/-- Entry point so this can register as a `lean_exe` and run in CI.
    The work is done at typecheck time — by the time `main` runs,
    every `example` above has already been kernel-checked. -/
def main : IO Unit :=
  IO.println "OK: 11 dischargeability examples typechecked."
