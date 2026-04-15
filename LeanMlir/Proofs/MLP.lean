import LeanMlir.Proofs.Tensor
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# MLP VJP Proofs

Formal VJP correctness for the layers of a 3-layer MLP.
All definitions over `ℝ`, proofs use Mathlib's `Finset.sum`.
-/

open Finset BigOperators Classical

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Dense Layer:  y = xW + b
-- ════════════════════════════════════════════════════════════════

noncomputable def dense {m n : Nat} (W : Mat m n) (b : Vec n) (x : Vec m) : Vec n :=
  fun j => (∑ i : Fin m, x i * W i j) + b j

axiom pdiv_dense {m n : Nat} (W : Mat m n) (b : Vec n)
    (x : Vec m) (i : Fin m) (j : Fin n) :
    pdiv (dense W b) x i j = W i j

/-- Dense VJP — proved. -/
noncomputable def dense_has_vjp {m n : Nat} (W : Mat m n) (b : Vec n) :
    HasVJP (dense W b) where
  backward := fun _x dy => Mat.mulVec W dy
  correct := by
    intro x dy i
    simp only [Mat.mulVec]
    congr 1; ext j; rw [pdiv_dense]

theorem dense_weight_grad {m n : Nat} (x : Vec m) (dy : Vec n) :
    Mat.outer x dy = (fun i j => x i * dy j) := rfl

-- ════════════════════════════════════════════════════════════════
-- § ReLU:  y = max(x, 0)
-- ════════════════════════════════════════════════════════════════

noncomputable def relu (n : Nat) (x : Vec n) : Vec n :=
  fun i => if x i > 0 then x i else 0

axiom pdiv_relu (n : Nat) (x : Vec n) (i j : Fin n) :
    pdiv (relu n) x i j =
      if i = j then (if x i > 0 then 1 else 0) else 0

/-- ReLU VJP — proved. -/
noncomputable def relu_has_vjp (n : Nat) : HasVJP (relu n) where
  backward := fun x dy i => if x i > 0 then dy i else 0
  correct := by
    intro x dy i
    simp [pdiv_relu]

-- ════════════════════════════════════════════════════════════════
-- § Softmax Cross-Entropy Loss
-- ════════════════════════════════════════════════════════════════

noncomputable def softmax (c : Nat) (z : Vec c) : Vec c :=
  let e : Vec c := fun j => Real.exp (z j)
  let total := ∑ k : Fin c, e k
  fun j => e j / total

noncomputable def oneHot (c : Nat) (label : Fin c) : Vec c :=
  fun j => if j = label then 1 else 0

noncomputable def crossEntropy (c : Nat) (logits : Vec c) (label : Fin c) : ℝ :=
  -(Real.log (softmax c logits label))

axiom softmaxCE_grad (c : Nat) (logits : Vec c) (label : Fin c) (j : Fin c) :
    sdiv (fun z => crossEntropy c z label) logits j
    = softmax c logits j - oneHot c label j

-- ════════════════════════════════════════════════════════════════
-- § MLP Composition
-- ════════════════════════════════════════════════════════════════

noncomputable def mlpForward {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) :
    Vec d₀ → Vec d₃ :=
  dense W₂ b₂ ∘ relu d₂ ∘ dense W₁ b₁ ∘ relu d₁ ∘ dense W₀ b₀

noncomputable def mlp_has_vjp {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) :
    HasVJP (mlpForward W₀ b₀ W₁ b₁ W₂ b₂) := by
  unfold mlpForward
  have h1 := vjp_comp (dense W₀ b₀) (relu d₁) (dense_has_vjp W₀ b₀) (relu_has_vjp d₁)
  have h2 := vjp_comp _ (dense W₁ b₁) h1 (dense_has_vjp W₁ b₁)
  have h3 := vjp_comp _ (relu d₂) h2 (relu_has_vjp d₂)
  exact vjp_comp _ (dense W₂ b₂) h3 (dense_has_vjp W₂ b₂)

end Proofs
