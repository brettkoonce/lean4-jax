import LeanMlir.Proofs.Tensor
import LeanMlir.Proofs.MLP
import LeanMlir.Proofs.BatchNorm
import LeanMlir.Proofs.LayerNorm
import LeanMlir.Proofs.Attention

open Proofs
open scoped Real

-- Foundation: structural calculus rules ────────────────────────────

theorem chk_pdiv_comp {m n p : Nat} (f : Vec m → Vec n) (g : Vec n → Vec p)
    (x : Vec m) (hf : DifferentiableAt ℝ f x)
    (hg : DifferentiableAt ℝ g (f x))
    (i : Fin m) (k : Fin p) :
    pdiv (g ∘ f) x i k =
    ∑ j : Fin n, pdiv f x i j * pdiv g (f x) j k := by sorry

theorem chk_pdiv_add {m n : Nat} (f g : Vec m → Vec n) (x : Vec m)
    (hf : DifferentiableAt ℝ f x) (hg : DifferentiableAt ℝ g x)
    (i : Fin m) (j : Fin n) :
    pdiv (fun y k => f y k + g y k) x i j
    = pdiv f x i j + pdiv g x i j := by sorry

theorem chk_pdiv_mul {m n : Nat} (f g : Vec m → Vec n) (x : Vec m)
    (hf : DifferentiableAt ℝ f x) (hg : DifferentiableAt ℝ g x)
    (i : Fin m) (j : Fin n) :
    pdiv (fun y k => f y k * g y k) x i j
    = pdiv f x i j * g x j + f x j * pdiv g x i j := by sorry

theorem chk_pdiv_id {n : Nat} (x : Vec n) (i j : Fin n) :
    pdiv (fun y : Vec n => y) x i j = if i = j then 1 else 0 := by sorry

theorem chk_pdiv_const {m n : Nat} (c : Vec n) (x : Vec m)
    (i : Fin m) (j : Fin n) :
    pdiv (fun _ : Vec m => c) x i j = 0 := by sorry

theorem chk_pdiv_reindex {a b : Nat} (σ : Fin b → Fin a) (x : Vec a)
    (i : Fin a) (j : Fin b) :
    pdiv (fun y : Vec a => fun k : Fin b => y (σ k)) x i j =
    if i = σ j then 1 else 0 := by sorry

theorem chk_pdiv_finset_sum {m n : Nat} {α : Type*} [DecidableEq α]
    (S : Finset α) (f : α → Vec m → Vec n) (x : Vec m)
    (hdiff : ∀ s ∈ S, DifferentiableAt ℝ (f s) x)
    (i : Fin m) (j : Fin n) :
    pdiv (fun y k => ∑ s ∈ S, f s y k) x i j =
    ∑ s ∈ S, pdiv (f s) x i j := by sorry

theorem chk_pdivMat_rowIndep {m n p : Nat} (g : Vec n → Vec p)
    (h_g_diff : Differentiable ℝ g)
    (A : Mat m n) (i : Fin m) (j : Fin n) (k : Fin m) (l : Fin p) :
    pdivMat (fun M : Mat m n => fun r => g (M r)) A i j k l =
    if i = k then pdiv g (A i) j l else 0 := by sorry

-- Mat-level structural rules ───────────────────────────────────────

theorem chk_pdivMat_comp {a b c d e f : Nat}
    (F : Mat a b → Mat c d) (G : Mat c d → Mat e f)
    (A : Mat a b)
    (hF_diff : DifferentiableAt ℝ
      (fun v : Vec (a * b) => Mat.flatten (F (Mat.unflatten v))) (Mat.flatten A))
    (hG_diff : DifferentiableAt ℝ
      (fun u : Vec (c * d) => Mat.flatten (G (Mat.unflatten u))) (Mat.flatten (F A)))
    (i : Fin a) (j : Fin b) (k : Fin e) (l : Fin f) :
    pdivMat (G ∘ F) A i j k l =
    ∑ p : Fin c, ∑ q : Fin d,
      pdivMat F A i j p q * pdivMat G (F A) p q k l := by sorry

theorem chk_pdivMat_matmul_left_const {m p q : Nat} (C : Mat m p) (B : Mat p q)
    (i : Fin p) (j : Fin q) (k : Fin m) (l : Fin q) :
    pdivMat (fun B' : Mat p q => Mat.mul C B') B i j k l =
    if l = j then C k i else 0 := by sorry

theorem chk_pdivMat_scalarScale {m n : Nat} (s : ℝ) (A : Mat m n)
    (i : Fin m) (j : Fin n) (k : Fin m) (l : Fin n) :
    pdivMat (fun M : Mat m n => fun r c => s * M r c) A i j k l =
    if i = k ∧ j = l then s else 0 := by sorry

theorem chk_pdivMat_transpose {m n : Nat} (A : Mat m n)
    (i : Fin m) (j : Fin n) (k : Fin n) (l : Fin m) :
    pdivMat (fun M : Mat m n => Mat.transpose M) A i j k l =
    if j = k ∧ i = l then 1 else 0 := by sorry

-- Ch 3 MLP ──────────────────────────────────────────────────────────

theorem chk_pdiv_dense {m n : Nat} (W : Mat m n) (b : Vec n)
    (x : Vec m) (i : Fin m) (j : Fin n) :
    pdiv (dense W b) x i j = W i j := by sorry

theorem chk_pdiv_dense_W {m n : Nat} (b : Vec n) (x : Vec m) (W : Mat m n)
    (i : Fin m) (j' : Fin n) (j : Fin n) :
    pdiv (fun v : Vec (m * n) => dense (Mat.unflatten v) b x)
         (Mat.flatten W) (finProdFinEquiv (i, j')) j =
      if j = j' then x i else 0 := by sorry

theorem chk_pdiv_dense_b {m n : Nat} (W : Mat m n) (b : Vec n) (x : Vec m)
    (i j : Fin n) :
    pdiv (fun b' : Vec n => dense W b' x) b i j = if i = j then 1 else 0 := by sorry

theorem chk_dense_weight_grad_correct {m n : Nat} (W : Mat m n) (b : Vec n)
    (x : Vec m) (dy : Vec n) (i : Fin m) (j : Fin n) :
    Mat.outer x dy i j =
      ∑ k : Fin n,
        pdiv (fun v : Vec (m * n) => dense (Mat.unflatten v) b x)
             (Mat.flatten W) (finProdFinEquiv (i, j)) k * dy k := by sorry

theorem chk_dense_bias_grad_correct {m n : Nat} (W : Mat m n) (b : Vec n)
    (x : Vec m) (dy : Vec n) (i : Fin n) :
    dy i =
      ∑ j : Fin n, pdiv (fun b' : Vec n => dense W b' x) b i j * dy j := by sorry

-- Ch 5 BN ───────────────────────────────────────────────────────────

theorem chk_pdiv_bnAffine (n : Nat) (γ β : ℝ) (v : Vec n) (i j : Fin n) :
    pdiv (bnAffine n γ β) v i j =
      if i = j then γ else 0 := by sorry

theorem chk_pdiv_bnCentered (n : Nat) (x : Vec n) (i j : Fin n) :
    pdiv (bnCentered n) x i j =
      (if i = j then (1 : ℝ) else 0) - 1 / (n : ℝ) := by sorry

theorem chk_pdiv_bnIstdBroadcast (n : Nat) (ε : ℝ) (hε : 0 < ε) (x : Vec n)
    (i j : Fin n) :
    pdiv (bnIstdBroadcast n ε) x i j =
      -(bnIstd n x ε)^3 * (x i - bnMean n x) / (n : ℝ) := by sorry

theorem chk_pdiv_bnNormalize (n : Nat) (ε : ℝ) (hε : 0 < ε)
    (x : Vec n) (i j : Fin n) :
    pdiv (bnNormalize n ε) x i j =
      bnIstd n x ε / (n : ℝ) *
        ((n : ℝ) * (if i = j then 1 else 0) - 1 - bnXhat n ε x i * bnXhat n ε x j) := by sorry

-- Ch 9 LayerNorm + GELU ─────────────────────────────────────────────

theorem chk_pdiv_gelu (n : Nat) (x : Vec n) (i j : Fin n) :
    pdiv (gelu n) x i j =
    if i = j then geluScalarDeriv (x i) else 0 := by sorry

-- Ch 10 Attention ───────────────────────────────────────────────────

theorem chk_pdiv_softmax (c : Nat) (z : Vec c) (i j : Fin c) :
    pdiv (softmax c) z i j =
    softmax c z j * ((if i = j then 1 else 0) - softmax c z i) := by sorry

theorem chk_softmaxCE_grad (c : Nat) (logits : Vec c) (label : Fin c) (j : Fin c) :
    pdiv (fun (z : Vec c) (_ : Fin 1) => crossEntropy c z label) logits j 0
    = softmax c logits j - oneHot c label j := by sorry

theorem chk_sdpa_back_Q_correct (n d : Nat) (Q K V dOut : Mat n d)
    (i : Fin n) (j : Fin d) :
    sdpa_back_Q n d Q K V dOut i j =
    ∑ k : Fin n, ∑ l : Fin d,
      pdivMat (fun Q' => sdpa n d Q' K V) Q i j k l * dOut k l := by sorry

theorem chk_sdpa_back_K_correct (n d : Nat) (Q K V dOut : Mat n d)
    (i : Fin n) (j : Fin d) :
    sdpa_back_K n d Q K V dOut i j =
    ∑ k : Fin n, ∑ l : Fin d,
      pdivMat (fun K' => sdpa n d Q K' V) K i j k l * dOut k l := by sorry

theorem chk_sdpa_back_V_correct (n d : Nat) (Q K V dOut : Mat n d)
    (i : Fin n) (j : Fin d) :
    sdpa_back_V n d Q K V dOut i j =
    ∑ k : Fin n, ∑ l : Fin d,
      pdivMat (fun V' => sdpa n d Q K V') V i j k l * dOut k l := by sorry
