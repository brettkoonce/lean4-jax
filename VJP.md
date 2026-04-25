# VJP.md — Foundation Flip Landed; What's Left

**Branch:** merged into `main` (commit `10c0aaf`). Blueprint sync at
`bde9313`. **23 project axioms.**

> **Strategy summary.** Attempt #3 used "guarded ReLU" — the soundness
> analysis from attempts #1 and #2 (still preserved below) showed why
> the unconditional bilinear rules were inconsistent with `fderiv` and
> why `pdiv_relu` was inconsistent at non-smooth multi-D points. The
> fix that worked: guard `pdiv_relu` with `(∀ k, x k ≠ 0)`,
> axiomatize `relu_has_vjp` (subgradient routing is now an honest
> convention), and axiomatize `mlp_has_vjp` (composes through ReLU).
> Foundation flipped cleanly; chapter migrations followed.

---

## What landed

### Stage 1 — Foundation (Tensor.lean, commit `8290155`)

- `axiom pdiv` → `noncomputable def pdiv f x i j := fderiv ℝ f x (basisVec i) j`.
- `axiom pdiv_id` / `_const` / `_reindex` → unconditional theorems.
- `axiom pdiv_add` / `_mul` / `_comp` → theorems with `DifferentiableAt` hypotheses.
- `pdiv_finset_sum` now requires `∀ s ∈ S, DifferentiableAt ℝ (f s) x`.
- Internal Tensor.lean cascade threaded: `vjp_comp`, `biPath_has_vjp`,
  `elemwiseProduct_has_vjp`, `pdivMat_comp`, `pdivMat_add`, `vjpMat_comp`,
  `biPathMat_has_vjp`, `pdivMat_matmul_left/right_const`,
  `pdivMat_scalarScale`, `pdiv3_comp`, `pdiv3_add`, `vjp3_comp`,
  `biPath3_has_vjp` all take new `Differentiable` arguments.
- Parallel-track `pdivFD*`, `pdivFDMat*`, `pdivFD3*` blocks deleted
  (consolidated with the now-flipped main blocks).
- `pdivFDMat_rowIndep` and `pdivMat_rowIndep` consolidated into one
  `pdivMat_rowIndep` axiom (the rowIndep proof via `fderiv_pi` is itself
  a deferred follow-up).

**Verification:** `#print axioms Proofs.pdiv_id` / `pdiv_add` / `pdivMat_comp`
shows only `propext` / `Classical.choice` / `Quot.sound`. No project axioms.

### Stage 2 — Per-chapter migrations

| Chapter | Commit | Notable change |
|---|---|---|
| MLP | `d02fcd9` | Diff threaded through `pdiv_dense*`. **`pdiv_relu` guarded** with `(∀ k, x k ≠ 0)`. **`relu_has_vjp` and `mlp_has_vjp` axiomatized** (subgradient routing). |
| CNN | `5f3519f` | Diff threaded through `conv2d_weight_grad_has_vjp` and `conv2d_bias_grad_has_vjp` via `unfold + fun_prop`. `conv2d_has_vjp3` and `maxPool2_has_vjp3` stay axiom (non-smooth ops). |
| Depthwise | `8c8bc0b` | Same recipe as CNN. `depthwise_weight_grad_has_vjp3` and `depthwise_bias_grad_has_vjp` now in pure Mathlib closure. |
| BatchNorm | `972681c` | `pdiv_bnAffine` and `pdiv_bnCentered` from foundation. `pdiv_bnNormalize`, `bnNormalize_has_vjp`, `bn_has_vjp`, `bn_input_grad_correct` now take `(hε : 0 < ε)`. **New axiom `bnIstdBroadcast_diff`** for sqrt/recip smoothness. |
| Residual + SE + LayerNorm | `aa6807f` | Added `Differentiable` arguments to `residual_has_vjp`, `residualProj_has_vjp`, `seBlock_has_vjp`. `layerNorm_has_vjp` takes `(hε : 0 < ε)`. |
| Attention | `7112ce3` | Diff helpers (`matmul_left_const_flat_diff`, `matmul_right_const_flat_diff`, `scalarScale_flat_diff`, `transpose_flat_diff`) via `fun_prop`. **New axiom `rowSoftmax_flat_diff`** for the smooth softmax. `sdpa_Q_chain_has_vjp` and `sdpa_K_chain_has_vjp` threaded properly. **7 high-level chains axiomatized** as a deferral (see "Mechanical follow-ups" below). |

### Stage 3 — Blueprint sync

- `content.tex` updated: every declaration now matches its actual
  Lean state (axiom / theorem / definition). 30 axioms → 20 in the
  blueprint.
- Macro definitions added to `macros/common.tex` for the new
  notation (`\fderiv`, `\Differentiable`, `\bnIstdBroadcast`, etc.).
  Without these the PDF build stops early; xelatex chokes on
  undefined commands while plasTeX (web build) was permissive.

---

## Project axiom inventory (23)

**Foundation (1):**
- `pdivMat_rowIndep` — provable via `fderiv_pi`, deferred.

**MLP (4):**
- `pdiv_relu` — guarded subgradient axiom (subgradient convention).
- `relu_has_vjp` — existence of HasVJP at non-smooth points.
- `softmaxCE_grad` — gradient of cross-entropy loss.
- `mlp_has_vjp` — composes through ReLU.

**CNN (2):**
- `conv2d_has_vjp3` — input-path VJP through non-smooth boundary handling.
- `maxPool2_has_vjp3` — argmax routing convention.

**Depthwise (1):**
- `depthwise_has_vjp3` — input-path VJP, parallel to conv2d.

**BatchNorm (2):**
- `pdiv_bnIstdBroadcast` — derivative of `1/√(σ²+ε)`.
- `bnIstdBroadcast_diff` — smoothness of same function (added by Stage 2d).

**LayerNorm (1):**
- `pdiv_gelu` — diagonal Jacobian of GELU.

**Attention (12):**
- `pdiv_softmax` — Jacobian of softmax.
- `mhsa_has_vjp_mat` — multi-head SDPA.
- `patchEmbed_flat_has_vjp` — opaque-codegen patch embedding.
- `rowSoftmax_flat_diff` — smoothness via Real.exp.
- `transformerMlp_has_vjp_mat` — deferred composition.
- `transformerAttnSublayer_has_vjp_mat` — deferred composition.
- `transformerMlpSublayer_has_vjp_mat` — deferred composition.
- `transformerBlock_has_vjp_mat` — deferred composition.
- `transformerTower_has_vjp_mat` — deferred (was an induction).
- `vit_body_has_vjp_mat` — deferred composition.
- `classifier_flat_has_vjp` — deferred (linear, easy to thread).
- `vit_full_has_vjp` — deferred composition.

---

## Mechanical follow-ups (the realistic next session)

These get the count back to ~13 with no architectural decisions left.

### A. Thread Diff through the 7 deferred Attention chains

Pattern is established by `sdpa_Q_chain_has_vjp` and
`sdpa_K_chain_has_vjp`. Each `vjpMat_comp` / `biPathMat_has_vjp` /
`vjp_comp` call needs `Differentiable` evidence for both factors,
discharged by:
- The four `*_flat_diff` helpers at the top of `Attention.lean`
  (already in place: matmul-left/right, scalar-scale, transpose).
- `rowSoftmax_flat_diff` (axiom — that one stays).
- For composed chains: `Differentiable.comp` after a `simp [Mat.unflatten_flatten]`
  rewrite that pushes the flatten/unflatten through `∘`.

Order to thread (innermost to outermost — each compose builds on the previous):
1. `transformerMlp_has_vjp_mat` (3 levels of `vjpMat_comp`).
2. `transformerAttnSublayer_has_vjp_mat` (`biPathMat` of identity and
   `mhsa ∘ LN1`).
3. `transformerMlpSublayer_has_vjp_mat` (parallel to attn sublayer).
4. `transformerBlock_has_vjp_mat` (one `vjpMat_comp`).
5. `transformerTower_has_vjp_mat` (induction on k via `vjpMat_comp`).
6. `vit_body_has_vjp_mat` (one `vjpMat_comp`).
7. `classifier_flat_has_vjp` (`vjp_comp` of two linear functions; trivial).
8. `vit_full_has_vjp` (three-level `vjp_comp` over patchEmbed, body, classifier).

Each `Differentiable` proof wants `(hε : 0 < ε)` plumbed through —
the per-token wrappers (`layerNorm_per_token_has_vjp_mat`) already
take it. **Net axiom delta: −7.**

Estimate: 4-6 hours, mechanical. Each axiom unrolls into roughly the
proof body that was already there before, plus the Diff hypotheses.

### B. Prove `rowSoftmax_flat_diff` from Mathlib calculus

`Real.exp` is C^∞ everywhere; the denominator `∑ⱼ exp(M r j)` is
positive everywhere; division by a positive function is smooth.
Mathlib has `Real.differentiable_exp`, `Differentiable.fun_sum`, and
`DifferentiableAt.div` with positivity hypotheses. The proof is:
```lean
unfold rowSoftmax softmax
intro v
-- Each entry: exp(M r c) / Σⱼ exp(M r j). Differentiable.div with
-- denom positivity (from Real.exp_pos) + Differentiable.exp.
fun_prop (disch := positivity)
```
**Net axiom delta: −1.** Estimate: 1-2 hours.

### C. Prove `bnIstdBroadcast_diff` from Mathlib calculus

`1/√(bnVar n x + ε)` with `ε > 0`:
- `bnVar n x + ε ≥ ε > 0` by sum-of-squares + positive ε.
- `Real.sqrt` differentiable at positive argument.
- `Inv.inv` (or `1/x`) differentiable at non-zero argument.
- Compose. Mathlib lemmas: `Real.sqrt`'s `hasDerivAt` family +
  `DifferentiableAt.inv`.

**Net axiom delta: −1.** Estimate: 1-2 hours.

### D. Prove `pdivMat_rowIndep` via `fderiv_pi`

VJP.md flagged this earlier as "provable via `fderiv_pi` but
non-trivial — defer." The argument: `fderiv_pi` says the fderiv of a
Pi-valued function decomposes coordinate-wise. For row-independent
functions (each row's output depends only on the corresponding input
row), the decomposition gives the block-diagonal Jacobian directly.

**Net axiom delta: −1.** Estimate: half a day.

### E. (Optional) Prove `pdiv_gelu` and `softmaxCE_grad`

`pdiv_gelu`: gelu is C^∞ (uses `Real.erf` or smooth tanh approximation
depending on the def). `fderiv_pi` decomposes to per-coordinate
scalars; per-coord derivative is `geluScalarDeriv`.

`softmaxCE_grad`: standard log-sum-exp identity unwinding. Uses
`Real.log_div` and `Real.exp` calculus.

**Net axiom delta: −2.** Estimate: 2-3 hours each.

**After A+B+C+D**: 23 → 13 axioms.
**After A+B+C+D+E**: 23 → 11 axioms.

---

## Hard residuals (likely staying axiomatic)

These 8-10 axioms will likely stay regardless of follow-up effort:

- `pdiv_relu` — DL community subgradient convention. Not a theorem of standard analysis.
- `relu_has_vjp` — same reason. Could be derived if `HasVJP.correct` is weakened to "smooth subset only" (project-wide rewrite, not in scope).
- `mhsa_has_vjp_mat` — multi-head attention is structurally a vmap over the head axis; needs a vmap-VJP framework lemma not in scope.
- `patchEmbed_flat_has_vjp` — opaque-codegen interface. The actual computation lives in MLIR; axiomatized that the forward+backward pair is consistent.
- `conv2d_has_vjp3`, `maxPool2_has_vjp3`, `depthwise_has_vjp3` — input-path VJPs through padding boundary / argmax non-smoothness. Like ReLU, these are subgradient conventions.

Below the floor (~10 axioms) the remaining are all of the form "the ML
framework treats this op's gradient by convention X" and aren't
provable from standard analysis without weakening `HasVJP.correct`.

---

## Soundness analysis (carry-forward, still load-bearing)

Two soundness wells from prior attempts. Both are now resolved by the
guarded-ReLU strategy + conditional bilinear rules, but anyone who
considers a future restructure should re-read these.

### 1. `pdiv_relu` (unguarded) is incompatible with `fderiv`-based pdiv

For `n ≥ 2`, take `x = (1, 0) : Vec 2`. The function `relu 2` is **not**
`Differentiable` at `x` — coordinate `y₁ ↦ if y₁ > 0 then y₁ else 0` is
not differentiable at `y₁ = 0`, and `fderiv_pi` says a Pi-valued
function is differentiable iff each coordinate is. So
`fderiv ℝ (relu 2) (1, 0) = 0` (Mathlib junk default), making
`pdiv (relu 2) (1, 0) 0 0 = 0`. The unguarded axiom asserts `1`. **0 ≠ 1.**

The guarded form (`∀ k, x k ≠ 0`) excludes this counterexample. ✅

### 2. Unconditional `pdiv_add` / `_mul` / `_comp` are incompatible with `fderiv`

Counterexample for `pdiv_add`: take `f y = fun _ => abs (y 0)` (not
`DifferentiableAt 0`) and `g y = fun _ => y 0` (= identity). At `x_0 = 0`:

- `f + g` has a kink at 0; `fderiv (f+g)` is junk = 0; LHS = `pdiv (f+g) x 0 0 = 0`.
- `pdiv f x 0 0` is junk = 0; `pdiv g x 0 0 = 1`. RHS = `0 + 1 = 1`.
- Unconditional axiom claims LHS = RHS, i.e., `0 = 1`. ✗

The conditional form (with `DifferentiableAt f x` ∧ `DifferentiableAt g x`)
excludes this counterexample. ✅

---

## Pitfalls (encountered during the migration)

1. **Lambda-form vs CLM-coercion in `rw`.** Passing
   `(reindexCLM σ).differentiableAt` directly to a `pdiv_*` theorem
   inside a rewrite generates a pattern with the `⇑(reindexCLM σ)`
   coercion that doesn't unify with goals containing the lambda form.
   **Fix:** name the diff hypothesis with an explicit lambda type:
   ```lean
   have h_reindex_diff : DifferentiableAt ℝ (fun w idx => w (σ idx)) x :=
     (reindexCLM σ).differentiableAt
   rw [pdiv_mul _ _ _ h_const_diff h_reindex_diff]
   ```

2. **`fun_prop` on division by `Nat`-coerced denominators.** `fun_prop`
   doesn't currently know `HDiv.hDiv` is `Differentiable` in general
   (would need a non-zero denominator hypothesis). Workaround: rewrite
   `x / (↑n : ℝ)` as `x * (↑n)⁻¹` before `fun_prop`. (Hit in `bnCentered`'s
   Diff proof during BN migration.)

3. **`fun_prop` on `Real.sqrt` / `Real.exp` chains.** Doesn't auto-handle
   the positivity-conditional smoothness of `1/√(...)` or
   `exp(...) / Σ exp(...)`. Use manual `DifferentiableAt.inv` +
   `Real.sqrt` lemmas with positivity, or axiomatize the Diff lemma
   for now.

4. **Big surgery via `sed` on Lean files is risky.** When deleting
   parallel-track blocks, find the correct opening and closing line
   boundaries (look at section headers, not line counts) — `sed` doesn't
   know about Lean's nesting. (Hit when deleting `pdivFDMat` block.)

5. **Doc-comment + axiom interaction.** `axiom foo` after `/-- ... -/`
   is fine, but back-to-back `/-- ... -//-- ... -/` (e.g. after
   converting a theorem with its own doc-comment to an axiom and
   adding a new doc-comment) gives "unexpected token '/--'" — Lean
   wants exactly one doc-comment per declaration.

---

## Discipline notes (what the staged plan got right)

- **Tree-green at every commit.** Stages 1, 2a-2e all landed with
  `lake build` passing. The two earlier flip attempts violated this
  and got tangled; staged commits are the way.
- **`#print axioms` after each stage.** Confirmed pure-Mathlib
  closure on key theorems at every chapter migration. This is the
  real success metric, not the raw `^axiom` count.
- **Don't optimize for axiom count alone.** Two cosmetic
  trivial-form commits (`9c03889`, `f94bc03`) were tried and
  reverted as noise — they removed the literal `axiom` declaration
  but added no proof content. The recipe is in
  `feedback_axiom_count_metric.md` (memory).

---

## Time estimate for follow-up

- **A (Attention threading):** 4-6 hours. Mechanical, pattern established.
- **B (rowSoftmax Diff):** 1-2 hours.
- **C (bnIstdBroadcast Diff):** 1-2 hours.
- **D (pdivMat_rowIndep):** half a day.
- **E (pdiv_gelu, softmaxCE_grad):** 2-3 hours each (optional).

**Total to reach the floor:** one focused session of ~8-12 hours.

---

## After the floor

13 axioms (or 11 with stage E). Of those:
- 1 is foundation-related (`pdivMat_rowIndep` if D is skipped, or 0 if landed).
- ~3-4 are subgradient conventions (`pdiv_relu`, `relu_has_vjp`, possibly `mlp_has_vjp` and the analogous CNN/Depthwise/MaxPool).
- ~3 are opaque-codegen interfaces (`patchEmbed_flat_has_vjp`, possibly `mhsa_has_vjp_mat`).
- ~3 are derivative formulas that could be Mathlib-derived with effort (`pdiv_gelu`, `pdiv_softmax`, `softmaxCE_grad`).

Below 11 starts requiring framework-level changes: weakening
`HasVJP.correct` to a "smooth subset" formulation so the subgradient
axioms can be deduced from `pdiv_relu`-like base axioms, and adding a
vmap-VJP framework lemma to derive `mhsa_has_vjp_mat` from the
single-head SDPA proof. Those are project-wide rewrites — separate
multi-week efforts, not this one's continuation.
