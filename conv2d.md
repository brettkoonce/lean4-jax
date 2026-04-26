# conv2d.md — depthwise input-VJP plan (Phase 2)

**Phase 1 (conv2d) status:** LANDED on `colslab-vmap-framework`.
~470 LOC, pure-Mathlib closure verified. See **"Phase 1 lessons"** below
for the recipe.

**Phase 2 (depthwise) status:** LANDED on `colslab-vmap-framework`
(Apr 2026, this session). ~470 LOC including 30 LOC of new outer-Σ-co
collapse logic. Pure-Mathlib closure verified via `#print axioms`
(only `propext`, `Classical.choice`, `Quot.sound`). The proof template
mirrored conv2d directly — single-shot build, no Lean errors, in part
because the three private helpers from Phase 1 (now public in CNN.lean)
generalized cleanly.

Axiom count: **7 → 6**. The 6 remaining are pure "framework convention"
axioms:
- 3 ReLU subgradient (`pdiv_relu`, `relu_has_vjp`, `mlp_has_vjp`)
- 1 maxPool2 subgradient (`maxPool2_has_vjp3`)
- 2 patchEmbed opaque-codegen (`patchEmbed_flat_has_vjp`,
  `patchEmbed_flat_diff`)

---

## What's different from conv2d

| Aspect | conv2d | depthwise |
|---|---|---|
| Output dims | `oc` separate from `ic` | output `c` = input `c` |
| Kernel shape | `(oc, ic, kH, kW)` | `(c, kH, kW)` |
| Channel sum | `Σ c kh kw` (3-level) | `Σ kh kw` (2-level, no `Σ c`) |
| Per-summand pdiv | `W o c kh kw * indicator` | `W co kh kw * indicator` |
| Channel match | `c = ci` (extracted via `Prod.mk.inj`) | `co = ci` (the output channel matches input) |

Concrete forward (Depthwise.lean:67):
```lean
depthwiseConv2d W b x ch hi wi
  = b ch + ∑ kh kw, W ch kh kw *
      (if hpad : pad(kh, hi, kw, wi) then x ch ⟨hh-pH, _⟩ ⟨ww-pW, _⟩ else 0)
```

The key observation: the FORWARD output channel `ch` IS the input
channel for `x`. There's no cross-channel mixing. So in `pdiv F v idx_in
idx_out` (with idx_in = (ci, hi_in, wi_in), idx_out = (co, ho, wo)):
- For `co ≠ ci`: pdiv = 0 (the W and indicator both reference `co`, but
  the v-eval is at `co` too — for the indicator to match `idx_in = (ci, ...)`,
  we'd need the v-eval index's first component = ci, but it's co. So `co ≠ ci`
  forces no match).
- For `co = ci`: pdiv = `∑ kh kw, W ci kh kw * (if pad ∧ kh+ho-pH = hi.val ∧
  kw+wo-pW = wi.val then 1 else 0)`.

---

## Reusable helpers from Phase 1 (CNN.lean:145–242)

**Important:** the three helpers are currently `private` in CNN.lean.
For depthwise, either:
- (a) change `private` → public (recommended; they're generic enough to
  live as utility lemmas), or
- (b) move them to `Tensor.lean` near the existing `pdiv_*` rules (better
  long-term home), or
- (c) re-declare them in Depthwise.lean (worst — code duplication).

The helpers (signatures, all `Vec n` / `Vec m` parameterized — no
conv2d-specific types):

```lean
/-- Differentiability of `if hpad : P then v(σ hpad) else 0`. -/
private lemma differentiableAt_pad_eval {n : Nat} (P : Prop) [Decidable P]
    (σ : P → Fin n) (v : Vec n) :
    DifferentiableAt ℝ (fun y : Vec n => if h : P then y (σ h) else (0 : ℝ)) v

/-- pdiv of a per-output dependent if-eval-or-zero family. -/
private lemma pdiv_pi_pad_eval {n m : Nat}
    (P : Fin m → Prop) [∀ k, Decidable (P k)]
    (σ : (k : Fin m) → P k → Fin n)
    (v : Vec n) (idx_in : Fin n) (idx_out : Fin m) :
    pdiv (fun (v' : Vec n) (k' : Fin m) =>
            if h : P k' then v' (σ k' h) else (0 : ℝ))
          v idx_in idx_out =
    if h : P idx_out then (if σ idx_out h = idx_in then (1 : ℝ) else 0) else 0

/-- pdiv of `c_const * pad-eval` family — the conv2d per-summand pattern. -/
private lemma pdiv_const_mul_pi_pad_eval {n m : Nat}
    (c_const : Fin m → ℝ)
    (P : Fin m → Prop) [∀ k, Decidable (P k)]
    (σ : (k : Fin m) → P k → Fin n)
    (v : Vec n) (idx_in : Fin n) (idx_out : Fin m) :
    pdiv (fun (v' : Vec n) (k' : Fin m) =>
            c_const k' *
            (if h : P k' then v' (σ k' h) else (0 : ℝ)))
          v idx_in idx_out =
    c_const idx_out *
    (if h : P idx_out then (if σ idx_out h = idx_in then (1 : ℝ) else 0) else 0)
```

For depthwise, instantiate `c_const idx_out := W ci kh kw` (or the
appropriate decoded form), and `σ idx_out hpad := finProdFinEquiv (...)`
encoding the input position.

---

## Recommended formula form: `(ho, wo)` loop, NOT `(kh, kw)`

The conv2d formula uses `∑ co ho wo, body` (output-position loop). For
depthwise, drop the `Σ co` (since output channel = input channel = ci):

```lean
noncomputable def depthwiseConv2d_input_grad_formula {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (dy : Tensor3 c h w) : Tensor3 c h w :=
  fun ci hi wi =>
    ∑ ho : Fin h, ∑ wo : Fin w,
      let pH := (kH - 1) / 2
      let pW := (kW - 1) / 2
      let kh_nat := hi.val + pH - ho.val
      let kw_nat := wi.val + pW - wo.val
      if hpad : ho.val ≤ hi.val + pH ∧ kh_nat < kH ∧
                 wo.val ≤ wi.val + pW ∧ kw_nat < kW then
        W ci ⟨kh_nat, hpad.2.1⟩ ⟨kw_nat, hpad.2.2.2⟩ * dy ci ho wo
      else 0
```

**Why `(ho, wo)` not `(kh, kw)`:** the `(ho, wo)` form falls out
directly from the pdiv expansion. The `(kh, kw)` form (which the original
conv2d.md proposed) would require a partial bijection between `Fin h`
and `Fin kH` — ~50 LOC of `Finset.sum_bij` bookkeeping to no benefit.
Phase 1 switched mid-proof to `(ho, wo)` after this realization.

---

## Proof skeleton (mirror conv2d's structure)

The full conv2d proof is at `LeanMlir/Proofs/CNN.lean` lines 250–706.
Adapt with these substitutions:

| conv2d | depthwise |
|---|---|
| `Kernel4 oc ic kH kW` | `DepthwiseKernel c kH kW` |
| `oc * ic * kH * kW` flat | (no flat — kernel is 3D, same shape as `Tensor3`) |
| `∑ c kh kw` triple sum | `∑ kh kw` double sum |
| `W o(k') c kh kw` | `W o(k') kh kw` (no separate c index) |
| Per-(c, kh, kw) summand | Per-(kh, kw) summand |
| Channel match: `c = ci` extracted via Prod.mk.inj on inner pair | Channel match: built into the auto-`co = c` since v-eval uses output channel directly |

### Step-by-step

```lean
noncomputable def depthwise_has_vjp3 ... : HasVJP3 (depthwiseConv2d W b) where
  backward := fun _x dy => depthwiseConv2d_input_grad_formula W dy
  correct := by
    intro x dy ci hi wi
    set idx_in : Fin (c * h * w) :=
      finProdFinEquiv (finProdFinEquiv (ci, hi), wi) with hidx_in

    -- Step 1: per-(idx_in, idx_out) pdiv lemma (UN-collapsed).
    have h_pdiv : ∀ idx_out : Fin (c * h * w),
        pdiv (fun v' : Vec (c * h * w) =>
                Tensor3.flatten (depthwiseConv2d W b (Tensor3.unflatten v')))
              (Tensor3.flatten x) idx_in idx_out =
        ∑ kh : Fin kH, ∑ kw : Fin kW,
          W (ohw_o := decoded) kh kw *
            (if hpad : pad(kh, ohw_hi, kw, ohw_wi) then
              (if idx_in = finProdFinEquiv (finProdFinEquiv
                  (ohw_o, ⟨hh - pH, hpad.2.1⟩), ⟨ww - pW, hpad.2.2.2⟩)
                then 1 else 0)
             else 0) := by
      intro idx_out
      set ohw_wi := ...; set ohw_hi := ...; set ohw_o := ...
      -- Decompose F = b + var_part (same as conv2d).
      rw [show F = (fun v' k' => b_part v' k' + var_part v' k') from ...]
      have h_b_diff := differentiableAt_const _
      have h_lin_diff : ... := by
        rw [differentiableAt_pi]; intro k'
        apply DifferentiableAt.fun_sum; intro kh _
        apply DifferentiableAt.fun_sum; intro kw _
        apply DifferentiableAt.mul (differentiableAt_const _)
        unfold Tensor3.unflatten
        exact differentiableAt_pad_eval _ (...) _
      rw [pdiv_add _ _ _ h_b_diff h_lin_diff]
      rw [show pdiv b_part _ _ _ = 0 from pdiv_const _ _ _ _]
      rw [zero_add]
      -- Distribute over kh-sum.
      rw [show var_part = (fun v'' k' => ∑ kh, summand_kh) from rfl]
      have h_kh_diff : ∀ khh ∈ univ, DifferentiableAt ℝ (summand_kh khh) _ := ...
      rw [pdiv_finset_sum _ _ _ h_kh_diff]
      congr 1; ext khh
      -- Distribute over kw-sum.
      rw [show summand_kh khh = (fun v''' k'' => ∑ kw, summand_kw khh kw) from rfl]
      have h_kw_diff : ∀ kww ∈ univ, DifferentiableAt ℝ (summand_kw khh kww) _ := ...
      rw [pdiv_finset_sum _ _ _ h_kw_diff]
      congr 1; ext kww
      -- Per-(kh, kw) summand: factor as W (constant) * (dite).
      -- Note: σ for depthwise references `ohw_o` (the output channel = input channel).
      rw [show summand_kw khh kww = (fun v k' => 
            (W-factor k') * (if hpad : pad then v(σ(...)) else 0)) from by
        funext v k'; unfold Tensor3.unflatten; rfl]
      rw [pdiv_const_mul_pi_pad_eval ...]

    -- Step 2: closing collapse.
    show depthwiseConv2d_input_grad_formula W dy ci hi wi =
      ∑ co ho wo, pdiv3 (depthwiseConv2d W b) x ci hi wi co ho wo * dy co ho wo
    unfold depthwiseConv2d_input_grad_formula pdiv3
    -- Note: LHS has no Σ co. RHS has Σ co ho wo.
    -- Collapse Σ co on co = ci first.
    rw [Finset.sum_eq_single ci ?_ ?_].symm
    rotate_left
    · -- For co ≠ ci: ∑ ho wo, pdiv * dy = 0 because pdiv = 0.
      intro co _ hco_ne
      apply Finset.sum_eq_zero; intro ho _
      apply Finset.sum_eq_zero; intro wo _
      rw [h_pdiv (finProdFinEquiv (finProdFinEquiv (co, ho), wo))]
      simp only [Equiv.symm_apply_apply]
      -- Show ∑ kh kw, W co kh kw * (if pad ∧ idx_in = flat(co, ...) then 1 else 0) = 0.
      -- For co ≠ ci, idx_in = flat(co, ...) is false (first component mismatch),
      -- so all indicators are 0.
      apply Finset.sum_eq_zero; intro kh _
      apply Finset.sum_eq_zero; intro kw _
      rw [show ((let pH := ...; if hpad : pad then (if idx_in = ... then 1 else 0) else 0) : ℝ) = 0 from ?_]
      · ring
      by_cases hpad : pad
      · rw [dif_pos hpad, if_neg ?_]
        intro h_eq
        rw [hidx_in] at h_eq
        have h_inj := finProdFinEquiv.injective h_eq
        have h_inj_pair := Prod.mk.inj h_inj
        have h_inj_inner := finProdFinEquiv.injective h_inj_pair.1
        have h_inj_inner_pair := Prod.mk.inj h_inj_inner
        exact hco_ne h_inj_inner_pair.1.symm
      · rw [dif_neg hpad]
    · intro hni; exact absurd (Finset.mem_univ ci) hni
    -- Now: LHS = ∑ ho wo, pdiv(at co=ci) * dy ci ho wo
    apply Finset.sum_congr rfl; intro ho _
    apply Finset.sum_congr rfl; intro wo _
    -- Per-(ho, wo): same recipe as conv2d (pull dy out, congr 1, h_indicator,
    -- by_cases back_cond, Σ kh kw collapse via Finset.sum_eq_single).
    -- The h_indicator now has 2 conjuncts (kh+ho = hi+pH ∧ kw+wo = wi+pW),
    -- not 3 (no `c = ci` since c isn't a sum index).
    sorry
```

The closing's per-(ho, wo) inner equality is structurally a 2-level
collapse (Σ kh kw) instead of 3-level (Σ c kh kw). Otherwise the same
shape as conv2d's.

---

## Pitfalls (from Phase 1 — pre-mitigate)

1. **`fun_prop` does not handle `dite`.** Use the helpers; do NOT try
   to `fun_prop` directly on the body containing `if hpad : pad then ...`.
   Phase 1 hit `No theorems found for dite` on the very first attempt.

2. **`congr 1` does NOT split Prod equality automatically.** When proving
   `finProdFinEquiv (a, b) = finProdFinEquiv (c, d)`, `congr 1` opens to
   `(a, b) = (c, d)` but a SECOND `congr 1` may not split into `a = c`
   and `b = d`. Use `Prod.mk.inj` on the hypothesis side (extracting from
   `h_inj_pair := Prod.mk.inj h_inj`) and explicit substitution
   `rw [← h_c, ← h_hi, ← h_wi]` on the goal side.

3. **Beta reduction needed before `dif_pos`/`dif_neg`.** After `funext v'`
   the goal has form `(fun v'' k' => ...) v' idx_out = ...`. Lean does NOT
   auto-beta-reduce. Use `show (if h : P idx_out then v' (σ idx_out h) else 0) = ...`
   to force beta, then `rw [dif_pos hP]`.

4. **`Fin.ext_iff.mp` direction.** `h_inj_inner_pair.2 : hi = ⟨..., _⟩`
   gives `Fin.ext_iff.mp this : hi.val = (⟨_⟩).val = kh.val + ho.val - pH`.
   Do NOT add `.symm` if the target is `hi.val = ...`. Phase 1 spent two
   build-error rounds chasing this.

5. **`set` abbreviations don't propagate into bound function bodies.**
   `set ohw_o := ... with hohw_o` rewrites the OUTER goal but NOT the
   `fun v'' k' => ... (decoded k')` lambda body. Inside the lambda, k' is
   bound and the decoding is fresh. For depthwise, set the abbreviations
   (ohw_o, ohw_hi, ohw_wi) but expect to handle decoded forms separately
   inside the function body.

6. **Indicator simplification: prove a non-dependent form first.** Phase 1
   used a sub-lemma `h_indicator` that converts the dependent-if indicator
   to a non-dependent conjunction `c = ci ∧ kh+ho = hi+pH ∧ kw+wo = wi+pW`.
   For depthwise: drop the `c = ci` conjunct (channel is forced by the
   structure). The conjunction becomes 2 conjuncts.

7. **Σ over a Fin not in scope: `intro hni; exact absurd (Finset.mem_univ ci) hni`.**
   `Finset.sum_eq_single` has a side condition "if the chosen point is not
   in the Finset, the result is the default". For `Finset.univ`, this is
   always vacuous; the explicit dispatch is `intro hni; exact absurd
   (Finset.mem_univ ci) hni`.

---

## Estimated LOC and time

- **Formula def:** ~12 LOC.
- **Per-coord pdiv lemma (`h_pdiv`):** ~150 LOC (about 60% of conv2d's
  due to one fewer sum level).
- **Closing collapse:** ~150 LOC (similar to conv2d, plus an outer
  `Σ co` collapse on `co = ci` at the start).
- **Helper migration** (private → public, or move to Tensor.lean): ~5 LOC
  of edits.

**Total: ~300–400 LOC.**

**Time estimate: 4–6 focused hours.** Faster than Phase 1 (12–16 hours)
because:
- The proof template is locked in (mirror conv2d).
- The 3 helpers are reused (no new diff/pdiv infrastructure).
- One fewer sum level (Σ kh kw vs Σ c kh kw).

---

## When to pick this back up

The right time:
- A clean ~half-day window.
- Phase 1's CNN.lean (lines 145–706) recently reviewed.
- You're OK changing 3 `private lemma` → public (or moving them to
  Tensor.lean).

The bad time: as a side-quest. The pitfall #4 alone (Fin.ext_iff
direction) can eat an hour if you're not focused.

---

## What landing this buys

- Axiom count: **7 → 6**.
- The 6 remaining are pure framework-convention axioms:
  - 3 ReLU subgradient (`pdiv_relu`, `relu_has_vjp`, `mlp_has_vjp`).
  - 1 maxPool2 subgradient (`maxPool2_has_vjp3`).
  - 2 patchEmbed opaque-codegen (`patchEmbed_flat_has_vjp`,
    `patchEmbed_flat_diff`).
- Book's "verified MobileNet" claim becomes literal (no caveat about
  the depthwise input-VJP axiom).

The 4 ReLU/maxPool axioms need a `HasVJP.correct` weakening to
"smooth subset only" — a project-wide rewrite, separate multi-week
effort. The 2 patchEmbed axioms could come down by de-opaquing
`patchEmbed_flat` (~500 LOC). Neither is in scope here.

---

## Phase 1 lessons (the conv2d recipe — preserved here for Phase 2 reference)

The crux insight that made Phase 1 work was the helper trio for the
dependent-if pattern:

1. **`fun_prop` failure is the first wall.** The conv2d body has
   `if hpad : pad then v(σ hpad) else 0` — a `dite` whose then-branch
   uses the `hpad` proof to construct a Fin. `fun_prop` has no `dite`
   rules; it errors with `No theorems found for dite`. Workaround:
   `differentiableAt_pad_eval` does the by_cases manually.

2. **The pi-fderiv extraction.** For `pdiv (fun v k' => g v k')` at
   `(idx_in, idx_out)`, you need `fderiv (fun v => g v idx_out) v
   (basisVec idx_in)`. Phase 1 used `fderiv_apply` (Mathlib lemma) for
   this conversion — same pattern as `LayerNorm.lean:182` for `pdiv_gelu`.
   Then `by_cases` on the pad condition + `ContinuousLinearMap.proj`
   for the eval-CLM in the pad-true branch.

3. **The product structure: `(constant in v) * (dite-in-v)`.** The
   per-summand has this exact shape. `pdiv_const_mul_pi_pad_eval`
   bundles `pdiv_mul` + `pdiv_const(constant) = 0` +
   `pdiv_pi_pad_eval(dite)` into one rule application.

4. **The closing's structure (per-(co, ho, wo) inner equality).** After
   substituting `h_pdiv`, for each `(co, ho, wo)` you need
   `LHS_inner = (∑ c kh kw, W * indicator) * dy co ho wo`. The recipe:
   - Pull dy out of the LHS if-true branch via the lemma
     `(if h then a*b else 0) = (if h then a else 0) * b`.
   - `congr 1` on the `* dy` factor.
   - Convert the indicator (dep-if of Fin equality) to a non-dep
     conjunction via a sub-lemma `h_indicator`.
   - `by_cases back_cond`; in pos case, triple `Finset.sum_eq_single` for
     `(c, kh, kw)`; in neg case, all sums = 0 (via Nat reasoning + omega).

5. **`omega` covers the Nat arithmetic.** The Fin bound proofs and Nat
   subtractions (`hi.val + pH - ho.val < kH` etc.) all close via `omega`,
   given the right hypotheses in context (`kh.isLt`, `hi.isLt`, etc.).

6. **Don't bother with `IsBoundedBilinearMap`.** Conv2d IS bilinear, and
   in principle `IsBoundedBilinearMap.hasFDerivAt` gives joint diff for
   free. But you'd still need to extract the per-input partial derivatives
   AND prove the bilinear bound (a calc on operator norm). Net: comparable
   length to the direct pdiv chain. Phase 1 considered this approach and
   rejected it.
