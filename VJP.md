# VJP.md — Attempt #3: Foundation Flip with Guarded ReLU

**Branch:** `axiom-elimination` (head: `c5ec77a`, 19 axioms)
**Goal:** make the foundation **sound**, not lower the axiom count.

> The count reduction is a side effect of doing the work right. Optimizing
> count directly is what gave us attempts #1 and #2 (both reverted), plus
> the trivial-form cosmetic commits 9c03889/f94bc03 (also reverted).

---

## Why this is attempt #3

The 7 foundation axioms in `Proofs/Tensor.lean` (`pdiv` + 6 rules) claim more
than they can deliver. Two prior attempts to swap `axiom pdiv` for an
`fderiv`-grounded `def pdiv` derived `False` because:

1. **`pdiv_relu` is incompatible with `fderiv`-based pdiv at non-smooth
   multi-D points.** `relu 2` at `x = (1, 0)` is not `Differentiable`
   (the second coordinate has a kink at 0), so `fderiv ℝ (relu 2) (1, 0) = 0`
   (Mathlib junk default), giving `pdiv (relu 2) (1, 0) 0 0 = 0`. But the
   axiom says `1`. **Contradiction.**

2. **Unconditional `pdiv_add` / `pdiv_mul` / `pdiv_comp` are incompatible
   with `fderiv`-based pdiv when summands aren't differentiable.** The
   axioms claim the chain/sum/product rule holds for all `f, g`. fderiv
   only obeys these rules at points where `f` and `g` are
   `DifferentiableAt`. At non-smooth points the axioms force `0 = 1`.

The prior plans (and this file as it existed before today) tried to flip
the foundation without resolving these issues, then ran into them
mid-flip and reverted.

---

## Strategy: (a) guarded `pdiv_relu`

Three soundness fixes were identified. Picking **(a)** because it has the
smallest blast radius:

- **(a) Guard `pdiv_relu` with `(∀ k, x k ≠ 0)`.** Axiom only constrains
  `pdiv (relu n) x` at points where `relu n` is differentiable. No
  contradiction with `fderiv`. ✅
- **(b) Redefine ReLU as a smooth approximation.** Disconnects from real
  ML semantics; codegen would no longer match the implementation.
- **(c) Weaken `HasVJP.correct` to "smooth subset only".** Project-wide
  framework rewrite. Touches every consumer of `HasVJP`.

(a) is the right choice for now. (c) is the principled long-term fix but
its scope justifies its own dedicated effort once the foundation is
proven sound.

### Cost of (a): `relu_has_vjp` becomes axiomatic

Currently `relu_has_vjp` is a `def` whose `correct` proof uses
`pdiv_relu` (`MLP.lean:222-226`). With guarded `pdiv_relu`, the proof
no longer goes through for arbitrary `x` — at non-smooth points there's
no fact about `pdiv (relu n) x` to substitute. So the def becomes an
axiom. Net axiom count for ReLU: the same axiom, plus one.

This is a feature, not a bug. The axiom now honestly admits that ReLU's
backward at the kink is a *convention* (the subgradient `1` at `x = 0`),
not a theorem. That's how every ML framework already treats it.

`maxPool2` has the same issue but it already lives behind an axiomatic
`maxPool2_has_vjp3`, so no new axiom needed.

---

## Pre-flight

- [ ] `git status` clean, on `axiom-elimination`.
- [ ] `lake build` green at start.
- [ ] Skim the parallel-track `pdivFD_*` proofs in `LeanMlir/Proofs/Tensor.lean`
      — these are the smooth-case pdiv theorems already proven against
      Mathlib's `fderiv`. They're the bodies of the new theorems.

---

## Stage 1 — Foundation flip in `Proofs/Tensor.lean`

**One commit, no chapter files touched.** External chapters will be
broken at the end of this commit; that's expected.

### A. Replace axiom block (lines 67-114)

| Was | Becomes | Body source |
|---|---|---|
| `axiom pdiv f x i j : ℝ` | `noncomputable def pdiv f x i j := fderiv ℝ f x (basisVec i) j` | new def |
| `axiom pdiv_id` | `theorem pdiv_id` (unconditional) | `pdivFD_id` |
| `axiom pdiv_const` | `theorem pdiv_const` (unconditional) | `pdivFD_const` |
| `axiom pdiv_reindex` | `theorem pdiv_reindex` (unconditional) | `pdivFD_reindex` |
| `axiom pdiv_add` | `theorem pdiv_add` w/ `DifferentiableAt` hyps | `pdivFD_add_of_diff` |
| `axiom pdiv_mul` | `theorem pdiv_mul` w/ `DifferentiableAt` hyps | `pdivFD_mul_of_diff` |
| `axiom pdiv_comp` | `theorem pdiv_comp` w/ `DifferentiableAt` hyps | `pdivFD_comp_of_diff` |

`basisVec` and `reindexCLM` move from the parallel-track section into
the foundation block.

### B. Surgery on duplicates and downstream framework

The parallel `pdivFD_*` proofs become redundant after (A). Specifically:

1. `pdiv_finset_sum` (currently a theorem using unconditional `pdiv_add`):
   restate with `Differentiable` hypothesis on each summand. Body
   reuses `pdivFD_finset_sum`.
2. Delete `pdivFD`, `pdivFD_id`, `pdivFD_const`, `pdivFD_reindex`,
   `pdivFD_add_of_diff`, `pdivFD_mul_of_diff`, `pdivFD_comp_of_diff`,
   `pdivFD_finset_sum` — now identical to the post-flip foundation.
3. Same for the `pdivFDMat` block — consolidate with `pdivMat`.
4. Same for the `pdivFD3` block — consolidate with `pdiv3`.
5. `vjp_comp`, `biPath_has_vjp`, `elemwiseProduct_has_vjp` (and their
   Mat / 3-tensor analogs) take new `Differentiable ℝ f` and
   `Differentiable ℝ g` arguments — thread through their existing proof
   bodies which now call the conditional `pdiv_comp` etc.
6. Consolidate `axiom pdivFDMat_rowIndep` and `axiom pdivMat_rowIndep`
   into one `axiom pdivMat_rowIndep` (rowIndep proof from `fderiv_pi`
   is itself a multi-day project; defer).

### C. Build check + commit

```bash
lake build LeanMlir.Proofs.Tensor   # must be green
```

Chapter files will be red — that's Stage 2.

Commit message:
```
Proofs/Tensor: foundation flip — pdiv as fderiv-grounded def, bilinear
rules now require Differentiable hypotheses
```

---

## Stage 2 — Per-chapter migration

**One commit per chapter file. Tree-green at each commit.** This is
the discipline that attempts #1 and #2 lacked.

For each chapter, the work is the same shape:
- Every `rw [pdiv_add]` / `rw [pdiv_mul]` / `rw [pdiv_comp]` /
  `rw [pdiv_finset_sum]` now needs `Differentiable` hypotheses.
- The functions in question are linear (sums, projections, scalar
  multiplications) so the hypotheses are easy via
  `(reindexCLM σ).differentiableAt`,
  `Differentiable.const`, `Differentiable.sub`, etc.
- Threaded `Differentiable` arguments to `vjp_comp` /
  `biPath_has_vjp` calls.

Migration order (smallest → largest):

1. **`MLP.lean`** (also: guard `pdiv_relu`, axiomatize `relu_has_vjp`)
2. **`CNN.lean`** (12 existing-proof theorems get `Differentiable` threading)
3. **`Depthwise.lean`** (parallel to CNN)
4. **`BatchNorm.lean`** (heavier — `bnIstdBroadcast` Diff requires `ε > 0`)
5. **`Residual.lean`**
6. **`SE.lean`**
7. **`LayerNorm.lean`**
8. **`Attention.lean`**

### MLP.lean migration — special handling for ReLU

Two changes specific to this file:

1. Guard `pdiv_relu`:
   ```lean
   axiom pdiv_relu (n : Nat) (x : Vec n)
       (h_smooth : ∀ k, x k ≠ 0)
       (i j : Fin n) :
       pdiv (relu n) x i j =
         if i = j then (if x i > 0 then 1 else 0) else 0
   ```
2. Convert `relu_has_vjp` from `def` to `axiom`:
   ```lean
   axiom relu_has_vjp (n : Nat) : HasVJP (relu n)
   ```
   (We lose the proof; the existence of the VJP at non-smooth points is
   asserted, matching the ML-conventional subgradient routing.)

Other MLP theorems (`pdiv_dense`, `pdiv_dense_W`, `pdiv_dense_b`,
`dense_has_vjp`) need `Differentiable` threaded through their existing
proofs.

---

## Stage 3 — Bonus axiom removals (optional, follows naturally)

After stages 1+2, the foundation is `fderiv`-grounded and the bilinear
rules properly hypothesized. Three additional axioms become provable
because we can now compose foundation theorems with Mathlib's calculus:

### 3a. `pdiv_gelu` (LayerNorm.lean)

`gelu n x = fun i => geluScalar (x i)` is C^∞. `fderiv_pi` decomposes
to per-coordinate, and `geluScalar`'s derivative is `geluScalarDeriv`
by `fderiv_pi` again. ~30-50 lines.

### 3b. `softmaxCE_grad` (MLP.lean)

`crossEntropy c z label = log(∑ k, exp(z k)) - z label`. Standard
`Real.log_div`, `Real.exp` calculus. ~80 lines.

### 3c. `pdiv_bnIstdBroadcast` (BatchNorm.lean)

`1/√(σ²+ε)` is C^∞ when `ε > 0`. `Real.hasDerivAt_sqrt` + `HasDerivAt.inv`
+ chain rule. May need a small `BNSmooth.lean` helper file. ~100 lines.

---

## Net axiom count

| Stage | Removed | Added | Cumulative |
|---|---|---|---|
| Start | — | — | 19 |
| Stage 1 (flip) | -7 (`pdiv` + 6 rules) | — | 12 |
| Stage 2 (MLP) | — | +1 (`relu_has_vjp` axiom) | 13 |
| Stage 3a (`pdiv_gelu`) | -1 | — | 12 |
| Stage 3b (`softmaxCE_grad`) | -1 | — | 11 |
| Stage 3c (`pdiv_bnIstdBroadcast`) | -1 | — | 10 |

**Floor: 10 axioms** if all of stage 3 lands.

What's left (the irreducible residual):
- `pdiv_relu`, `relu_has_vjp` — ReLU subgradient convention.
- `maxPool2_has_vjp3` — max-pool subgradient routing.
- `pdiv_softmax`, `mhsa_has_vjp_mat`, `patchEmbed_flat_has_vjp` — softmax /
  attention; need `Real.exp` calculus + smooth dispatch (could be future work).
- `pdivMat_rowIndep` — provable via `fderiv_pi` but deferred.

---

## Discipline (the actual lessons from attempts #1 + #2)

1. **Tree-green at every commit.** The flip + per-chapter migration is
   atomic *for soundness*; that does **not** mean it has to be one
   commit. One commit per **stage** (foundation, then per-chapter).
   Both prior attempts violated this and got tangled.

2. **`#print axioms` after each stage.** Check the dependency closure
   of a key theorem (e.g., `dense_weight_grad_correct`) is the expected
   set. Catches accidental additional axiom dependencies early.

3. **Don't optimize for axiom count alone.** The trivial-form trick
   (replacing `axiom foo_has_vjp` with `def foo_has_vjp { backward :=
   Σ pdiv · dy; correct := rfl }`) reduces the count without adding any
   proof content. We tried it; it was reverted. The count is a
   downstream signal, not the objective.

4. **Don't try to combine stage 3 with stage 2.** Stage 3's bonus
   removals depend on the foundation flip being done *cleanly*. Land
   stages 1 and 2 first, verify tree-green, then attempt 3.

---

## Soundness analysis (carry-forward from prior attempts)

### 1. `pdiv_relu` (unguarded) contradicts fderiv-based `pdiv` in multi-D

For `n ≥ 2`, take `x = (1, 0) : Vec 2`. The function `relu 2` is **not**
`Differentiable` at this point — coordinate `y_1 ↦ if y_1 > 0 then y_1 else 0`
is not differentiable at `y_1 = 0`, and Mathlib's `fderiv_pi` says a
Pi-valued function is differentiable iff each coordinate is. So
`fderiv ℝ (relu 2) (1, 0) = 0` (junk default), making
`pdiv (relu 2) (1, 0) 0 0 = 0`. The unguarded axiom asserts `1`. **0 ≠ 1.**

The guarded form (`∀ k, x k ≠ 0`) excludes this counterexample. ✓

### 2. Unconditional `pdiv_add` / `_mul` / `_comp` contradict fderiv

Counterexample for `pdiv_add`: take `f y = fun _ => abs (y 0)` (not
DiffAt 0) and `g y = fun _ => y 0` (= identity). At `x_0 = 0`:

- `f + g` has a kink at 0; `fderiv (f+g)` is junk = 0; LHS = `pdiv (f+g) x 0 0 = 0`.
- `pdiv f x 0 0` is junk = 0; `pdiv g x 0 0 = 1`. RHS = `0 + 1 = 1`.
- Unconditional axiom claims LHS = RHS, i.e., `0 = 1`. ✗

The conditional form (with `DifferentiableAt f x` ∧ `DifferentiableAt g x`)
excludes this counterexample. ✓

---

## Risk areas / known pitfalls

1. **Lambda-form vs CLM-coercion in `rw`.** Passing
   `(reindexCLM σ).differentiableAt` directly to a `pdiv_*` theorem
   inside a rewrite generates a pattern with the `⇑(reindexCLM σ)`
   coercion that doesn't unify with goals containing the lambda form.
   **Fix:** name the diff hypothesis with an explicit lambda type
   before the rewrite:
   ```lean
   have h_reindex_diff : DifferentiableAt ℝ (fun w idx => w (σ idx)) x :=
     (reindexCLM σ).differentiableAt
   rw [pdiv_mul _ _ _ h_const_diff h_reindex_diff]
   ```

2. **`finProdFinEquiv` round-trip inside sum binders.** Needs
   `simp_rw [Equiv.apply_symm_apply]`, not `rw` (which won't reach
   inside the binder).

3. **`(if c then 1 else 0) * y → if c then y else 0`.** Needs
   `simp_rw [ite_mul, one_mul, zero_mul]` before `Finset.sum_ite_eq`.
   (Lesson from `conv2d_bias_grad_has_vjp`.)

4. **BN's `Differentiable bnIstdBroadcast`.** Smoothness of `1/√(σ²+ε)`
   requires `ε > 0`. Check whether BN's framework already carries this
   as an invariant or whether the migration adds it.

5. **Soundness gotcha (already hit twice):** never have `pdiv` as a
   `def` AND keep the bilinear rules unconditional — that combination
   is provably unsound. The flip is atomic *for soundness*; the
   bilinear rules MUST become conditional in the same commit.

---

## Time estimate

- Stage 1: 1-2 hours (mostly mechanical surgery; bodies already exist
  as `pdivFD_*`).
- Stage 2: 2-4 hours total (each chapter ~30-60 min, BN heaviest).
- Stage 3: 2-3 hours (optional bonus).

**Total: 5-9 hours.** Worth a dedicated weekend session.

---

## After this session

Branch is in shape for review or merge to `main`. Foundation is
`fderiv`-grounded, bilinear rules properly hypothesized, axiom count
down to the irreducible residual (~10-13 axioms). Future sessions
tackle the remaining axioms one at a time: `pdivMat_rowIndep` via
`fderiv_pi`, `pdiv_softmax` via `Real.exp` calculus, the attention
trio.
