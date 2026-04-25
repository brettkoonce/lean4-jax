# VJP.md — Atomic Flip Plan for the Next Session

**Branch:** `axiom-elimination` (head: `e033da0`)
**Project axiom count:** 23. **Goal:** drop to ~15 after the flip, ~13 if bonus removals land.

---

## Why this needs its own session

The atomic flip — replacing `axiom pdiv` with the fderiv-grounded `def pdiv` and converting the 6 rule axioms to theorems-with-hypotheses — is mechanically big. The foundation rewrite itself is small (the parallel-track buildup at commits `f5ca56d`-`c29b2c1` already pre-validated every replacement proof), but every existing `rw [pdiv_add]` / `rw [pdiv_mul]` / `rw [pdiv_comp]` / `rw [pdiv_finset_sum]` in the project breaks because those rules now require `DifferentiableAt` hypotheses. Spread across 7 chapter files that's ~25-40 individual proof updates. Confirmed in a flip-attempt sub-branch on 2026-04-25 — too big to bash through at the tail of an already-long session.

## Pre-flight checks

- [ ] On branch `axiom-elimination`, head `e033da0`, tree clean.
- [ ] `lake build` passes (no rebuild needed if already fresh).
- [ ] Read `~/.claude/projects/-home-skoonce-lean-klawd-sandbox-demo-sandbox-lean4-mlir/memory/project_axiom_elimination.md` for full context.
- [ ] Skim the parallel-track proofs in `LeanMlir/Proofs/Tensor.lean` (`pdivFD_*`, `pdivFDMat_*`, `pdivFD3_*`) — these are the proofs we're about to promote.

---

## Stage 1 — Foundation flip in Tensor.lean (single commit)

### A. Replace axiom block (lines 67-114) with proven theorems

The proofs already exist verbatim as `pdivFD_*`. Mechanical promotion:

| Was | Becomes | Proof source |
|---|---|---|
| `axiom pdiv ... : ℝ` | `noncomputable def pdiv ... := fderiv ℝ f x (basisVec i) j` | `pdivFD` def body |
| `axiom pdiv_id` | `theorem pdiv_id` (unconditional) | `pdivFD_id` |
| `axiom pdiv_const` | `theorem pdiv_const` (unconditional) | `pdivFD_const` |
| `axiom pdiv_reindex` | `theorem pdiv_reindex` (unconditional) | `pdivFD_reindex` |
| `axiom pdiv_add` | `theorem pdiv_add` (with `DifferentiableAt` hyps) | `pdivFD_add_of_diff` |
| `axiom pdiv_mul` | `theorem pdiv_mul` (with hyps) | `pdivFD_mul_of_diff` |
| `axiom pdiv_comp` | `theorem pdiv_comp` (with hyps) | `pdivFD_comp_of_diff` |

`basisVec` and `reindexCLM` definitions move from the parallel-track section into the foundation block.

### B. Delete obsolete + rename parallels

After (A), the file has duplicates. Surgery:

1. Delete the original `theorem pdiv_finset_sum` (uses `pdiv_add` unconditionally — won't compile).
2. Rename `pdivFD_finset_sum` → `pdiv_finset_sum`.
3. Update `vjp_comp` / `biPath_has_vjp` / `elemwiseProduct_has_vjp` to take additional `Differentiable ℝ f` and `Differentiable ℝ g` args. The parallel-track work didn't touch these (HasVJP-using framework deferred to flip), so this is genuinely new but the proof structure is straightforward — just thread `(hf_diff x)` / `(hg_diff x)` to the renamed `pdiv_comp` / `pdiv_add` / `pdiv_mul` calls inside `correct`.
4. Delete the original `pdivMat_comp` and `pdivMat_add` (use unconditional rules in their proofs — broken post-flip).
5. Delete `pdivFDMat` def, `pdivFDMat_id`, `pdivFDMat_transpose` (duplicates of `pdivMat` def, `pdivMat_id`, `pdivMat_transpose` which still work post-flip).
6. Replace deleted `pdivMat_comp` / `pdivMat_add` with new versions taking `Differentiable` hypotheses; copy proofs from `pdivFDMat_comp_of_diff` / `pdivFDMat_add_of_diff`.
7. Delete the original `pdivMat_matmul_left_const` / `_right_const` / `_scalarScale` (cascade-broken).
8. Replace with new versions; copy proofs from `pdivFDMat_matmul_left_const` / `_right_const` / `_scalarScale`.
9. Delete `axiom pdivMat_rowIndep` AND `axiom pdivFDMat_rowIndep` — consolidate into one `axiom pdivMat_rowIndep` (the rowIndep proof from `fderiv_pi` is itself non-trivial; defer to a later session).
10. Update `vjpMat_comp` / `biPathMat_has_vjp` to take `Differentiable` args (HasVJPMat-framework analog of step 3).
11. Repeat 4-10 for the 3-tensor block: delete original `pdiv3_comp` / `pdiv3_add`, delete `pdivFD3` def + `pdivFD3_id` / `_add_of_diff` / `_comp_of_diff`, insert new `pdiv3_comp` / `_add` with hyps, update `vjp3_comp` / `biPath3_has_vjp`.

### C. Build check + commit

```bash
lake build LeanMlir.Proofs.Tensor
```

Expected: green. External chapters will fail — that's Stage 2.

Commit message: `Proofs/Tensor: atomic flip — foundation theorem-grounded, framework threaded with Differentiable`.

---

## Stage 2 — Per-chapter migration (one commit per file, tree-green at each)

After Stage 1, every chapter file's proofs that called the (formerly axiomatic) rules unconditionally now fail. Migrate in this order:

### 2a. MLP.lean

Proofs to update:
- `pdiv_dense` (theorem, e146aca) — uses `pdiv_add`, `pdiv_finset_sum`, `pdiv_mul`. Provide diff hypotheses for `(fun b' k => ∑ i', b' i * W i k)` (linear, sum of `(reindexCLM _).differentiableAt`-style projections), `(fun _ k => b k)` (constant), each `(fun b' k => x i' * b' k')` summand (constant × reindex).
- `pdiv_dense_W` (theorem, 61612df) — same shape over flatten bijection.
- `pdiv_dense_b` (theorem) — uses `pdiv_add` on `(constant) + (identity)` factors.
- `dense_has_vjp` — uses now-changed framework `vjp_comp` / `biPath_has_vjp`. May need `Differentiable ℝ (dense W b)` proof (linear, easy).
- `relu_has_vjp` — uses `pdiv_relu` (still axiom); the HasVJP construction itself may break if framework signature changed.

Commit: `Proofs/MLP: thread Differentiable through dense + relu proofs`.

### 2b. CNN.lean

- `conv2d_has_vjp3`, `conv2d_weight_grad_has_vjp` — still axioms. Just touched if framework signatures changed.
- `conv2d_bias_grad_has_vjp` (theorem, 9e9d37e) — uses `pdiv_add`, `pdiv_reindex`, `pdiv_const`. Provide diff hyps for the channel-reindex (linear via `reindexCLM`) and the constant W,x term.

Commit: `Proofs/CNN: thread Differentiable through conv2d_bias_grad`.

### 2c. Depthwise.lean

- `depthwise_bias_grad_has_vjp` (theorem, e033da0) — same as conv2d_bias_grad case.
- Other axioms unchanged.

Commit: `Proofs/Depthwise: thread Differentiable through bias_grad`.

### 2d. BatchNorm.lean — the hardest

`pdiv_bnNormalize` (line 341) uses `pdiv_mul` on `bnCentered × bnIstdBroadcast`. Needs:
- `Differentiable ℝ (bnCentered n)` — linear (`x - mean(x)`), straightforward via Mathlib's `Differentiable.sub` and `Differentiable.const`.
- `Differentiable ℝ (bnIstdBroadcast n ε)` — involves `1/√(var(x) + ε)`. Smooth when `ε > 0`. Requires `Real.sqrt`, division, and BN's variance computation to all be diff. **May need a new helper file `LeanMlir/Proofs/BNSmooth.lean`** with these proofs. Possibly add `ε_pos : ε > 0` hypothesis if not already present.

The 3 BN axioms (`pdiv_bnAffine`, `pdiv_bnCentered`, `pdiv_bnIstdBroadcast`) stay axiomatic — proving them from foundation is the hard work for a future session.

Commit: `Proofs/BatchNorm: thread Differentiable through bnNormalize` (possibly + `add BNSmooth.lean helper`).

### 2e. Residual.lean / SE.lean / LayerNorm.lean / Depthwise.lean / Attention.lean

Each has fewer call sites. Walk through sequentially. Attention.lean uses `pdiv_softmax` (still axiom) plus standard chain-rule pieces.

One commit per file. Tree-green at each commit.

---

## Stage 3 — Bonus axioms (optional, same session if time)

After flip, these become provable from Mathlib calculus:

### 3a. `pdiv_gelu` (LayerNorm.lean)

Statement: `pdiv (gelu n) x i j = if i = j then geluScalarDeriv (x i) else 0`.

`gelu n x = fun i => geluScalar (x i)` is elementwise. Use `fderiv_pi` to decompose: each output coordinate's fderiv is the per-coordinate scalar fderiv. Per-coordinate, `(fun x => geluScalar (x i))` differentiates to `geluScalarDeriv (x i)` via `deriv geluScalar` (which is exactly the def). Then `fderiv_pi` gives the diagonal structure.

Estimate: 30-50 lines.

### 3b. `softmaxCE_grad` (MLP.lean)

Standard fact: `∂(-log softmax(z)_label)/∂z_j = softmax(z)_j - δ(j, label)`. Decompose `crossEntropy c z label = log(∑ k, exp(z k)) - z label` (provable by unfolding `softmax` and `Real.log_div`). Then `pdiv_add` + per-term derivatives via Mathlib's `Real.exp`/`Real.log` calculus.

Estimate: ~80 lines (most of it is showing the log-sum-exp identity).

### 3c. `pdiv_relu` (MLP.lean) — STAYS axiom

ReLU isn't differentiable at 0; the axiom commits to the subgradient convention `if x > 0 then 1 else 0`. This is a DL-community convention, not a theorem of standard analysis. Stays as a narrowly-scoped axiom. **Don't try to prove it.**

---

## Net axiom count

| Stage | Removed | Cumulative count |
|---|---|---|
| Start | — | 23 |
| Stage 1 (flip) | -8 (pdiv + 6 rules + rowIndep consolidation) | 15 |
| Stage 3a (pdiv_gelu) | -1 | 14 |
| Stage 3b (softmaxCE_grad) | -1 | 13 |

What's left after this session (13 axioms):
- 3 BN reduction axioms.
- 3 CNN VJPs (conv2d_has_vjp3, conv2d_weight_grad_has_vjp, maxPool2_has_vjp3).
- 2 Depthwise VJPs.
- 3 Attention axioms.
- 1 ReLU subgradient axiom (kept).
- 1 pdivMat_rowIndep (provable via fderiv_pi but non-trivial — defer).

These 13 are the long tail. Each is multi-day proof work.

---

## Risk areas / known pitfalls

1. **Lambda-form vs CLM-coercion in `rw`**: passing `(reindexCLM σ).differentiableAt` directly to a `pdiv_*` theorem inside a rewrite generates a pattern with the `⇑(reindexCLM σ)` coercion that doesn't unify with goals containing the lambda form. **Fix:** always name the diff hypothesis with an explicit lambda type before the rewrite. Pattern from `pdivFDMat_matmul_left_const`:

   ```lean
   have h_reindex_diff : DifferentiableAt ℝ (fun w idx => w (σ idx)) x :=
     (reindexCLM σ).differentiableAt
   rw [pdiv_mul _ _ _ h_const_diff h_reindex_diff]
   ```

2. **`finProdFinEquiv (finProdFinEquiv.symm x) = x` inside sum binders**: needs `simp_rw [Equiv.apply_symm_apply]`, not `rw`. The `rw` won't reach inside the binder.

3. **`(if c then 1 else 0) * y → if c then y else 0`**: needs `simp_rw [ite_mul, one_mul, zero_mul]` before `Finset.sum_ite_eq`. (Lesson from `conv2d_bias_grad_has_vjp`.)

4. **BN's `Differentiable bnIstdBroadcast`**: smoothness of `1/√(var(x) + ε)` requires `ε > 0`. Check whether BN's framework already carries this or needs an addition.

5. **Soundness gotcha (already hit twice)**: never have `pdiv` as a `def` AND keep the unconditional rules as axioms — that combination is provably unsound. The flip is atomic for soundness reasons.

---

## Success criteria

- [ ] Stage 1 lands as one commit; `lake build LeanMlir.Proofs.Tensor` green.
- [ ] Stages 2a-2e each land as their own commit; `lake build` green at each commit.
- [ ] (Optional) Stages 3a-3b land as one commit each.
- [ ] `grep -rc "^axiom" LeanMlir/Proofs/*.lean | awk -F: '{s+=$2} END {print s}'` shows 13-15 (depending on whether stage 3 ran).
- [ ] No new `sorry` anywhere.
- [ ] `#print axioms dense_weight_grad_correct` (and similar top-level theorems) confirms axiom dependencies are now down to the expected residual list.

---

## Time estimate

- Stage 1: 1-2 hours (mostly mechanical surgery; the proofs already exist).
- Stage 2: 2-4 hours total (each chapter ~30-60 min, BN being the heaviest).
- Stage 3 (optional): 2-3 hours.

**Total: 5-9 hours.** Worth a dedicated weekend session.

---

## After this session

Branch is in shape for review or merge to main. Project axiom count down to the irreducible residual. Future sessions tackle the hard VJPs one at a time — each is its own multi-day project: BN trio, conv2d_has_vjp3, maxPool2_has_vjp3, attention's `pdiv_softmax` + `mhsa_has_vjp_mat`, depthwise weight grad, pdivMat_rowIndep.
