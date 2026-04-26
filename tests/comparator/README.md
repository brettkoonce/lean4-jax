# Comparator-based independent kernel re-check

This directory holds an end-to-end verification of 27 theorems from the
proof suite using
[leanprover/comparator](https://github.com/leanprover/comparator) — the
trustworthy-judge tool the Lean Zulip community recommended for projects
that claim "zero project axioms".

`#print axioms` and `lake build` both run via Lean's elaborator, so they
share a trust path with whatever the elaborator did. **comparator
re-runs Lean's kernel typechecker independently and verifies the
transitive axiom closure of each theorem statically**, sandboxed in
landlock, with the option to pile on the
[nanoda](https://github.com/ammkrn/nanoda_lib) kernel as a second
opinion.

## What gets verified

The 27 theorems span foundation rules + every chapter's headline
Jacobian:

| Bucket | Theorems |
|---|---|
| Foundation calculus rules | `pdiv_comp`, `pdiv_add`, `pdiv_mul`, `pdiv_id`, `pdiv_const`, `pdiv_reindex`, `pdiv_finset_sum`, `pdivMat_rowIndep` |
| Mat-level structural rules | `pdivMat_comp`, `pdivMat_matmul_left_const`, `pdivMat_scalarScale`, `pdivMat_transpose` |
| Ch 3 MLP | `pdiv_dense`, `pdiv_dense_W`, `pdiv_dense_b`, `dense_weight_grad_correct`, `dense_bias_grad_correct` |
| Ch 5 BN | `pdiv_bnAffine`, `pdiv_bnCentered`, `pdiv_bnIstdBroadcast`, `pdiv_bnNormalize` (the famous 3-term cancellation) |
| Ch 9 LN+GELU | `pdiv_gelu` |
| Ch 10 Attention | `pdiv_softmax`, `softmaxCE_grad`, `sdpa_back_Q/K/V_correct` |

For each, comparator confirms:

1. The Solution's theorem statement is bit-identical to the Challenge's
   (which has `:= by sorry`) — the prover didn't redefine the goal.
2. The Solution's proof uses **only** `propext`, `Quot.sound`,
   `Classical.choice` — no project axioms anywhere in the transitive
   closure.
3. The Solution typechecks against Lean's kernel, re-run from the
   compiled `.olean` independently of the elaborator.

## Prerequisites (one-time)

- **Linux kernel ≥ 6.10** for Landlock ABI v5. Check with `uname -r`. On
  Ubuntu 24.04, the HWE kernel works:
  ```
  sudo apt install linux-image-generic-hwe-24.04
  sudo reboot
  ```
- **landrun** ≥ v0.1.13 — sandbox runner using Linux Landlock LSM:
  ```
  curl -fsSL -o ~/.local/bin/landrun \
    https://github.com/Zouuup/landrun/releases/latest/download/landrun-linux-amd64
  chmod +x ~/.local/bin/landrun
  ```
- **lean4export** — Lean's `.olean` → text exporter:
  ```
  git clone --depth=1 https://github.com/leanprover/lean4export ~/lean4export
  (cd ~/lean4export && lake build)
  ln -sf ~/lean4export/.lake/build/bin/lean4export ~/.local/bin/lean4export
  ```
- **comparator** itself:
  ```
  git clone --depth=1 https://github.com/leanprover/comparator ~/comparator
  (cd ~/comparator && lake build)
  ln -sf ~/comparator/.lake/build/bin/comparator ~/.local/bin/comparator
  ```

`landrun`, `lean4export`, and `comparator` must be on `PATH` (they
should be after the `ln -sf` lines above if `~/.local/bin` is on PATH).

## Running

```
./run.sh
```

Expected output (full Mathlib decompress on the first run, ~5 minutes;
seconds on subsequent runs):

```
[1/3] lake update (resolving Mathlib)…
[2/3] lake build Solution (outside sandbox)…
[3/3] lake env comparator config.json…
Building Challenge
…
Exporting #[chk_pdiv_comp, …, propext, Quot.sound, Classical.choice, …] from Solution
Running Lean default kernel on solution.
Lean default kernel accepts the solution
Your solution is okay!
```

The `Exporting` line is the audit list. The only allowed axioms are
`propext`, `Quot.sound`, `Classical.choice` (Lean core); everything
else in that list is a `Nat.*` primitive or `String.ofList` (also Lean
core, present in any non-trivial Lean program).

## Notes on the wrapper hack

`run.sh` injects a shim around `landrun` that does two things:

1. Translates single-dash flags `-ldd` and `-add-exec` (which comparator
   emits) into the double-dash form (`--ldd`, `--add-exec`) that
   landrun's CLI parser expects.
2. Prepends `--rox /usr` to the sandbox config — comparator only allows
   `/usr/bin/git`, but `lake` does generic path lookups in `/usr` and
   needs that broader exec permission.

Both look like upstream comparator/landrun polish items; the workaround
keeps the audit reproducible until they land.

## What's *not* covered

- **53 other theorems** in the proof suite (downstream compositions,
  `_diff` smoothness lemmas, `_eq_compose` rewrites). They share the
  same foundation as the 27 above, so methodologically there's nothing
  new to discover — just a lot of mechanical signature-extraction.
- **`noncomputable def` artifacts** like `vit_full_has_vjp`,
  `mhsa_layer_has_vjp_mat`, etc. comparator's `theorem_names` matches
  `Lean.ConstantInfo.thm`, not `defn`. `#print axioms vit_full_has_vjp`
  in `lake env lean -- ` confirms the same allowlist closure for the
  composition shortcuts.
- **nanoda second-kernel re-check.** Set `enable_nanoda: true` in
  `config.json` and add nanoda to PATH (Rust build, ~5 min) for that
  upgrade.
