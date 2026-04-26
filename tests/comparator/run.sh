#!/bin/bash
# Run leanprover/comparator against the 27 theorems in Solution.lean.
#
# What it verifies (per Lean's kernel, independently of the elaborator):
#   1. Solution's theorem statements bit-identically match Challenge's
#      (so the prover didn't redefine the goal).
#   2. Each proof's transitive axiom closure is a subset of
#      [propext, Quot.sound, Classical.choice] — i.e. zero project axioms.
#   3. The proof typechecks against Lean's kernel.
#
# Prerequisites (one-time setup):
#   - Linux kernel ≥ 6.10 (for Landlock ABI v5 — `uname -r`)
#   - landrun: https://github.com/Zouuup/landrun (prebuilt binary)
#   - lean4export: https://github.com/leanprover/lean4export (lake build)
#   - comparator: https://github.com/leanprover/comparator (lake build)
#
# All three binaries must be on PATH. See README.md for a setup recipe.
#
# Usage: ./run.sh

set -e

cd "$(dirname "$0")"

# landrun v0.1.13 doesn't recognize the single-dash flags comparator emits
# (-ldd / -add-exec) and comparator's `--rox /usr/bin/git` is too narrow
# for lake's path lookup. Wrap landrun in a shim that fixes both.
WRAPPER_DIR="$(mktemp -d)"
trap 'rm -rf "$WRAPPER_DIR"' EXIT
cat > "$WRAPPER_DIR/landrun" <<'WRAP'
#!/bin/bash
# Translate single-dash flags + ensure /usr is exec-able.
args=(--rox /usr)
for a in "$@"; do
  case "$a" in
    -ldd)      args+=(--ldd) ;;
    -add-exec) args+=(--add-exec) ;;
    *)         args+=("$a") ;;
  esac
done
exec landrun-real "${args[@]}"
WRAP
chmod +x "$WRAPPER_DIR/landrun"

# Stash the real landrun under a different name and prepend the wrapper.
REAL_LANDRUN="$(command -v landrun)"
ln -sf "$REAL_LANDRUN" "$WRAPPER_DIR/landrun-real"

# Pre-resolve Mathlib via the parent project's manifest, then build
# Solution outside the sandbox (so the sandbox doesn't try to write to
# the parent's .lake/build/).
echo "[1/3] lake update (resolving Mathlib)…"
lake update >/dev/null

echo "[2/3] lake build Solution (outside sandbox)…"
lake build Solution

echo "[3/3] lake env comparator config.json…"
PATH="$WRAPPER_DIR:$PATH" lake env comparator config.json
