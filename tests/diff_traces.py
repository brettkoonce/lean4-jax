#!/usr/bin/env python3
"""Compare two training-trace .jsonl files for numerical agreement.

Usage:
    python3 tests/diff_traces.py <trace_a.jsonl> <trace_b.jsonl> [--mode=MODE]

Modes (controls numeric tolerance):
    strict       atol=1e-4, rtol=1e-3   (default; same platform, same seed)
    cross-gpu    atol=1e-3, rtol=1e-2   (different GPU HALs — accommodates
                                         IEEE-754 reduction-order drift
                                         between matmul kernels)
    cross-comp   atol=1e-2, rtol=1e-1   (different compilers entirely, e.g.
                                         Lean→IREE vs Lean→JAX+XLA)

Exits 0 on agreement, 1 on any mismatch. Prints a compact summary.

See traces/TRACE_FORMAT.md for the trace contract.
"""
import json
import math
import sys
from pathlib import Path

MODES = {
    "strict":      (1e-4, 1e-3),
    "cross-gpu":   (1e-3, 1e-2),
    "cross-comp":  (1e-2, 1e-1),
}
DEFAULT_MODE = "strict"
# Required numeric fields on every step record.
NUMERIC_REQUIRED = ("loss", "lr")
# Optional numeric fields — compared only when present in BOTH traces.
NUMERIC_OPTIONAL = ("grad_norm", "param_norm")
HEADER_IDENTICAL_FIELDS = (
    "netspec_name", "netspec_hash", "config", "total_params", "dataset",
)


def load_trace(path: Path) -> tuple[dict, list[dict]]:
    """Return (header, [step_records])."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        sys.exit(f"{path}: empty trace")
    header, *steps = records
    if header.get("kind") != "header":
        sys.exit(f"{path}: first record is not a header")
    for i, s in enumerate(steps):
        if s.get("kind") != "step":
            sys.exit(f"{path}: record {i+1} is not a step record")
    return header, steps


def check_headers(ha: dict, hb: dict) -> list[str]:
    """Return list of diff messages; empty if headers match on identity fields."""
    diffs = []
    for key in HEADER_IDENTICAL_FIELDS:
        if ha.get(key) != hb.get(key):
            diffs.append(f"header.{key}: {ha.get(key)!r} vs {hb.get(key)!r}")
    return diffs


def close(a: float, b: float, atol: float, rtol: float) -> bool:
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


def check_steps(sa: list[dict], sb: list[dict], atol: float, rtol: float) -> list[str]:
    diffs = []
    if len(sa) != len(sb):
        diffs.append(f"step count mismatch: {len(sa)} vs {len(sb)}")
        return diffs
    for i, (a, b) in enumerate(zip(sa, sb)):
        # Required fields must be present on both sides.
        for field in NUMERIC_REQUIRED:
            va, vb = a.get(field), b.get(field)
            if va is None or vb is None:
                diffs.append(f"step {i} {field}: missing ({va} vs {vb})")
                continue
            if not close(va, vb, atol, rtol):
                delta = abs(va - vb)
                diffs.append(
                    f"step {i} {field}: {va:.6f} vs {vb:.6f} (delta={delta:.2e})"
                )
        # Optional fields compared only when BOTH sides have them.
        for field in NUMERIC_OPTIONAL:
            va, vb = a.get(field), b.get(field)
            if va is None or vb is None:
                continue  # skip — missing from one side is OK
            if not close(va, vb, atol, rtol):
                delta = abs(va - vb)
                diffs.append(
                    f"step {i} {field}: {va:.6f} vs {vb:.6f} (delta={delta:.2e})"
                )
    return diffs


def parse_args(argv: list[str]) -> tuple[Path, Path, str]:
    args = [a for a in argv[1:] if not a.startswith("--")]
    flags = [a for a in argv[1:] if a.startswith("--")]
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    mode = DEFAULT_MODE
    for f in flags:
        if f.startswith("--mode="):
            mode = f.split("=", 1)[1]
        elif f in ("-h", "--help"):
            print(__doc__); sys.exit(0)
        else:
            print(f"unknown flag: {f}", file=sys.stderr); sys.exit(2)
    if mode not in MODES:
        print(f"unknown mode {mode!r}; valid: {list(MODES)}", file=sys.stderr)
        sys.exit(2)
    return Path(args[0]), Path(args[1]), mode


def main() -> int:
    pa, pb, mode = parse_args(sys.argv)
    atol, rtol = MODES[mode]

    ha, sa = load_trace(pa)
    hb, sb = load_trace(pb)

    header_diffs = check_headers(ha, hb)
    step_diffs   = check_steps(sa, sb, atol, rtol)

    print(f"Comparing (mode={mode}):")
    print(f"  A: {pa}  (phase={ha.get('phase')}, steps={len(sa)})")
    print(f"  B: {pb}  (phase={hb.get('phase')}, steps={len(sb)})")
    print(f"Tolerance: atol={atol}, rtol={rtol}")
    print()

    if not header_diffs and not step_diffs:
        print(f"✓ PASS — {len(sa)} steps agree across phases")
        return 0

    if header_diffs:
        print("✗ HEADER MISMATCH:")
        for d in header_diffs:
            print(f"    {d}")
    if step_diffs:
        print(f"✗ STEP MISMATCHES ({len(step_diffs)}):")
        for d in step_diffs[:20]:      # cap output at 20 diffs
            print(f"    {d}")
        if len(step_diffs) > 20:
            print(f"    ...and {len(step_diffs) - 20} more")
    return 1


if __name__ == "__main__":
    sys.exit(main())
