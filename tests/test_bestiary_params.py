#!/usr/bin/env python3
"""Bestiary param-count regression test.

Runs every `bestiary-<name>` binary, parses `(variant_name, params)` pairs
out of the output, and compares against the golden table in
`tests/bestiary_params.yml`. Fails loudly on any:

  * count mismatch (a pinned variant's param count changed)
  * missing variant (a previously-pinned variant has disappeared)
  * new variant   (a variant appeared that isn't in the golden table)

Why this shape (vs `.totalParams` in Lean): 168 variants across 37
bestiary files; a YAML golden table next to a 50-line harness is a lot
easier to maintain and review than a comparable Lean test file.

Usage:
  tests/test_bestiary_params.py                   # run full suite
  tests/test_bestiary_params.py --update-golden   # regenerate the yaml
                                                  # (use only after deliberate
                                                  # architectural changes)

Exit 0 on all-pass, 1 on any failure.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "tests" / "bestiary_params.yml"
BIN_DIR = REPO / ".lake" / "build" / "bin"

BESTIARY_PAT = re.compile(r"^bestiary-[a-z0-9-]+$")
VARIANT_RE = re.compile(r"^\s*──\s+(.+?)\s+──\s*$")
PARAMS_RE = re.compile(r"^\s*params\s*:\s*(\d+)")


def load_golden(path: Path) -> dict[str, dict[str, int]]:
    """Tiny YAML-subset parser. The golden file only uses:
      top-level `key:` lines, nested `  "quoted key": int` lines, and
      `#` comments. No anchors / flows / multi-docs. Keeping this local
      avoids a PyYAML dep for a test that lives next to diff_step.py."""
    out: dict[str, dict[str, int]] = {}
    current: str | None = None
    for line in path.read_text().splitlines():
        stripped = line.split("#", 1)[0].rstrip()
        if not stripped:
            continue
        if not stripped.startswith(" "):
            current = stripped.rstrip(":")
            out.setdefault(current, {})
            continue
        m = re.match(r'\s+"(.+)":\s*(\d+)\s*$', line.split("#", 1)[0])
        if m and current is not None:
            out[current][m.group(1)] = int(m.group(2))
    return out


def run_binary(bin_path: Path) -> dict[str, int]:
    """Run one bestiary binary and return the {variant_name: params} it printed."""
    proc = subprocess.run([str(bin_path)], capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"{bin_path.name} exited {proc.returncode}\n{proc.stderr[:500]}")
    text = proc.stdout
    variants: dict[str, int] = {}
    current_name: str | None = None
    for line in text.splitlines():
        m = VARIANT_RE.match(line)
        if m:
            current_name = m.group(1).strip()
            continue
        m = PARAMS_RE.match(line)
        if m and current_name is not None:
            variants[current_name] = int(m.group(1))
            current_name = None
    return variants


def dump_golden(data: dict[str, dict[str, int]], path: Path) -> None:
    lines = [
        "# Golden table of NetSpec.totalParams for every bestiary variant.",
        "# Regenerated with `tests/test_bestiary_params.py --update-golden`.",
        "# Update only when a count change is intentional.",
        "#",
        "# Long term: back-fill a `# source:` annotation per row pointing at",
        "# the paper / torchvision / timm / HF reference count the spec aims",
        "# to match. For now these are pinned to whatever the current code",
        "# emits.",
        "",
    ]
    for bin_name in sorted(data):
        lines.append(f"{bin_name}:")
        if not data[bin_name]:
            lines.append("  # (no variants parsed)")
            continue
        for variant, count in data[bin_name].items():
            safe = variant.replace('"', '\\"')
            lines.append(f'  "{safe}": {count}')
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    update = "--update-golden" in sys.argv[1:]

    if not BIN_DIR.exists():
        print(f"FAIL  {BIN_DIR} missing — run `lake build` first", file=sys.stderr)
        return 1

    bins = sorted(p for p in BIN_DIR.iterdir() if BESTIARY_PAT.match(p.name) and p.is_file())
    if not bins:
        print(f"FAIL  no bestiary-* binaries in {BIN_DIR}", file=sys.stderr)
        return 1

    live: dict[str, dict[str, int]] = {}
    for bin_path in bins:
        try:
            live[bin_path.name] = run_binary(bin_path)
        except Exception as e:
            print(f"FAIL  {bin_path.name}: {e}", file=sys.stderr)
            return 1

    if update:
        dump_golden(live, GOLDEN)
        total = sum(len(v) for v in live.values())
        print(f"UPDATED  {GOLDEN} — {len(live)} binaries, {total} variants")
        return 0

    golden = load_golden(GOLDEN)
    failures: list[str] = []

    for bin_name in sorted(set(live) | set(golden)):
        live_map = live.get(bin_name, {})
        gold_map = golden.get(bin_name, {})
        for variant in sorted(set(live_map) | set(gold_map)):
            if variant not in gold_map:
                failures.append(f"NEW      {bin_name} :: {variant!r} = {live_map[variant]} (not in golden)")
            elif variant not in live_map:
                failures.append(f"MISSING  {bin_name} :: {variant!r} expected {gold_map[variant]} (not emitted)")
            elif live_map[variant] != gold_map[variant]:
                failures.append(
                    f"CHANGED  {bin_name} :: {variant!r}  {gold_map[variant]} → {live_map[variant]}"
                )

    if failures:
        for f in failures:
            print(f)
        print(f"\nFAIL  {len(failures)} differences vs {GOLDEN.name}")
        print("If a change is intentional, rerun with --update-golden.")
        return 1

    total = sum(len(v) for v in live.values())
    print(f"PASS  {len(live)} binaries, {total} variants, all counts match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
