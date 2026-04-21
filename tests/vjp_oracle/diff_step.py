#!/usr/bin/env python3
"""VJP oracle step-2 loss differ.

Step 1 loss depends only on the forward pass (both sides start from
identical init). Step 2 is the first step whose value depends on the
backward pass + optimizer. Small abs(step-2-Δ) means the Lean
hand-derived VJP agrees with JAX `value_and_grad` at f32 precision.

Usage: diff_step.py <phase3.jsonl> <phase2.jsonl> <name> [tolerance]

Exit 0 on pass (|step 2 delta| < tol), exit 1 on fail.
"""
import json, sys


def load_step(path, step):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.endswith('}'):
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get('kind') == 'step' and d.get('step') == step:
                return d['loss']
    return None


def main():
    if len(sys.argv) < 4:
        print("usage: diff_step.py <phase3.jsonl> <phase2.jsonl> <name> [tolerance]")
        sys.exit(2)
    p3_path, p2_path, name = sys.argv[1], sys.argv[2], sys.argv[3]
    tol = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-4

    l3_1 = load_step(p3_path, 1)
    l2_1 = load_step(p2_path, 1)
    l3_2 = load_step(p3_path, 2)
    l2_2 = load_step(p2_path, 2)

    if None in (l3_1, l2_1, l3_2, l2_2):
        print(f"FAIL  {name:30s} missing step 1 or 2 in one of the traces")
        sys.exit(1)

    d1 = abs(l3_1 - l2_1)
    d2 = abs(l3_2 - l2_2)
    status = "PASS" if d2 < tol else "FAIL"
    print(f"{status}  {name:30s} step1 Δ={d1:.2e}  step2 Δ={d2:.2e}  (tol={tol:.0e})")
    sys.exit(0 if d2 < tol else 1)


if __name__ == "__main__":
    main()
