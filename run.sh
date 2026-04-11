#!/bin/bash
# Run a phase-3 trainer with the right env vars set.
#
# Usage:
#   ./run.sh <trainer> [gpu] [backend]
#
#   trainer  - the lean exec name without "-train" (e.g. "resnet34", "mnv4")
#              or with it ("resnet34-train"). Also accepts the binary name
#              ("vit-tiny").
#   gpu      - GPU index to expose (default: 0)
#   backend  - "rocm" (default) or "cuda"
#
# Examples:
#   ./run.sh resnet34
#   ./run.sh mobilenet-v4 1
#   ./run.sh efficientnet-v2 0 rocm
#   ./run.sh vit-tiny 0 cuda
#
# Output is teed to <trainer>.log in the repo root.

set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <trainer> [gpu] [backend]" >&2
  echo "  trainer: e.g. resnet34, mobilenet-v4, vit-tiny, efficientnet-v2" >&2
  exit 1
fi

# Normalize the trainer name. Accept both "resnet34" and "resnet34-train".
trainer="$1"
case "$trainer" in
  *-train) bin="$trainer" ;;
  *)       bin="$trainer-train" ;;
esac

binpath=".lake/build/bin/$bin"
if [ ! -x "$binpath" ]; then
  echo "Binary not found at $binpath. Try: lake build $bin" >&2
  exit 1
fi

gpu="${2:-0}"
backend="${3:-rocm}"

logfile="$(echo "$trainer" | tr '/' '_').log"

case "$backend" in
  rocm) export HIP_VISIBLE_DEVICES="$gpu" ;;
  cuda) export CUDA_VISIBLE_DEVICES="$gpu" ;;
  *)    echo "Unknown backend: $backend (expected rocm or cuda)" >&2; exit 1 ;;
esac

export IREE_BACKEND="$backend"

echo "→ $bin (GPU $gpu, $backend) → $logfile"
exec "$binpath" 2>&1 | tee "$logfile"
