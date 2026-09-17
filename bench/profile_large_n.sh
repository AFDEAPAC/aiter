#!/usr/bin/env bash
# Per-kernel rocprofv3 breakdown for three large-N representative shapes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="$ROOT/benchmark_topk"
OUT="$ROOT/log/large_n_profile"
mkdir -p "$OUT"

run_shape() {
  local tag=$1 m=$2 n=$3
  local prof="$OUT/${tag}_m${m}_n${n}"
  rm -rf "$prof"
  mkdir -p "$prof"
  echo "=== $tag M=$m N=$n ==="
  rocprofv3 --kernel-trace --output-format csv -d "$prof" -- \
    "$BENCH" --mode time --m "$m" --n "$n" --topk 2048 \
    --warmup 5 --iters 20 --repeats 1 2>&1 | tail -3
  csv=$(find "$prof" -name '*kernel_trace.csv' | head -1)
  if [[ -z "$csv" ]]; then
    echo "no kernel_trace.csv under $prof" >&2
    return 1
  fi
  python3 - "$csv" "$tag" <<'PY'
import csv, sys
from collections import defaultdict
csv_path, tag = sys.argv[1], sys.argv[2]
by = defaultdict(list)
with open(csv_path, newline="") as f:
    for row in csv.DictReader(f):
        name = row.get("Kernel_Name") or row.get("Name") or ""
        if not name or "phase_" not in name:
            continue
        if "Start_Timestamp" in row and "End_Timestamp" in row:
            ns = float(row["End_Timestamp"]) - float(row["Start_Timestamp"])
        else:
            ns = float(row.get("Duration (ns)") or row.get("Duration_ns") or 0)
        by[name.split("(")[0]].append(ns)
print("kernel_us_median:")
for k in sorted(by):
    xs = sorted(by[k])
    med = xs[len(xs)//2] / 1e3
    print("  %-40s %8.1f us  (n=%d)" % (k, med, len(xs)))
PY
}

[[ -x "$BENCH" ]] || make -C "$ROOT" benchmark_topk

run_shape anchor 4096 131072
run_shape prefill_ref 4096 1048576
run_shape decode 128 65536

echo "profiles under $OUT"
