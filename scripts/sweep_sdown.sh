#!/bin/bash
# The joint (S, margin) sweep only moved S UP (11520, 16384). phase_a's gather
# is now measured to scale cleanly with S at large M (18.77 / 35.00 / 52.86us at
# S = 4096 / 8192 / 16384, m=4096 n=131072), so going DOWN is worth 16us of
# phase_a -- if the coarser threshold does not cost more in candidates.
# attempts.tsv 24 ("sample count back to S=4096") rejected this before
# CAP_SAFE_FILL and before auto_margin; re-asking with the full pipeline timed.
cd /topk
run() {
  local M=$1 N=$2; shift 2
  ./benchmark_topk --mode time --m $M --n $N --topk 2048 --dist gaussian --seed 0 \
    --warmup 20 --iters 100 --repeats 3 "$@" 2>/dev/null | grep '^RESULT' \
  | sed -n 's/.*wall_ms=\([0-9.]*\).*fallback_rows=\([0-9]*\).*/\1 \2/p'
}
for MN in "4096 131072" "4096 262144" "1024 131072" "512 262144"; do
  set -- $MN
  B=$(run $1 $2); BW=$(echo $B|cut -d' ' -f1)
  echo "=== M=$1 N=$2  (auto = $B) ==="
  printf "   %-10s %10s %6s %9s\n" S wall_ms fb "vs auto"
  for S in 4096 6144 8192 12288; do
    R=$(run $1 $2 --sample-s $S); W=$(echo $R|cut -d' ' -f1); F=$(echo $R|cut -d' ' -f2)
    [ -z "$W" ] && continue
    printf "   %-10s %10s %6s %9s\n" "$S" "$W" "$F" \
      "$(awk -v a=$BW -v b=$W 'BEGIN{printf "%.3fx",b/a}')"
  done
done
