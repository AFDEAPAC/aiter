#!/bin/bash
# A2: the (S, margin) pair has never been swept JOINTLY. margin sets the
# expected candidate count (= margin * K) and so the write volume, which the
# anchor ledger prices at 83.68 us of 581 -- writes cost 5.6x reads per byte.
# S sets phase_a's cost AND the threshold precision that lets margin fall:
# safety needs margin * (1 - 3/sqrt(R)) >= 1 with R = margin*K*S/N.
# Sweeping either alone hides the trade. auto is the shipped point.
cd /topk
run() {  # $1=M $2=N $3=margin $4=S  -> "wall fb"
  local A="--margin $3"; local B=""
  [ "$3" = auto ] && A=""
  [ "$4" != auto ] && B="--sample-s $4"
  ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian --seed 0 \
     --warmup 20 --iters 100 --repeats 3 $A $B 2>/dev/null | grep '^RESULT' \
   | sed -n 's/.*wall_ms=\([0-9.]*\).*fallback_rows=\([0-9]*\).*/\1 \2/p'
}
for MN in "4096 131072" "4096 262144" "512 262144"; do
  set -- $MN
  echo "=== M=$1 N=$2 (shipped = auto/auto) ==="
  printf "%8s %8s %10s %6s %8s\n" margin S wall_ms fb "vs auto"
  BASE=$(run $1 $2 auto auto); BW=$(echo $BASE | cut -d' ' -f1)
  printf "%8s %8s %10s %6s %8s\n" auto auto "$BW" "$(echo $BASE|cut -d' ' -f2)" "1.000x"
  for M in 1.20 1.25 1.30 1.55 1.70; do
    for S in auto 11520 16384; do
      R=$(run $1 $2 $M $S); W=$(echo $R|cut -d' ' -f1); F=$(echo $R|cut -d' ' -f2)
      [ -z "$W" ] && continue
      printf "%8s %8s %10s %6s %8s\n" "$M" "$S" "$W" "$F" \
        "$(awk -v a="$BW" -v b="$W" 'BEGIN{printf "%.3fx",b/a}')"
    done
  done
done
