#!/bin/bash
# A/B phase_a's radix pass count now that CAP_SAFE_FILL gives the candidate
# window 4.94 sigma of room under the cap instead of 3.09. known_bad.md:1476
# falsified `--phase-a-passes 2` because the coarser threshold pushed the
# candidate count past the cap; that premise changed today.
cd /topk
printf "%6s %9s %10s %10s %9s %10s\n" M N "passes3" "passes2" "ratio" "fb 3/2"
for MN in "1 131072" "1 1048576" "16 131072" "16 1048576" "32 524288" \
          "64 262144" "64 1048576" "128 524288" "256 1048576" \
          "1024 262144" "4096 131072" "4096 1048576"; do
  set -- $MN
  for P in 3 2; do
    R=$(./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
        --seed 0 --warmup 20 --iters 100 --repeats 3 --phase-a-passes $P 2>/dev/null \
        | grep '^RESULT')
    W=$(echo "$R" | sed -n 's/.*wall_ms=\([0-9.]*\).*/\1/p')
    F=$(echo "$R" | sed -n 's/.*fallback_rows=\([0-9]*\).*/\1/p')
    if [ "$P" = 3 ]; then W3=$W; F3=$F; else W2=$W; F2=$F; fi
  done
  RAT=$(awk -v a="$W3" -v b="$W2" 'BEGIN{if(a+0>0)printf "%.3fx",b/a; else printf "n/a"}')
  printf "%6d %9d %10s %10s %9s %10s\n" $1 $2 "$W3" "$W2" "$RAT" "$F3/$F2"
done
