#!/bin/bash
# The shipped coop_g table is non-monotone at N=131072: M=256 takes 8, M=512 and
# M=1024 drop to 2, M=2048/4096 return to 8. Those two dips are also the worst
# cells of the M>=512 band there (55.3% and 53.0% of the pipe101 floor). The
# anchor ledger marks "phase_b above its own floor" (+32.56us, 5.6% of wall) as
# the one recoverable item, and coop_g is the knob on it.
cd /topk
printf "%6s %9s %7s %10s %9s %s\n" M N coop_g wall_ms ratio note
for MN in "512 131072" "1024 131072" "512 262144" "4096 262144" "4096 524288" "256 262144"; do
  set -- $MN
  BW=""
  for G in 0 1 2 4 8 16; do
    FLAG=""; [ "$G" != 0 ] && FLAG="--coop-g $G"
    R=$(./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian --seed 0 \
        --warmup 20 --iters 100 --repeats 3 $FLAG 2>/dev/null | grep '^RESULT')
    W=$(echo "$R" | sed -n 's/.*wall_ms=\([0-9.]*\).*/\1/p')
    [ -z "$W" ] && continue
    if [ "$G" = 0 ]; then BW=$W; NOTE="shipped"; else NOTE=""; fi
    printf "%6d %9d %7s %10s %9s %s\n" $1 $2 "$([ $G = 0 ] && echo auto || echo $G)" "$W" \
      "$(awk -v a="$BW" -v b="$W" 'BEGIN{if(a+0>0)printf "%.3fx",b/a; else printf "-"}')" "$NOTE"
  done
  echo
done
