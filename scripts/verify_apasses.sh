#!/bin/bash
# known_bad.md:1483: a pass count is a distribution-sensitive knob, so testing
# it on one distribution proves nothing. The 3-pass column is the control.
cd /topk
printf "%6s %9s %-12s %-26s %-26s\n" M N dist "passes=3" "passes=2"
for MN in "16 524288" "64 262144" "256 1048576" "4096 131072"; do
  set -- $MN
  for D in gaussian uniform equal inf adversarial; do
    for P in 3 2; do
      R=$(./benchmark_topk --mode verify --m $1 --n $2 --topk 2048 --dist $D \
          --seed 0 --phase-a-passes $P 2>/dev/null | grep -E '^VERIFY|^VERDICT' | tr '\n' ' ')
      V=$(echo "$R" | sed -n 's/.*rows_fail=\([0-9]*\).*fallback_rows=\([0-9]*\).*\(PASS\|FAIL\).*/rows_fail=\1 fb=\2 \3/p')
      if [ "$P" = 3 ]; then V3="$V"; else V2="$V"; fi
    done
    printf "%6d %9d %-12s %-26s %-26s\n" $1 $2 "$D" "$V3" "$V2"
  done
done
