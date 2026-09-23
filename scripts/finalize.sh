#!/bin/bash
cd /topk
echo "=== shipped path still correct? (ABLATE_COMPACT defaults to 0) ==="
make clean >/dev/null 2>&1; make -j 2>&1 | tail -1
for MN in "4096 131072" "16 524288" "128 1048576"; do
  set -- $MN
  for D in gaussian uniform equal inf adversarial; do
    R=$(./benchmark_topk --mode verify --m $1 --n $2 --topk 2048 --dist $D --seed 0 2>/dev/null \
        | grep -E '^VERIFY|^VERDICT' | tr '\n' ' ')
    printf "   m=%-5d n=%-8d %-12s %s\n" $1 $2 "$D" \
      "$(echo $R | sed -n 's/.*rows_fail=\([0-9]*\).*\(PASS\|FAIL\).*/rows_fail=\1 \2/p')"
  done
done
