#!/bin/bash
# F2: A+B fusion, measured fairly for the first time. known_bad.md:440 recorded
# --fuse-ab 1 as faulting on any coop_g > 1 shape and said the +89% ablation
# "must not be trusted". The fault was a layout mismatch, not the fusion.
cd /topk
echo "--- correctness first: the fault this used to take ---"
for MN in "64 131072" "128 524288"; do
  set -- $MN
  printf "  m=%-5d n=%-8d fuse=1: " $1 $2
  ./benchmark_topk --mode verify --m $1 --n $2 --topk 2048 --dist gaussian --seed 0 \
      --fuse-ab 1 2>&1 | grep -E '^VERIFY|^VERDICT|error' | tr '\n' ' '; echo
done
echo
echo "--- A/B ---"
printf "%6s %9s %10s %10s %9s %10s\n" M N "fuse=0" "fuse=1" "ratio" "fb 0/1"
for MN in "1 131072" "16 131072" "16 1048576" "64 262144" "64 1048576" \
          "128 524288" "256 262144" "512 262144" "4096 131072"; do
  set -- $MN
  for F in 0 1; do
    R=$(./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian --seed 0 \
        --warmup 20 --iters 100 --repeats 3 --fuse-ab $F 2>/dev/null | grep '^RESULT')
    W=$(echo "$R" | sed -n 's/.*wall_ms=\([0-9.]*\).*/\1/p')
    B=$(echo "$R" | sed -n 's/.*fallback_rows=\([0-9]*\).*/\1/p')
    if [ "$F" = 0 ]; then W0=$W; B0=$B; else W1=$W; B1=$B; fi
  done
  printf "%6d %9d %10s %10s %9s %10s\n" $1 $2 "$W0" "$W1" \
    "$(awk -v a="$W0" -v b="$W1" 'BEGIN{if(a+0>0)printf "%.3fx",b/a; else printf "n/a"}')" "$B0/$B1"
done
