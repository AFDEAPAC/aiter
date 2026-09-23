#!/bin/bash
cd /topk
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -5
fail=0
for m in 1 16 64 128 256 1024 4096; do for n in 131072 262144 524288 1048576 131073 262146 1048577; do for d in gaussian adversarial all_equal uniform inf; do
  r=$(./benchmark_topk --mode verify --m $m --n $n --topk 2048 --dist $d --seed 0 2>&1 | grep -o "VERDICT [A-Z]*" | head -1)
  [ "$r" = "VERDICT PASS" ] || { echo "  FAIL m=$m n=$n dist=$d -> $r"; fail=1; }
done; done; done
[ $fail = 0 ] && echo "verify: all 245 PASS"
