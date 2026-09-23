#!/bin/bash
cd /topk
echo "=== the pushed aiter tree carries the gate ==="
grep -c ragged_eff /aiter/csrc/kernels/topk_per_row_sampled_kernels.cu
echo
echo "=== local rebased state still builds and verifies ==="
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -3
fail=0
for m in 8 128 1024 4096; do for n in 131072 262146 1048577; do for d in gaussian all_equal inf; do
  r=$(./benchmark_topk --mode verify --m $m --n $n --topk 2048 --dist $d --seed 0 2>&1|grep -o "VERDICT [A-Z]*"|head -1)
  [ "$r" = "VERDICT PASS" ] || { echo "  FAIL m=$m n=$n dist=$d -> $r"; fail=1; }
done; done; done
[ $fail = 0 ] && echo "  36 spot checks PASS"
echo
echo "=== export contract on the rebased local state ==="
pip -q install clang-format==23.1.1 2>&1|tail -0
python3 scripts/export_aiter_op.py --aiter /aiter >/dev/null
python3 scripts/export_aiter_op.py --aiter /aiter --check && echo "  EXPORT-CHECK-CLEAN"
