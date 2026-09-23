#!/bin/bash
cd /topk
echo "=== 1. benchmark verify, 245 cases ==="
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -3
fail=0
for m in 1 16 64 128 256 1024 4096; do for n in 131072 262144 524288 1048576 131073 262146 1048577; do for d in gaussian adversarial all_equal uniform inf; do
  r=$(./benchmark_topk --mode verify --m $m --n $n --topk 2048 --dist $d --seed 0 2>&1|grep -o "VERDICT [A-Z]*"|head -1)
  [ "$r" = "VERDICT PASS" ] || { echo "  FAIL m=$m n=$n dist=$d -> $r"; fail=1; }
done; done; done
[ $fail = 0 ] && echo "  all 245 PASS"
echo
echo "=== 2. aiter op_tests ==="
pip -q install tabulate 2>&1|tail -0
cd /aiter && timeout 2000 python3 -u op_tests/test_topk_select.py > /topk/log/op5.log 2>&1
echo "  exit=$?"; grep -iE "all hold" /topk/log/op5.log | sed "s/^/  /"
echo
echo "=== 3. style ==="
pip -q install black ruff==0.16.0 2>&1|tail -0
black --check . 2>&1|tail -2|sed "s/^/  /"; ruff check . 2>&1|tail -1|sed "s/^/  /"
echo
echo "=== 4. fallback rate ==="
cd /topk && timeout 2400 python3 -u bench/fbrate.py 2>&1 | grep -iE "TOTAL fallback" | sed "s/^/  /"
