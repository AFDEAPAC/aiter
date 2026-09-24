#!/bin/bash
cd /topk
pip -q install clang-format==23.1.1 2>&1|tail -0
python3 scripts/export_aiter_op.py --aiter /aiter >/dev/null
python3 scripts/export_aiter_op.py --aiter /aiter --check && echo "EXPORT-CHECK-CLEAN"
rm -rf /aiter/aiter/jit/build/module_top*k* /aiter/aiter/jit/module_top*k*.so
cd /aiter && pip -q install -e . 2>&1|tail -0
cd /topk
echo "=== fullverify 350 ==="; STAGE=all python3 -u scripts/fullverify.py 2>&1 | grep -E "FAILURES|m=" | sed "s/^/  /"
echo "=== mod3 191 ==="; python3 -u scripts/mod3.py 2>&1 | grep -E "FAILURES|m=" | sed "s/^/  /"
echo "=== benchmark verify 245 ==="
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -3
fail=0
for m in 1 16 64 128 256 1024 4096; do for n in 131072 262144 524288 1048576 131073 262146 1048577; do for d in gaussian adversarial all_equal uniform inf; do
  r=$(./benchmark_topk --mode verify --m $m --n $n --topk 2048 --dist $d --seed 0 2>&1|grep -o "VERDICT [A-Z]*"|head -1)
  [ "$r" = "VERDICT PASS" ] || { echo "  FAIL m=$m n=$n dist=$d"; fail=1; }
done; done; done
[ $fail = 0 ] && echo "  all 245 PASS"
echo "=== op_tests ==="
pip -q install tabulate 2>&1|tail -0
cd /aiter && timeout 2000 python3 -u op_tests/test_topk_select.py > /topk/log/op7.log 2>&1
echo "  exit=$?"; grep -iE "all hold" /topk/log/op7.log|sed "s/^/  /"
