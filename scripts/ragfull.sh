#!/bin/bash
pip -q install clang-format==23.1.1 2>/dev/null
python3 scripts/export_aiter_op.py --aiter /aiter
python3 scripts/export_aiter_op.py --aiter /aiter --check && echo EXPORT-CHECK-CLEAN
rm -rf /aiter/aiter/jit/build/module_top_k_per_row /aiter/aiter/jit/module_top_k_per_row.so
cd /aiter && pip -q install -e . 2>&1 | tail -0
cd /topk && python3 -u scripts/ragfull.py 2>&1 | grep -E "AITER|MISMATCH" 
echo "=== op_tests ==="
cd /aiter && python3 -m pytest op_tests/test_topk_select.py -q -k "lds_sizing or invariants" 2>&1 | tail -4
