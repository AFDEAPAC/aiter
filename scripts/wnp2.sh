#!/bin/bash
cd /topk
pip -q install clang-format==23.1.1 2>&1|tail -0
python3 scripts/export_aiter_op.py --aiter /aiter >/dev/null
python3 scripts/export_aiter_op.py --aiter /aiter --check && echo EXPORT-CHECK-CLEAN
rm -rf /aiter/aiter/jit/build/module_top*k* /aiter/aiter/jit/module_top*k*.so
cd /aiter && pip -q install -e . 2>&1|tail -0
cd /topk && python3 -u scripts/wide_np2.py 2>&1 | grep -vE "^\[aiter\]|WARNING|warn" | tail -20
