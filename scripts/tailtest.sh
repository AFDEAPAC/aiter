#!/bin/bash
cd /topk
echo "=== export + rebuild ==="
pip -q install clang-format==23.1.1 2>&1|tail -0
python3 scripts/export_aiter_op.py --aiter /aiter >/dev/null
python3 scripts/export_aiter_op.py --aiter /aiter --check && echo "  EXPORT-CHECK-CLEAN"
rm -rf /aiter/aiter/jit/build/module_top*k* /aiter/aiter/jit/module_top*k*.so
cd /aiter && pip -q install -e . 2>&1|tail -0
echo
echo "=== the probe: is the tail read now? ==="
cd /topk && python3 -u scripts/probe72.py 2>&1 | grep -vE "^\[aiter\]|WARNING|warn|NUMA"
echo
echo "=== the 350-case gate ==="
STAGE=all python3 -u scripts/fullverify.py 2>&1 | grep -E "FAILURES|m=|cases checked"
