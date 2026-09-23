#!/bin/bash
cd /topk
mkdir -p /topk/log/att
rm -rf /topk/log/att/out
timeout 1500 rocprofv3 --att --att-target-cu 0 --att-simd-select 0xF --att-activity 8 \
  --att-library-path /opt/rocm/lib \
  --kernel-include-regex "phase_a_threshold" \
  -d /topk/log/att/out --output-format csv -- \
  ./benchmark_topk --mode time --m 8 --n 524288 --topk 2048 --dist gaussian \
  --seed 0 --warmup 0 --iters 1 --repeats 1 > /topk/log/att/run.log 2>&1
echo "exit=$?"
grep -iE "error|abort|Traceback|no such" /topk/log/att/run.log | head -5
echo "--- produced ---"
find /topk/log/att/out -maxdepth 4 -type f 2>/dev/null | head -14
du -sh /topk/log/att/out 2>/dev/null
