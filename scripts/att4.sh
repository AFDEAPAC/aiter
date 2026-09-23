#!/bin/bash
cd /topk
mkdir -p /topk/log/att4 && rm -rf /topk/log/att4/out
timeout 1500 rocprofv3 --att --att-target-cu 0 --att-simd-select 0xF --att-activity 8 \
  --att-shader-engine-mask 0xF --att-library-path /opt/rocm/lib \
  --kernel-include-regex "phase_a_threshold" \
  -d /topk/log/att4/out --output-format csv -- \
  ./benchmark_topk --mode time --m 4096 --n 131072 --topk 2048 --dist gaussian \
  --seed 0 --warmup 0 --iters 1 --repeats 1 > /topk/log/att4/run.log 2>&1
echo "exit=$?"; du -sh /topk/log/att4/out 2>/dev/null
for f in /topk/log/att4/out/stats_ui_output_*.csv; do
  echo "--- $(basename $f): $(wc -l < $f) lines ---"
  head -1 $f
  tail -n +2 $f | sort -t, -k5 -g -r | head -18
done
