#!/bin/bash
cd /topk
mkdir -p /topk/log/attc && rm -rf /topk/log/attc/out
timeout 1500 rocprofv3 --att --att-target-cu 0 --att-simd-select 0xF --att-activity 8 \
  --att-shader-engine-mask 0xF --att-library-path /opt/rocm/lib \
  --kernel-include-regex "phase_c_select_contig" \
  -d /topk/log/attc/out --output-format csv -- \
  ./benchmark_topk --mode time --m 4096 --n 131072 --topk 2048 --dist gaussian \
  --seed 0 --warmup 0 --iters 1 --repeats 1 > /topk/log/attc/run.log 2>&1
echo "exit=$?"
for f in /topk/log/attc/out/stats_ui_output_*.csv; do
  echo "--- phase_c, top stalls ---"
  python3 -c "
import csv,sys
rows=list(csv.DictReader(open('$f')))
rows=[r for r in rows if r['Latency']]
rows.sort(key=lambda r:-int(r['Latency']))
tot=sum(int(r['Latency']) for r in rows)
print('   total latency cycles in trace: %d' % tot)
print('   %-42s %8s %12s %12s %6s' % ('instruction','hits','latency','stall','%'))
for r in rows[:16]:
    print('   %-42s %8s %12s %12s %5.1f%%' % (r['Instruction'][:42], r['Hitcount'], r['Latency'], r['Stall'], 100*int(r['Latency'])/tot))
"
done
