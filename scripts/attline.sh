#!/bin/bash
cd /topk
hipcc -O3 -gline-tables-only -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk_g benchmark_topk.hip.cpp 2>&1|grep -i error|head -3
mkdir -p /topk/log/attl && rm -rf /topk/log/attl/out
timeout 1500 rocprofv3 --att --att-target-cu 0 --att-simd-select 0xF --att-activity 8 \
  --att-shader-engine-mask 0xF --att-library-path /opt/rocm/lib \
  --kernel-include-regex "phase_c_select_contig" \
  -d /topk/log/attl/out --output-format csv -- \
  ./benchmark_topk_g --mode time --m 4096 --n 131072 --topk 2048 --dist gaussian \
  --seed 0 --warmup 0 --iters 1 --repeats 1 > /topk/log/attl/run.log 2>&1
echo "exit=$?"
for f in /topk/log/attl/out/stats_ui_output_*.csv; do
python3 -c "
import csv
rows=[r for r in csv.DictReader(open('$f')) if r.get('Latency')]
rows.sort(key=lambda r:-int(r['Latency']))
tot=sum(int(r['Latency']) for r in rows)
print('phase_c, top stalls WITH source attribution (total %d cycles)'%tot)
print('%-34s %7s %10s %6s  %s'%('instruction','hits','latency','%','source'))
for r in rows[:18]:
    src=(r.get('Source') or '').strip().strip('\"')
    print('%-34s %7s %10s %5.1f%%  %s'%(r['Instruction'][:34],r['Hitcount'],r['Latency'],100*int(r['Latency'])/tot,src[-70:]))
"
done
