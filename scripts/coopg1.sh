#!/bin/bash
# Feasibility gate for fusing phase_b into phase_c: fusion needs the whole row
# owned by one block (coop_g=1) so the candidates never leave LDS. That drops the
# readers of a row from 8 blocks to 1, so the question is whether one block can
# still saturate the read.
cd /topk
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1 | grep -i error | head -3
ks() {
  rm -rf /tmp/kg
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kg -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 5 --iters 15 --repeats 1 ${3:+--coop-g $3} >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/kg/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        k=r['Kernel_Name']
        t=('a' if 'phase_a' in k else 'b' if 'phase_b' in k else 'c' if 'phase_c' in k else None)
        if t: agg[t].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
o=[];tot=0
for t in 'abc':
    v=sorted(agg[t])
    if not v: continue
    m=sum(v[len(v)//4:])/len(v[len(v)//4:]); tot+=m; o.append('%s=%7.2f'%(t,m))
print('  '.join(o)+'  TOTAL=%8.2f'%tot)
"
}
for MN in "4096 131072" "4096 262144" "128 131072" "16 131072"; do
  set -- $MN
  echo "m=$1 n=$2"
  for G in "" 1 2 4 8; do
    printf "  coop_g=%-6s %s\n" "${G:-auto}" "$(ks $1 $2 $G)"
  done
done
