#!/bin/bash
cd /topk
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -3
ks() {
  rm -rf /tmp/kng
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kng -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 5 --iters 20 --repeats 1 >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/kng/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        k=r['Kernel_Name']
        t=('a' if 'phase_a' in k else 'b' if 'phase_b' in k else 'c' if 'phase_c' in k else 'd' if 'phase_d' in k else None)
        if t: agg[t].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
o={};tot=0
for t in 'abcd':
    v=sorted(agg[t])
    if v: o[t]=sum(v[len(v)//4:])/len(v[len(v)//4:]); tot+=o[t]
print('a=%7.2f b=%8.2f c=%7.2f  sum=%8.2f   a%%=%4.1f c%%=%4.1f'%(
  o.get('a',0),o.get('b',0),o.get('c',0),tot,
  100*o.get('a',0)/tot if tot else 0, 100*o.get('c',0)/tot if tot else 0))
"
}
echo "the near-green cluster: where does the time go, and is the NT gate on?"
printf "%6s %9s %6s %5s  %s\n" M N "M*N" "gate" "per-kernel"
for MN in "256 262144" "512 262144" "2048 131072" "512 131072" "64 1048576" "128 524288" "1024 131072" "256 131072"; do
  set -- $MN
  W=$(( $1 * $2 ))
  G=$([ $W -ge 134217728 ] && echo ON || echo off)
  printf "%6d %9d %6s %5s  %s\n" $1 $2 "2^$(python3 -c "import math;print(int(math.log2($W)))")" "$G" "$(ks $1 $2)"
done
