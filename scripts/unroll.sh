#!/bin/bash
cd /topk
ka() {
  rm -rf /tmp/kur
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kur -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 5 --iters 20 --repeats 1 >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/kur/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        k=r['Kernel_Name']
        t=('a' if 'phase_a' in k else 'b' if 'phase_b' in k else 'c' if 'phase_c' in k else 'd' if 'phase_d' in k else None)
        if t: agg[t].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
o={};tot=0
for t in 'abcd':
    v=sorted(agg[t])
    if v: o[t]=sum(v[len(v)//4:])/len(v[len(v)//4:]); tot+=o[t]
print('%6.2f %8.2f'%(o.get('a',0),tot))
"
}
echo "phase_a and the three-kernel total, by unroll depth on the sample load"
printf "%6s %9s %16s %16s %16s %16s\n" M N "unroll 1" "unroll 2" "unroll 4" "unroll 8"
for MN in "256 131072" "512 131072" "1024 131072" "2048 131072" "4096 131072" "256 262144" "128 524288" "64 1048576"; do
  set -- $MN
  out=""
  for U in 1 2 4 8; do
    hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -DPA_UNROLL=$U -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
    out="$out $(printf '%16s' "$(ka $1 $2)")"
  done
  printf "%6d %9d%s\n" $1 $2 "$out"
done
