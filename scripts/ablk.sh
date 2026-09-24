#!/bin/bash
cd /topk
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -3
ks() {
  rm -rf /tmp/ab2
  rocprofv3 --kernel-trace --output-format csv -d /tmp/ab2 -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 8 --iters 25 --repeats 1 "${@:3}" >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/ab2/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        k=r['Kernel_Name']
        t=('a' if 'phase_a' in k else 'b' if 'phase_b' in k else 'c' if 'phase_c' in k else 'd' if 'phase_d' in k else None)
        if t: agg[t].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
o={};tot=0
for t in 'abcd':
    v=sorted(agg[t])
    if v: o[t]=sum(v[len(v)//4:])/len(v[len(v)//4:]); tot+=o[t]
print('%5.2f/%6.2f'%(o.get('a',0),tot))
"
}
echo "phase_a / three-kernel total, by phase_a block width. At S=8192 with 1024"
echo "threads the sample loop runs 2 iterations, so PA_UNROLL=4 cannot apply;"
echo "a narrower block gives it something to unroll."
printf "%6s %9s %15s %15s %15s %15s\n" M N "auto" "a-block 256" "512" "1024"
for MN in "512 131072" "1024 131072" "64 1048576" "256 131072" "128 262144" "2048 131072"; do
  set -- $MN
  printf "%6d %9d %15s %15s %15s %15s\n" $1 $2 "$(ks $1 $2)" \
    "$(ks $1 $2 --phase-a-block 256)" "$(ks $1 $2 --phase-a-block 512)" "$(ks $1 $2 --phase-a-block 1024)"
done
