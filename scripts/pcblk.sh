#!/bin/bash
cd /topk
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -3
ks() {
  rm -rf /tmp/kb3
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kb3 -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 8 --iters 25 --repeats 1 "${@:3}" >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/kb3/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        k=r['Kernel_Name']
        t=('a' if 'phase_a' in k else 'b' if 'phase_b' in k else 'c' if 'phase_c' in k else 'd' if 'phase_d' in k else None)
        if t: agg[t].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
o={};tot=0
for t in 'abcd':
    v=sorted(agg[t])
    if v: o[t]=sum(v[len(v)//4:])/len(v[len(v)//4:]); tot+=o[t]
print('%6.2f/%8.2f'%(o.get('c',0),tot))
"
}
echo "phase_c / three-kernel total, by phase_c block width."
echo "the candidate count is ~2867 per row, so 1024 threads get 3 elements each"
echo "and the post-histogram barrier synchronises 16 waves to do it."
printf "%6s %9s %17s %17s %17s %17s\n" M N "auto" "256" "512" "1024"
for MN in "512 131072" "256 262144" "128 524288" "64 1048576" "2048 131072" "1024 131072" "4096 131072" "16 524288"; do
  set -- $MN
  printf "%6d %9d %17s %17s %17s %17s\n" $1 $2 "$(ks $1 $2)" \
    "$(ks $1 $2 --phase-c-block 256)" "$(ks $1 $2 --phase-c-block 512)" "$(ks $1 $2 --phase-c-block 1024)"
done
