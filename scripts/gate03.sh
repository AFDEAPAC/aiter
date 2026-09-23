#!/bin/bash
cd /topk
ks() {
  rm -rf /tmp/g3
  rocprofv3 --kernel-trace --output-format csv -d /tmp/g3 -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 8 --iters 25 --repeats 1 "${@:3}" >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/g3/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        k=r['Kernel_Name']
        t=('a' if 'phase_a' in k else 'c' if 'phase_c' in k else None)
        if t: agg[t].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
o={}
for t in 'ac':
    v=sorted(agg[t]); o[t]=sum(v[len(v)//4:])/len(v[len(v)//4:]) if v else 0
print('%6.2f %6.2f'%(o['a'],o['c']))
"
}
echo "Gate 0.3 -- what the per-block floor of phase_a and phase_c is made of."
echo "null-kernel ramp at these block counts is 1.39-1.44 us (gate 0.1)."
echo
printf "%6s %9s  %-14s %-14s %-14s %-14s\n" M N "shipped" "no select" "S=256" "S=256,no sel"
for MN in "8 524288" "8 1048576" "32 1048576" "128 131072" "128 524288"; do
  set -- $MN
  hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
  A=$(ks $1 $2); C=$(ks $1 $2 --sample-s 256 --s-rule 0)
  hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -DABLATE_PA=1 -DABLATE_PC=1 -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
  B=$(ks $1 $2); D=$(ks $1 $2 --sample-s 256 --s-rule 0)
  printf "%6d %9d  a/c %-10s a/c %-10s a/c %-10s a/c %-10s\n" $1 $2 "$A" "$B" "$C" "$D"
done
