#!/bin/bash
cd /topk
ks() {
  rm -rf /tmp/kfo
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kfo -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 8 --iters 25 --repeats 1 >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/kfo/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        k=r['Kernel_Name']
        t=('a' if 'phase_a' in k else 'b' if 'phase_b' in k else 'c' if 'phase_c' in k else 'd' if 'phase_d' in k else None)
        if t: agg[t].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
o={};tot=0
for t in 'abcd':
    v=sorted(agg[t])
    if v: o[t]=sum(v[len(v)//4:])/len(v[len(v)//4:]); tot+=o[t]
print('%6.2f/%8.2f'%(o.get('a',0),tot))
"
}
echo "phase_a / three-kernel total, pass 0 folded into the sample load"
printf "%6s %9s %16s %16s\n" M N "PA_FOLD=0" "PA_FOLD=1"
for MN in "4096 131072" "2048 131072" "1024 131072" "512 131072" "256 262144" "128 524288" "64 1048576" "16 524288"; do
  set -- $MN
  hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -DPA_FOLD=0 -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -3
  A=$(ks $1 $2)
  hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -DPA_FOLD=1 -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -3
  B=$(ks $1 $2)
  printf "%6d %9d %16s %16s\n" $1 $2 "$A" "$B"
done
echo
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -3
for d in gaussian adversarial all_equal uniform inf; do
  printf "  verify m=4096 n=131072 %-12s %s\n" $d "$(./benchmark_topk --mode verify --m 4096 --n 131072 --topk 2048 --dist $d --seed 0 2>&1 | grep -o 'VERDICT [A-Z]*' | head -1)"
done
