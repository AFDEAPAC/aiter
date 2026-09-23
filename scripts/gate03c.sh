#!/bin/bash
cd /topk
ks() {
  rm -rf /tmp/g3c
  rocprofv3 --kernel-trace --output-format csv -d /tmp/g3c -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 8 --iters 25 --repeats 1 >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/g3c/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        if 'phase_c' in r['Kernel_Name']:
            agg['c'].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
v=sorted(agg['c']); print('%.2f'%(sum(v[len(v)//4:])/len(v[len(v)//4:])) if v else 'n/a')
"
}
echo "Gate 0.3b -- phase_c: how much of it is block_select_lds?"
echo "null-kernel ramp at these block counts is 1.39-1.44 us."
printf "%6s %9s %10s %12s %10s %10s\n" M N "shipped" "no select" "select" "ramp share"
for MN in "8 524288" "8 1048576" "32 1048576" "128 131072" "128 524288" "512 131072" "256 131072"; do
  set -- $MN
  hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
  A=$(ks $1 $2)
  hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -DABLATE_PC=1 -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
  B=$(ks $1 $2)
  printf "%6d %9d %10s %12s %10s %10s\n" $1 $2 "$A" "$B" \
    "$(awk -v a=$A -v b=$B 'BEGIN{printf "%.2f",a-b}')" \
    "$(awk -v a=$A 'BEGIN{printf "%.0f%%",100*1.4/a}')"
done
