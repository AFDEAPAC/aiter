#!/bin/bash
cd /topk
ka() {
  rm -rf /tmp/ku2
  rocprofv3 --kernel-trace --output-format csv -d /tmp/ku2 -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 5 --iters 25 --repeats 1 >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/ku2/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        k=r['Kernel_Name']
        t=('a' if 'phase_a' in k else 'b' if 'phase_b' in k else 'c' if 'phase_c' in k else 'd' if 'phase_d' in k else None)
        if t: agg[t].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
tot=0
for t in 'abcd':
    v=sorted(agg[t])
    if v: tot+=sum(v[len(v)//4:])/len(v[len(v)//4:])
print('%.2f'%tot)
"
}
echo "three-kernel total, unroll 1 vs 4, interleaved, 3 rounds each"
printf "%6s %9s %26s %26s %8s\n" M N "unroll 1 (3 runs)" "unroll 4 (3 runs)" "ratio"
for MN in "4096 131072" "2048 131072" "1024 131072" "512 131072" "256 131072" "64 1048576" "4096 262144"; do
  set -- $MN
  A=(); B=()
  for r in 1 2 3; do
    hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -DPA_UNROLL=1 -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
    A+=("$(ka $1 $2)")
    hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -DPA_UNROLL=4 -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
    B+=("$(ka $1 $2)")
  done
  ma=$(python3 -c "print('%.2f'%(sum([${A[0]},${A[1]},${A[2]}])/3))")
  mb=$(python3 -c "print('%.2f'%(sum([${B[0]},${B[1]},${B[2]}])/3))")
  printf "%6d %9d %26s %26s %8s\n" $1 $2 "${A[*]}" "${B[*]}" \
    "$(awk -v a=$ma -v b=$mb 'BEGIN{printf "%.4f",b/a}')"
done
