#!/bin/bash
cd /topk
ks() {
  rm -rf /tmp/dyab
  rocprofv3 --kernel-trace --output-format csv -d /tmp/dyab -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 8 --iters 25 --repeats 1 >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/dyab/**/*kernel_trace.csv',recursive=True):
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
echo "static wbuf against dynamic, interleaved, 3 rounds. The shipped path picks"
echo "512 threads everywhere, so this must be a no-op on it."
printf "%6s %9s %24s %24s %9s\n" M N "static" "dynamic" ratio
for MN in "512 131072" "1024 131072" "4096 131072" "256 131072" "64 1048576" "8 1048576" "128 262144"; do
  set -- $MN; A=(); B=()
  for r in 1 2 3; do
    git stash -q 2>/dev/null; hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
    A+=("$(ks $1 $2)")
    git stash pop -q 2>/dev/null; hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
    B+=("$(ks $1 $2)")
  done
  ma=$(python3 -c "print('%.2f'%(sum([${A[0]},${A[1]},${A[2]}])/3))")
  mb=$(python3 -c "print('%.2f'%(sum([${B[0]},${B[1]},${B[2]}])/3))")
  printf "%6d %9d %24s %24s %9s\n" $1 $2 "${A[*]}" "${B[*]}" \
    "$(python3 -c "print('%.4f'%($mb/$ma))")"
done
