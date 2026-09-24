#!/bin/bash
cd /topk
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -3
ks() {
  rm -rf /tmp/nte
  rocprofv3 --kernel-trace --output-format csv -d /tmp/nte -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 8 --iters 25 --repeats 1 "${@:3}" >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/nte/**/*kernel_trace.csv',recursive=True):
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
echo "The NT gate fires at M*pitch >= 2^27. Several near-green cells sit just"
echo "under it. Interleaved, 3 rounds, gate-off against forced-on."
printf "%6s %9s %8s %22s %22s %9s\n" M N "M*N" "gate decision" "--nt-gate 1" "ratio"
for MN in "512 131072" "64 1048576" "1024 131072" "256 262144" "128 524288" "256 131072" "128 262144" "2048 131072"; do
  set -- $MN
  W=$(python3 -c "import math;print('2^%d'%round(math.log2($1*$2)))")
  A=(); B=()
  for r in 1 2 3; do A+=("$(ks $1 $2)"); B+=("$(ks $1 $2 --nt-gate 1)"); done
  ma=$(python3 -c "print('%.2f'%(sum([${A[0]},${A[1]},${A[2]}])/3))")
  mb=$(python3 -c "print('%.2f'%(sum([${B[0]},${B[1]},${B[2]}])/3))")
  printf "%6d %9d %8s %22s %22s %9s\n" $1 $2 "$W" "${A[*]}" "${B[*]}" \
    "$(python3 -c "print('%.4f'%($mb/$ma))")"
done
