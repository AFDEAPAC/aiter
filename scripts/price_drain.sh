#!/bin/bash
# Separate the drain's three jobs. The 100.35us (m=4096 n=131072) that vanishes
# when nothing passes the threshold is the whole drain; this says which part.
#   shipped : atomicAdd + LDS read + global write
#   D=1     : atomicAdd + global write        (LDS read removed)
#   D=2     : atomicAdd only                  (LDS read + global write removed)
cd /topk
pb() {
  rm -rf /tmp/kt
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kt -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 5 --iters 15 --repeats 1 >/dev/null 2>&1
  python3 -c "
import csv,glob
v=[]
for f in glob.glob('/tmp/kt/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        if 'phase_b' in r['Kernel_Name']:
            v.append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
v=sorted(v)[len(v)//4:]
print('%.2f'%(sum(v)/len(v)) if v else 'n/a')
"
}
for D in 0 3; do
  hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -DABLATE_DRAIN=$D \
    -o benchmark_topk benchmark_topk.hip.cpp 2>&1 | grep -i error | head -3
  case $D in
    0) L="shipped: atomicAdd + LDS read + global write";;
    1) L="D=1: atomicAdd + global write (no LDS read)";;
    3) L="D=3: reservation atomic replaced by a constant";;
  esac
  printf "%-46s n=131072 %8s   n=262144 %8s\n" "$L" "$(pb 4096 131072)" "$(pb 4096 262144)"
done
echo
echo "reference: nothing passes the threshold at all = 363.52 / 717.40"
