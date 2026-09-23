#!/bin/bash
# The 2x2 my earlier ladder missed. ABLATE_COMPACT removes the staging write but
# the drain still reads buf and writes global; ABLATE_DRAIN removes the drain
# copy but the staging write still runs. Neither alone can look expensive, which
# is exactly what they showed. Crossing them is the only way to see the pair.
cd /topk
pb() {
  rm -rf /tmp/kt
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kt -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 5 --iters 15 --repeats 1 "${@:3}" >/dev/null 2>&1
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
printf "%-14s %-26s %9s %9s\n" compaction drain n=131072 n=262144
for C in 0 2; do
  for D in 0 2; do
    hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -DABLATE_COMPACT=$C -DABLATE_DRAIN=$D \
      -o benchmark_topk benchmark_topk.hip.cpp 2>&1 | grep -i error | head -2
    cl=$([ $C = 0 ] && echo "ds_write kept" || echo "ds_write GONE")
    dl=$([ $D = 0 ] && echo "ds_read+global write kept" || echo "ds_read+global write GONE")
    printf "%-14s %-26s %9s %9s\n" "$cl" "$dl" "$(pb 4096 131072)" "$(pb 4096 262144)"
  done
done
echo
echo "reference: nothing passes the threshold = 363.42 / 718.09"
