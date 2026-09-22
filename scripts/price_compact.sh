#!/bin/bash
# Price the two halves of the candidate compaction. Both ablations give WRONG
# results; they exist only to bound what a perfect version of each half is worth.
cd /topk
pbtime() {
  rm -rf /tmp/kt
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kt -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 3 --iters 10 --repeats 1 >/dev/null 2>&1
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
for A in 0 1 2; do
  make clean >/dev/null 2>&1
  EXTRA="-DABLATE_COMPACT=$A" make -j >/dev/null 2>&1 || \
    hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -DABLATE_COMPACT=$A \
      -o benchmark_topk benchmark_topk.hip.cpp 2>&1 | tail -2
  case $A in
    0) L="shipped (full compaction)";;
    1) L="ABLATE 1: fixed-slot write (no prefix arithmetic)";;
    2) L="ABLATE 2: arithmetic only (no ds_write)";;
  esac
  printf "%-50s n=131072 %9s   n=262144 %9s\n" "$L" "$(pbtime 4096 131072)" "$(pbtime 4096 262144)"
done
