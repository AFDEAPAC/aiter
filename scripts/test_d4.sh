#!/bin/bash
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
run() { hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc $2 -o benchmark_topk benchmark_topk.hip.cpp 2>&1 | grep -i error | head -2
        printf "%-48s %9s %9s\n" "$1" "$(pb 4096 131072)" "$(pb 4096 262144)"; }
run "shipped"                                   ""
run "no staging write"                          "-DABLATE_COMPACT=2"
run "no staging write, no drain copy"           "-DABLATE_COMPACT=2 -DABLATE_DRAIN=2"
run "no staging write, no drain CHECK/barrier"  "-DABLATE_COMPACT=2 -DABLATE_DRAIN=4"
run "no drain CHECK/barrier only"               "-DABLATE_DRAIN=4"
echo
echo "reference: nothing passes the threshold = 363.42 / 718.09"
