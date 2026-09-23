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
build() { hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc $1 -o benchmark_topk benchmark_topk.hip.cpp 2>&1 | grep -i "error" | head -3; }
build "";                        printf "%-52s %8s %8s\n" "shipped" "$(pb 4096 131072)" "$(pb 4096 262144)"
build "-DABLATE_COMPACT=3";      printf "%-52s %8s %8s\n" "C=3: the guarded block and the drain never run" "$(pb 4096 131072)" "$(pb 4096 262144)"
build "";                        printf "%-52s %8s %8s\n" "shipped, but --margin 0.02 (nothing passes)" "$(pb 4096 131072 --margin 0.02)" "$(pb 4096 262144 --margin 0.02)"
