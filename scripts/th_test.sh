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
        printf "%-52s %9s %9s\n" "$1" "$(pb 4096 131072)" "$(pb 4096 262144)"; }
printf "%-52s %9s %9s\n" "" "n=131072" "n=262144"
run "shipped"                                        ""
run "th=inf, everything else identical"              "-DABLATE_TH=1"
run "th=inf + no staging write"                      "-DABLATE_TH=1 -DABLATE_COMPACT=2"
echo "  (the old --margin 0.02 reference read 363.42 / 718.09, but it also"
echo "   moves rank and cap, so it was never the same configuration)"
