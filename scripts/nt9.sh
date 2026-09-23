#!/bin/bash
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
vf() { for d in gaussian adversarial all_equal uniform inf; do
         r=$(./benchmark_topk --mode verify --m $1 --n $2 --topk 2048 --dist $d --seed 0 2>&1 | grep -o "VERDICT [A-Z]*" | head -1)
         [ "$r" = "VERDICT PASS" ] || { echo "FAIL:$d"; return; }
       done; echo "5-dist PASS"; }
run() { hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc $2 -o benchmark_topk benchmark_topk.hip.cpp 2>&1 | grep -i error | head -3
        printf "%-46s %9s %9s  %s\n" "$1" "$(pb 4096 131072)" "$(pb 4096 262144)" "$(vf 4096 131072)"; }
printf "%-46s %9s %9s  %s\n" "" "n=131072" "n=262144" "verify"
run "shipped"                                     ""
run "per-wave copy + non-temporal store"          "-DABLATE_EPI=9"
run "block-wide walk + non-temporal store"        "-DABLATE_EPI=8"
run "shipped again (run-to-run spread)"           ""
