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
run() { hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc $2 -o benchmark_topk benchmark_topk.hip.cpp 2>&1 | grep -i error | head -3
        printf "%-46s %9s %9s\n" "$1" "$(pb 4096 131072 ${@:3})" "$(pb 4096 262144 ${@:3})"; }
printf "%-46s %9s %9s\n" "8-byte record, the shipped footprint" "n=131072" "n=262144"
run "shipped"                                    ""
run "4-byte record (wrong, prices the halving)"  "-DABLATE_EPI=7"
run "no copy loop at all"                        "-DABLATE_EPI=1"
echo
echo "fewer candidates instead: margin scales the write linearly"
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1 | grep -i error
for MG in 1.4 1.25 1.15 1.08; do
  printf "%-46s %9s %9s\n" "  --margin $MG (auto is 1.4 here)" "$(pb 4096 131072 --margin $MG)" "$(pb 4096 262144 --margin $MG)"
done
