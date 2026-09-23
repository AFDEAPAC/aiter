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
        printf "%-48s %9s %9s\n" "$1" "$(pb 4096 131072)" "$(pb 4096 262144)"; }
printf "%-48s %9s %9s\n" "" "n=131072" "n=262144"
run "shipped"                                      ""
run "one block-wide walk, unaligned head"          "-DABLATE_EPI=5"
run "one block-wide walk, head forced to 128 B"    "-DABLATE_EPI=6"
run "no copy loop at all"                          "-DABLATE_EPI=1"
echo
echo "--- what the write actually costs, measured at the memory system ---"
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1 | grep -i error | head -2
for A in 1 0; do
  hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -DABLATE_EPI=$A -o benchmark_topk benchmark_topk.hip.cpp 2>&1 | grep -i error
  rm -rf /tmp/pm
  rocprofv3 -i scripts/pmc_mem.txt --output-format csv -d /tmp/pm -- \
    ./benchmark_topk --mode time --m 4096 --n 131072 --topk 2048 --dist gaussian \
    --seed 0 --warmup 2 --iters 3 --repeats 1 >/dev/null 2>&1
  python3 -c "
import csv,glob,sys
acc={}
for f in glob.glob('/tmp/pm/**/*counter_collection.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        if 'phase_b' not in r['Kernel_Name']: continue
        acc[r['Counter_Name']]=acc.get(r['Counter_Name'],0.0)+float(r['Counter_Value'])
n=3+2
wr=acc.get('TCC_EA0_WRREQ_64B_sum',0)/n; wt=acc.get('TCC_EA0_WRREQ_sum',0)/n
print('  EPI=$A  write reqs %10.0f  of which 64B %10.0f (%5.1f%%)  bytes %7.2f MB'%(wt,wr,100*wr/wt if wt else 0,(wr*64+(wt-wr)*32)/1e6))
"
done
