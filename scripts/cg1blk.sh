#!/bin/bash
cd /topk
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -3
ks() {
  rm -rf /tmp/kb2
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kb2 -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 5 --iters 15 --repeats 1 "${@:3}" >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/kb2/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        if 'phase_b' in r['Kernel_Name']:
            agg['b'].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
v=sorted(agg['b'])
print('%.2f'%(sum(v[len(v)//4:])/len(v[len(v)//4:])) if v else 'n/a')
"
}
echo "phase_b alone. coop_g=1 is what a fused phase_b/c needs; the question is"
echo "whether a wider block gives the row back the threads coop_g was supplying."
printf "%6s %9s %10s %10s %10s %10s %10s\n" M N "auto" "g1 b256" "g1 b512" "g1 b1024" "g8 b1024"
for MN in "4096 131072" "4096 262144" "1024 524288" "128 1048576"; do
  set -- $MN
  printf "%6d %9d %10s %10s %10s %10s %10s\n" $1 $2 \
    "$(ks $1 $2)" \
    "$(ks $1 $2 --coop-g 1 --cf-block 256)" \
    "$(ks $1 $2 --coop-g 1 --cf-block 512)" \
    "$(ks $1 $2 --coop-g 1 --cf-block 1024)" \
    "$(ks $1 $2 --coop-g 8 --cf-block 1024)"
done
