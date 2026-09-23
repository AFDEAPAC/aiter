#!/bin/bash
cd /topk
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -3
for MN in "2048 131072" "256 131072"; do
  set -- $MN
  rm -rf /tmp/kall
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kall -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 5 --iters 20 --repeats 1 >/dev/null 2>&1
  echo "m=$1 n=$2  every kernel in the trace:"
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/kall/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        agg[r['Kernel_Name'].split('(')[0][:58]].append(
            (int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
tot=0
for k,v in sorted(agg.items(), key=lambda z:-sum(z[1])):
    v=sorted(v); m=sum(v[len(v)//4:])/len(v[len(v)//4:])
    n=len(v)
    tot+=m
    print('   %-58s x%-4d mean %8.3f us' % (k, n, m))
print('   %-58s      %8.3f us' % ('SUM OF KERNEL MEANS', tot))
"
  echo
done
