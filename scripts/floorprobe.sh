#!/bin/bash
cd /topk
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -3
ks() {
  rm -rf /tmp/kf
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kf -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 5 --iters 20 --repeats 1 "${@:3}" >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/kf/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        k=r['Kernel_Name']
        t=('a' if 'phase_a' in k else 'c' if 'phase_c' in k else None)
        if t: agg[t].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
o={}
for t in 'ac':
    v=sorted(agg[t])
    o[t]=sum(v[len(v)//4:])/len(v[len(v)//4:]) if v else 0
print('a=%6.2f c=%6.2f'%(o['a'],o['c']))
"
}
echo "How much of phase_a is work and how much is one block's serial latency."
echo "m=256 n=131072, S walked down. Rule 0 picks S=8192 here."
printf "%8s  %s\n" S "per-kernel"
for S in 8192 4096 2048 1024 512 256; do
  printf "%8d  %s\n" $S "$(ks 256 131072 --sample-s $S)"
done
echo
echo "and phase_a with no select at all, S=8192 (ABLATE_PA=1):"
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -DABLATE_PA=1 -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
printf "%8s  %s\n" "no sel" "$(ks 256 131072)"
echo
echo "same probe at m=4096 for contrast (blocks far exceed the 512 resident):"
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
for S in 8192 2048 512; do
  printf "%8d  %s\n" $S "$(ks 4096 131072 --sample-s $S)"
done
