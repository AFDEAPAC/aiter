#!/bin/bash
cd /topk
ks() {
  rm -rf /tmp/s1
  rocprofv3 --kernel-trace --output-format csv -d /tmp/s1 -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 8 --iters 25 --repeats 1 "${@:3}" >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/s1/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        k=r['Kernel_Name']
        t=('a' if 'phase_a' in k else 'b' if 'phase_b' in k else 'c' if 'phase_c' in k else 'd' if 'phase_d' in k else None)
        if t: agg[t].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
o={};tot=0
for t in 'abcd':
    v=sorted(agg[t])
    if v: o[t]=sum(v[len(v)//4:])/len(v[len(v)//4:]); tot+=o[t]
print('%5.1f/%6.1f'%(o.get('b',0),tot) if tot else 'n/a')
"
}
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -DWSTAGE_WAVES_OVERRIDE=16 -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -3
echo "phase_b / three-kernel total, by coop_g and phase_b block width (WSTAGE_WAVES=16)."
echo "The floor probe prefers wg1024 at small M; the shipped build is capped at 512."
for MN in "8 524288" "8 1048576" "32 1048576" "128 131072"; do
  set -- $MN
  echo "m=$1 n=$2  (shipped auto: $(ks $1 $2))"
  printf "   %8s" "coop_g"
  for B in 256 512 1024; do printf " %14s" "cf$B"; done; echo
  for G in 8 16 32 64; do
    printf "   %8d" $G
    for B in 256 512 1024; do printf " %14s" "$(ks $1 $2 --coop-g $G --cf-block $B)"; done
    echo
  done
done
