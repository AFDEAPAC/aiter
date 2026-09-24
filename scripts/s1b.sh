#!/bin/bash
cd /topk
ks() {
  rm -rf /tmp/s1b
  rocprofv3 --kernel-trace --output-format csv -d /tmp/s1b -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 8 --iters 25 --repeats 1 "${@:3}" >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/s1b/**/*kernel_trace.csv',recursive=True):
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
echo "phase_b / three-kernel total, at the shapes that are closest to green."
echo "The near-green cells need: m=1024 n=131072 4.9us, m=512 n=131072 1.2us,"
echo "m=64 n=1048576 0.9us."
for MN in "1024 131072" "512 131072" "2048 131072" "64 1048576" "256 131072" "128 262144"; do
  set -- $MN
  echo "m=$1 n=$2  (shipped auto: $(ks $1 $2))"
  printf "   %8s" "coop_g"
  for B in 256 512 1024; do printf " %14s" "cf$B"; done; echo
  for G in 2 4 8 16; do
    printf "   %8d" $G
    for B in 256 512 1024; do printf " %14s" "$(ks $1 $2 --coop-g $G --cf-block $B)"; done
    echo
  done
done
