#!/bin/bash
cd /topk
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -3
ks() {
  rm -rf /tmp/kpa
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kpa -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 5 --iters 20 --repeats 1 "${@:3}" >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/kpa/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        k=r['Kernel_Name']
        t=('a' if 'phase_a' in k else 'b' if 'phase_b' in k else 'c' if 'phase_c' in k else 'd' if 'phase_d' in k else None)
        if t: agg[t].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
o={};tot=0
for t in 'abcd':
    v=sorted(agg[t])
    if v: o[t]=sum(v[len(v)//4:])/len(v[len(v)//4:]); tot+=o[t]
print('a=%7.2f TOT=%8.2f'%(o.get('a',0),tot))
"
}
echo "phase_a: default vs --phase-a-compact 1 vs a narrower block"
printf "%6s %9s  %-22s %-22s %-22s\n" M N "default" "compact" "a-block 256"
for MN in "4096 131072" "4096 262144" "1024 524288" "128 1048576" "16 131072"; do
  set -- $MN
  printf "%6d %9d  %-22s %-22s %-22s\n" $1 $2 \
    "$(ks $1 $2)" "$(ks $1 $2 --phase-a-compact 1)" "$(ks $1 $2 --phase-a-block 256)"
done
echo
echo "correctness with compact:"
for MN in "4096 131072" "128 1048576"; do set -- $MN
  for d in gaussian adversarial all_equal; do
    r=$(./benchmark_topk --mode verify --m $1 --n $2 --topk 2048 --dist $d --seed 0 --phase-a-compact 1 2>&1|grep -o "VERDICT [A-Z]*"|head -1)
    echo "  m=$1 n=$2 $d -> $r"
  done
done
