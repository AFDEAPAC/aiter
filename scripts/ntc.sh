#!/bin/bash
cd /topk
ks() {
  rm -rf /tmp/kc
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kc -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 5 --iters 15 --repeats 1 >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/kc/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        k=r['Kernel_Name']
        t=('a' if 'phase_a' in k else 'b' if 'phase_b' in k else 'c' if 'phase_c' in k else 'd' if 'phase_d' in k else None)
        if t: agg[t].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
o={};tot=0
for t in 'abcd':
    v=sorted(agg[t])
    if v: o[t]=sum(v[len(v)//4:])/len(v[len(v)//4:]); tot+=o[t]
print('c=%7.2f  TOTAL=%8.2f'%(o.get('c',0),tot))
"
}
vf() { for d in gaussian adversarial all_equal uniform inf; do
         r=$(./benchmark_topk --mode verify --m $1 --n $2 --topk 2048 --dist $d --seed 0 2>&1 | grep -o "VERDICT [A-Z]*" | head -1)
         [ "$r" = "VERDICT PASS" ] || { echo "FAIL:$d"; return; }
       done; echo "5-dist PASS"; }
echo "phase_c reading the candidate array non-temporally"
printf "%6s %9s   %-28s %-28s\n" M N "NT_CAND=0" "NT_CAND=1"
for MN in "4096 131072" "4096 1048576" "1024 524288" "128 1048576" "128 131072"; do
  set -- $MN
  hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -DNT_CAND=0 -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
  A=$(ks $1 $2)
  hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
  B=$(ks $1 $2)
  printf "%6d %9d   %-28s %-28s\n" $1 $2 "$A" "$B"
done
printf "verify: %s\n" "$(vf 4096 131072)"
