#!/bin/bash
cd /topk
ks() {
  rm -rf /tmp/kl
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kl -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 5 --iters 15 --repeats 1 >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/kl/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        k=r['Kernel_Name']
        t=('a' if 'phase_a' in k else 'b' if 'phase_b' in k else 'c' if 'phase_c' in k else None)
        if t: agg[t].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
o=[];tot=0
for t in 'abc':
    v=sorted(agg[t])
    if not v: continue
    m=sum(v[len(v)//4:])/len(v[len(v)//4:]); tot+=m; o.append('%s=%7.2f'%(t,m))
print('  '.join(o)+'  TOTAL=%8.2f'%tot)
"
}
vf() { for d in gaussian adversarial all_equal uniform inf; do
         r=$(./benchmark_topk --mode verify --m $1 --n $2 --topk 2048 --dist $d --seed 0 2>&1 | grep -o "VERDICT [A-Z]*" | head -1)
         [ "$r" = "VERDICT PASS" ] || { echo "FAIL:$d"; return; }
       done; echo "5-dist PASS"; }
run() { hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc $2 -o benchmark_topk benchmark_topk.hip.cpp 2>&1 | grep -i error | head -3
        for MN in "4096 131072" "4096 262144" "128 1048576"; do set -- $MN
          printf "%-34s m=%-5s n=%-8s %s\n" "$1x" "$1" "$2" "$(ks $1 $2)"; done
        printf "%-34s %s\n" "" "$(vf 4096 131072)"; }
p() { hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc $1 -o benchmark_topk benchmark_topk.hip.cpp 2>&1 | grep -i error | head -3
      echo "$2"
      for MN in "4096 131072" "4096 262144" "128 1048576"; do set -- $MN
        printf "   m=%-5s n=%-8s %s\n" "$1" "$2" "$(ks $1 $2)"; done
      printf "   verify: %s\n" "$(vf 4096 131072)"; }
p ""            "cached loads (shipped)"
p "-DNT_LOAD=1" "non-temporal loads"
