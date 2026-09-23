#!/bin/bash
cd /topk
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -3
ks() {
  rm -rf /tmp/ksm
  rocprofv3 --kernel-trace --output-format csv -d /tmp/ksm -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 10 --iters 30 --repeats 1 "${@:3}" >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/ksm/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        k=r['Kernel_Name']
        t=('a' if 'phase_a' in k else 'b' if 'phase_b' in k else 'c' if 'phase_c' in k else 'd' if 'phase_d' in k else None)
        if t: agg[t].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
o={};tot=0
for t in 'abcd':
    v=sorted(agg[t])
    if v: o[t]=sum(v[len(v)//4:])/len(v[len(v)//4:]); tot+=o[t]
print('a=%6.2f b=%6.2f c=%6.2f d=%6.2f  sum=%7.2f'%(o.get('a',0),o.get('b',0),o.get('c',0),o.get('d',0),tot))
"
}
echo "=== where the time goes at the worst red cells ==="
printf "%6s %9s  %s\n" M N "per kernel"
for MN in "8 524288" "8 1048576" "16 524288" "16 1048576" "1 131072" "32 1048576"; do
  set -- $MN; printf "%6d %9d  %s\n" $1 $2 "$(ks $1 $2)"
done
echo
echo "=== the shipped shape plan there ==="
cat > /tmp/sp4.cpp <<'EOF'
#include "topk_common.hip.hpp"
#include "topk_shape.hip.hpp"
#include <cstdio>
int main(){int K=2048;
 printf("%6s %9s %7s %8s %6s %6s %8s %10s\n","M","N","S","margin","cap","coop","rule","blocks_b");
 for(int M:{1,8,16,32,64}) for(int N:{131072,524288,1048576}){
  ShapeParams p=derive_shape_params(M,N,K,0.f,0,0,PATH_AUTO);
  printf("%6d %9d %7d %8.3f %6d %6d %8d %10d\n",M,N,p.S,p.margin,p.cap,p.coop_g,
         effective_s_rule(M), p.coop_g*M);}
 return 0;}
EOF
hipcc -O2 --offload-arch=gfx950 -Icsrc /tmp/sp4.cpp -o /tmp/sp4 2>&1|grep -iE "^.*error"|head -3
/tmp/sp4
