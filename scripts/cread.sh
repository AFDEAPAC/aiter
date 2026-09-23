#!/bin/bash
cd /topk
ks() {
  rm -rf /tmp/kr
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kr -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 5 --iters 15 --repeats 1 >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/kr/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        k=r['Kernel_Name']
        t=('a' if 'phase_a' in k else 'b' if 'phase_b' in k else 'c' if 'phase_c' in k else 'd' if 'phase_d' in k else None)
        if t: agg[t].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
o={};tot=0
for t in 'abcd':
    v=sorted(agg[t])
    if v: o[t]=sum(v[len(v)//4:])/len(v[len(v)//4:]); tot+=o[t]
print('b=%8.2f c=%7.2f TOTAL=%8.2f'%(o.get('b',0),o.get('c',0),tot))
"
}
echo "the fusion prize, priced from both ends (all arms give wrong results except the first)"
printf "%6s %9s  %-38s\n" M N arm
for MN in "4096 131072" "4096 262144" "1024 524288"; do
  set -- $MN
  for A in "0 0 shipped" "0 1 phase_c without the candidate read" "1 0 phase_b without the candidate write" "1 1 both, which is what fusion removes"; do
    set -- $MN $A
    hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -DABLATE_EPI=$3 -DABLATE_CREAD=$4 -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
    printf "%6d %9d  %-38s %s\n" $1 $2 "$5 $6 $7 $8 $9" "$(ks $1 $2)"
  done
  echo
done
