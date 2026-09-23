#!/bin/bash
cd /topk
ks() {
  rm -rf /tmp/ee
  rocprofv3 --kernel-trace --output-format csv -d /tmp/ee -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 8 --iters 25 --repeats 1 >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/ee/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        k=r['Kernel_Name']
        t=('a' if 'phase_a' in k else 'b' if 'phase_b' in k else 'c' if 'phase_c' in k else 'd' if 'phase_d' in k else None)
        if t: agg[t].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
o={};tot=0
for t in 'abcd':
    v=sorted(agg[t])
    if v: o[t]=sum(v[len(v)//4:])/len(v[len(v)//4:]); tot+=o[t]
print('%5.2f %5.2f %7.2f'%(o.get('a',0),o.get('c',0),tot))
"
}
echo "phase_a / phase_c / three-kernel total, stopping when the remaining passes"
echo "cannot change the answer."
printf "%6s %9s %22s %22s %9s\n" M N "SELECT_EARLY_EXIT=0" "=1" "total x"
for MN in "512 131072" "256 131072" "128 524288" "32 1048576" "8 524288" "1024 131072" "2048 131072" "4096 131072" "64 1048576"; do
  set -- $MN
  hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -DSELECT_EARLY_EXIT=0 -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -3
  A=$(ks $1 $2); at=$(echo $A|awk '{print $3}')
  hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -DSELECT_EARLY_EXIT=1 -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -3
  B=$(ks $1 $2); bt=$(echo $B|awk '{print $3}')
  printf "%6d %9d %22s %22s %9s\n" $1 $2 "$A" "$B" "$(awk -v a=$at -v b=$bt 'BEGIN{printf "%.4f",b/a}')"
done
echo
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -3
fail=0
for m in 1 16 64 128 256 1024 4096; do for n in 131072 262144 524288 1048576 131073 262146 1048577; do for d in gaussian adversarial all_equal uniform inf; do
  r=$(./benchmark_topk --mode verify --m $m --n $n --topk 2048 --dist $d --seed 0 2>&1|grep -o "VERDICT [A-Z]*"|head -1)
  [ "$r" = "VERDICT PASS" ] || { echo "  FAIL m=$m n=$n dist=$d -> $r"; fail=1; }
done; done; done
[ $fail = 0 ] && echo "verify: all 245 PASS"
