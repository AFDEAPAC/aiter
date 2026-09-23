#!/bin/bash
cd /topk
echo "=== 1. rebuild after the PC_UNROLL removal, confirm it was a no-op ==="
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -3
ks() {
  rm -rf /tmp/kfin
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kfin -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 8 --iters 25 --repeats 1 >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/kfin/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        k=r['Kernel_Name']
        t=('a' if 'phase_a' in k else 'b' if 'phase_b' in k else 'c' if 'phase_c' in k else 'd' if 'phase_d' in k else None)
        if t: agg[t].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
tot=0
for t in 'abcd':
    v=sorted(agg[t])
    if v: tot+=sum(v[len(v)//4:])/len(v[len(v)//4:])
print('%.2f'%tot)
"
}
for MN in "512 131072" "1024 131072" "256 262144"; do
  set -- $MN; printf "  m=%-5d n=%-8d total %s us\n" $1 $2 "$(ks $1 $2)"
done
echo
echo "=== 2. correctness, 245 cases ==="
fail=0
for m in 1 16 64 128 256 1024 4096; do for n in 131072 262144 524288 1048576 131073 262146 1048577; do for d in gaussian adversarial all_equal uniform inf; do
  r=$(./benchmark_topk --mode verify --m $m --n $n --topk 2048 --dist $d --seed 0 2>&1 | grep -o "VERDICT [A-Z]*" | head -1)
  [ "$r" = "VERDICT PASS" ] || { echo "  FAIL m=$m n=$n dist=$d -> $r"; fail=1; }
done; done; done
[ $fail = 0 ] && echo "  all 245 PASS"
echo
echo "=== 3. export contract ==="
pip -q install clang-format==23.1.1 2>&1|tail -0
python3 scripts/export_aiter_op.py --aiter /aiter >/dev/null
python3 scripts/export_aiter_op.py --aiter /aiter --check && echo "  EXPORT-CHECK-CLEAN"
