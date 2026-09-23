#!/bin/bash
cd /topk
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1 | grep -i error | head -3
ks() {
  rm -rf /tmp/kg2
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kg2 -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 5 --iters 15 --repeats 1 "${@:3}" >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/kg2/**/*kernel_trace.csv',recursive=True):
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
echo "three-kernel device total, gate off vs the size gate. M*pitch >= 2^27 picks NT."
printf "%6s %9s %12s %10s %10s %8s\n" M N "M*pitch" off gate ratio
for MN in "1 131072" "1 262144" "1 1048576" "16 131072" "16 524288" "16 1048576" "64 131072" "64 262144" "64 1048576" "128 131072" "128 262144" "128 1048576" "256 131072" "256 524288" "512 262144" "1024 131072" "1024 524288" "1024 1048576" "2048 131072" "2048 524288" "4096 131072" "4096 262144" "4096 524288" "4096 1048576" "64 131073" "256 262146" "4096 1048577"; do
  set -- $MN
  A=$(ks $1 $2 --nt-gate 0); B=$(ks $1 $2)
  printf "%6d %9d %12d %10s %10s %8s\n" $1 $2 $(( $1 * $2 )) "$A" "$B" \
    "$(awk -v a=$A -v b=$B 'BEGIN{printf "%.3fx",b/a}')"
done
echo
echo "verify with the gate on: 5 distributions"
fail=0
for m in 1 16 64 128 256 1024 4096; do for n in 131072 262144 524288 1048576 131073 262146 1048577; do for d in gaussian adversarial all_equal uniform inf; do
  r=$(./benchmark_topk --mode verify --m $m --n $n --topk 2048 --dist $d --seed 0 2>&1 | grep -o "VERDICT [A-Z]*" | head -1)
  [ "$r" = "VERDICT PASS" ] || { echo "  FAIL m=$m n=$n dist=$d -> $r"; fail=1; }
done; done; done
[ $fail = 0 ] && echo "  all 245 PASS"
