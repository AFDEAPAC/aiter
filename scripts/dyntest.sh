#!/bin/bash
cd /topk
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -5
ks() {
  rm -rf /tmp/dy
  rocprofv3 --kernel-trace --output-format csv -d /tmp/dy -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 8 --iters 25 --repeats 1 "${@:3}" >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/dy/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        k=r['Kernel_Name']
        t=('a' if 'phase_a' in k else 'b' if 'phase_b' in k else 'c' if 'phase_c' in k else 'd' if 'phase_d' in k else None)
        if t: agg[t].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
tot=0
for t in 'abcd':
    v=sorted(agg[t])
    if v: tot+=sum(v[len(v)//4:])/len(v[len(v)//4:])
print('%.1f'%tot if tot else '99999')
"
}
echo "Dynamic staging: the shipped 512-thread path must be unchanged, and 1024"
echo "must now work without the global penalty."
printf "%6s %9s %9s %11s %11s %9s %8s\n" M N "auto(512)" "cf1024 g2" "cf1024 g4" "cf1024 g8" "best/auto"
for MN in "32 1048576" "128 131072" "512 131072" "1024 131072" "64 1048576" "4096 131072" "256 131072" "128 262144" "8 1048576"; do
  set -- $MN
  A=$(ks $1 $2)
  b2=$(ks $1 $2 --coop-g 2 --cf-block 1024); b4=$(ks $1 $2 --coop-g 4 --cf-block 1024)
  b8=$(ks $1 $2 --coop-g 8 --cf-block 1024)
  best=$(python3 -c "print(min([$b2,$b4,$b8]))")
  printf "%6d %9d %9s %11s %11s %9s %8s\n" $1 $2 "$A" "$b2" "$b4" "$b8" \
    "$(python3 -c "print('%.4f'%($best/$A))")"
done
echo
for d in gaussian adversarial all_equal uniform inf; do
  printf "  verify m=512 n=131072 %-12s %s\n" $d "$(./benchmark_topk --mode verify --m 512 --n 131072 --topk 2048 --dist $d --seed 0 2>&1|grep -o 'VERDICT [A-Z]*'|head -1)"
done
