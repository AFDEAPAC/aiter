#!/bin/bash
cd /topk
ks() {
  rm -rf /tmp/kcp
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kcp -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 5 --iters 20 --repeats 1 "${@:3}" >/dev/null 2>&1
  python3 -c "
import csv,glob
v=[]
for f in glob.glob('/tmp/kcp/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        if 'phase_a' in r['Kernel_Name']:
            v.append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
v=sorted(v)
print('%.2f'%(sum(v[len(v)//4:])/len(v[len(v)//4:])) if v else 'n/a')
"
}
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -DABLATE_PA=1 -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
echo "floor (no select): m=4096 n=131072 $(ks 4096 131072)   n=262144 $(ks 4096 262144)"
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
echo
echo "marginal cost of each radix pass. If the compact variant is compacting,"
echo "its passes 2-4 should collapse; the plain one re-scans all 16384 each time."
for MN in "4096 131072" "4096 262144"; do
  set -- $MN
  for VAR in "0 plain" "1 compact"; do
    set -- $MN $VAR
    printf "  m=%-5d n=%-8d %-8s" $1 $2 "$4"
    prev=""
    for P in 1 2 3 4; do
      T=$(ks $1 $2 --phase-a-compact $3 --phase-a-passes $P)
      if [ -n "$prev" ]; then D=$(awk -v a=$prev -v b=$T 'BEGIN{printf "%+6.2f",b-a}'); else D="      "; fi
      printf " %8s(%s)" "$T" "$D"
      prev=$T
    done
    echo
  done
done
