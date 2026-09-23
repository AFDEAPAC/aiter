#!/bin/bash
cd /topk
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -3
ks() {
  rm -rf /tmp/gcp
  rocprofv3 --kernel-trace --output-format csv -d /tmp/gcp -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 8 --iters 25 --repeats 1 "${@:3}" >/dev/null 2>&1
  python3 -c "
import csv,glob
v=[]
for f in glob.glob('/tmp/gcp/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        if 'phase_c' in r['Kernel_Name']:
            v.append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
v=sorted(v); print('%.2f'%(sum(v[len(v)//4:])/len(v[len(v)//4:])) if v else 'n/a')
"
}
echo "phase_c device time by radix pass count. 4 is shipped and is what exactness"
echo "needs; fewer is a TIMING ABLATION with wrong results. This prices what an"
echo "early exit could be worth if the active set collapses before pass 4."
printf "%6s %9s %9s %9s %9s %9s %11s\n" M N "no select" "1 pass" "2" "3" "4 (ship)" "pass 4 costs"
for MN in "512 131072" "256 131072" "128 524288" "32 1048576" "8 524288" "1024 131072" "4096 131072"; do
  set -- $MN
  hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -DABLATE_PC=1 -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
  Z=$(ks $1 $2)
  hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
  P1=$(ks $1 $2 --phase-c-passes 1); P2=$(ks $1 $2 --phase-c-passes 2)
  P3=$(ks $1 $2 --phase-c-passes 3); P4=$(ks $1 $2 --phase-c-passes 4)
  printf "%6d %9d %9s %9s %9s %9s %9s %11s\n" $1 $2 "$Z" "$P1" "$P2" "$P3" "$P4" \
    "$(awk -v a=$P3 -v b=$P4 'BEGIN{printf "%.2f",b-a}')"
done
