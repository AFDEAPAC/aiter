#!/bin/bash
cd /topk
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -3
ks() {
  rm -rf /tmp/kpp
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kpp -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 5 --iters 20 --repeats 1 "${@:3}" >/dev/null 2>&1
  python3 -c "
import csv,glob
v=[]
for f in glob.glob('/tmp/kpp/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        if 'phase_a' in r['Kernel_Name']:
            v.append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
v=sorted(v)
print('%.2f'%(sum(v[len(v)//4:])/len(v[len(v)//4:])) if v else 'n/a')
"
}
echo "phase_a device time by radix pass count. ABLATE_PA=1 (read + LDS fill, no"
echo "select at all) is the floor each column is walking up from."
printf "%6s %9s %9s %9s %9s %9s %9s\n" M N "no select" "1 pass" "2" "3" "4 (ship)"
for MN in "4096 131072" "4096 262144" "1024 524288"; do
  set -- $MN
  hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -DABLATE_PA=1 -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
  Z=$(ks $1 $2)
  hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
  printf "%6d %9d %9s %9s %9s %9s %9s\n" $1 $2 "$Z" \
    "$(ks $1 $2 --phase-a-passes 1)" "$(ks $1 $2 --phase-a-passes 2)" \
    "$(ks $1 $2 --phase-a-passes 3)" "$(ks $1 $2 --phase-a-passes 4)"
done
echo
grep -n "RADIX_PASSES\|HIST_REP\|HIST_SLOTS\|HIST_AGG_ROUNDS\|SELECT_CLEAR_ON_READ" csrc/topk_common.hip.hpp | grep -i "constexpr\|define" | head
