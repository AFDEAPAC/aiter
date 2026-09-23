#!/bin/bash
cd /topk
ks() {
  rm -rf /tmp/kw
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kw -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 5 --iters 15 --repeats 1 "${@:3}" >/dev/null 2>&1
  python3 -c "
import csv,glob
v=[]
for f in glob.glob('/tmp/kw/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        if 'phase_b' in r['Kernel_Name']:
            v.append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
v=sorted(v)
print('%.2f'%(sum(v[len(v)//4:])/len(v[len(v)//4:])) if v else 'n/a')
"
}
echo "WSTAGE_WAVES=16 so a 1024-thread block has staging for all its waves."
echo "phase_b alone; the fused kernel needs coop_g=1 to keep candidates in LDS."
printf "%6s %9s %10s %10s %10s %10s\n" M N "g8 b512" "g1 b512" "g1 b1024" "g2 b1024"
for MN in "4096 131072" "4096 262144" "1024 524288"; do
  set -- $MN
  hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
  A=$(ks $1 $2)
  hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -DWSTAGE_WAVES_OVERRIDE=16 -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
  printf "%6d %9d %10s %10s %10s %10s\n" $1 $2 "$A" \
    "$(ks $1 $2 --coop-g 1 --cf-block 512)" \
    "$(ks $1 $2 --coop-g 1 --cf-block 1024)" \
    "$(ks $1 $2 --coop-g 2 --cf-block 1024)"
done
