#!/bin/bash
cd /topk
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -3
ka() {
  rm -rf /tmp/kst
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kst -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 5 --iters 20 --repeats 1 "${@:3}" >/dev/null 2>&1
  python3 -c "
import csv,glob
v=[]
for f in glob.glob('/tmp/kst/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        if 'phase_a' in r['Kernel_Name']:
            v.append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
v=sorted(v)
print('%.2f'%(sum(v[len(v)//4:])/len(v[len(v)//4:])) if v else 'n/a')
"
}
echo "phase_a at a sample count small enough that the work is negligible."
echo "If the cost is one block's serial latency, this is a staircase in M/512."
printf "%6s %9s %9s %9s %9s\n" M "S=512" "S=2048" "S=8192" "a-block 512, S=512"
for M in 64 128 256 512 1024 2048 4096; do
  printf "%6d %9s %9s %9s %9s\n" $M \
    "$(ka $M 131072 --sample-s 512 --s-rule 0)" \
    "$(ka $M 131072 --sample-s 2048 --s-rule 0)" \
    "$(ka $M 131072 --sample-s 8192 --s-rule 0)" \
    "$(ka $M 131072 --sample-s 512 --s-rule 0 --phase-a-block 512)"
done
