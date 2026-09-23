#!/bin/bash
cd /topk
ks() {
  rm -rf /tmp/kp
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kp -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 5 --iters 15 --repeats 1 "${@:3}" >/dev/null 2>&1
  python3 -c "
import csv,glob
v=[]
for f in glob.glob('/tmp/kp/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        if 'phase_a' in r['Kernel_Name']:
            v.append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
v=sorted(v)
print('%.2f'%(sum(v[len(v)//4:])/len(v[len(v)//4:])) if v else 'n/a')
"
}
echo "phase_a alone: how much of it is the sample read, how much the LDS select"
printf "%6s %9s %8s %10s %10s %9s %9s\n" M N S full "read only" "select" "GB/s read"
for MN in "4096 131072 8192" "4096 262144 16384" "4096 1048576 16384" "1024 524288 16384" "128 1048576 16384"; do
  set -- $MN
  hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
  A=$(ks $1 $2)
  hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -DABLATE_PA=1 -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
  B=$(ks $1 $2)
  printf "%6d %9d %8d %10s %10s %9s %9s\n" $1 $2 $3 "$A" "$B" \
    "$(awk -v a=$A -v b=$B 'BEGIN{printf "%.2f",a-b}')" \
    "$(awk -v m=$1 -v s=$3 -v b=$B 'BEGIN{printf "%.2f",m*s*4/(b*1e3)}')"
done
