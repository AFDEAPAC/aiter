#!/bin/bash
cd /topk
ks() {
  rm -rf /tmp/ks
  rocprofv3 --kernel-trace --output-format csv -d /tmp/ks -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 5 --iters 20 --repeats 1 >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/ks/**/*kernel_trace.csv',recursive=True):
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
echo "three-kernel total. The NT STORE in phase_b's epilogue went in without a"
echo "size gate; the NT LOAD needed one. Checking whether the store needs one too."
printf "%6s %9s %11s %11s %11s\n" M N "old walk" "per-wave" "per-wave+NT"
for MN in "1 131072" "4 524288" "16 131072" "16 1048576" "64 131072" "128 262144" "1024 131072" "4096 131072" "4096 1048576"; do
  set -- $MN
  hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -DABLATE_EPI=91 -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
  A=$(ks $1 $2)
  hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -DABLATE_EPI=4 -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
  B=$(ks $1 $2)
  hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
  C=$(ks $1 $2)
  printf "%6d %9d %11s %11s %11s\n" $1 $2 "$A" "$B" "$C"
done
