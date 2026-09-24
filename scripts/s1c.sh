#!/bin/bash
cd /topk
ks() {
  rm -rf /tmp/s1c
  rocprofv3 --kernel-trace --output-format csv -d /tmp/s1c -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 8 --iters 25 --repeats 1 "${@:3}" >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/s1c/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        k=r['Kernel_Name']
        t=('a' if 'phase_a' in k else 'b' if 'phase_b' in k else 'c' if 'phase_c' in k else 'd' if 'phase_d' in k else None)
        if t: agg[t].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
o={};tot=0
for t in 'abcd':
    v=sorted(agg[t])
    if v: o[t]=sum(v[len(v)//4:])/len(v[len(v)//4:]); tot+=o[t]
print('%6.1f'%tot if tot else 'n/a')
"
}
echo "three-kernel total. W8 is the shipped build. W16 doubles wbuf to 40 KB in"
echo "every block, which is the price a global WSTAGE_WAVES makes 512-thread"
echo "blocks pay; a template parameter would not."
printf "%6s %9s %10s %10s %12s %10s\n" M N "W8 auto" "W16 auto" "W16 best1024" "best/W8"
for MN in "1024 131072" "512 131072" "2048 131072" "64 1048576" "256 131072" "128 262144" "4096 131072"; do
  set -- $MN
  hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
  A=$(ks $1 $2)
  hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -DWSTAGE_WAVES_OVERRIDE=16 -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
  B=$(ks $1 $2)
  best=""; bg=""
  for G in 2 4 8 16; do
    t=$(ks $1 $2 --coop-g $G --cf-block 1024)
    [ "$t" = "n/a" ] && continue
    if [ -z "$best" ] || [ "$(awk -v a=$t -v b=$best 'BEGIN{print (a<b)?1:0}')" = "1" ]; then best=$t; bg=$G; fi
  done
  printf "%6d %9d %10s %10s %12s %10s\n" $1 $2 "$A" "$B" "$best (g$bg)" \
    "$(awk -v a=$A -v b=$best 'BEGIN{printf "%.4f",b/a}')"
done
