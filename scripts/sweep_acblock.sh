#!/bin/bash
# phase_a and phase_c are barrier-bound, not traffic-bound: at m=16 n=131072
# phase_a reads 512 KB (0.08us at 6.3 TB/s) and takes 6.60us, and a+c is 68% of
# that pipeline. Barrier cost rises with waves per block, and at small M
# occupancy_block_threads maxes the block (1024 threads = 16 waves) because it
# is sizing to fill a machine that 16 rows cannot fill anyway. Never swept.
cd /topk
pb() {
  rm -rf /tmp/kt
  local M=$1 N=$2; shift 2
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kt -- \
    ./benchmark_topk --mode time --m $M --n $N --topk 2048 --dist gaussian \
    --seed 0 --warmup 5 --iters 20 --repeats 1 "$@" >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
d=collections.defaultdict(list)
for f in glob.glob('/tmp/kt/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        n=r['Kernel_Name'].split('(')[0]
        if 'phase_' in n: d[n].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
tot=0; parts=[]
for n,v in d.items():
    v=sorted(v)[len(v)//4:]; m=sum(v)/len(v); tot+=m
    parts.append((n.split('phase_')[1][0], m))
parts.sort()
print('%7.2f  (%s)' % (tot, ' '.join('%s=%.2f'%p for p in parts)))
"
}
for MN in "16 131072" "64 524288" "128 1048576" "4096 131072"; do
  set -- $MN
  echo "=== M=$1 N=$2 ==="
  printf "   %-22s %s\n" "auto/auto" "$(pb $1 $2)"
  for B in 128 256 512; do
    printf "   %-22s %s\n" "phase-a-block $B" "$(pb $1 $2 --phase-a-block $B)"
  done
  for B in 128 256 512; do
    printf "   %-22s %s\n" "phase-c-block $B" "$(pb $1 $2 --phase-c-block $B)"
  done
  printf "   %-22s %s\n" "both 256" "$(pb $1 $2 --phase-a-block 256 --phase-c-block 256)"
done
