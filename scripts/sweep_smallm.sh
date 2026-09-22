#!/bin/bash
# known_bad.md:1446 swept coop_g at M=16 N=32768 and found auto (8) best, then
# concluded "block count is not what costs us". That sweep was at ONE N. At
# N>=131072 the row is 4-32x longer, so each block does that much more work and
# the balance can move. phase_b at m=16 n=131072 runs 8.4 MB in 9.28us = 0.9 TB/s
# on 12.5% of the machine's thread slots, which is what prompted re-asking.
cd /topk
make clean >/dev/null 2>&1; make -j >/dev/null 2>&1
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
print('%.2f  (%s)' % (tot, ' '.join('%s=%.2f'%p for p in parts)))
"
}
for MN in "16 131072" "16 1048576" "64 524288" "128 1048576"; do
  set -- $MN
  echo "=== M=$1 N=$2 : pipeline us by --coop-g ==="
  printf "   %-8s %s\n" auto "$(pb $1 $2)"
  for G in 4 8 16 32 64; do
    printf "   %-8s %s\n" "$G" "$(pb $1 $2 --coop-g $G)"
  done
done
