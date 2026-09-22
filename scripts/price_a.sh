#!/bin/bash
# Price phase_a's two halves. The repo's model is "~4us fixed + ~1.0us per radix
# pass, independent of block count". What IS the fixed part? npasses=0 runs the
# sample gather and the LDS staging and skips the radix walk entirely (wrong
# threshold, price only). The gap between passes=0 and passes=3 is the walk.
cd /topk
pa() {
  rm -rf /tmp/kt
  local M=$1 N=$2; shift 2
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kt -- \
    ./benchmark_topk --mode time --m $M --n $N --topk 2048 --dist gaussian \
    --seed 0 --warmup 5 --iters 15 --repeats 1 "$@" >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
d=collections.defaultdict(list)
for f in glob.glob('/tmp/kt/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        n=r['Kernel_Name'].split('(')[0]
        if 'phase_a' in n: d[n].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
for n,v in d.items():
    v=sorted(v)[len(v)//4:]
    print('%.2f'%(sum(v)/len(v)))
    break
else: print('n/a')
"
}
printf "%6s %9s %9s %9s %9s %9s %12s\n" M N p0 p1 p2 p3 "walk=p3-p0"
for MN in "16 131072" "64 524288" "128 1048576" "4096 131072" "4096 1048576"; do
  set -- $MN
  A0=$(pa $1 $2 --phase-a-passes 0); A1=$(pa $1 $2 --phase-a-passes 1)
  A2=$(pa $1 $2 --phase-a-passes 2); A3=$(pa $1 $2 --phase-a-passes 3)
  printf "%6d %9d %9s %9s %9s %9s %12s\n" $1 $2 "$A0" "$A1" "$A2" "$A3" \
    "$(awk -v a=$A0 -v b=$A3 'BEGIN{printf "%.2f", b-a}')"
done
