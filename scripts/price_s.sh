#!/bin/bash
# Is phase_a's "fixed" part the sample gather or the kernel launch? p0 (no radix
# walk) is 3.94us at m=16 n=131072 while the samples are 512 KB = 0.08us of
# traffic, and an empty dispatch on this box is 2.64us. If p0 is flat in S the
# gather is not the cost and the launch is.
cd /topk
pa() {
  rm -rf /tmp/kt
  local M=$1 N=$2; shift 2
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kt -- \
    ./benchmark_topk --mode time --m $M --n $N --topk 2048 --dist gaussian \
    --seed 0 --warmup 5 --iters 15 --repeats 1 --phase-a-passes 0 "$@" >/dev/null 2>&1
  python3 -c "
import csv,glob
v=[]
for f in glob.glob('/tmp/kt/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        if 'phase_a' in r['Kernel_Name']:
            v.append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
v=sorted(v)[len(v)//4:]
print('%.2f'%(sum(v)/len(v)) if v else 'n/a')
"
}
printf "%6s %9s %9s %9s %9s %9s %9s\n" M N S=4096 S=8192 S=16384 auto "bytes"
for MN in "16 131072" "64 524288" "4096 131072"; do
  set -- $MN
  printf "%6d %9d %9s %9s %9s %9s %9s\n" $1 $2 \
    "$(pa $1 $2 --sample-s 4096)" "$(pa $1 $2 --sample-s 8192)" \
    "$(pa $1 $2 --sample-s 16384)" "$(pa $1 $2)" \
    "$(awk -v m=$1 'BEGIN{printf "%.0fKB", m*8192*4/1024}')"
done
echo
echo "reference: an empty dispatch on this box is 2.64us (knowledge/g0_floor_model.json)"
