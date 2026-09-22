#!/bin/bash
# A 2% claim on a box whose unchanged-cell p95 is 1.077x needs interleaved
# repeats, not one reading each. A/B/A/B x 6, same process, same clock state.
cd /topk
run() {
  ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian --seed 0 \
    --warmup 20 --iters 100 --repeats 3 ${3:+--sample-s $3} 2>/dev/null \
  | grep '^RESULT' | sed -n 's/.*wall_ms=\([0-9.]*\).*/\1/p'
}
for MN in "4096 262144" "512 262144" "4096 131072"; do
  set -- $MN
  A=(); B=()
  for i in 1 2 3 4 5 6; do
    A+=($(run $1 $2)); B+=($(run $1 $2 12288))
  done
  python3 -c "
import statistics as st
a=[float(x) for x in '${A[*]}'.split()]
b=[float(x) for x in '${B[*]}'.split()]
print('M=$1 N=$2')
print('   auto     median %.4f  min %.4f  max %.4f  (n=%d)'%(st.median(a),min(a),max(a),len(a)))
print('   S=12288  median %.4f  min %.4f  max %.4f'%(st.median(b),min(b),max(b)))
print('   ratio of medians %.4fx   best-vs-best %.4fx'%(st.median(b)/st.median(a),min(b)/min(a)))
"
done
