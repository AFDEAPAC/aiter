#!/bin/bash
cd /topk
echo "=== build ==="
make -j 2>&1 | tail -2
echo "=== correctness across distributions (must all be VERDICT PASS, rows_fail=0) ==="
for MN in "4096 131072" "4096 262144" "64 1048576" "16 524288"; do
  set -- $MN
  for D in gaussian uniform equal inf adversarial; do
    R=$(./benchmark_topk --mode verify --m $1 --n $2 --topk 2048 --dist $D --seed 0 2>/dev/null \
        | grep -E '^VERIFY|^VERDICT' | tr '\n' ' ')
    printf "   m=%-5d n=%-8d %-12s %s\n" $1 $2 "$D" "$(echo $R | sed -n 's/.*rows_fail=\([0-9]*\).*fallback_rows=\([0-9]*\).*\(PASS\|FAIL\).*/rows_fail=\1 fb=\2 \3/p')"
  done
done
echo "=== phase_b time (kernel trace) ==="
for MN in "4096 131072" "4096 262144" "4096 524288" "1024 262144" "128 524288"; do
  set -- $MN
  rm -rf /tmp/kt
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kt -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian --seed 0 \
    --warmup 3 --iters 10 --repeats 1 >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
d=collections.defaultdict(list)
for f in glob.glob('/tmp/kt/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        n=r['Kernel_Name'].split('(')[0]
        d[n].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
tot=0
out=[]
for n,v in d.items():
    if 'phase_' not in n: continue
    v=sorted(v)[len(v)//4:]
    m=sum(v)/len(v); tot+=m
    out.append((m,n))
out.sort(reverse=True)
print('   m=$1 n=$2  pipeline %.2f us   ' % tot + '  '.join('%s=%.2f'%(n.split('phase_')[1][:1],m) for m,n in out))
"
done
