#!/bin/bash
cd /topk
for MN in "2048 131072" "256 131072"; do
  set -- $MN
  rm -rf /tmp/kai
  M=$1 N=$2 rocprofv3 --kernel-trace --output-format csv -d /tmp/kai -- \
    python3 -u /topk/scripts/trace_aiter.py > /tmp/ta.out 2>&1
  echo "m=$1 n=$2 through aiter.topk_select:"
  grep -E "^AITER|Error|Traceback" /tmp/ta.out | head -3
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/kai/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        agg[r['Kernel_Name'].split('(')[0][:56]].append(
            (int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
tot=0
for k,v in sorted(agg.items(), key=lambda z:-sum(z[1])):
    v=sorted(v); m=sum(v[len(v)//4:])/len(v[len(v)//4:]); n=len(v)
    per_iter = n/25.0
    tot += m*per_iter if n>=20 else 0
    print('   %-56s x%-4d mean %8.3f us  %s' % (k,n,m,'per-call %.2f'%per_iter if n>=20 else 'setup'))
print('   %-56s      %8.3f us' % ('PER-CALL SUM',tot))
"
  echo
done
