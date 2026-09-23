#!/bin/bash
cd /topk
rm -rf /tmp/krg2
rocprofv3 --kernel-trace --output-format csv -d /tmp/krg2 -- python3 -u /topk/scripts/ragtest2.py 2>&1 | grep -E "AITER|matches|Error" | head -8
python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/krg2/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        k=r['Kernel_Name'].split('(')[0][:50]
        if 'phase_' in k: agg[k].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
# 3 shapes x 29 calls each, in order; split by shape
for k,v in sorted(agg.items()):
    n=len(v); per=n//3
    parts=[]
    for i in range(3):
        c=sorted(v[i*per:(i+1)*per])
        parts.append('%8.2f'%(sum(c[len(c)//4:])/len(c[len(c)//4:])))
    print('   %-50s %s'%(k,' '.join(parts)))
print('   %-50s   m=2048   m=256   m=512'%'(shapes: n=131072, n=131072, n=262144)')
"
