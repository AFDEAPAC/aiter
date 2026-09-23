#!/bin/bash
set -e
cd /aiter && python3 /topk/scripts/ragpatch.py
pip -q install -e . 2>&1 | tail -1
rm -rf /aiter/aiter/jit/build/module_top_k_per_row /aiter/aiter/jit/module_top_k_per_row.so
cd /topk
rm -rf /tmp/krg
rocprofv3 --kernel-trace --output-format csv -d /tmp/krg -- python3 -u /topk/scripts/ragtest.py 2>&1 | grep -E "AITER|matches|Error|Traceback" | head -8
python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/krg/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        k=r['Kernel_Name'].split('(')[0][:52]
        if 'phase_' in k: agg[k].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
for k,v in sorted(agg.items()):
    v=sorted(v)
    print('   %-52s x%-4d mean %8.3f us'%(k,len(v),sum(v[len(v)//4:])/len(v[len(v)//4:])))
"
