#!/bin/bash
cd /topk
python3 - <<'PY'
import csv, glob, collections
agg = collections.defaultdict(list)
for f in glob.glob('/topk/log/ramp/out/**/*kernel_trace.csv', recursive=True):
    for r in csv.DictReader(open(f)):
        if 'nop_kernel' not in r['Kernel_Name']: continue
        wg = int(r['Workgroup_Size_X'])
        blocks = int(r['Grid_Size_X']) // max(wg, 1)
        lds = int(r['LDS_Block_Size'])
        d = (int(r['End_Timestamp']) - int(r['Start_Timestamp'])) / 1e3
        agg[(lds, wg, blocks)].append(d)
print("A kernel that stores one byte per block. Device time, us.")
print("Upper three quartiles, the same rule the sweeps use.")
print()
ldss = sorted({k[0] for k in agg}); wgs = sorted({k[1] for k in agg})
blks = sorted({k[2] for k in agg})
for lds in ldss:
    print("dynamic LDS = %d B" % lds)
    print("  %8s" % "blocks", end="")
    for w in wgs: print(" %12s" % ("wg%d" % w), end="")
    print()
    for b in blks:
        print("  %8d" % b, end="")
        for w in wgs:
            v = sorted(agg.get((lds, w, b), []))
            if not v: print(" %12s" % "-", end=""); continue
            v = v[len(v)//4:]
            print(" %12.3f" % (sum(v)/len(v)), end="")
        print()
    print()
PY
