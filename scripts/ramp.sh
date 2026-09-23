#!/bin/bash
cd /topk
hipcc -O3 -std=c++17 --offload-arch=gfx950 scripts/null_kernel_ramp.hip -o /tmp/nullramp 2>&1 | grep -i error | head -5
rm -rf /tmp/kramp
rocprofv3 --kernel-trace --output-format csv -d /tmp/kramp -- /tmp/nullramp 40 > /tmp/ramp.out 2>&1
grep -i "LAUNCH-FAILED" /tmp/ramp.out | head -3
python3 - <<'PY'
import csv, glob, itertools
ev = []
for f in glob.glob('/tmp/kramp/**/*kernel_trace.csv', recursive=True):
    for r in csv.DictReader(open(f)):
        if 'nop_kernel' in r['Kernel_Name']:
            ev.append((int(r['Start_Timestamp']),
                       (int(r['End_Timestamp']) - int(r['Start_Timestamp'])) / 1e3,
                       int(r['Grid_Size']), int(r['Workgroup_Size']),
                       int(r.get('LDS_Block_Size', 0) or 0)))
ev.sort()
agg = {}
for _, d, gs, wg, lds in ev:
    key = (gs // max(wg,1), wg, lds)
    agg.setdefault(key, []).append(d)
print("device time of a kernel that stores one byte, by grid / block / dynamic LDS")
print("%8s %8s %9s %10s %10s" % ("blocks", "block", "lds_B", "median_us", "n"))
for key in sorted(agg):
    v = sorted(agg[key])
    v = v[len(v)//4:]           # drop the cold quarter, same rule as the sweeps
    print("%8d %8d %9d %10.3f %10d" % (key[0], key[1], key[2], sum(v)/len(v), len(agg[key])))
PY
