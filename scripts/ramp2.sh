#!/bin/bash
cd /topk
mkdir -p /topk/log/ramp && rm -rf /topk/log/ramp/out
hipcc -O3 -std=c++17 --offload-arch=gfx950 scripts/null_kernel_ramp.hip -o /tmp/nullramp 2>&1 | grep -i error | head -5
rocprofv3 --kernel-trace --output-format csv -d /topk/log/ramp/out -- /tmp/nullramp 40 > /topk/log/ramp/prog.out 2>&1
grep -i "LAUNCH-FAILED" /topk/log/ramp/prog.out | head -3
echo "--- trace columns ---"
head -1 $(find /topk/log/ramp/out -name "*kernel_trace.csv" | head -1)
echo
python3 - <<'PY'
import csv, glob
rows = []
for f in glob.glob('/topk/log/ramp/out/**/*kernel_trace.csv', recursive=True):
    for r in csv.DictReader(open(f)):
        if 'nop_kernel' in r['Kernel_Name']:
            rows.append(r)
print("nop_kernel dispatches captured:", len(rows))
if rows:
    print("available keys:", ", ".join(k for k in rows[0].keys()))
PY
