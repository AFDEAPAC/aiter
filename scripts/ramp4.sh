#!/bin/bash
cd /topk
python3 - <<'PY'
import csv, glob
rows = []
for f in glob.glob('/topk/log/ramp/out/**/*kernel_trace.csv', recursive=True):
    for r in csv.DictReader(open(f)):
        if 'nop_kernel' in r['Kernel_Name']:
            rows.append((int(r['Dispatch_Id']),
                         (int(r['End_Timestamp']) - int(r['Start_Timestamp'])) / 1e3,
                         int(r['Grid_Size_X']) // max(int(r['Workgroup_Size_X']), 1),
                         int(r['Workgroup_Size_X'])))
rows.sort()
grids  = [8, 16, 32, 64, 128, 256, 512, 1024, 4096]
blocks = [256, 512, 1024]
ldsb   = [0, 8192, 32768, 65536]
PER = 45
n = len(rows) // PER
print("captured %d dispatches = %d configs of %d" % (len(rows), n, PER))
res = {}
ok = True
for ci in range(n):
    li, rem = divmod(ci, len(blocks) * len(grids))
    bi, gi = divmod(rem, len(grids))
    chunk = rows[ci*PER:(ci+1)*PER]
    # the trace must agree with the order we think we launched in
    if chunk[0][2] != grids[gi] or chunk[0][3] != blocks[bi]:
        ok = False
        print("ORDER MISMATCH at config %d: trace says %d blocks x %d, expected %d x %d"
              % (ci, chunk[0][2], chunk[0][3], grids[gi], blocks[bi]))
        break
    v = sorted(d for _, d, _, _ in chunk[5:])
    v = v[len(v)//4:]
    res[(ldsb[li], blocks[bi], grids[gi])] = sum(v)/len(v)
if ok:
    print("order verified against the trace's own grid and block fields")
    print()
    print("device time us, a kernel that stores one byte per block")
    print("%8s %8s" % ("blocks", "block"), end="")
    for L in ldsb: print(" %11s" % ("lds%dKB" % (L//1024)), end="")
    print()
    for b in blocks:
        for g in grids:
            print("%8d %8d" % (g, b), end="")
            for L in ldsb:
                x = res.get((L, b, g))
                print(" %11s" % ("%.3f" % x if x else "-"), end="")
            print()
PY
