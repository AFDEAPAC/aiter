#!/bin/bash
cd /topk
hipcc -O3 -std=c++17 --offload-arch=gfx950 scripts/bw_gx_floor.hip -o /tmp/bwf 2>&1|grep -i error|head -3
: > /tmp/floors.txt
while read M N PIPE NOW; do
  F=$(/tmp/bwf $M $N 2>/dev/null | sed -n 's/.*median_us= *\([0-9.]*\).*/\1/p' | sort -g | head -1)
  echo "$M $N $PIPE $NOW ${F:-0}" >> /tmp/floors.txt
done <<EOF
1 131072 3.7 19.8
8 131072 3.7 23.7
8 524288 4.5 31.3
8 1048576 6.7 34.9
16 131072 3.8 24.7
16 524288 5.5 34.9
16 1048576 10.6 40.9
32 1048576 19.6 49.6
64 1048576 38.1 65.4
128 131072 11.5 36.0
128 1048576 76.7 108.6
256 131072 21.1 46.6
512 131072 41.2 71.7
1024 131072 82.9 149.2
2048 131072 168.8 281.5
4096 131072 339.4 521.1
EOF
python3 - <<'PY'
print("read-the-row + write-the-candidates at phase_b's grid, best over gx and block width.")
print("target = pipe101_floor / 0.60, which is what the 60% line demands.")
print()
print("%6s %9s %9s %10s %9s %12s  %s" % ("M","N","floor","target","now","target/floor","verdict"))
for line in open("/tmp/floors.txt"):
    m, n, pipe, now, f = line.split()
    m, n = int(m), int(n); pipe, now, f = float(pipe), float(now), float(f)
    t = pipe / 0.6
    r = t / f if f else 0
    v = ("UNREACHABLE: 60% sits below one bare read" if r < 1.0 else
         "no room: 3 kernels inside 1.3x of one read" if r < 1.3 else
         "tight" if r < 2.0 else "REACHABLE")
    print("%6d %9d %9.2f %10.2f %9.2f %11.2fx  %s" % (m, n, f, t, now, r, v))
PY
