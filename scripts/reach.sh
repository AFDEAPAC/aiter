#!/bin/bash
cd /topk
hipcc -O3 -std=c++17 --offload-arch=gfx950 scripts/bw_gx_floor.hip -o /tmp/bwf 2>&1|grep -i error|head -3
floor() { /tmp/bwf $1 $2 2>/dev/null | sed -n 's/.*median_us= *\([0-9.]*\).*/\1/p' | sort -g | head -1; }
echo "read-the-row + write-the-candidates, swept over gx and block width, best config."
echo "target = pipe101_floor / 0.60, i.e. what the 60% line demands."
printf "%6s %9s %10s %10s %10s %9s  %s\n" M N "floor us" "target us" "now us" "target/floor" verdict
while read M N PIPE NOW; do
  F=$(floor $M $N)
  awk -v m=$M -v n=$N -v f="$F" -v p=$PIPE -v u=$NOW 'BEGIN{
    t=p/0.6; r=(f>0)?t/f:0;
    v = (r < 1.0) ? "UNREACHABLE: 60% is below a bare read" :
        (r < 1.3) ? "no room: needs 3 kernels inside 1.3x of one read" :
        (r < 2.0) ? "tight" : "REACHABLE";
    printf "%6d %9d %10s %10.2f %10.2f %9.2fx  %s\n", m, n, f, t, u, r, v}'
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
