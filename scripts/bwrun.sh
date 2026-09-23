#!/bin/bash
cd /topk
hipcc -O3 -std=c++17 --offload-arch=gfx950 scripts/bw_gx_floor.hip -o /tmp/bwfloor 2>&1 | grep -i error | head -3
echo "read the whole row + write 2867 candidates per row, at phase_b's grid shape."
echo "this is the floor the shipped pipeline is actually competing with."
printf "%6s %9s %12s %12s %10s\n" M N "bytes" "best us" "GB/s"
for MN in "1 131072" "8 131072" "8 524288" "8 1048576" "16 524288" "16 1048576" "32 1048576" "64 1048576" "128 131072" "512 131072" "1024 131072" "4096 131072"; do
  set -- $MN
  OUT=$(/tmp/bwfloor $1 $2 2>/dev/null | grep -oE "[0-9]+\.[0-9]+ *us" | tr -d " us" | sort -g | head -1)
  B=$(( $1 * $2 * 4 ))
  printf "%6d %9d %12d %12s %10s\n" $1 $2 $B "${OUT:-n/a}" \
    "$(awk -v b=$B -v t="${OUT:-0}" 'BEGIN{if(t>0)printf "%.0f",b/(t*1e3); else printf "-"}')"
done
