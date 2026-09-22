#!/bin/bash
# Rule 0's S = R_TARGET * N / (margin * K) saturates at SAMPLE_S_MAX for
# N >= 262144 -- the law asks for 16365 at N=262144 and 65462 at N=1048576 and
# gets 16384 either way, so it stops being a law and becomes a constant. The
# file already says R_TARGET "does not transfer" and that rule 1 was written to
# derive the requirement instead, but rule 1 is gated to M <= 32.
#
# Sweep S per cell at M >= 64, N >= 128K, to see whether the measured optimum
# follows a rule or is per-cell noise.
cd /topk
run() {
  ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian --seed 0 \
    --warmup 20 --iters 100 --repeats 3 ${3:+--sample-s $3} 2>/dev/null \
  | grep '^RESULT' | sed -n 's/.*wall_ms=\([0-9.]*\).*fallback_rows=\([0-9]*\).*/\1 \2/p'
}
printf "%6s %9s %10s %s\n" M N auto "S=6144 8192 10240 12288 14336 16384 (ratio, fb)"
for M in 64 128 256 512 1024 2048 4096; do
  for N in 131072 262144 524288 1048576; do
    B=$(run $M $N); BW=$(echo $B|cut -d' ' -f1)
    [ -z "$BW" ] && continue
    line=$(printf "%6d %9d %10s " $M $N "$BW")
    best=999; bestS=auto
    for S in 6144 8192 10240 12288 14336 16384; do
      R=$(run $M $N $S); W=$(echo $R|cut -d' ' -f1); F=$(echo $R|cut -d' ' -f2)
      [ -z "$W" ] && { line="$line   -   "; continue; }
      r=$(awk -v a=$BW -v b=$W 'BEGIN{printf "%.3f",b/a}')
      line="$line $r$([ "$F" != 0 ] && echo "!" || echo " ")"
      awk -v r=$r -v b=$best 'BEGIN{exit !(r<b)}' && { best=$r; bestS=$S; }
    done
    echo "$line  best=$bestS($best)"
  done
done
