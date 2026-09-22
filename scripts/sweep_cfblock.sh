#!/bin/bash
# phase_b's block width is the one knob on it never swept. It sets waves/block,
# which sets how many wave-private LDS staging buffers share a block and how
# often each wave drains (the guard fires at bcnt > WSTAGE_CAP - 4*WAVE_SIZE
# = 64 entries). The ablation says 100-152us of phase_b is the compaction and
# that the part which scales with ITERATIONS rather than candidates is ~52us.
cd /topk
pbtime() {  # M N extra...
  local M=$1 N=$2; shift 2
  rm -rf /tmp/kt
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kt -- \
    ./benchmark_topk --mode time --m $M --n $N --topk 2048 --dist gaussian \
    --seed 0 --warmup 3 --iters 10 --repeats 1 "$@" >/dev/null 2>&1
  python3 -c "
import csv,glob
v=[]
for f in glob.glob('/tmp/kt/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        if 'phase_b' in r['Kernel_Name']:
            v.append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
v=sorted(v)[len(v)//4:]
print('%.2f'%(sum(v)/len(v)) if v else 'n/a')
"
}
for MN in "4096 131072" "4096 262144"; do
  set -- $MN
  echo "=== M=$1 N=$2 : phase_b us by --cf-block ==="
  BASE=$(pbtime $1 $2)
  printf "   %-10s %10s %9s\n" auto "$BASE" "1.000x"
  for B in 64 128 256 512 1024; do
    T=$(pbtime $1 $2 --cf-block $B)
    printf "   %-10s %10s %9s\n" "$B" "$T" \
      "$(awk -v a=$BASE -v b=$T 'BEGIN{if(a+0>0&&b+0>0)printf "%.3fx",b/a; else printf "-"}')"
  done
done
