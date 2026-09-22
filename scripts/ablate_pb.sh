#!/bin/bash
# Ablation: how much of phase_b is the candidate COMPACTION (ballot prefix +
# LDS staging + drain) rather than the load+compare?
#
# No code change. A tiny margin drives the sampled threshold so high that almost
# nothing passes, so `wtotal > 0` is false every iteration and the compact/drain
# block is skipped -- while the load, the 4 ballots and the 4 popcounts still
# run. The rest of the pipeline goes to the exact fallback and is ignored:
# rocprofv3 --kernel-trace gives phase_b's own duration.
cd /topk
trace() {  # $1=M $2=N $3=margin-flag...
  rm -rf /tmp/kt; shift_m=$1; shift_n=$2; shift 2
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kt -- \
    ./benchmark_topk --mode time --m $shift_m --n $shift_n --topk 2048 \
    --dist gaussian --seed 0 --warmup 3 --iters 10 --repeats 1 "$@" >/dev/null 2>&1
  python3 - <<PY
import csv,glob,collections
d=collections.defaultdict(list)
for f in glob.glob("/tmp/kt/**/*kernel_trace.csv",recursive=True):
    for r in csv.DictReader(open(f)):
        n=r["Kernel_Name"].split("(")[0]
        d[n].append((int(r["End_Timestamp"])-int(r["Start_Timestamp"]))/1e3)
for n,v in sorted(d.items(), key=lambda e:-sum(e[1])):
    if "phase_b" in n:
        v=sorted(v)[len(v)//4:]   # drop cold
        print("      %-34s %8.2f us  x%d" % (n[:34], sum(v)/len(v), len(v)))
PY
}
for MN in "4096 131072" "4096 262144"; do
  set -- $MN
  echo "=== M=$1 N=$2 ==="
  echo "   normal (margin auto):"; trace $1 $2
  echo "   ablated (margin 0.02 -> almost nothing passes):"; trace $1 $2 --margin 0.02
done
