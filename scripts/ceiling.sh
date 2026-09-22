#!/bin/bash
# Ceiling-first, the way this repo priced ABLATE_HIST_ATOMIC: what is the
# efficiency of every N>=128K cell if phase_a and phase_c cost NOTHING? That
# bounds every remaining direction at once, because phase_b is already at 94% of
# its streaming floor and its excess is the candidate write, which is closed.
cd /topk
echo "M N floor_us a_us b_us c_us total_us eff_now eff_if_ac_free"
for M in 1 4 16 64 128 256 512 1024 2048 4096; do
  for N in 131072 262144 524288 1048576; do
    rm -rf /tmp/kt
    rocprofv3 --kernel-trace --output-format csv -d /tmp/kt -- \
      ./benchmark_topk --mode time --m $M --n $N --topk 2048 --dist gaussian \
      --seed 0 --warmup 5 --iters 15 --repeats 1 >/dev/null 2>&1
    python3 -c "
import csv,glob,collections
d=collections.defaultdict(list)
for f in glob.glob('/tmp/kt/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        n=r['Kernel_Name'].split('(')[0]
        if 'phase_' in n: d[n].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
g={}
for n,v in d.items():
    v=sorted(v)[len(v)//4:]
    g[n.split('phase_')[1][0]]=sum(v)/len(v)
a,b,c=g.get('a',0),g.get('b',0),g.get('c',0)
print('$M $N %.3f %.3f %.3f'%(a,b,c))
"
  done
done
