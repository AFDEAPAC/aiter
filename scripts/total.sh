#!/bin/bash
cd /topk
ks() {
  rm -rf /tmp/kt$3
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kt$3 -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 5 --iters 15 --repeats 1 >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/kt$3/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        k=r['Kernel_Name']
        tag=('a' if 'phase_a' in k else 'b' if 'phase_b' in k else 'c' if 'phase_c' in k else 'd' if 'phase_d' in k else None)
        if tag: agg[tag].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
out=[]; tot=0.0
for t in 'abcd':
    v=sorted(agg[t])
    if not v: continue
    m=sum(v[len(v)//4:])/len(v[len(v)//4:]); tot+=m
    out.append('%s=%7.2f'%(t,m))
print('  '.join(out)+'   TOTAL=%8.2f'%tot)
"
}
run() { hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc $2 -o benchmark_topk benchmark_topk.hip.cpp 2>&1 | grep -i error | head -3
        printf "%-42s n=131072  %s\n" "$1" "$(ks 4096 131072 x)"
        printf "%-42s n=262144  %s\n" ""    "$(ks 4096 262144 y)"; }
run "new: per-wave + non-temporal (shipped now)" ""
run "old: block-serial walk, ordinary store"     "-DABLATE_EPI=91"
