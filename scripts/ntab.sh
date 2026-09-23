#!/bin/bash
cd /topk
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1 | grep -i error | head -5
ks() {
  rm -rf /tmp/kq
  rocprofv3 --kernel-trace --output-format csv -d /tmp/kq -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 5 --iters 15 --repeats 1 "${@:3}" >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/kq/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        k=r['Kernel_Name']
        t=('a' if 'phase_a' in k else 'b' if 'phase_b' in k else 'c' if 'phase_c' in k else 'd' if 'phase_d' in k else None)
        if t: agg[t].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
tot=0
for t in 'abcd':
    v=sorted(agg[t])
    if v: tot+=sum(v[len(v)//4:])/len(v[len(v)//4:])
print('%.2f'%tot)
"
}
echo "three-kernel device total. --nt-load is the repo's runtime knob in load_f4"
echo "(all phases, branch in the innermost load); --nt-gate 1 is the new"
echo "compile-time template on phase_b only."
printf "%6s %9s %10s %10s %10s %10s\n" M N cached "runtime" "phase_b CT" "gate(-1)"
for MN in "4096 131072" "4096 1048576" "128 1048576" "128 131072" "64 262144"; do
  set -- $MN
  printf "%6d %9d %10s %10s %10s %10s\n" $1 $2 \
    "$(ks $1 $2 --nt-gate 0 --nt-load 0)" \
    "$(ks $1 $2 --nt-gate 0 --nt-load 1)" \
    "$(ks $1 $2 --nt-gate 1 --nt-load 0)" \
    "$(ks $1 $2 --nt-load 0)"
done
