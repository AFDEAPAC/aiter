#!/bin/bash
cd /topk
ka() {
  rm -rf /tmp/klf
  rocprofv3 --kernel-trace --output-format csv -d /tmp/klf -- \
    ./benchmark_topk --mode time --m $1 --n $2 --topk 2048 --dist gaussian \
    --seed 0 --warmup 5 --iters 20 --repeats 1 "${@:3}" >/dev/null 2>&1
  python3 -c "
import csv,glob
v=[]
for f in glob.glob('/tmp/klf/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        if 'phase_a' in r['Kernel_Name']:
            v.append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
v=sorted(v)
print('%.2f'%(sum(v[len(v)//4:])/len(v[len(v)//4:])) if v else 'n/a')
"
}
echo "phase_a's LOAD alone (ABLATE_PA=1: sample, fill LDS, no select)."
echo "S sets both how many bytes are read and how they are spread:"
echo "chunks = S/64 contiguous 256B chunks, spaced N/chunks apart."
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -DABLATE_PA=1 -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
printf "%8s %8s %10s %12s %12s %10s\n" S chunks "stride(el)" "bytes m=4096" "load us" "GB/s"
for S in 8192 4096 2048 1024 512 256; do
  T=$(ka 4096 131072 --sample-s $S --s-rule 0)
  printf "%8d %8d %10d %12d %12s %10s\n" $S $((S/64)) $((131072/(S/64))) $((4096*S*4)) "$T" \
    "$(awk -v b=$((4096*S*4)) -v t=$T 'BEGIN{printf "%.0f",b/(t*1e3)}')"
done
echo
echo "the same, m=256:"
for S in 8192 2048 512; do
  T=$(ka 256 131072 --sample-s $S --s-rule 0)
  printf "%8d %8d %10d %12d %12s %10s\n" $S $((S/64)) $((131072/(S/64))) $((256*S*4)) "$T" \
    "$(awk -v b=$((256*S*4)) -v t=$T 'BEGIN{printf "%.0f",b/(t*1e3)}')"
done
