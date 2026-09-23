#!/bin/bash
cd /topk
echo "=== the generated .cu carries the gate ==="
grep -n "ragged_eff" /aiter/csrc/kernels/topk_per_row_sampled_kernels.cu | head -3
echo
echo "=== what the gate costs: widths that lose the plain path ==="
ks() {
  rm -rf /tmp/fx
  rocprofv3 --kernel-trace --output-format csv -d /tmp/fx -- \
    python3 -c "
import torch, aiter
M,N,K=$1,$2,2048
torch.manual_seed(0); torch.cuda.manual_seed_all(0)
x=torch.randn(M,N,device='cuda',dtype=torch.float32)
for _ in range(8): aiter.topk_select(x,K)
torch.cuda.synchronize()
for _ in range(20): aiter.topk_select(x,K)
torch.cuda.synchronize()
" >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
agg=collections.defaultdict(list)
for f in glob.glob('/tmp/fx/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        k=r['Kernel_Name']
        if 'phase_' in k: agg[k.split('<')[0]].append((int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3)
tot=0
for k,v in agg.items():
    v=sorted(v)[8:]
    if v: tot+=sum(v)/len(v)
print('%.2f'%tot)
"
}
printf "%6s %9s %6s %10s\n" M N "N%4" "per-call us"
for MN in "512 131072" "512 131073" "512 131074" "512 131076" "256 262144" "256 262145" "256 262148"; do
  set -- $MN
  printf "%6d %9d %6d %10s\n" $1 $2 $(( $2 % 4 )) "$(ks $1 $2)"
done
echo
echo "=== benchmark 245-case verify (unchanged path, sanity) ==="
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -3
fail=0
for m in 1 16 64 128 256 1024 4096; do for n in 131072 262144 524288 1048576 131073 262146 1048577; do for d in gaussian adversarial all_equal uniform inf; do
  r=$(./benchmark_topk --mode verify --m $m --n $n --topk 2048 --dist $d --seed 0 2>&1|grep -o "VERDICT [A-Z]*"|head -1)
  [ "$r" = "VERDICT PASS" ] || { echo "  FAIL m=$m n=$n dist=$d -> $r"; fail=1; }
done; done; done
[ $fail = 0 ] && echo "  all 245 PASS"
