#!/bin/bash
cd /topk
echo "=== WSTAGE_WAVES=8 (shipped), cf-block 1024 ==="
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
./benchmark_topk --mode time --m 8 --n 524288 --topk 2048 --dist gaussian --seed 0 \
  --warmup 2 --iters 3 --repeats 1 --coop-g 1 --cf-block 1024 2>&1 | tail -6
echo
echo "=== WSTAGE_WAVES=16, cf-block 1024, coop-g 1 ==="
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -DWSTAGE_WAVES_OVERRIDE=16 -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -2
./benchmark_topk --mode time --m 8 --n 524288 --topk 2048 --dist gaussian --seed 0 \
  --warmup 2 --iters 3 --repeats 1 --coop-g 1 --cf-block 1024 2>&1 | tail -6
echo
echo "=== WSTAGE_WAVES=16, cf-block 1024, coop-g 2 (this one worked before) ==="
./benchmark_topk --mode time --m 8 --n 524288 --topk 2048 --dist gaussian --seed 0 \
  --warmup 2 --iters 3 --repeats 1 --coop-g 2 --cf-block 1024 2>&1 | tail -4
echo
echo "=== what LDS each kernel reserves at WSTAGE_WAVES=16 ==="
python3 -c "
W=16; C=320
print('  wbuf  = WSTAGE_WAVES * WSTAGE_CAP * 8 B = %d B = %.1f KB' % (W*C*8, W*C*8/1024))
print('  phase_c dynamic = cap*4 (keys) + cap*4 (idx); cap=8192 -> %.1f KB' % (8192*8/1024))
print('  gfx950 LDS per CU = 160 KB')
"
