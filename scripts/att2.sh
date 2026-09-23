#!/bin/bash
cd /topk
mkdir -p /topk/log/att
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -3
cat > /topk/log/att/in.txt <<'EOF'
att: TARGET_CU=0 SIMD_SELECT=0xF ISA_CAPTURE_MODE=2 KERNEL=phase_a_threshold
EOF
rm -rf /topk/log/att/out
timeout 1200 rocprofv3 -i /topk/log/att/in.txt -d /topk/log/att/out --output-format csv -- \
  ./benchmark_topk --mode time --m 8 --n 524288 --topk 2048 --dist gaussian \
  --seed 0 --warmup 0 --iters 1 --repeats 1 > /topk/log/att/run.log 2>&1
echo "exit=$?"
echo "--- tail ---"
tail -20 /topk/log/att/run.log
echo "--- produced ---"
find /topk/log/att/out -maxdepth 3 2>/dev/null | head -12
