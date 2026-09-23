#!/bin/bash
cd /topk
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -3
rm -rf /tmp/att && mkdir -p /tmp/att
cat > /tmp/att_in.txt <<'EOF'
att: TARGET_CU=0
SIMD_SELECT=0xF
ISA_CAPTURE_MODE=2
KERNEL=phase_a_threshold
EOF
timeout 1200 rocprofv3 -i /tmp/att_in.txt --att-library-path /opt/rocm/lib -d /tmp/att --output-format csv -- \
  ./benchmark_topk --mode time --m 8 --n 524288 --topk 2048 --dist gaussian \
  --seed 0 --warmup 0 --iters 1 --repeats 1 > /tmp/att.log 2>&1
echo "att exit=$?"
grep -iE "error|fail|abort" /tmp/att.log | head -5
find /tmp/att -maxdepth 3 | head -20
