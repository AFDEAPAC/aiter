#!/bin/bash
cd /topk
hipcc -O3 -std=c++17 --offload-arch=gfx950 -Icsrc -o benchmark_topk benchmark_topk.hip.cpp 2>&1|grep -i error|head -3
rm -rf /topk/workloads/pa8
timeout 2400 rocprof-compute profile -n pa8 --no-roof -k phase_a_threshold -- \
  ./benchmark_topk --mode time --m 8 --n 524288 --topk 2048 --dist gaussian \
  --seed 0 --warmup 3 --iters 5 --repeats 1 > /tmp/rpc_prof.log 2>&1
echo "profile exit=$?"
tail -6 /tmp/rpc_prof.log
echo
ls -d /topk/workloads/pa8/* 2>/dev/null
