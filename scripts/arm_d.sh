#!/bin/bash
# Arm D: both the load and the store behind the 2^27 gate. Same card, after C.
set -e
while pgrep -f select_ab_swee[p] > /dev/null; do sleep 20; done
cd /home/mh/topk-prefill-avo
docker run --rm -v /home/mh/topk-prefill-avo:/topk -v /home/mh/aiter-topk:/aiter -w /topk \
  rocm/ali-private:ubuntu22.04_rocm10.0.0_cp313_torch2.12.0_vllm_dsv4_20260916 \
  bash -c "pip -q install clang-format==23.1.1 2>/dev/null
           python3 scripts/export_aiter_op.py --aiter /aiter
           python3 scripts/export_aiter_op.py --aiter /aiter --check && echo EXPORT-CHECK-CLEAN"
cd /home/mh/aiter-topk && git add -A csrc && git commit -q -m "topk: regenerate for the gated candidate store

Regenerated from topk-prefill-avo by scripts/export_aiter_op.py; the source
change is that repo's csrc/topk_generalize.hip.hpp.

phase_b's candidate store now takes the same M*pitch >= 2^27 gate as its loads.
Measured per-wave-with-an-ordinary-store against per-wave-with-a-non-temporal
one, the store crosses at the same size: 18.49us against 19.72us at m=1
n=131072, and 2802.07us against 2761.22us at m=4096 n=1048576. Writing each
wave's own run rather than walking all eight is a win everywhere, so only the
non-temporal part is gated." || echo "nothing to commit"
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  --shm-size 32G -e HIP_VISIBLE_DEVICES=0 -e PYTHONPATH=/aiter \
  -v /home/mh/aiter-topk:/aiter -v /home/mh/topk-prefill-avo:/topk -w /topk \
  rocm/ali-private:ubuntu22.04_rocm10.0.0_cp313_torch2.12.0_vllm_dsv4_20260916 \
  bash -c "rm -rf /aiter/aiter/jit/build/module_top_k_per_row /aiter/aiter/jit/module_top_k_per_row.so
           cd /aiter && pip -q install -e . 2>&1|tail -1
           cd /topk && python3 -c 'import aiter; print(\"AITER:\", aiter.__file__)'
           python3 bench/select_ab_sweep.py --mode backends --seed 0 --out /topk/log/clean_gated.json"
echo ARM-D-DONE
