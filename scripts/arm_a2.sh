#!/bin/bash
# Re-baseline immediately after arm D. The 234 cells neither change can reach
# drifted to a median of 1.0429 between arm A at 11:45 and arm D at 12:25, so
# the two arms were not measured on the same machine conditions. This runs the
# pre-tonight state again, back to back with D, on the same card.
set -e
while pgrep -f select_ab_swee[p] > /dev/null; do sleep 20; done
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  --shm-size 32G -e HIP_VISIBLE_DEVICES=0 -e PYTHONPATH=/aiter \
  -v /home/mh/aiter-base:/aiter -v /home/mh/topk-prefill-avo:/topk -w /topk \
  rocm/ali-private:ubuntu22.04_rocm10.0.0_cp313_torch2.12.0_vllm_dsv4_20260916 \
  bash -c "rm -rf /aiter/aiter/jit/build/module_top_k_per_row /aiter/aiter/jit/module_top_k_per_row.so
           cd /aiter && pip -q install -e . 2>&1|tail -1
           cd /topk && python3 -c 'import aiter; print(\"AITER:\", aiter.__file__)'
           python3 bench/select_ab_sweep.py --mode backends --seed 0 --out /topk/log/clean_base2.json"
echo REBASELINE-DONE
