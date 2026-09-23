#!/bin/bash
# The fallback gate on the FINAL state. The earlier run predates the phase_a
# unroll and the ragged dispatch. Neither can move the shape plan -- the unroll
# only reorders loads and params_for still sizes by geometry_k_ragged -- but
# "cannot" is an argument, not a measurement.
set -e
while pgrep -f select_ab_swee[p] > /dev/null; do sleep 30; done
sleep 10
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  --shm-size 32G -e HIP_VISIBLE_DEVICES=0 -e PYTHONPATH=/aiter \
  -v /home/mh/aiter-topk:/aiter -v /home/mh/topk-prefill-avo:/topk -w /topk \
  rocm/ali-private:ubuntu22.04_rocm10.0.0_cp313_torch2.12.0_vllm_dsv4_20260916 \
  bash -c "cd /aiter && pip -q install -e . 2>&1|tail -1
           cd /topk && python3 -c 'import aiter; print(\"AITER:\", aiter.__file__)'
           python3 -u bench/fbrate.py"
echo FINAL-FBRATE-DONE
