#!/bin/bash
set -e
run() {
  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
    --shm-size 32G -e HIP_VISIBLE_DEVICES=0 -e PYTHONPATH=/aiter \
    -v $1:/aiter -v /home/mh/topk-prefill-avo:/topk -w /topk \
    rocm/ali-private:ubuntu22.04_rocm10.0.0_cp313_torch2.12.0_vllm_dsv4_20260916 \
    bash -c "rm -rf /aiter/aiter/jit/build/module_top*k* /aiter/aiter/jit/module_top*k*.so
             cd /aiter && pip -q install -e . 2>&1|tail -1
             cd /topk && python3 -c 'import aiter; print(\"AITER:\", aiter.__file__)'
             python3 bench/select_ab_sweep.py --mode backends --seed 0 --out /topk/log/$2"
}
echo "=== with the tail read ==="
run /home/mh/aiter-topk  tail_new.json
echo "=== upstream tip, same card, right after ==="
run /home/mh/aiter-base  tail_base.json
echo TAIL-SWEEP-DONE
