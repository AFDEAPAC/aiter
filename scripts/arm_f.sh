#!/bin/bash
# Final paired sweep: all five changes, then the pre-tonight state on the same
# card right after, so the two arms share machine conditions. The A/A control
# earlier showed cells above 100us reproduce to 0.15% while small cells drift up
# to 4% across an hour, which is why the pair has to be back to back.
set -e
while pgrep -f fbrat[e] > /dev/null; do sleep 30; done
sleep 10
run() {
  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
    --shm-size 32G -e HIP_VISIBLE_DEVICES=0 -e PYTHONPATH=/aiter \
    -v $1:/aiter -v /home/mh/topk-prefill-avo:/topk -w /topk \
    rocm/ali-private:ubuntu22.04_rocm10.0.0_cp313_torch2.12.0_vllm_dsv4_20260916 \
    bash -c "rm -rf /aiter/aiter/jit/build/module_top_k_per_row /aiter/aiter/jit/module_top_k_per_row.so
             cd /aiter && pip -q install -e . 2>&1|tail -1
             cd /topk && python3 -c 'import aiter; print(\"AITER:\", aiter.__file__)'
             python3 bench/select_ab_sweep.py --mode backends --seed 0 --out /topk/log/$2"
}
echo "=== arm F: all five changes ==="
run /home/mh/aiter-topk  five_new.json
echo "=== arm F0: pre-tonight, same card, right after ==="
run /home/mh/aiter-base  five_base.json
echo ARM-F-DONE
