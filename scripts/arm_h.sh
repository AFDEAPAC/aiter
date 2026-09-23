#!/bin/bash
# Paired sweep after the rebase. The baseline is the NEW upstream tip, not the
# one tonight started from: three of the six commits pulled in are stream-backend
# work plus 110 lines of routing in topk_select.py, so which backend serves a
# cell can have moved and the old 41-green baseline no longer describes it.
set -e
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
echo "=== rebased, with my five changes ==="
run /home/mh/aiter-topk  reb_new.json
echo "=== new upstream tip, same card, right after ==="
run /home/mh/aiter-base  reb_base.json
echo REBASE-SWEEP-DONE
