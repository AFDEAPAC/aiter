#!/bin/bash
# Same pair, but every topk JIT module is cleared, not just module_top_k_per_row.
# The previous run left module_topk_plain.so from before the rebase in the
# with-changes worktree, so its 41 `plain` cells ran a kernel that predates the
# upstream work they were being compared against. .evo/config-v5.yaml already
# recorded this class of bug once; this is the same one through a different door.
set -e
run() {
  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
    --shm-size 32G -e HIP_VISIBLE_DEVICES=0 -e PYTHONPATH=/aiter \
    -v $1:/aiter -v /home/mh/topk-prefill-avo:/topk -w /topk \
    rocm/ali-private:ubuntu22.04_rocm10.0.0_cp313_torch2.12.0_vllm_dsv4_20260916 \
    bash -c "rm -rf /aiter/aiter/jit/build/module_top*k* /aiter/aiter/jit/module_top*k*.so
             ls /aiter/aiter/jit/*.so 2>/dev/null | sed 's|.*/|  left: |'
             cd /aiter && pip -q install -e . 2>&1|tail -1
             cd /topk && python3 -c 'import aiter; print(\"AITER:\", aiter.__file__)'
             python3 bench/select_ab_sweep.py --mode backends --seed 0 --out /topk/log/$2"
}
echo "=== rebased, with my five changes ==="
run /home/mh/aiter-topk  cl2_new.json
echo "=== new upstream tip, same card, right after ==="
run /home/mh/aiter-base  cl2_base.json
echo CLEAN-PAIR-DONE
