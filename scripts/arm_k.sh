#!/bin/bash
set -e
cd /home/mh/topk-prefill-avo
docker run --rm -v /home/mh/topk-prefill-avo:/topk -v /home/mh/aiter-topk:/aiter -w /topk \
  rocm/ali-private:ubuntu22.04_rocm10.0.0_cp313_torch2.12.0_vllm_dsv4_20260916 \
  bash -c "pip -q install clang-format==23.1.1 2>&1|tail -0
           python3 scripts/export_aiter_op.py --aiter /aiter
           python3 scripts/export_aiter_op.py --aiter /aiter --check && echo EXPORT-CHECK-CLEAN"
cd /home/mh/aiter-topk && git add -A csrc && git commit -q -m "topk: regenerate for the phase_c pass-0 fold

Regenerated from topk-prefill-avo by scripts/export_aiter_op.py.

Pass 1 of phase_c's radix select is the only unfiltered scan of all c keys and
costs 4.38us at m=512 n=131072 against 1.62 / 1.16 / 1.12 for passes 2, 3 and 4.
Counting pass 0's digits during the candidate read, which already holds the keys
in registers, removes most of it. use_prefill guards on start == 0 because
phase_c passes prefix_skip and common_prefix_passes can move the first executed
pass off 0.

Interleaved, three rounds each, three-kernel device total: 0.9611x at m=8
n=524288, 0.9618x at m=32 n=1048576, 0.9811x at m=256 n=131072, up to 0.9916x at
m=4096 n=131072. No shape slower." || echo "nothing to commit"
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
echo "=== seven changes ==="
run /home/mh/aiter-topk  sev_new.json
echo "=== upstream tip, same card, right after ==="
run /home/mh/aiter-base  sev_base.json
echo SEVEN-DONE
