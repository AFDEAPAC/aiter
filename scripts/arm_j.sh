#!/bin/bash
# Paired sweep for the pass-0 fold. Every topk JIT module cleared in both arms.
set -e
cd /home/mh/topk-prefill-avo
docker run --rm -v /home/mh/topk-prefill-avo:/topk -v /home/mh/aiter-topk:/aiter -w /topk \
  rocm/ali-private:ubuntu22.04_rocm10.0.0_cp313_torch2.12.0_vllm_dsv4_20260916 \
  bash -c "pip -q install clang-format==23.1.1 2>&1|tail -0
           python3 scripts/export_aiter_op.py --aiter /aiter
           python3 scripts/export_aiter_op.py --aiter /aiter --check && echo EXPORT-CHECK-CLEAN"
cd /home/mh/aiter-topk && git add -A csrc && git commit -q -m "topk: regenerate for the phase_a pass-0 fold

Regenerated from topk-prefill-avo by scripts/export_aiter_op.py.

An ATT trace with line tables puts 40% of phase_a's latency on the wait for the
first sample load and 37% on the barriers inside block_select_lds; rocprof-compute
reports 1.38% CU utilisation at m=8, so nothing hides it. Counting pass 0's
digits during that load removes a full scan of s_keys and the barrier ending it.
Interleaved, three rounds each: 0.9767x at m=16 n=524288 up to 1.0001x at m=4096
n=131072, no shape slower." || echo "nothing to commit"
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
echo "=== six changes ==="
run /home/mh/aiter-topk  six_new.json
echo "=== upstream tip, same card, right after ==="
run /home/mh/aiter-base  six_base.json
echo SIX-DONE
