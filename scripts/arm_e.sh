#!/bin/bash
# Arm E: all four changes. Waits for fbrate so the box is quiet, then exports,
# sweeps, and immediately re-sweeps the pre-tonight state on the same card so the
# pair is contemporaneous. The A/A control earlier showed cells above 100us
# reproduce to 0.15% but small cells drift up to 4% across an hour.
set -e
while pgrep -f fbrat[e] > /dev/null; do sleep 30; done
sleep 10
cd /home/mh/topk-prefill-avo
docker run --rm -v /home/mh/topk-prefill-avo:/topk -v /home/mh/aiter-topk:/aiter -w /topk \
  rocm/ali-private:ubuntu22.04_rocm10.0.0_cp313_torch2.12.0_vllm_dsv4_20260916 \
  bash -c "pip -q install clang-format==23.1.1 2>/dev/null
           python3 scripts/export_aiter_op.py --aiter /aiter
           python3 scripts/export_aiter_op.py --aiter /aiter --check && echo EXPORT-CHECK-CLEAN"
cd /home/mh/aiter-topk && git add -A csrc && git commit -q -m "topk: regenerate for the phase_a sample-load unroll

Regenerated from topk-prefill-avo by scripts/export_aiter_op.py.

phase_a's sampler is latency-bound: with the select ablated away it measures
18.10us at m=4096 n=131072 with S=4096 and 18.09us at S=512, eight times fewer
bytes for the same time, because each thread issues one load and the block waits
out a single round trip. Unrolling the sample loop by four puts several rounds
in flight: 0.9875x at m=4096 n=131072 and 0.9868x at n=262144 on the
three-kernel total, interleaved three rounds each and non-overlapping, and
neutral at every M where one block-round covers all the rows." || echo "nothing to commit"
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
echo "=== arm E: all four changes ==="
run /home/mh/aiter-topk  final_new.json
echo "=== arm E0: pre-tonight, same card, right after ==="
run /home/mh/aiter-base  final_base.json
echo ARM-E-DONE
