#!/bin/bash
# Arm C: the non-temporal load gate, exported into aiter and swept on the same
# card as arms A and B. Runs only after arm B has written its file, because
# touching aiter's csrc mid-sweep would change what arm B is measuring.
set -e
while pgrep -f select_ab_swee[p] > /dev/null; do sleep 20; done
cd /home/mh/topk-prefill-avo
docker run --rm -v /home/mh/topk-prefill-avo:/topk -v /home/mh/aiter-topk:/aiter -w /topk \
  rocm/ali-private:ubuntu22.04_rocm10.0.0_cp313_torch2.12.0_vllm_dsv4_20260916 \
  bash -c "pip -q install clang-format==23.1.1 2>/dev/null
           python3 scripts/export_aiter_op.py --aiter /aiter
           python3 scripts/export_aiter_op.py --aiter /aiter --check && echo EXPORT-CHECK-CLEAN"
cd /home/mh/aiter-topk && git add -A csrc && git commit -q -m "topk: regenerate for the phase_b non-temporal load gate

Regenerated from topk-prefill-avo by scripts/export_aiter_op.py; the source
changes are that repo's csrc/topk_common.hip.hpp, csrc/topk_generalize.hip.hpp
and benchmark_topk.hip.cpp.

phase_b_filter_coop takes NT as a template parameter and the launch picks it on
M*pitch >= 2^27, which is 512MB, the first input that cannot sit in gfx950's
256MB MALL. Above it the row data is read once per block and never reused, so a
cache line only evicts what other blocks are still reading; below it the input
can stay resident and the caching is the whole benefit. Measured over 27 shapes:
0.884x to 0.969x at or above the gate, and below it the same device code as
before.

Also carries ABLATE_CREAD, ABLATE_PA and WSTAGE_WAVES_OVERRIDE, which are
pricing knobs that default to the shipped behaviour." || echo "nothing to commit"
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  --shm-size 32G -e HIP_VISIBLE_DEVICES=0 -e PYTHONPATH=/aiter \
  -v /home/mh/aiter-topk:/aiter -v /home/mh/topk-prefill-avo:/topk -w /topk \
  rocm/ali-private:ubuntu22.04_rocm10.0.0_cp313_torch2.12.0_vllm_dsv4_20260916 \
  bash -c "rm -rf /aiter/aiter/jit/build/module_top_k_per_row /aiter/aiter/jit/module_top_k_per_row.so
           cd /aiter && pip -q install -e . 2>&1|tail -1
           cd /topk && python3 -c 'import aiter; print(\"AITER:\", aiter.__file__)'
           python3 bench/select_ab_sweep.py --mode backends --seed 0 --out /topk/log/clean_nt.json"
echo ARM-C-DONE
