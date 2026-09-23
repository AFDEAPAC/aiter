#!/bin/bash
# One card, one at a time, nothing else timing. The first attempt put the two
# arms on different cards while fbrate and a kernel A/B ran alongside, and the
# contamination showed: the worst "regressions" were n=2048 cells that the
# sampled path never touches.
set -e
D=${1:-0}
run() {
  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
    --shm-size 32G -e HIP_VISIBLE_DEVICES=$D -e PYTHONPATH=$2 \
    -v $2:/aiter -v /home/mh/topk-prefill-avo:/topk -w /topk \
    rocm/ali-private:ubuntu22.04_rocm10.0.0_cp313_torch2.12.0_vllm_dsv4_20260916 \
    bash -c "cd /aiter && pip -q install -e . 2>&1|tail -1; cd /topk && python3 bench/select_ab_sweep.py --mode backends --seed 0 --out /topk/log/$3"
}
echo "=== arm A: cd486fbd0, the state that measured 41 green ==="
run x /home/mh/aiter-base    clean_base.json
echo "=== arm B: d63d34517, the epilogue change ==="
run x /home/mh/aiter-topk    clean_new.json
echo DONE
