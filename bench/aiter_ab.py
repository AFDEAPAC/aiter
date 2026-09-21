#!/usr/bin/env python3
"""A/B the AVO prefill op against aiter's own, on aiter's data, timed here.

Exists because the op test's `us` column comes from @perftest(), and a number
measured through a different harness than the standalone benchmark cannot be
compared to it: the two disagreed by 18% at M=256 and the cause was not
attributable while the harness, the timer AND the data generator all differed.
This pins the harness and the data and times only the op, so the remaining
difference is the op.

The two helpers are inlined from op_tests/test_topk_per_row.py rather than
imported: that module runs its whole benchmark sweep at import time.

Run inside the ROCm container:
  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    --ipc=host --shm-size 16G -e PYTHONPATH=/aiter \
    -v /home/mh/aiter-topk:/aiter -v /home/mh:/home/mh -w /aiter <image> \
    python3 /home/mh/topk-prefill-avo/bench/aiter_ab.py
"""

import argparse

import torch

import aiter


def boundaries(num_rows, num_prefix):
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device="cuda")
    row_ends = torch.arange(
        num_prefix + 1,
        num_prefix + num_rows + 1,
        dtype=torch.int32,
        device="cuda",
    )
    return row_starts, row_ends


def logits_for(row_starts, row_ends, seed=42):
    torch.manual_seed(seed)
    width = int(row_ends.max())
    logits = torch.randn(
        row_starts.shape[0], width, dtype=torch.float32, device="cuda"
    )
    for i, end in enumerate(row_ends.tolist()):
        logits[i, end:] = float("-inf")
    return logits


def bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        stop.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(stop) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[64, 256, 1024, 4096])
    ap.add_argument("--num-prefix", type=int, default=131072)
    ap.add_argument("--topk", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    print(
        "num_prefix=%d topk=%d warmup=%d iters=%d"
        % (args.num_prefix, args.topk, args.warmup, args.iters)
    )
    print("%6s %8s %11s %11s %8s" % ("M", "width", "sampled_us", "aiter_us", "ratio"))
    for m in args.rows:
        row_starts, row_ends = boundaries(m, args.num_prefix)
        logits = logits_for(row_starts, row_ends)
        indices = torch.empty((m, args.topk), dtype=torch.int32, device="cuda")
        stride0 = logits.stride(0)
        if not aiter.topk_sampled_supports(m, stride0, args.topk):
            print("%6d %8d %11s" % (m, logits.shape[1], "declined"))
            continue
        # The Python wrapper owns the workspace (get_topk_scratch_workspace), so
        # it is inside the timed region for both ops, as it is in production.
        sampled = bench(
            lambda: aiter.top_k_per_row_prefill_sampled(
                logits,
                row_starts,
                row_ends,
                indices,
                None,
                m,
                stride0,
                1,
                args.topk,
            ),
            args.warmup,
            args.iters,
        )
        ref = bench(
            lambda: aiter.top_k_per_row_prefill(
                logits,
                row_starts,
                row_ends,
                indices,
                None,
                m,
                stride0,
                1,
                args.topk,
            ),
            args.warmup,
            args.iters,
        )
        print(
            "%6d %8d %11.2f %11.2f %7.2fx"
            % (m, logits.shape[1], sampled, ref, ref / sampled)
        )


if __name__ == "__main__":
    main()
