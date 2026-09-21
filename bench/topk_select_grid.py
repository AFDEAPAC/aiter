#!/usr/bin/env python3
"""Reproduce the topk_select k=2048 M x N table with the AVO prefill op.

The table being matched is a DENSE per-row top-k: topk_select takes no row
boundaries. The AVO op is a prefill op, so the equivalent shape is rowStarts=0
and rowEnds=N on every row -- full width, nothing ragged.

`topk_select` is measured HERE as well, with the same timer and the same data,
because a number from another harness is not comparable to one from this one:
@perftest and a cuda-event loop disagreed by 18% on one shape earlier in this
work. The `sel_us` column is therefore the like-for-like baseline, and `ref_us`
is only the figure quoted in the table.

Run inside the ROCm container:
  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    --ipc=host --shm-size 16G -e PYTHONPATH=/aiter \
    -v /home/mh/aiter-topk:/aiter -v /home/mh:/home/mh -w /aiter <image> \
    python3 /home/mh/topk-prefill-avo/bench/topk_select_grid.py
"""

import argparse
import gc

import torch

import aiter
from aiter.test_common import perftest

MS = [1, 2, 4, 8, 4096]
NS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
TOPK = 2048

# The parenthesised column of the quoted table, transcribed. Reference only:
# measured on an unknown harness, so it is printed but never used to judge a
# regression. None where the table gives no value.
REFERENCE = {
    1: [1.5, 5.5, 9.5, 12.1, 17.7, 25.5, 30.7, 39.9, 46.3, 48.7],
    2: [1.8, 6.0, 9.6, 12.6, 18.0, 25.3, 41.5, 51.8, 62.7, 44.9],
    4: [2.0, 8.6, 10.2, 12.7, 18.2, 28.5, 43.8, 61.1, 62.1, 69.3],
    8: [2.4, 9.0, 10.7, 16.1, 18.7, 29.2, 49.2, 59.8, 66.4, 72.8],
    4096: [13.4, 73.4, 98.0, 135.3, 239.3, 456.0, 946.9, 2200.0, 4500.0, 9000.0],
}

# The quoted table's own measured column, for context on the harness offset.
QUOTED_SELECT = {
    1: [2.0, 6.2, 9.8, 15.3, 29.7, 32.0, 32.3, 32.2, 33.4, 40.2],
    2: [2.0, 6.5, 10.2, 15.9, 30.4, 32.7, 32.6, 32.0, 34.0, 41.5],
    4: [2.0, 6.9, 10.5, 16.1, 30.0, 32.4, 32.3, 32.3, 34.4, 41.2],
    8: [2.8, 7.7, 11.2, 16.7, 30.1, 32.3, 32.2, 32.8, 35.6, 48.9],
    4096: [9.2, 61.2, 98.8, 148.7, 249.0, 458.5, 986.7, 1686.0, 2961.0, 5312.0],
}


# aiter's own decorator, which is how the quoted table was produced. It reports
# DEVICE time: the default path profiles with torch.profiler and divides the
# summed kernel time by num_iters, so host-side cost is excluded entirely.
#
# That distinction is the whole reason this file does not roll its own timer.
# Two home-made attempts read 39.4 us and then 34.6 us for topk_select at
# M=1 N=2K, against the 2.0 us quoted; the residual was a flat ~32 us of
# per-call Python dispatch and allocation, which at 2 us of GPU work makes the
# HOST the bottleneck and the measurement meaningless for this table.
@perftest()
def run_sampled(logits, row_starts, row_ends, indices, values, m, stride0, k):
    return aiter.top_k_per_row_prefill_sampled(
        logits, row_starts, row_ends, indices, values, m, stride0, 1, k
    )


@perftest()
def run_select(logits, k, return_value):
    return aiter.topk_select(logits, k, return_value=return_value)


def label(n):
    return "%dK" % (n // 1024)


def measure_cell(m, n, k, want_values):
    torch.manual_seed(42)
    logits = torch.randn(m, n, dtype=torch.float32, device="cuda")
    row_starts = torch.zeros(m, dtype=torch.int32, device="cuda")
    row_ends = torch.full((m,), n, dtype=torch.int32, device="cuda")
    indices = torch.empty((m, k), dtype=torch.int32, device="cuda")
    values = (
        torch.empty((m, k), dtype=torch.float32, device="cuda")
        if want_values
        else None
    )
    stride0 = logits.stride(0)

    sampled = None
    if aiter.topk_sampled_supports(m, stride0, k):
        _, sampled = run_sampled(
            logits, row_starts, row_ends, indices, values, m, stride0, k
        )

    try:
        _, sel = run_select(logits, k, want_values)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        sel = "err: %s" % str(exc)[:40]

    del logits, row_starts, row_ends, indices, values
    gc.collect()
    torch.cuda.empty_cache()
    return sampled, sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=MS)
    ap.add_argument("--widths", type=int, nargs="+", default=NS)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument(
        "--no-values",
        action="store_true",
        help="time the indices-only path instead",
    )
    args = ap.parse_args()

    want_values = not args.no_values
    print(
        "AVO top_k_per_row_prefill_sampled, k=%d, values=%s, rowEnds=N (dense)"
        % (args.topk, want_values)
    )
    print("timer: aiter @perftest() -- DEVICE time, host cost excluded")
    print(
        "cells: sampled_us | sel_us (topk_select, same timer) | ref_us (quoted table)"
    )
    print()

    head = "%6s" % "M\\N" + "".join("%22s" % label(n) for n in args.widths)
    print(head)
    for m in args.rows:
        cells = []
        for j, n in enumerate(args.widths):
            if args.topk > n:
                cells.append("%22s" % "k>N")
                continue
            sampled, sel = measure_cell(m, n, args.topk, want_values)
            ref = REFERENCE.get(m, [None] * len(NS))
            r = ref[NS.index(n)] if n in NS and m in REFERENCE else None
            a = "declined" if sampled is None else "%.1f" % sampled
            s = sel if isinstance(sel, str) else "%.1f" % sel
            cells.append("%22s" % ("%s | %s | %s" % (a, s, "-" if r is None else r)))
        print("%6d" % m + "".join(cells), flush=True)


if __name__ == "__main__":
    main()
