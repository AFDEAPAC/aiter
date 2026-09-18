#!/usr/bin/env python3
"""Host-side cost of one top_k_per_row_prefill call, separated from GPU time.

The two numbers that matter are different things and both are needed:

  enqueue  wall time of the call with no synchronise, i.e. what the caller's CPU
           spends before it can do anything else. Every Python-level lookup in
           the wrapper lands here and none of it lands in a profiler kernel
           trace, which is why @perftest never showed it.
  e2e      wall time of the same loop with one synchronise at the end, divided
           by the iteration count, so the GPU work is in it too.

Run inside the correctness image with the mounted aiter ahead of the image's own.
"""
import os
import statistics as st
import sys
import time

sys.path.insert(0, "/home/mh/topk-prefill-avo/bench")

import torch  # noqa: E402

import aiter  # noqa: E402
from aiter_ab import boundaries, logits_for  # noqa: E402

SHAPES = [(16, 32768), (64, 65536), (256, 65536), (64, 131072), (4096, 131072)]
K = 2048
ITERS = 200
REPS = 7


def bench(fn, iters, reps):
    enq, e2e = [], []
    for _ in range(3):                       # warm
        fn()
    torch.cuda.synchronize()
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        t1 = time.perf_counter()             # enqueue only, no sync
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        enq.append((t1 - t0) / iters * 1e6)
        e2e.append((t2 - t0) / iters * 1e6)
    return st.median(enq), st.median(e2e)


def main():
    os.environ["AITER_DISABLE_TOPK_AVO"] = "0"
    print("%6s %9s | %11s %11s %11s" % ("M", "N", "enqueue_us", "e2e_us", "host_share"))
    for m, n in SHAPES:
        rs, re = boundaries(m, n - m)
        lg = logits_for(rs, re)
        s0 = lg.stride(0)
        idx = torch.empty((m, K), dtype=torch.int32, device="cuda")
        args = (lg, rs, re, idx, None, m, s0, 1, K)

        def call():
            aiter.top_k_per_row_prefill(*args)

        enq, e2e = bench(call, ITERS, REPS)
        print("%6d %9d | %11.2f %11.2f %10.1f%%"
              % (m, s0, enq, e2e, 100 * enq / e2e))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
