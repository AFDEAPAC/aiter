#!/usr/bin/env python3
"""Measure AVO over the (M, N) grid at three widths, for the ceiling report.

Separate from bench/parity_sweep.py on purpose. That file answers the odd-pitch
question on RAGGED rows, because ragged is what the aiter entry actually builds.
This one answers "how far are we from the ideal-selector floor", and the floor
(scripts/select_grid.hip) reads FULL uniform rows -- so here every row is
[0, N) too. Otherwise tables 2 and 4 of the report would be computed from
different traffic and could not be divided into each other: ragged rows read up
to 25% fewer bytes at large M and small N.

Everything else is inherited from parity_sweep.py because it was the right
answer there and is the right answer here:

  - aiter's own @perftest, which rotates arguments to defeat L2 and reads GPU
    time from a profiler trace, rather than a hand-written loop over one input;
  - all three widths built and warmed BEFORE any of them is timed, then
    interleaved rounds. Timing them one after another charges run-order drift
    to whichever width went first, which is how an earlier sweep reported a
    +5.60% power-of-two effect that was +0.34% once interleaved.

AITER_DISABLE_TOPK_AVO is irrelevant here -- top_k_per_row_prefill_avo is called
directly and never reads it -- but it is left unset so nothing is ambiguous.

  python /home/mh/topk-prefill-avo/bench/ceiling_sweep.py
"""

import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import aiter  # noqa: E402
from aiter.ops.topk import topk_avo_supports  # noqa: E402
from aiter.test_common import perftest  # noqa: E402

MS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
BASES = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
WIDTH_OFFSETS = (0, 1, 2)
TOPK = 2048
ROUNDS = 3


@perftest()
def run_avo(logits, row_starts, row_ends, indices, values,
            num_rows, stride_row, stride_col, k):
    return aiter.top_k_per_row_prefill_avo(
        logits, row_starts, row_ends, indices, values,
        num_rows, stride_row, stride_col, k=k)


def full_rows(m, n):
    """Uniform rows: every row is the whole [0, N), matching select_grid."""
    starts = torch.zeros(m, dtype=torch.int32, device="cuda")
    ends = torch.full((m,), n, dtype=torch.int32, device="cuda")
    return starts, ends


def triple(m, base, topk, rounds):
    widths = (base + o for o in WIDTH_OFFSETS)
    widths = tuple(widths)
    args = {}
    g = torch.Generator(device="cuda")
    for w in widths:
        g.manual_seed(42)
        lg = torch.randn((m, w), generator=g, dtype=torch.float32, device="cuda")
        assert lg.stride(0) == w, (lg.stride(0), w)
        rs, re = full_rows(m, w)
        idx = torch.empty((m, topk), dtype=torch.int32, device="cuda")
        args[w] = (lg, rs, re, idx, None, m, w, 1, topk)

    for w in widths:                       # warm every width before timing any
        run_avo(*args[w])

    samples = {w: [] for w in widths}
    for _ in range(rounds):
        for w in widths:                   # interleaved
            samples[w].append(run_avo(*args[w])[1])

    out = []
    for w in widths:
        v = samples[w]
        out.append({"m": m, "base": base, "width": w, "topk": topk,
                    "odd": bool(w & 1), "pow2": w == base,
                    "us": st.median(v),
                    "spread_pct": (max(v) - min(v)) / st.median(v) * 100,
                    "runs": [round(x, 3) for x in v],
                    "bytes": m * w * 4,
                    "supports": bool(topk_avo_supports(m, w, topk))})
    del args
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--out", default="/home/mh/topk-prefill-avo/reports/ceiling_measured.json")
    args = ap.parse_args()

    out = []
    print("%6s %9s | %10s %7s %9s" % ("M", "stride0", "us", "sd%", "TB/s"))
    for m in MS:
        for base in BASES:
            try:
                recs = triple(m, base, TOPK, args.rounds)
            except Exception as e:
                msg = str(e).strip().splitlines()[-1][:110]
                print("%6d %9d | FAILED: %s" % (m, base, msg))
                out.append({"m": m, "base": base, "error": msg})
                torch.cuda.empty_cache()
                sys.stdout.flush()
                continue
            for r in recs:
                print("%6d %9d | %10.2f %7.2f %9.2f"
                      % (r["m"], r["width"], r["us"], r["spread_pct"],
                         r["bytes"] / (r["us"] * 1e-6) / 1e12))
            out.extend(recs)
            sys.stdout.flush()
    json.dump({"ms": MS, "bases": BASES, "width_offsets": list(WIDTH_OFFSETS),
               "topk": TOPK, "rounds": args.rounds, "rows": "full [0, N)",
               "harness": "aiter.test_common.perftest", "cells": out},
              open(args.out, "w"), indent=1)
    print("WROTE %s  (%d cells)" % (args.out, len(out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
