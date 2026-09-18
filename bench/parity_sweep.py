#!/usr/bin/env python3
"""Even/odd row-pitch parity sweep for the AVO prefill op, on aiter's @perftest.

The question: now that an odd stride0 is served (g_25) and hostile extents are
clamped (g_26), does an odd pitch cost anything against an even one?

Three design decisions, and the sweep is worthless without any of them.

1. aiter's own @perftest(), not a hand-written timing loop. It sizes an argument
   rotation from the measured input size so the working set defeats L2, and it
   reads GPU kernel time out of a torch profiler trace rather than wall clock.
   A loop that reuses one input measures a warm cache. The two harnesses do not
   agree -- at M=256 stride0=1048577 a plain loop gives 234.42 us for AVO and
   666.61/690 us for the reference where @perftest gives 217.56 and 666.61 --
   so numbers from the two must never be mixed in one table.

   The two runners below are inlined from op_tests/test_topk_per_row.py rather
   than imported, because that module runs its whole benchmark sweep at import
   time. Same reason bench/aiter_ab.py inlines its helpers.

2. THREE widths, not two. Each base B is a power of two, and each M is measured
   at B, B+1 and B+2:

       B     even, power of two
       B+1   odd,  not a power of two
       B+2   even, not a power of two

   Three widths within two columns of each other, under 0.0031% of the work
   apart. Two comparisons fall out:

       (B+1) vs (B+2)   odd vs even, both off the power of two  -> PARITY
       (B+2) vs B       non-pow2 vs pow2, both even             -> POW2 BOUNDARY

   Comparing only B against B+1 is not a parity measurement: B is a power of two
   and B+1 is not, so leaving the power of two rides along and gets charged to
   parity.

3. INTERLEAVED, all three warmed first. Measuring the widths one after another
   -- build, warm, time, free, next -- does not work here. Run that way, an
   earlier version of this sweep reported the pow2 boundary costing +5.60% at
   M=256 B=1048576; interleaved, the same cell was +0.34%. The tell was that
   B+1 and B+2 came out nearly equal to each other (513.08 and 513.08 at M=1024
   B=524288; 1956.29 and 1956.25 at M=4096 B=524288) while both sat the same
   distance above the B timed before them. That is drift between positions in
   the run order, not a property of the width. Parity survived only by luck:
   B+1 and B+2 are adjacent, so the drift between them cancels; the pow2
   comparison spans the whole triple and does not. Same trap
   .evo/config-v5.yaml records as baseline_decay_warning.

AITER_DISABLE_TOPK_AVO=1 is required, and it does NOT disable the op under test.
It appears at exactly one executable site, aiter/ops/topk.py:419, inside the
dispatch condition of top_k_per_row_prefill, so it only forces the REFERENCE
side onto the original mb/ob path; top_k_per_row_prefill_avo never reads it.
Confirmed by construction: with the variable set, top_k_per_row_prefill is
666.61 us at M=256 stride0=1048577 while top_k_per_row_prefill_avo is 217.56,
and with it unset top_k_per_row_prefill returns to the AVO number.

Run inside the correctness image:

  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    --ipc=host --shm-size 16G -e PYTHONPATH=/aiter -e AITER_DISABLE_TOPK_AVO=1 \
    -v /home/mh/aiter-topk:/aiter -v /home/mh:/home/mh -w /aiter <image> \
    python /home/mh/topk-prefill-avo/bench/parity_sweep.py
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
from aiter_ab import boundaries, logits_for  # noqa: E402

MS = [1, 8, 64, 256, 1024, 4096]
BASES = [65536, 131072, 262144, 524288, 1048576]
TOPK = 2048
ROUNDS = 4


@perftest()
def run_avo(logits, row_starts, row_ends, indices, values,
            num_rows, stride_row, stride_col, k):
    return aiter.top_k_per_row_prefill_avo(
        logits, row_starts, row_ends, indices, values,
        num_rows, stride_row, stride_col, k=k)


@perftest()
def run_ref(logits, row_starts, row_ends, indices, values,
            num_rows, stride_row, stride_col, k):
    return aiter.top_k_per_row_prefill(
        logits, row_starts, row_ends, indices, values,
        num_rows, stride_row, stride_col, k=k)


RUNNERS = {"avo": run_avo, "mbob": run_ref}


def triple(m, base, topk, rounds=ROUNDS):
    """Time B, B+1, B+2 interleaved. Returns one record per width."""
    widths = (base, base + 1, base + 2)
    args = {}
    for w in widths:
        rs, re = boundaries(m, w - m)
        lg = logits_for(rs, re)
        assert lg.stride(0) == w, (lg.stride(0), w)
        idx = torch.empty((m, topk), dtype=torch.int32, device="cuda")
        args[w] = (lg, rs, re, idx, None, m, w, 1, topk)

    # Warm every width and both ops before timing any of them, so that no width
    # is measured in a state the others were not. @perftest warms internally
    # too, but only for the call it is in.
    for w in widths:
        for op in RUNNERS:
            RUNNERS[op](*args[w])

    samples = {w: {op: [] for op in RUNNERS} for w in widths}
    for _ in range(rounds):
        for op in RUNNERS:
            for w in widths:
                samples[w][op].append(RUNNERS[op](*args[w])[1])

    out = []
    for w in widths:
        r = {"m": m, "base": base, "width": w, "topk": topk, "odd": bool(w & 1),
             "pow2": w == base, "rounds": rounds, "harness": "aiter.perftest",
             "supports": bool(topk_avo_supports(m, w, topk))}
        for op in RUNNERS:
            v = samples[w][op]
            r[op + "_us"] = st.median(v)
            r[op + "_spread_pct"] = (max(v) - min(v)) / st.median(v) * 100
            r[op + "_runs"] = [round(x, 2) for x in v]
        r["speedup"] = r["mbob_us"] / r["avo_us"]
        out.append(r)
    del args
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--out", default="/home/mh/topk-prefill-avo/reports/parity_sweep.json")
    args = ap.parse_args()

    if os.environ.get("AITER_DISABLE_TOPK_AVO", "0") != "1":
        print("REFUSING: set AITER_DISABLE_TOPK_AVO=1, or the reference side is "
              "also AVO and every ratio is 1.00x by construction")
        return 2

    out = []
    print("harness: aiter.test_common.perftest (rotated args, GPU time from trace)")
    print("%6s %9s %9s | %10s %6s | %10s %6s | %8s"
          % ("M", "stride0", "kind", "avo_us", "sd%", "mbob_us", "sd%", "speedup"))
    for m in MS:
        for base in BASES:
            try:
                recs = triple(m, base, args.topk, args.rounds)
            except Exception as e:
                msg = str(e).strip().splitlines()[-1][:110]
                print("%6d %9d | FAILED: %s" % (m, base, msg))
                out.append({"m": m, "base": base, "error": msg})
                torch.cuda.empty_cache()
                sys.stdout.flush()
                continue
            for r in recs:
                kind = "pow2 even" if r["pow2"] else ("odd" if r["odd"] else "even")
                print("%6d %9d %9s | %10.2f %6.2f | %10.2f %6.2f | %7.2fx"
                      % (r["m"], r["width"], kind, r["avo_us"], r["avo_spread_pct"],
                         r["mbob_us"], r["mbob_spread_pct"], r["speedup"]))
            out.extend(recs)
            sys.stdout.flush()

    by = {(r["m"], r["width"]): r for r in out if "avo_us" in r}
    print("\nAVO side, aiter @perftest. parity = (B+1) vs (B+2), both off the")
    print("power of two; pow2 = (B+2) vs B, both even. x = AVO over mb/ob.")
    print("\n%6s %9s | %9s %9s %9s | %8s %8s | %7s %7s"
          % ("M", "B", "B(pow2)", "B+1 odd", "B+2 even",
             "parity", "pow2", "x@B+1", "x@B+2"))
    par, pw2 = [], []
    for m in MS:
        for base in BASES:
            a, o, e = by.get((m, base)), by.get((m, base + 1)), by.get((m, base + 2))
            if not (a and o and e):
                continue
            dp = (o["avo_us"] - e["avo_us"]) / e["avo_us"] * 100
            d2 = (e["avo_us"] - a["avo_us"]) / a["avo_us"] * 100
            par.append(dp)
            pw2.append(d2)
            print("%6d %9d | %9.2f %9.2f %9.2f | %+7.2f%% %+7.2f%% | %6.2fx %6.2fx"
                  % (m, base, a["avo_us"], o["avo_us"], e["avo_us"], dp, d2,
                     o["speedup"], e["speedup"]))
    if par:
        print("\nparity (odd vs even, both non-pow2): mean %+.2f%%  max %+.2f%%  min %+.2f%%"
              % (st.mean(par), max(par), min(par)))
        print("pow2   (non-pow2 vs pow2, both even): mean %+.2f%%  max %+.2f%%  min %+.2f%%"
              % (st.mean(pw2), max(pw2), min(pw2)))
        print("cells  %d, %d interleaved rounds each" % (len(par), args.rounds))
    json.dump(out, open(args.out, "w"), indent=1)
    print("WROTE %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
