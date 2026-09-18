#!/usr/bin/env python3
"""Even/odd row-pitch parity sweep for the AVO prefill op.

The question: now that an odd stride0 is served (g_25) and hostile extents are
clamped (g_26), does an odd pitch cost anything against an even one?

The triple design is the whole point. Each base B is a power of two, and each M
is measured at B, B+1 and B+2 in the SAME pass:

    B     even, power of two
    B+1   odd,  not a power of two
    B+2   even, not a power of two

Three widths within two columns of each other, so they differ by under 0.0031%
of the work and any real gap is structural, not size. Two comparisons fall out,
and keeping them apart matters:

    B+1 vs B+2   odd vs even, both off the power of two  -> PARITY
    B+2 vs B     non-pow2 vs pow2, both even             -> POWER-OF-TWO-NESS

A first version of this sweep compared only B against B+1 and reported odd
widths costing up to +5.7%. That number conflated the two effects: B is a power
of two and B+1 is not, so leaving the power of two (which changes the sampling
stride from exact to masked, and moves the kCoopLog2G and phase_a S lookups)
was being charged to parity. All three widths must be in one pass, because the
effect being resolved is a few percent and cross-run drift is the same size.

Both sides are timed in one process on one dataset with one timer, reusing
bench/aiter_ab.py's helpers, because a number measured through a different
harness is not comparable to one measured here (that file's docstring records
two harnesses disagreeing by 18% at M=256).

AITER_DISABLE_TOPK_AVO=1 is required so that `top_k_per_row_prefill` takes the
original mb/ob path; the AVO side calls `top_k_per_row_prefill_avo` directly.
Without it both sides would be AVO and every ratio would be 1.00x.

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
from aiter_ab import bench, boundaries, logits_for  # noqa: E402

MS = [1, 8, 64, 256, 1024, 4096]
BASES = [65536, 131072, 262144, 524288, 1048576]
TOPK = 2048


def argv_for(m, width):
    """Fewer iterations once a shape is big enough that 100 of them is minutes."""
    return (10, 30, 3) if m * width > (1 << 28) else (20, 100, 3)


def one(m, width, topk):
    """Median-of-run-medians for both ops at stride0 = width."""
    num_prefix = width - m
    row_starts, row_ends = boundaries(m, num_prefix)
    logits = logits_for(row_starts, row_ends)
    stride0 = logits.stride(0)
    assert stride0 == width, (stride0, width)
    rec = {"m": m, "width": width, "topk": topk, "odd": bool(width & 1),
           "supports": bool(topk_avo_supports(m, stride0, topk))}
    if not rec["supports"]:
        rec["note"] = "declined by topk_avo_supports"
        return rec
    indices = torch.empty((m, topk), dtype=torch.int32, device="cuda")
    warmup, iters, repeats = argv_for(m, width)
    rec["argv"] = [warmup, iters, repeats]

    def avo():
        aiter.top_k_per_row_prefill_avo(
            logits, row_starts, row_ends, indices, None, m, stride0, 1, topk)

    def mbob():
        aiter.top_k_per_row_prefill(
            logits, row_starts, row_ends, indices, None, m, stride0, 1, topk)

    for name, fn in (("avo", avo), ("mbob", mbob)):
        runs = [bench(fn, warmup, iters) for _ in range(repeats)]
        rec[name + "_us"] = st.median(runs)
        rec[name + "_spread_pct"] = (max(runs) - min(runs)) / st.median(runs) * 100
    rec["speedup"] = rec["mbob_us"] / rec["avo_us"]
    del logits, indices, row_starts, row_ends
    torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--out", default="/home/mh/topk-prefill-avo/reports/parity_sweep.json")
    args = ap.parse_args()

    if os.environ.get("AITER_DISABLE_TOPK_AVO", "0") != "1":
        print("REFUSING: set AITER_DISABLE_TOPK_AVO=1, or the reference side is "
              "also AVO and every ratio is 1.00x by construction")
        return 2

    out = []
    print("%6s %9s %4s | %10s %6s | %10s %6s | %8s"
          % ("M", "stride0", "par", "avo_us", "sd%", "mbob_us", "sd%", "speedup"))
    for m in MS:
        for base in BASES:
            for width in (base, base + 1, base + 2):
                try:
                    r = one(m, width, args.topk)
                except Exception as e:  # keep the sweep alive, record the shape
                    r = {"m": m, "width": width, "odd": bool(width & 1),
                         "error": str(e).strip().splitlines()[-1][:120]}
                out.append(r)
                if "error" in r:
                    print("%6d %9d %4s | %s" % (m, width, "odd" if width & 1 else "even",
                                                r["error"]))
                elif "avo_us" not in r:
                    print("%6d %9d %4s | %s" % (m, width, "odd" if width & 1 else "even",
                                                r.get("note", "skipped")))
                else:
                    print("%6d %9d %4s | %10.2f %6.2f | %10.2f %6.2f | %7.2fx"
                          % (m, width, "odd" if width & 1 else "even",
                             r["avo_us"], r["avo_spread_pct"],
                             r["mbob_us"], r["mbob_spread_pct"], r["speedup"]))
                sys.stdout.flush()

    # Two comparisons per (M, base). Keeping them apart is the point: see the
    # module docstring on why B vs B+1 alone is not a parity measurement.
    by = {(r["m"], r["width"]): r for r in out if "avo_us" in r}
    print("\nAVO side. B = power of two. parity = (B+1) vs (B+2), both off the pow2;")
    print("pow2 = (B+2) vs B, both even. Speedups are AVO over the mb/ob path.")
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
        print("\nparity  (odd vs even, both non-pow2): mean %+.2f%%  max %+.2f%%  min %+.2f%%"
              % (st.mean(par), max(par), min(par)))
        print("pow2    (non-pow2 vs pow2, both even): mean %+.2f%%  max %+.2f%%  min %+.2f%%"
              % (st.mean(pw2), max(pw2), min(pw2)))
        print("cells   %d" % len(par))
    json.dump(out, open(args.out, "w"), indent=1)
    print("WROTE %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
