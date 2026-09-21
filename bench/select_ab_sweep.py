#!/usr/bin/env python3
"""Measure `aiter.topk_select` over the (M, N) grid at three widths, two ways.

Two modes, because two different questions need two different tables.

  --mode backends   For each cell, force each backend that can serve it and time
                    it, and time the `sampled` kernel alongside. This is the
                    table a routing rule gets fitted to. `topk_select.py:90` and
                    `:367` say to re-fit with `topk_backend_fit.py` over
                    `topk_backend_sweep.py`; neither file exists anywhere on this
                    machine, so this mode replaces them.

  --mode ab         Time the real router twice, once with `sampled` available and
                    once without. This is the before/after report.

Why not `bench/ceiling_sweep.py`: that file calls the kernel entry directly and
says so -- `AITER_DISABLE_TOPK_SAMPLED` never reaches it. It also answers a
different question, "how far from the ideal-selector floor". This one answers
"what does the router pick, and what does that cost".

Three things are inherited from `ceiling_sweep.py` because they were right there
and are right here:

  - aiter's own `@perftest()`, which sizes an argument rotation from the measured
    input so the working set defeats L2, and reads GPU kernel time from a
    profiler trace rather than wall clock. A hand-written loop over one input
    measures a warm cache.
  - All three widths built and warmed BEFORE any of them is timed, then
    interleaved rounds. Timing them one after another charges run-order drift to
    whichever width went first, which is how an earlier sweep reported a +5.60%
    parity effect that was +0.34% once interleaved.
  - Full uniform rows [0, N). The earlier heatmap used full rows, so cells stay
    comparable with it. Ragged rows read up to 25% fewer bytes at large M and
    small N.

Forcing a backend: `topk_select` has no override parameter, so this replaces
`topk_select_backend` and clears `_choose`, which is `@lru_cache(maxsize=1024)`
and would otherwise keep handing back the name chosen before the patch. The
whole `topk_select` path still runs -- argument rejection, allocation, dispatch
-- so a forced number is comparable with a routed one.

Run inside the correctness image with both repos mounted:

  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \\
    --ipc=host --shm-size 16G -e PYTHONPATH=/aiter \\
    -v /home/mh/aiter-topk:/aiter -v /home/mh/topk-prefill-avo:/topk -w /aiter \\
    rocm/ali-private:ubuntu22.04_rocm10.0.0_cp313_torch2.12.0_vllm_dsv4_20260916 \\
    python /topk/bench/select_ab_sweep.py --mode backends
"""
import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import aiter  # noqa: E402

# `python /topk/bench/x.py` puts sys.path[0] at this file's directory, not the
# cwd, so `import aiter` can silently resolve to the site-packages copy and
# produce a green run against a module nobody changed. Caught exactly that once.
if not aiter.__file__.startswith("/aiter/"):
    raise SystemExit(
        "WRONG AITER: imported %s, expected one under /aiter/. "
        "Re-run with -e PYTHONPATH=/aiter." % aiter.__file__
    )

import grid  # noqa: E402
from aiter.ops import topk_select as ts  # noqa: E402
from aiter.test_common import perftest  # noqa: E402

MS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
BASES = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
WIDTH_OFFSETS = (0, 1, 2)
TOPK = 2048
ROUNDS = 3


@perftest()
def run_select(inp, k, out_idx):
    return aiter.topk_select(inp, k, output_idx=out_idx)


@perftest()
def run_sampled(logits, row_starts, row_ends, indices, values,
                num_rows, stride_row, stride_col, k):
    return aiter.top_k_per_row_prefill_sampled(
        logits, row_starts, row_ends, indices, values,
        num_rows, stride_row, stride_col, k=k)


def force(name):
    """Pin `topk_select` to one backend. Pass None to restore the real rule."""
    if name is None:
        ts.topk_select_backend = force._real
    else:
        ts.topk_select_backend = lambda rows, width, k, available: name
    ts._choose.cache_clear()


force._real = ts.topk_select_backend


def available_for(m, width, k):
    """Backends that can serve this geometry, as `topk_select` computes it."""
    try:
        wave = ts.wave_size_of(torch.device("cuda", 0).index or 0)
    except Exception:
        wave = 64
    avail = ts._available(width, k, wave, False, True)
    allowed = frozenset(ts._BACKENDS_BY_TIE[None])
    return sorted(avail & allowed)


def check(inp, idx, k, rows_to_check=(0,)):
    """Index set against torch.topk on the same row. Values, not positions:
    ties are common at these widths and two backends may break them differently
    while both being correct."""
    want = min(k, inp.shape[1])
    for r in rows_to_check:
        g = idx[r][:want].to(torch.int64)
        if not bool(((g >= 0) & (g < inp.shape[1])).all().item()):
            return "row %d: index out of range" % r
        if len(set(g.tolist())) != want:
            return "row %d: duplicate indices" % r
        got = torch.sort(inp[r][g]).values
        ref = torch.sort(torch.topk(inp[r], want).values).values
        if not torch.equal(got, ref):
            return "row %d: value multiset != torch.topk" % r
    return None


def build(m, widths, topk):
    """One tensor set per width, all built before any is timed."""
    args, g = {}, torch.Generator(device="cuda")
    for w in widths:
        g.manual_seed(42)
        lg = torch.randn((m, w), generator=g, dtype=torch.float32, device="cuda")
        assert lg.stride(0) == w, (lg.stride(0), w)
        idx = torch.empty((m, topk), dtype=torch.int32, device="cuda")
        rs = torch.zeros(m, dtype=torch.int32, device="cuda")
        re = torch.full((m,), w, dtype=torch.int32, device="cuda")
        args[w] = {"logits": lg, "idx": idx, "rs": rs, "re": re}
    return args


def time_backends(m, base, topk, rounds, verify):
    """Force each serving backend in turn, plus the sampled kernel direct."""
    widths = tuple(base + o for o in WIDTH_OFFSETS)
    args = build(m, widths, topk)
    rows = []

    cands = available_for(m, base, topk)
    supported = bool(aiter.topk_sampled_supports(m, base, topk))

    try:
        for name in cands:
            force(name)
            bad = None
            for w in widths:                       # warm every width first
                a = args[w]
                run_select(a["logits"], topk, a["idx"])
            if verify:
                a = args[widths[0]]
                bad = check(a["logits"], a["idx"], topk)
            samples = {w: [] for w in widths}
            for _ in range(rounds):
                for w in widths:                   # interleaved
                    a = args[w]
                    samples[w].append(run_select(a["logits"], topk, a["idx"])[1])
            for w in widths:
                rows.append({"m": m, "base": base, "width": w, "topk": topk,
                             "backend": name, "us": st.median(samples[w]),
                             "runs": [round(x, 3) for x in samples[w]],
                             "correct": bad is None, "detail": bad})

        if supported:
            force(None)
            for w in widths:
                a = args[w]
                run_sampled(a["logits"], a["rs"], a["re"], a["idx"], None,
                            m, w, 1, topk)
            bad = None
            if verify:
                a = args[widths[0]]
                bad = check(a["logits"], a["idx"], topk)
            samples = {w: [] for w in widths}
            for _ in range(rounds):
                for w in widths:
                    a = args[w]
                    samples[w].append(run_sampled(
                        a["logits"], a["rs"], a["re"], a["idx"], None,
                        m, w, 1, topk)[1])
            for w in widths:
                rows.append({"m": m, "base": base, "width": w, "topk": topk,
                             "backend": "sampled", "us": st.median(samples[w]),
                             "runs": [round(x, 3) for x in samples[w]],
                             "correct": bad is None, "detail": bad})
    finally:
        force(None)
        del args
        torch.cuda.empty_cache()
    return rows


def time_ab(m, base, topk, rounds, verify):
    """The real router, with and without `sampled` in the available set."""
    widths = tuple(base + o for o in WIDTH_OFFSETS)
    args = build(m, widths, topk)
    rows = []
    try:
        for side in ("before", "after"):
            if side == "before":
                os.environ["AITER_DISABLE_TOPK_SAMPLED"] = "1"
            else:
                os.environ.pop("AITER_DISABLE_TOPK_SAMPLED", None)
            ts._choose.cache_clear()
            ts._available.cache_clear()

            picked = {}
            for w in widths:
                a = args[w]
                run_select(a["logits"], topk, a["idx"])
                try:
                    picked[w] = ts._choose(
                        m, w, topk, ts.wave_size_of(0), False, None, False, True)
                except Exception:
                    picked[w] = "?"
            bad = None
            if verify:
                a = args[widths[0]]
                bad = check(a["logits"], a["idx"], topk)
            samples = {w: [] for w in widths}
            for _ in range(rounds):
                for w in widths:
                    a = args[w]
                    samples[w].append(run_select(a["logits"], topk, a["idx"])[1])
            for w in widths:
                rows.append({"m": m, "base": base, "width": w, "topk": topk,
                             "side": side, "backend": picked[w],
                             "us": st.median(samples[w]),
                             "runs": [round(x, 3) for x in samples[w]],
                             "correct": bad is None, "detail": bad})
    finally:
        os.environ.pop("AITER_DISABLE_TOPK_SAMPLED", None)
        ts._choose.cache_clear()
        ts._available.cache_clear()
        del args
        torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("backends", "ab"), default="backends")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--ms", help="comma list, overrides the M axis")
    ap.add_argument("--bases", help="comma list, overrides the N axis")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the torch.topk check (timing only)")
    ap.add_argument("--out", default="/topk/reports/select_ab.json")
    args = ap.parse_args()

    ms = [int(x) for x in args.ms.split(",")] if args.ms else MS
    bases = [int(x) for x in args.bases.split(",")] if args.bases else BASES
    verify = not args.no_verify

    print("aiter under test: %s" % aiter.__file__)
    print("mode=%s topk=%d rounds=%d cells=%d"
          % (args.mode, args.topk, args.rounds, len(ms) * len(bases)))
    print("%6s %9s | %s" % ("M", "base", "result"))

    out, skipped = [], []
    for m in ms:
        for base in bases:
            if not grid.fits_in_vram(m, base + max(WIDTH_OFFSETS), args.topk):
                skipped.append((m, base))
                print("%6d %9d | oom_skip" % (m, base))
                continue
            try:
                if args.mode == "backends":
                    rows = time_backends(m, base, args.topk, args.rounds, verify)
                else:
                    rows = time_ab(m, base, args.topk, args.rounds, verify)
            except Exception as e:  # noqa: BLE001 -- record, do not abort the sweep
                print("%6d %9d | ERROR %s" % (m, base, str(e)[:60]))
                out.append({"m": m, "base": base, "error": str(e)[:200]})
                continue
            out.extend(rows)
            best = min((r for r in rows if r.get("us")), key=lambda r: r["us"],
                       default=None)
            wrong = [r for r in rows if r.get("correct") is False]
            print("%6d %9d | %2d rows, fastest %s %.1fus%s"
                  % (m, base, len(rows),
                     best["backend"] if best else "-",
                     best["us"] if best else float("nan"),
                     "  WRONG:%d" % len(wrong) if wrong else ""))
            # Written every cell, so a killed run is resumable by hand and a
            # long sweep never loses everything to one bad shape.
            with open(args.out, "w") as f:
                json.dump({"mode": args.mode, "topk": args.topk,
                           "rounds": args.rounds, "ms": ms, "bases": bases,
                           "width_offsets": list(WIDTH_OFFSETS),
                           "oom_skipped": skipped, "rows": out}, f, indent=1)

    print()
    print("cells skipped for VRAM: %d %s" % (len(skipped), skipped if skipped else ""))
    wrong = [r for r in out if r.get("correct") is False]
    print("rows with a wrong result: %d   <-- must be 0" % len(wrong))
    for r in wrong[:10]:
        print("   m=%(m)s width=%(width)s backend=%(backend)s %(detail)s" % r)
    print("WROTE %s (%d rows)" % (args.out, len(out)))
    return 1 if wrong else 0


if __name__ == "__main__":
    raise SystemExit(main())
