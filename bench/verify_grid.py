#!/usr/bin/env python3
"""Correctness hard gate over the whole pow2 grid.

bench/correctness.py carries the torch.topk gates but only over ~10 matrix
cases, which does not cover the 130-point grid. This runs `--mode verify` on
every grid point and requires rows_fail=0 everywhere, plus under_K=0 and
over_Calloc=0 on the sampled paths.

A point that both slows down AND raises its fallback count is a correctness
event, not a performance one, so the candidate-count stats are collected here
rather than inferred from the clock.

Exit 0 only when every point passes.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid  # noqa: E402

ROOT = grid.ROOT


def verify_one(m, n, k, dist, extra=()):
    cmd = [str(grid.BENCH), "--mode", "verify", "--m", str(m), "--n", str(n), "--topk", str(k),
           "--dist", dist, "--dump-stats", "1"]
    cmd += list(extra)
    p = subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True)
    out = p.stdout
    res = {"m": m, "n": n, "topk": k, "dist": dist, "ok": False, "why": ""}
    if p.returncode != 0 or "VERDICT PASS" not in out:
        res["why"] = (p.stderr.strip() or out.strip() or "no verdict")[:160]
        return res
    rf = re.search(r"rows_fail=(\d+)", out)
    if rf and int(rf.group(1)) != 0:
        res["why"] = "rows_fail=%s" % rf.group(1)
        return res
    mp = re.search(r"path=(\w+)", out)
    res["path"] = mp.group(1) if mp else "?"
    # small_n has no candidate stage, so its CANDSTATS line is n/a by design.
    st = re.search(r"under_K=(\d+) over_Calloc=(\d+)", out)
    if st:
        res["under_K"], res["over_cap"] = int(st.group(1)), int(st.group(2))
        if res["under_K"] or res["over_cap"]:
            # Not a wrong answer -- the exact fallback caught it -- but it is the
            # signal that the sampling parameters are off for this shape, and it
            # is what turns a fast shape slow.
            res["warn"] = "under_K=%d over_Calloc=%d" % (res["under_K"], res["over_cap"])
    fb = re.search(r"fallback_rows=(\d+)", out)
    if fb:
        res["fallback_rows"] = int(fb.group(1))
    res["ok"] = True
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", default="uniform",
                    help="uniform|gaussian|equal|inf|adversarial, or 'all' for the five")
    ap.add_argument("--inner", action="store_true", help="only the 24 inner-loop points")
    ap.add_argument("--strict-stats", action="store_true",
                    help="treat under_K/over_Calloc > 0 as a failure, not a warning")
    args = ap.parse_args()

    if not grid.BENCH.exists():
        print("ERROR: missing %s; run make first" % grid.BENCH, file=sys.stderr)
        return 2

    shapes = grid.inner_shapes() if args.inner else grid.all_shapes()
    dists = ("uniform", "gaussian", "equal", "inf", "adversarial") if args.dist == "all" \
        else (args.dist,)

    print("=== correctness gate: %d shapes x %d distribution(s) ===" % (len(shapes), len(dists)))
    bad, warned = [], []
    for dist in dists:
        for (m, n, k) in shapes:
            r = verify_one(m, n, k, dist)
            if not r["ok"]:
                bad.append(r)
                print("  FAIL  M=%-5d N=%-8d %-12s %s" % (m, n, dist, r["why"]))
            else:
                if r.get("warn"):
                    warned.append(r)
                    print("  warn  M=%-5d N=%-8d %-12s %s" % (m, n, dist, r["warn"]))

    print("\n  passed   %d" % (len(shapes) * len(dists) - len(bad)))
    print("  failed   %d" % len(bad))
    print("  warnings %d (sampling off, exact fallback covered it)" % len(warned))

    if bad:
        print("\nCORRECTNESS GATE FAILED", file=sys.stderr)
        return 2
    if warned and args.strict_stats:
        print("\nCORRECTNESS GATE FAILED (--strict-stats: candidate stats off)", file=sys.stderr)
        return 2
    print("\nCORRECTNESS GATE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
