#!/usr/bin/env python3
"""Correctness gate for ragged (triangular) rows: row_len = row + 1.

Compares the AVO kernels against a CPU oracle over the five distributions,
checking value multiset, index validity (idx < row_len), uniqueness among
valid indices, and -1 padding when row_len < K.
"""

import argparse
import re
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark_topk"

DIST_MODES = {
    "uniform": 0,
    "gaussian": 1,
    "equal": 2,
    "inf": 3,
    "adversarial": 4,
}


def fp32_to_sortable_bits(v):
    u = struct.unpack(">I", struct.pack(">f", v))[0]
    if u & 0x80000000:
        return u ^ 0xFFFFFFFF
    return u | 0x80000000


def cpu_topk_row(row, k):
    n = len(row)
    k_take = min(k, n)
    pairs = [(fp32_to_sortable_bits(row[i]), i) for i in range(n)]
    pairs.sort(key=lambda x: x[0], reverse=True)
    idx = [pairs[i][1] for i in range(k_take)]
    idx += [-1] * (k - k_take)
    return idx


def verify_one(m, n, k, dist, seed=42):
    cmd = [
        str(BENCH),
        "--mode",
        "verify",
        "--ragged",
        "1",
        "--m",
        str(m),
        "--n",
        str(n),
        "--topk",
        str(k),
        "--dist",
        dist,
        "--seed",
        str(seed),
        "--verify-oracle",
        "cpu",
        "--verify-sample-rows",
        str(m),
    ]
    p = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    out = p.stdout
    if p.returncode != 0 or "VERDICT PASS" not in out:
        return False, (p.stderr.strip() or out.strip() or "no verdict")[:200]
    rf = re.search(r"rows_fail=(\d+)", out)
    if rf and int(rf.group(1)) != 0:
        return False, "rows_fail=%s" % rf.group(1)
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--n", type=int, default=131072,
                    help="row pitch (buffer width); row i has length i+1")
    ap.add_argument("--topk", type=int, default=2048)
    ap.add_argument("--dist", default="all",
                    help="uniform|gaussian|equal|inf|adversarial|all")
    args = ap.parse_args()

    if not BENCH.exists():
        print("ERROR: missing %s; run make first" % BENCH, file=sys.stderr)
        return 2

    dists = list(DIST_MODES.keys()) if args.dist == "all" else [args.dist]
    bad = []
    print("=== ragged verify: M=%d pitch=%d K=%d x %d dist ===" % (
        args.m, args.n, args.topk, len(dists)))
    for dist in dists:
        ok, why = verify_one(args.m, args.n, args.topk, dist)
        if ok:
            print("  PASS  %-12s" % dist)
        else:
            bad.append((dist, why))
            print("  FAIL  %-12s %s" % (dist, why))

    if bad:
        print("\nRAGGED GATE FAILED", file=sys.stderr)
        return 2
    print("\nRAGGED GATE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
