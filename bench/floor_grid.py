#!/usr/bin/env python3
"""Per-cell ideal-selector floor for the whole (M, N) grid, via scripts/select_grid.hip.

The floor is the time an ideal selector needs: read the row once, compare every
element against a threshold, write out the indices that pass. No radix, no
histogram. It is a lower bound for any correct per-row top-k on this hardware,
and unlike knowledge/g0_floor_model.json it is measured per cell rather than
interpolated from three anchors.

Two things the vendored file insists on, and this driver honours:

1. **Read and select have opposing optimal geometries**, so the floor must be
   the smallest END-TO-END select, never a best-case read plus a selection cost.
   Confirmed here on the reference cell: g=32/mlp=1/tb=1024 reads fastest
   (288.3 us) and selects worst (510.7 us), while g=2/mlp=4/tb=1024 selects best
   (359.8 us) on a slower read. Hence several candidate geometries per cell and
   min() over their select medians.

2. **hit_rate is read back from a counter**, so it catches a wrong threshold, a
   bad fill, or a cell that silently did nothing. Every cell is checked against
   k/n and a mismatch is recorded, not dropped.

The vendored file's own tuning heuristic is the starting point, but it was
measured on the toolchain its reference numbers came from and ours is not that
one (see the note in the report): its rule is mlp=4 only when g==1, yet here
g=2/mlp=4 beats g=1/mlp=4 on the reference cell. So mlp=4 variants at g>1 are in
the candidate set too.

One input buffer serves every cell -- the file sizes it for the largest spec
entry and every smaller cell reads a prefix, because the fill hashes the global
index and any prefix is still valid randn. So the whole grid costs one 17.2 GB
fill rather than one per cell.

  python3 bench/floor_grid.py [--out reports/floor_grid.json] [--raw log/floor_grid_raw.txt]
"""

import argparse
import json
import os
import statistics as st
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "select_grid")

MS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
BASES = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
WIDTH_OFFSETS = (0, 1, 2)
TOPK = 2048

# Reference cell from the vendored file's own header, for the build gate.
REF = (4096, 131072, 2048, 1, 4, 1024)
REF_READ_US, REF_SEL_US = 335.8, 342.4

# The file's column order, from its #COLS header.
COLS = ("m n k g mlp tb tail exact blocks per4 slots thr hit_rate hits_total "
        "hits_mean hits_min hits_max written dropped drop_pct "
        "rd1 rd2 rd3 sel1 sel2 sel3 d1 d2 d3").split()
INT_COLS = {"m", "n", "k", "g", "mlp", "tb", "tail", "exact", "blocks", "per4",
            "slots", "hits_total", "hits_min", "hits_max", "written", "dropped"}


def pow2_ceil(x):
    p = 1
    while p < x:
        p *= 2
    return p


def geometries(m, n):
    """Candidate (g, mlp, tb) for one cell, deduped, heuristic first."""
    tb = 256 if n <= 4096 else 1024        # the file: 256 is 15-17% faster at n<=4096
    g0 = max(1, min(32, pow2_ceil(max(1, 512 // max(m, 1)))))
    out = [(g0, 4 if g0 == 1 else 1, tb),  # the file's rule, verbatim
           (g0, 4, tb),                    # same g, mlp=4 -- what wins on this build
           (1, 4, tb),
           (2, 4, tb),
           (min(32, g0 * 2), 4, tb)]
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq                            # never mlp=8: its scalar tail loop makes
                                           # the read control slower than the kernel


def iters_for(m, n):
    return 10 if m * n > (1 << 28) else 50


def build_spec(path):
    lines = ["# m n k g mlp tb iters"]
    index = []
    # Dispatch probe: an almost-empty cell gives the launch floor directly, which
    # is what the small-M cells are actually measuring.
    lines.append("1 4 4 1 4 256 200")
    index.append(("probe", 1, 4, 4, 1, 4, 256))
    lines.append("%d %d %d %d %d %d 30" % REF)
    index.append(("ref",) + REF)
    for m in MS:
        for base in BASES:
            for off in WIDTH_OFFSETS:
                n = base + off
                for (g, mlp, tb) in geometries(m, n):
                    lines.append("%d %d %d %d %d %d %d"
                                 % (m, n, TOPK, g, mlp, tb, iters_for(m, n)))
                    index.append(("cell", m, n, TOPK, g, mlp, tb))
    open(path, "w").write("\n".join(lines) + "\n")
    return len(lines) - 1


def parse(raw):
    rows = []
    for ln in raw.splitlines():
        if not ln.startswith("#R "):
            continue
        f = ln.split()[1:]
        if len(f) != len(COLS):
            continue
        r = {}
        for name, v in zip(COLS, f):
            r[name] = int(v) if name in INT_COLS else float(v)
        r["rd_us"] = st.median([r["rd1"], r["rd2"], r["rd3"]])
        r["sel_us"] = st.median([r["sel1"], r["sel2"], r["sel3"]])
        r["d_us"] = r["sel_us"] - r["rd_us"]
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "floor_grid.json"))
    ap.add_argument("--raw", default=os.path.join(ROOT, "log", "floor_grid_raw.txt"))
    ap.add_argument("--spec", default="/tmp/floor_grid_spec.txt")
    args = ap.parse_args()

    if not os.path.exists(BIN):
        print("missing %s -- build it first:\n  hipcc --offload-arch=gfx950 -O3 "
              "-Wall -Wextra scripts/select_grid.hip -o select_grid" % BIN)
        return 2

    n_spec = build_spec(args.spec)
    print("spec lines: %d" % n_spec)
    sys.stdout.flush()
    p = subprocess.run([BIN, args.spec], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, universal_newlines=True)
    os.makedirs(os.path.dirname(args.raw), exist_ok=True)
    open(args.raw, "w").write(p.stdout)
    if p.returncode != 0:
        print("select_grid exit %d\n%s" % (p.returncode, p.stderr[-2000:]))
        return 1
    rows = parse(p.stdout)
    print("parsed %d rows" % len(rows))

    # --- gate 1: the build reproduces the file's reference cell ---------------
    ref = [r for r in rows if (r["m"], r["n"], r["g"], r["mlp"], r["tb"]) ==
           (REF[0], REF[1], REF[3], REF[4], REF[5])]
    gate = {"expected_read_us": REF_READ_US, "expected_sel_us": REF_SEL_US}
    if ref:
        r = ref[0]
        gate.update(read_us=r["rd_us"], sel_us=r["sel_us"], d_us=r["d_us"],
                    read_pct=100 * (r["rd_us"] - REF_READ_US) / REF_READ_US,
                    sel_pct=100 * (r["sel_us"] - REF_SEL_US) / REF_SEL_US)
        print("reference cell: read %.2f us (%+.1f%%)  select %.2f us (%+.1f%%)  d %.2f us"
              % (gate["read_us"], gate["read_pct"], gate["sel_us"], gate["sel_pct"],
                 gate["d_us"]))

    # --- gate 2: hit_rate against k/n on every cell ---------------------------
    bad_hit = [r for r in rows
               if abs(r["hit_rate"] - r["k"] / r["n"]) > 0.05 * (r["k"] / r["n"])]
    print("hit_rate mismatches: %d of %d" % (len(bad_hit), len(rows)))

    probe = [r for r in rows if r["m"] == 1 and r["n"] == 4]
    launch_us = probe[0]["sel_us"] if probe else None
    print("dispatch floor (m=1 n=4 probe): %s us"
          % ("%.2f" % launch_us if launch_us else "n/a"))

    # --- collapse candidates: smallest end-to-end select per (m, n) -----------
    best = {}
    for r in rows:
        if r["n"] == 4:
            continue
        key = (r["m"], r["n"])
        if key not in best or r["sel_us"] < best[key]["sel_us"]:
            best[key] = r
    cells = []
    for (m, n), r in sorted(best.items()):
        cands = [x for x in rows if x["m"] == m and x["n"] == n]
        cells.append({
            "m": m, "n": n, "k": r["k"], "base": n - (n % 4 if n % 4 <= 2 else 0),
            "floor_us": round(r["sel_us"], 3), "read_us": round(r["rd_us"], 3),
            "d_us": round(r["d_us"], 3), "g": r["g"], "mlp": r["mlp"], "tb": r["tb"],
            "blocks": r["blocks"], "tail": r["tail"], "hit_rate": r["hit_rate"],
            "hit_rate_expected": r["k"] / r["n"], "drop_pct": r["drop_pct"],
            "n_candidates": len(cands),
            "candidate_spread_pct": round(
                100 * (max(x["sel_us"] for x in cands) - r["sel_us"]) / r["sel_us"], 2),
        })
    out = {"gate_reference_cell": gate, "dispatch_floor_us": launch_us,
           "hit_rate_mismatches": len(bad_hit), "topk": TOPK,
           "ms": MS, "bases": BASES, "width_offsets": list(WIDTH_OFFSETS),
           "cells": cells}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("cells: %d   WROTE %s" % (len(cells), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
