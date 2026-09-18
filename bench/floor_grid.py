#!/usr/bin/env python3
"""Per-cell ideal-selector floor for the whole (M, N) grid, via scripts/select_grid.hip.

The floor is the time an ideal selector needs: read the row once, compare every
element against a threshold, write out the indices that pass. No radix, no
histogram. It is a lower bound for any correct per-row top-k on this hardware,
and unlike knowledge/g0_floor_model.json it is measured per cell rather than
interpolated from three anchors.

Two modes, and the difference matters more than it looks:

  --trace   (default) kernel duration from a rocprofv3 kernel trace
  --event             select_grid's own hipEvent timing, as the file ships it

select_grid brackets EVERY launch with a hipEvent pair (its lines 462-475), so
its number is kernel time plus the command-processor bubble around a single
dispatch. That bubble is about 2.5 us here, which is invisible on a 300 us cell
and is the entire measurement on a 3 us one -- it is why an empty kernel reads
"about 6 us" in the file's own notes and why the small-M half of the grid comes
out as a flat 6.2 us plateau. Measured directly at m=128 n=16384 g=4 mlp=4
tb=1024: event timing says read 5.96 / select 6.16 us, the trace says 3.44 /
3.64. The plateau is the ruler, not the kernel.

That matters beyond tidiness: our own numbers come from aiter @perftest, which
reads kernel duration out of a profiler trace. Comparing a kernel-only number
against a launch-inclusive one makes the kernel look faster than the floor in
exactly the region where the floor is mostly launch. Both sides have to be the
same ruler, so --trace is the default.

Everything else follows the vendored file's own instructions:

  * Read and select have OPPOSING optimal geometries, so the floor must be the
    smallest END-TO-END select, never a best-case read plus a selection cost.
    Confirmed on the reference cell: g=32/mlp=1/tb=1024 reads fastest (288.3 us)
    and selects worst (510.7 us). Hence several candidates per cell, min over
    their select times.
  * hit_rate is read back from a counter, so it catches a wrong threshold, a bad
    fill, or a cell that silently did nothing. Checked against k/n on every cell.
  * One input buffer serves every cell -- the file sizes it for the largest spec
    entry and every smaller cell reads a prefix -- so the whole grid costs one
    17.2 GB fill. It also means the absolute numbers depend on that allocation:
    the reference cell alone (2.15 GB) gives a different select than the same
    cell inside the full grid (17.2 GB). Measure the grid in one run.

  python3 bench/floor_grid.py [--event] [--out reports/floor_grid.json]
"""

import argparse
import csv
import glob
import json
import os
import shutil
import statistics as st
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "select_grid")

MS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
BASES = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
WIDTH_OFFSETS = (0, 1, 2)
TOPK = 2048

REF = (4096, 131072, 2048, 1, 4, 1024)
REF_READ_US, REF_SEL_US = 335.8, 342.4
PASSES = 3
WARMUP_PAIRS = 5

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
    """Candidate (g, mlp, tb) for one cell, deduped, the file's heuristic first."""
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


def iters_for(m, n, traced):
    # The same sample count in both modes. Cutting it to 5 under the profiler
    # looked free -- every dispatch is timed, so why take more -- and wrecked
    # reproducibility: re-running moved the median cell 12.3% and put 258 of 390
    # cells over 5%, against a 1.9% median in event mode. Short kernels need the
    # samples whichever ruler you hold them against.
    del traced
    return 10 if m * n > (1 << 28) else 50


def build_spec(path, traced):
    """Write the spec and return the ordered index of what each line is."""
    lines, index = ["# m n k g mlp tb iters"], []

    def add(kind, m, n, k, g, mlp, tb, it):
        lines.append("%d %d %d %d %d %d %d" % (m, n, k, g, mlp, tb, it))
        index.append({"kind": kind, "m": m, "n": n, "k": k, "g": g, "mlp": mlp,
                      "tb": tb, "iters": it})

    probe_it = 5 if traced else 200
    add("probe", 1, 4, 4, 1, 4, 256, probe_it)      # dispatch floor, on its own
    add("ref", *(REF + (5 if traced else 30,)))
    for m in MS:
        for base in BASES:
            for off in WIDTH_OFFSETS:
                n = base + off
                for (g, mlp, tb) in geometries(m, n):
                    add("cell", m, n, TOPK, g, mlp, tb, iters_for(m, n, traced))
    open(path, "w").write("\n".join(lines) + "\n")
    return index


def parse_r_rows(raw):
    rows = []
    for ln in raw.splitlines():
        if not ln.startswith("#R "):
            continue
        f = ln.split()[1:]
        if len(f) != len(COLS):
            continue
        r = {c: (int(v) if c in INT_COLS else float(v)) for c, v in zip(COLS, f)}
        r["rd_us"] = st.median([r["rd1"], r["rd2"], r["rd3"]])
        r["sel_us"] = st.median([r["sel1"], r["sel2"], r["sel3"]])
        rows.append(r)
    return rows


def trace_durations(csv_path):
    """(kind, duration_us) per kernel dispatch, in launch order."""
    out = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            nm = r.get("Kernel_Name", "")
            kind = "S" if "select_kern" in nm else ("R" if "read_kern" in nm else None)
            if kind is None:
                continue
            out.append((int(r["Dispatch_Id"]), kind,
                        (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1000.0))
    out.sort()
    return [(k, d) for _, k, d in out]


def attribute(disp, index):
    """Walk the trace in launch order and hand each spec line its own dispatches.

    The pattern is fixed by select_grid's loop: one select for the hit counts,
    then PASSES passes of WARMUP_PAIRS warm-up (read, select) pairs followed by
    `iters` timed pairs. Mismatching it means the trace and the spec have drifted
    apart, which would silently mis-assign every later cell, so it aborts.
    """
    i, out = 0, []
    for spec in index:
        need = 1 + PASSES * (WARMUP_PAIRS + spec["iters"]) * 2
        if i + need > len(disp):
            raise SystemExit("trace ran out at spec %r: need %d, have %d"
                             % (spec, need, len(disp) - i))
        chunk = disp[i:i + need]
        i += need
        if chunk[0][0] != "S":
            raise SystemExit("expected the hit-count select first at %r" % (spec,))
        p = 1
        rd_pass, se_pass = [], []
        for _ in range(PASSES):
            pairs = []
            for _ in range(WARMUP_PAIRS + spec["iters"]):
                r, s = chunk[p], chunk[p + 1]
                p += 2
                if r[0] != "R" or s[0] != "S":
                    raise SystemExit("pattern broke at %r" % (spec,))
                pairs.append((r[1], s[1]))
            timed = pairs[WARMUP_PAIRS:]          # the file times only these
            rd_pass.append(st.median(x[0] for x in timed))
            se_pass.append(st.median(x[1] for x in timed))
        out.append(dict(spec, rd_us=st.median(rd_pass), sel_us=st.median(se_pass)))
    if i != len(disp):
        raise SystemExit("trace has %d unassigned dispatches left" % (len(disp) - i))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", action="store_true",
                    help="use select_grid's own hipEvent timing instead of a trace")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "floor_grid.json"))
    ap.add_argument("--raw", default=os.path.join(ROOT, "log", "floor_grid_raw.txt"))
    ap.add_argument("--spec", default="/tmp/floor_grid_spec.txt")
    ap.add_argument("--tracedir", default="/tmp/floor_grid_trace")
    args = ap.parse_args()
    traced = not args.event

    if not os.path.exists(BIN):
        print("missing %s -- build it first:\n  hipcc --offload-arch=gfx950 -O3 "
              "-Wall -Wextra scripts/select_grid.hip -o select_grid" % BIN)
        return 2

    index = build_spec(args.spec, traced)
    print("spec lines: %d   timing: %s" % (len(index), "trace" if traced else "event"))
    sys.stdout.flush()

    if traced:
        shutil.rmtree(args.tracedir, ignore_errors=True)
        cmd = ["rocprofv3", "--kernel-trace", "-d", args.tracedir, "-o", "fg",
               "--output-format", "csv", "--", BIN, args.spec]
    else:
        cmd = [BIN, args.spec]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True)
    os.makedirs(os.path.dirname(args.raw), exist_ok=True)
    open(args.raw, "w").write(p.stdout)
    if p.returncode != 0:
        print("select_grid exit %d\n%s" % (p.returncode, p.stderr[-2000:]))
        return 1

    rrows = parse_r_rows(p.stdout)
    print("parsed %d #R rows" % len(rrows))
    if len(rrows) != len(index):
        print("spec/#R mismatch: %d vs %d" % (len(index), len(rrows)))
        return 1

    if traced:
        hits = glob.glob(os.path.join(args.tracedir, "**", "*kernel_trace.csv"),
                         recursive=True)
        if not hits:
            print("no kernel trace under %s" % args.tracedir)
            return 1
        disp = trace_durations(hits[0])
        print("trace dispatches: %d" % len(disp))
        timed = attribute(disp, index)
        # Carry the diagnostics from the #R row, replace only the timings.
        for r, t in zip(rrows, timed):
            assert (r["m"], r["n"], r["g"], r["mlp"], r["tb"]) == \
                   (t["m"], t["n"], t["g"], t["mlp"], t["tb"]), (r, t)
            r["rd_us"], r["sel_us"] = t["rd_us"], t["sel_us"]
    for r in rrows:
        r["d_us"] = r["sel_us"] - r["rd_us"]

    ref = [r for r in rrows if (r["m"], r["n"], r["g"], r["mlp"], r["tb"]) ==
           (REF[0], REF[1], REF[3], REF[4], REF[5])]
    gate = {"expected_read_us": REF_READ_US, "expected_sel_us": REF_SEL_US,
            "timing": "trace" if traced else "event"}
    if ref:
        r = ref[0]
        gate.update(read_us=r["rd_us"], sel_us=r["sel_us"], d_us=r["d_us"],
                    read_pct=100 * (r["rd_us"] - REF_READ_US) / REF_READ_US,
                    sel_pct=100 * (r["sel_us"] - REF_SEL_US) / REF_SEL_US)
        print("reference cell: read %.2f us (%+.1f%%)  select %.2f us (%+.1f%%)  d %+.2f us"
              % (gate["read_us"], gate["read_pct"], gate["sel_us"], gate["sel_pct"],
                 gate["d_us"]))

    bad_hit = [r for r in rrows
               if abs(r["hit_rate"] - r["k"] / r["n"]) > 0.05 * (r["k"] / r["n"])]
    print("hit_rate mismatches: %d of %d" % (len(bad_hit), len(rrows)))

    probe = [r for r in rrows if r["m"] == 1 and r["n"] == 4]
    launch_us = probe[0]["sel_us"] if probe else None
    print("dispatch floor (m=1 n=4 probe): %s us"
          % ("%.2f" % launch_us if launch_us else "n/a"))

    best = {}
    for r in rrows:
        if r["n"] == 4:
            continue
        key = (r["m"], r["n"])
        if key not in best or r["sel_us"] < best[key]["sel_us"]:
            best[key] = r
    cells = []
    for (m, n), r in sorted(best.items()):
        cands = [x for x in rrows if x["m"] == m and x["n"] == n]
        cells.append({
            "m": m, "n": n, "k": r["k"],
            "floor_us": round(r["sel_us"], 4), "read_us": round(r["rd_us"], 4),
            "d_us": round(r["d_us"], 4), "g": r["g"], "mlp": r["mlp"], "tb": r["tb"],
            "blocks": r["blocks"], "tail": r["tail"], "hit_rate": r["hit_rate"],
            "hit_rate_expected": r["k"] / r["n"], "drop_pct": r["drop_pct"],
            "n_candidates": len(cands),
            "candidate_spread_pct": round(
                100 * (max(x["sel_us"] for x in cands) - r["sel_us"]) / r["sel_us"], 2),
        })
    out = {"gate_reference_cell": gate, "dispatch_floor_us": launch_us,
           "hit_rate_mismatches": len(bad_hit), "topk": TOPK,
           "timing": "rocprofv3 kernel duration" if traced else "hipEvent per launch",
           "ms": MS, "bases": BASES, "width_offsets": list(WIDTH_OFFSETS),
           "cells": cells}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("cells: %d   WROTE %s" % (len(cells), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
