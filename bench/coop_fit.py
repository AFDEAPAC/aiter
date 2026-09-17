#!/usr/bin/env python3
"""Fit kCoopLog2G at half-octave N resolution from a coop sweep, and emit the C++.

Why half-octaves. The shipped table indexes N by `ilog2_floor(N) - 14`, so one
column serves a whole octave, and the v5 sweep found columns that cannot: at
M=256 the ni=1 column has to serve both N=32768, which wants G=1 (G=8 there is
+32%), and N=49152, which wants G=8 (1.09x). Same conflict at M=1024 in ni=2.
Splitting each octave at 1.5x separates them.

Reads log/coop_sweep_half.json for the per-bucket optimum and
log/coop_sweep_hole.json to cross-check the choice against N values inside the
bucket that were not the one fitted on.
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid  # noqa: E402

ROOT = grid.ROOT
MS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
NCOLS = 13          # half-octave buckets, N = 16384 .. 1048576
BASE_LOG2 = 14

# The shipped octave table, kept for the one bucket the v5 sweep did not cover
# (N in [16384, 24576), measured at N=16384 in the original 8x7 sweep).
LEGACY_NI0 = {1: 2, 2: 2, 4: 3, 8: 3, 16: 3, 32: 3, 64: 3, 128: 0,
              256: 0, 512: 0, 1024: 0, 2048: 0, 4096: 0}


def bucket_of(n):
    """Half-octave index: 0 = [16384, 24576), 1 = [24576, 32768), 2 = [32768, 49152) ..."""
    k = int(math.floor(math.log2(n)))
    lo = 1 << k
    return 2 * (k - BASE_LOG2) + (1 if n >= lo + (lo >> 1) else 0)


def load(*tags):
    out = {}
    for tag in tags:
        p = ROOT / "log" / ("coop_sweep_%s.json" % tag)
        if not p.exists():
            continue
        for r in json.loads(p.read_text())["rows"]:
            out.setdefault((r["m"], r["n"]), {})[r["coop_g"]] = r["us"]
    return out


def pick_minimax(cells):
    """Choose one G for a bucket from several measured N inside it.

    Minimax on relative cost, NOT the argmin at one N. A cell serves every N in
    its half-octave, so fitting it to a single point is how M=1024 bucket 11 got
    G=1: at the one N measured there (786432) G=1 led G=16 by 0.2%, while at
    N=1048572 in the same bucket G=1 costs +12.6%. Worst case over the bucket
    picks G=16, which costs 0.2% and 0.1% at the other two N.

    `cells` is {n: {g: us}}. Returns (g, worst_cost_pct, detail).
    """
    gs = set()
    for g_us in cells.values():
        gs |= set(g_us)
    best_per_n = {n: min(g_us.values()) for n, g_us in cells.items()}
    scored = []
    for g in sorted(gs):
        worst = 0.0
        ok = True
        for n, g_us in cells.items():
            if g not in g_us:
                ok = False
                break
            worst = max(worst, (g_us[g] - best_per_n[n]) / best_per_n[n] * 100)
        if ok:
            scored.append((worst, g))
    if not scored:
        return 1, 0.0, "no G measured at every N"
    scored.sort(key=lambda t: (round(t[0], 2), t[1]))
    worst, g = scored[0]
    return g, worst, "%d N in bucket" % len(cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance-pct", type=float, default=2.0,
                    help="how much a bucket's G may cost an N it was not fitted on")
    ap.add_argument("--tie-pct", type=float, default=1.0,
                    help="G values within this much of a cell's best count as tied")
    args = ap.parse_args()

    fit = load("half", "hole", "mid", "base16k")
    if not fit:
        print("no coop sweep data in log/", file=sys.stderr)
        return 2

    table = {}
    coverage = {}
    for m in MS:
        row = [None] * NCOLS
        row[0] = LEGACY_NI0.get(m, 0)
        buckets = {}
        for (mm, n), gs in sorted(fit.items()):
            if mm != m:
                continue
            b = bucket_of(n)
            if b < NCOLS:
                buckets.setdefault(b, {})[n] = gs
        for b, cells in sorted(buckets.items()):
            g, worst, detail = pick_minimax(cells)
            # Only claim a win when it clears the noise band against G=1;
            # otherwise keep G=1, the cheaper kernel and the safer default.
            if g != 1:
                base = {n: c.get(1) for n, c in cells.items() if 1 in c}
                if base and all(cells[n][g] > base[n] * (1.0 - grid.PATH_NOISE_BAND_PCT / 100.0)
                                for n in base):
                    g = 1
            row[b] = int(round(math.log2(g)))
            coverage[(m, b)] = (len(cells), worst)
        for i in range(NCOLS):
            if row[i] is None:
                row[i] = row[i - 1] if i else 0
        table[m] = row

    thin = sorted((b, m) for (m, b), (n, _w) in coverage.items() if n < 2)
    print("=== bucket coverage ===")
    print("  cells fitted on a single N (cannot see across their own range): %d of %d"
          % (len(thin), len(coverage)))
    if thin:
        print("  %s" % ", ".join("M=%d col%d" % (m, b) for b, m in thin[:12]))
    worst_cells = sorted(((w, m, b) for (m, b), (n, w) in coverage.items() if n >= 2),
                         reverse=True)[:6]
    print("  largest worst-case cost inside a multi-N bucket:")
    for w, m, b in worst_cells:
        print("    M=%-5d col%-3d %+.2f%%" % (m, b, w))

    # Monotonicity pass. For a fixed M a longer row has more work to split, so
    # the best block count per row should not FALL as N grows -- which is the
    # shape the original M=1..128 rows already have. Without this the argmin
    # picks noise: at M=512 N=786432 the whole G=2..32 range spans 1.9% and the
    # winner (G=2) sits between neighbours that both want G=8..16, and at
    # M=1024 N=393216 and N=786432 G=1 beats the smooth choice by 0.3% and 0.2%.
    # A noise pick does not stay local, because the cell serves every N in its
    # half-octave, so ties are resolved toward monotone instead.
    print("=== ties resolved toward monotonicity (within %.1f%% ACROSS the bucket) ==="
          % args.tie_pct)
    any_tie = False
    for m in MS:
        row = table[m]
        for i in range(1, NCOLS):
            if row[i] >= row[i - 1]:
                continue
            cells = {nn: gs for (mm, nn), gs in fit.items()
                     if mm == m and bucket_of(nn) == i}
            if not cells:
                row[i] = row[i - 1]
                continue
            want = 1 << row[i - 1]
            # Worst case over every measured N in the bucket, for the same reason
            # the selection above is minimax: checking a single N let the monotone
            # nudge move M=64 col1 to G=16, free at N=24576 and +9.4% at N=28672
            # in the same bucket.
            cost = 0.0
            for nn, gs in cells.items():
                if want not in gs:
                    cost = float("inf")
                    break
                b = min(gs.values())
                cost = max(cost, (gs[want] - b) / b * 100)
            any_tie = True
            if cost <= args.tie_pct:
                print("  M=%-5d col%-3d G=%d -> %d  (worst +%.2f%% over %d N)"
                      % (m, i, 1 << row[i], want, cost, len(cells)))
                row[i] = row[i - 1]
            else:
                print("  M=%-5d col%-3d KEPT non-monotone G=%d: G=%d costs +%.2f%% somewhere"
                      % (m, i, 1 << row[i], want, cost))
    if not any_tie:
        print("  none: every row is already monotone in N")

    # Cross-check: what does the fitted bucket cost at every measured N?
    print("=== cross-check: fitted G against each measured N's own optimum ===")
    worst = []
    for (m, n), gs in sorted(fit.items()):
        b = bucket_of(n)
        if b >= NCOLS or m not in table:
            continue
        g = 1 << table[m][b]
        if g not in gs:
            continue
        best = min(gs.values())
        cost = (gs[g] - best) / best * 100
        if cost > args.tolerance_pct:
            worst.append((cost, m, n, g, gs[g], best))
    if worst:
        for cost, m, n, g, us, best in sorted(worst, reverse=True):
            print("  M=%-5d N=%-7d table G=%-3d %8.2f us vs best %8.2f  +%.1f%%"
                  % (m, n, g, us, best, cost))
    else:
        print("  every cross-checked N is within %.1f%% of its own optimum" % args.tolerance_pct)

    print("\n=== fitted table (log2 G), columns are half-octaves of N ===")
    hdr = []
    for i in range(NCOLS):
        k = BASE_LOG2 + i // 2
        lo = 1 << k
        hdr.append("%d" % (lo if i % 2 == 0 else lo + (lo >> 1)))
    print("%-8s %s" % ("M", " ".join("%7s" % h for h in hdr)))
    for m in MS:
        print("%-8d %s" % (m, " ".join("%7d" % v for v in table[m])))

    print("\n=== C++ ===")
    print("constexpr int COOP_TAB_M = %d;" % len(MS))
    print("constexpr int COOP_TAB_N = %d;" % NCOLS)
    print("static const signed char kCoopLog2G[COOP_TAB_M][COOP_TAB_N] = {")
    for m in MS:
        print("    /* M=%-4d */ {%s}," % (m, ", ".join("%d" % v for v in table[m])))
    print("};")
    out = ROOT / "log" / "coop_table_half.json"
    out.write_text(json.dumps({"ms": MS, "ncols": NCOLS, "base_log2": BASE_LOG2,
                               "table": {str(k): v for k, v in table.items()}}, indent=1) + "\n")
    print("\nWROTE %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
