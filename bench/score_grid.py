#!/usr/bin/env python3
"""Score the pow2 grid and apply the acceptance rules.

  --tier inner   24 stratified points, drives every decision (~90 s)
  --tier outer   the full grid, the commit gate

  --save-baseline   write the result as the new baseline to compare against
  --baseline PATH   compare against a specific baseline file

Performance acceptance only. The correctness hard gate lives in
bench/verify_grid.py + bench/gate_selftest.py + bench/correctness.py and is
judged FIRST; a candidate that fails it is discarded without looking at these
numbers.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid  # noqa: E402

ROOT = grid.ROOT
BASELINE_INNER = ROOT / "knowledge" / "grid_baseline_inner.json"
BASELINE_OUTER = ROOT / "knowledge" / "grid_baseline_outer.json"


def measure(shapes, model, label, enforce_stddev=True, verbose=True):
    scale = grid.achievable_scale(model)
    recs = []
    noisy = []
    for (m, n, k) in shapes:
        us, sd, path, argv = grid.time_shape_auto(m, n, k, enforce_stddev=enforce_stddev)
        f = grid.floor_us(m, n, k, model, scale)
        # `regime` is the static scoring stratum; `path` is what actually ran.
        rec = {"m": m, "n": n, "topk": k, "regime": grid.regime_of(m, n), "path": path,
               "us": round(us, 2), "stddev_pct": sd, "floor_us": round(f, 2),
               "ratio": round(us / f, 3), "argv": list(argv)}
        recs.append(rec)
        if sd >= grid.MAX_STDDEV_PCT:
            noisy.append(rec)
        if verbose:
            print("  M=%-5d N=%-8d %-8s %9.2f us  floor %8.2f  %5.2fx  sd=%.2f%%"
                  % (m, n, path, us, f, us / f, sd))
    return recs, noisy


def summarize(recs):
    pp = grid.per_regime_geomean(recs)
    counts = {}
    for r in recs:
        key = r.get("regime") or grid.regime_of(r["m"], r["n"])
        counts[key] = counts.get(key, 0) + 1
    return {
        "n_points": len(recs),
        "geomean_us": round(grid.geomean([r["us"] for r in recs]), 3),
        "geomean_ratio": round(grid.geomean([r["ratio"] for r in recs]), 3),
        "per_path_geomean_us": {p: round(v, 3) for p, v in sorted(pp.items())},
        "regime_counts": counts,
        "worst_ratio": max(recs, key=lambda r: r["ratio"]) if recs else None,
    }


def _anchor_fail(cur):
    am, an, _ak = grid.ANCHOR
    anchor = next((r for r in cur if r["m"] == am and r["n"] == an), None)
    if anchor and anchor["us"] > grid.ANCHOR_LIMIT_US:
        return ("anchor M=%d N=%d %.2f us over the %.1f us limit"
                % (am, an, anchor["us"], grid.ANCHOR_LIMIT_US))
    return None


def _per_point_regressions(cur, base):
    """Return (fails, improved_cells, regressed_cells)."""
    fails = []
    improved = []
    regressed = []
    # Keyed on (m, n, topk), NOT (m, n). all_shapes() emits the same (m, n) at
    # k = 512, 1024 and 2048, so 134 of the outer tier's 279 distinct (m, n)
    # pairs collide; a two-part key kept whichever k came last and compared every
    # k's measurement against it. That produced a reproducible phantom -- the
    # whole N=65536 column reporting +5.4% to +12.7% while an interleaved A/B of
    # the old and new binaries at the baseline's own argv agreed to within
    # 0.2 us -- and it can hide a real regression just as easily.
    bmap = {(r["m"], r["n"], r.get("topk", grid.TOPK)): r for r in base}
    improve_band = grid.PATH_NOISE_BAND_PCT
    for r in cur:
        key = (r["m"], r["n"], r.get("topk", grid.TOPK))
        b = bmap.get(key)
        if not b:
            continue
        d = (r["us"] - b["us"]) / b["us"] * 100
        if d > grid.POINT_REGRESS_PCT:
            msg = ("M=%d N=%d K=%d %.2f -> %.2f us (+%.1f%%) exceeds the %.0f%% per-point limit"
                   % (r["m"], r["n"], key[2], b["us"], r["us"], d, grid.POINT_REGRESS_PCT))
            fails.append(msg)
            regressed.append(key + (d,))
        elif d < -improve_band:
            improved.append(key + (d,))
    return fails, improved, regressed


def compare(cur, base):
    """v3 acceptance: per-path geomean + overall geomean improvement."""
    fails = []
    cs, bs = summarize(cur), summarize(base)

    # Rule 2: every path's geomean must not get worse, beyond the noise band.
    band = 1.0 + grid.PATH_NOISE_BAND_PCT / 100.0
    for path, bg in bs["per_path_geomean_us"].items():
        cg = cs["per_path_geomean_us"].get(path)
        if cg is None:
            fails.append("regime %s disappeared from the result set" % path)
            continue
        cn = cs["regime_counts"].get(path)
        bn = bs.get("regime_counts", {}).get(path)
        if bn is not None and cn != bn:
            fails.append("regime %s changed size %d -> %d points; geomeans are not "
                         "comparable" % (path, bn, cn))
        if cg > bg * band:
            fails.append("regime %s geomean %.2f -> %.2f us (+%.2f%%), over the %.1f%% band"
                         % (path, bg, cg, (cg - bg) / bg * 100, grid.PATH_NOISE_BAND_PCT))

    point_fails, _, _ = _per_point_regressions(cur, base)
    fails.extend(point_fails)

    af = _anchor_fail(cur)
    if af:
        fails.append(af)

    improved = cs["geomean_us"] < bs["geomean_us"]
    return (not fails and improved), fails, cs, bs


def compare_per_cell(cur, base):
    """v4 acceptance: no cell regresses beyond POINT_REGRESS_PCT, anchor limit,
    and at least one cell improves beyond the noise band."""
    fails = []
    cs, bs = summarize(cur), summarize(base)

    point_fails, improved, _regressed = _per_point_regressions(cur, base)
    fails.extend(point_fails)

    af = _anchor_fail(cur)
    if af:
        fails.append(af)

    if not improved:
        fails.append("no cell improved beyond the %.1f%% noise band"
                       % grid.PATH_NOISE_BAND_PCT)

    return (not fails), fails, cs, bs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=("inner", "outer"), default="inner")
    ap.add_argument("--rule", choices=("geomean", "per-cell"), default="geomean",
                    help="geomean: v3 per-path + overall geomean (default); "
                         "per-cell: v4 no-regress + at-least-one-win")
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--allow-noisy", action="store_true",
                    help="record points over the stddev limit instead of failing")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="assert the baseline compares clean against itself, then exit")
    args = ap.parse_args()

    if args.self_test:
        # A gate that fires when a baseline is compared against ITSELF is broken,
        # and this one was: the per-point key dropped `topk`, so the 134 colliding
        # (m, n) pairs in the outer tier were scored across different k. Keeping
        # the check here because it needs no GPU and no measurement.
        bad = 0
        for tier, path in (("inner", BASELINE_INNER), ("outer", BASELINE_OUTER)):
            if not path.exists():
                print("  %-6s no baseline at %s" % (tier, path))
                continue
            recs = json.loads(path.read_text())["points"]
            fails, improved, regressed = _per_point_regressions(recs, recs)
            ok = not fails and not improved and not regressed
            print("  %-6s %d points: %d fails, %d improved, %d regressed -> %s"
                  % (tier, len(recs), len(fails), len(improved), len(regressed),
                     "clean" if ok else "BROKEN"))
            bad += 0 if ok else 1
        return 0 if bad == 0 else 1

    if not grid.BENCH.exists():
        print("ERROR: missing %s; run make first" % grid.BENCH, file=sys.stderr)
        return 2

    model = grid.load_model()
    shapes = grid.inner_shapes() if args.tier == "inner" else grid.all_shapes()
    default_base = BASELINE_INNER if args.tier == "inner" else BASELINE_OUTER
    base_path = Path(args.baseline) if args.baseline else default_base

    print("=== tier=%s  %d points ===" % (args.tier, len(shapes)))
    recs, noisy = measure(shapes, model, args.tier, verbose=not args.quiet)
    summary = summarize(recs)

    print("\n  points             %d" % summary["n_points"])
    print("  geomean            %.2f us" % summary["geomean_us"])
    print("  geomean ratio      %.2fx" % summary["geomean_ratio"])
    for p, v in summary["per_path_geomean_us"].items():
        print("  %-18s %.2f us" % (p + " geomean", v))
    w = summary["worst_ratio"]
    if w:
        print("  worst ratio        %.2fx at M=%d N=%d (%s)" % (w["ratio"], w["m"], w["n"], w["path"]))

    if noisy:
        print("\n  STDDEV OVER %.1f%% at %d points:" % (grid.MAX_STDDEV_PCT, len(noisy)))
        for r in noisy:
            print("    M=%d N=%d sd=%.2f%%" % (r["m"], r["n"], r["stddev_pct"]))
        if not args.allow_noisy:
            print("  -> these points cannot score a change smaller than their own noise", file=sys.stderr)

    payload = {"tier": args.tier, "summary": summary, "points": recs}

    if args.save_baseline:
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text(json.dumps(payload, indent=2) + "\n")
        print("\nWROTE baseline %s" % base_path)
        return 0 if (not noisy or args.allow_noisy) else 1

    if base_path.exists():
        base = json.loads(base_path.read_text())["points"]
        scorer = compare_per_cell if args.rule == "per-cell" else compare
        ok, fails, cs, bs = scorer(recs, base)
        print("\n=== acceptance (%s) vs %s ===" % (args.rule, base_path.name))
        print("  geomean %.2f -> %.2f us (%+.2f%%)"
              % (bs["geomean_us"], cs["geomean_us"],
                 (cs["geomean_us"] - bs["geomean_us"]) / bs["geomean_us"] * 100))
        if args.rule == "per-cell":
            _, improved, regressed = _per_point_regressions(recs, base)
            print("  cells improved     %d (beyond %.1f%% band)"
                  % (len(improved), grid.PATH_NOISE_BAND_PCT))
            print("  cells regressed    %d (within %.0f%% limit)"
                  % (len(regressed), grid.POINT_REGRESS_PCT))
        if fails:
            for f in fails:
                print("  REJECT: %s" % f)
        print("  VERDICT: %s" % ("ACCEPT (performance)" if ok else "REJECT"))
        print("  NOTE: correctness hard gate is judged separately and FIRST")
        return 0 if ok else 1

    print("\nno baseline at %s; run with --save-baseline first" % base_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
