#!/usr/bin/env python3
"""Sweep coop_g over the v4 extension region and record per-cell optima.

Stage 1a of the per-region overnight plan: M in {256..4096} x N in
{131072..1048576}, G in {1,2,4,8,16,32}. Writes log/coop_sweep.tsv and
log/coop_sweep.json.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid  # noqa: E402

ROOT = grid.ROOT
OUT_TSV = ROOT / "log" / "coop_sweep.tsv"
OUT_JSON = ROOT / "log" / "coop_sweep.json"
OUT_OPT = ROOT / "log" / "coop_sweep_optima.json"

SWEEP_MS = [256, 512, 1024, 2048, 4096]
SWEEP_NS = [131072, 262144, 524288, 1048576]
SWEEP_GS = [1, 2, 4, 8, 16, 32]
TOPK = grid.TOPK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer warmup/iters/repeats")
    args = ap.parse_args()

    if not grid.BENCH.exists():
        import subprocess
        subprocess.check_call(["make", "-C", str(ROOT), "benchmark_topk"])

    rows = []
    print("m\tn\tg\tus\tpath\tstddev_pct")
    for n in SWEEP_NS:
        for m in SWEEP_MS:
            if not grid.fits_in_vram(m, n, TOPK):
                continue
            for g in SWEEP_GS:
                extra = ["--coop-g", str(g)]
                if args.quick:
                    us, sd, path = grid.time_shape(m, n, TOPK, 5, 20, 3, extra)
                else:
                    us, sd, path, _argv = grid.time_shape_auto(
                        m, n, TOPK, extra=extra, enforce_stddev=False
                    )
                rec = {
                    "m": m,
                    "n": n,
                    "topk": TOPK,
                    "coop_g": g,
                    "us": round(us, 2),
                    "stddev_pct": round(sd, 3),
                    "path": path,
                }
                rows.append(rec)
                print("%d\t%d\t%d\t%.2f\t%s\t%.2f" % (m, n, g, us, path, sd))

    optima = {}
    for n in SWEEP_NS:
        for m in SWEEP_MS:
            cell = [r for r in rows if r["m"] == m and r["n"] == n]
            if not cell:
                continue
            best = min(cell, key=lambda r: r["us"])
            default = next(r for r in cell if r["coop_g"] == 1)
            speedup = default["us"] / best["us"] if best["us"] > 0 else 1.0
            optima["%d,%d" % (m, n)] = {
                "best_g": best["coop_g"],
                "best_us": best["us"],
                "default_us": default["us"],
                "speedup": round(speedup, 3),
                "log2_g": 0 if best["coop_g"] <= 1 else int(round(__import__("math").log2(best["coop_g"]))),
            }

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TSV, "w", encoding="utf-8") as f:
        f.write("m\tn\ttopk\tcoop_g\tus\tstddev_pct\tpath\n")
        for r in rows:
            f.write("%d\t%d\t%d\t%d\t%.2f\t%.3f\t%s\n"
                    % (r["m"], r["n"], r["topk"], r["coop_g"], r["us"],
                       r["stddev_pct"], r["path"]))

    payload = {"sweep_ms": SWEEP_MS, "sweep_ns": SWEEP_NS, "sweep_gs": SWEEP_GS,
               "rows": rows, "optima": optima}
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    OUT_OPT.write_text(json.dumps(optima, indent=2) + "\n")

    print("\n=== per-cell optima ===")
    for key in sorted(optima.keys(), key=lambda k: (int(k.split(",")[0]), int(k.split(",")[1]))):
        o = optima[key]
        print("  M=%-5s N=%-8s G=%-2d  %.2f us (was %.2f, %.2fx)"
              % (key.split(",")[0], key.split(",")[1], o["best_g"],
                 o["best_us"], o["default_us"], o["speedup"]))
    print("\nWROTE %s" % OUT_TSV)
    print("WROTE %s" % OUT_JSON)


if __name__ == "__main__":
    raise SystemExit(main())
