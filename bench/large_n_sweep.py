#!/usr/bin/env python3
"""Large-N (N >= 32768) perf sweep: us, floor, ratio, effective TB/s.

Uses the same traffic model and floor as bench/grid.py / reports/grid_report.json.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid  # noqa: E402

ROOT = grid.ROOT
OUT_TSV = ROOT / "log" / "large_n_sweep.tsv"
OUT_JSON = ROOT / "log" / "large_n_sweep.json"

LARGE_MS = [128, 256, 1024, 4096]
LARGE_NS = [32768, 65536, 131072, 262144, 524288, 1048576]
TOPK = grid.TOPK


def tb_s(m, n, k, us):
    if us <= 0:
        return 0.0
    tbytes, _ = grid.traffic_bytes(m, n, k)
    return tbytes / (us * 1e6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer warmup/iters/repeats")
    ap.add_argument("--json", action="store_true", help="also write log/large_n_sweep.json")
    args = ap.parse_args()

    if not grid.BENCH.exists():
        import subprocess
        subprocess.check_call(["make", "-C", str(ROOT), "benchmark_topk"])

    model = grid.load_model()
    scale = grid.achievable_scale(model)
    rows = []
    print("%5s %8s %9s %8s %6s %8s %8s" % ("M", "N", "us", "floor", "ratio", "TB/s", "path"))
    for n in LARGE_NS:
        for m in LARGE_MS:
            if not grid.fits_in_vram(m, n, TOPK):
                continue
            us, sd, path, argv = grid.time_shape_auto(
                m, n, TOPK, enforce_stddev=False
            )
            floor = grid.floor_us(m, n, TOPK, model, scale)
            ratio = us / floor if floor > 0 else 0.0
            bw = tb_s(m, n, TOPK, us)
            rec = {
                "m": m,
                "n": n,
                "topk": TOPK,
                "us": round(us, 2),
                "floor_us": round(floor, 2),
                "ratio": round(ratio, 3),
                "tb_s": round(bw, 3),
                "path": path,
                "regime": grid.regime_of(m, n),
                "stddev_pct": sd,
                "argv": list(argv),
            }
            rows.append(rec)
            print(
                "%5d %8d %9.2f %8.2f %6.3f %8.3f %8s"
                % (m, n, us, floor, ratio, bw, path)
            )

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TSV, "w", encoding="utf-8") as f:
        f.write("m\tn\ttopk\tus\tfloor_us\tratio\ttb_s\tpath\tregime\tstddev_pct\n")
        for r in rows:
            f.write(
                "%d\t%d\t%d\t%.2f\t%.2f\t%.3f\t%.3f\t%s\t%s\t%.2f\n"
                % (
                    r["m"],
                    r["n"],
                    r["topk"],
                    r["us"],
                    r["floor_us"],
                    r["ratio"],
                    r["tb_s"],
                    r["path"],
                    r["regime"],
                    r["stddev_pct"],
                )
            )
    print("\nwrote %s (%d shapes)" % (OUT_TSV, len(rows)))
    if args.json:
        OUT_JSON.write_text(json.dumps(rows, indent=2) + "\n")
        print("wrote %s" % OUT_JSON)


if __name__ == "__main__":
    main()
