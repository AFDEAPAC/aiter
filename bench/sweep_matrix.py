#!/usr/bin/env python3
"""Sweep M x N matrix, report us / floor_us / ratio sorted table."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid  # noqa: E402

# The grid, the floor model and the VRAM limit all live in bench/grid.py so this
# table and the scoring loop cannot disagree about what they are measuring.
ROOT = grid.ROOT
BENCH = grid.BENCH
OUT_TSV = ROOT / "log" / "grid_sweep.tsv"
OUT_JSON = ROOT / "knowledge" / "grid_sweep.json"

load_model = grid.load_model
achievable_scale = grid.achievable_scale
floor_us = grid.floor_us


def run_shape(m, n, k, quick):
    # The limit is the device, not a hardcoded constant: a previous 14 GB guard
    # wrongly marked M=4096 N=1M (16 GB) as oom_skip on a 309 GB card.
    if not grid.fits_in_vram(m, n, k):
        return {"m": m, "n": n, "topk": k, "status": "oom_skip"}
    kk = min(k, n)
    if kk < 1:
        return {"m": m, "n": n, "topk": k, "status": "invalid_k"}
    warmup, iters, repeats = (2, 5, 3) if quick else (5, 20, 3)
    cmd = [
        str(BENCH),
        "--mode",
        "time",
        "--m",
        str(m),
        "--n",
        str(n),
        "--topk",
        str(kk),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--repeats",
        str(repeats),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, universal_newlines=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip()
        if "incompatible with sampling geometry" in err or "incompatible with sampling geometry" in proc.stdout:
            return {"m": m, "n": n, "topk": kk, "status": "geom_fail", "error": err[:200]}
        if "ERROR" in err or "ERROR" in proc.stdout:
            return {"m": m, "n": n, "topk": kk, "status": "fail", "error": (err or proc.stdout)[:200]}
        return {"m": m, "n": n, "topk": kk, "status": "fail", "error": err[:200]}
    m_wall = re.search(r"wall_ms_median=([0-9.]+)", proc.stdout)
    m_sd = re.search(r"stddev_pct=([0-9.]+)", proc.stdout)
    if not m_wall:
        return {"m": m, "n": n, "topk": kk, "status": "parse_fail", "raw": proc.stdout[:300]}
    us = float(m_wall.group(1)) * 1000.0
    sd = float(m_sd.group(1)) if m_sd else 0.0
    return {"m": m, "n": n, "topk": kk, "status": "ok", "us": us, "stddev_pct": sd}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--propose-scoring", action="store_true")
    args = parser.parse_args()

    if not BENCH.exists():
        subprocess.check_call(["make", "-C", str(ROOT), "benchmark_topk"])

    model = load_model()
    scale = achievable_scale(model)
    print("achievable_scale=%.3f (1.0 == peak stream BW)" % scale)
    ms, ns = grid.MS, grid.NS
    rows = []
    for n in ns:
        for m in ms:
            k = grid.TOPK
            rec = run_shape(m, n, k, args.quick)
            if rec.get("status") == "ok":
                f_us = floor_us(m, n, rec["topk"], model, scale)
                rec["floor_us"] = f_us
                rec["ratio"] = rec["us"] / f_us if f_us > 0 else 0.0
            rows.append(rec)
            tag = rec.get("status", "?")
            if tag == "ok":
                print(f"M={m:5d} N={n:7d} us={rec['us']:9.1f} floor={rec['floor_us']:9.1f} ratio={rec['ratio']:.2f}x")
            else:
                print(f"M={m:5d} N={n:7d} status={tag}")

    ok = [r for r in rows if r.get("status") == "ok"]
    ok.sort(key=lambda r: r.get("ratio", 0.0), reverse=True)

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w") as f:
        f.write("rank\tm\tn\ttopk\tus\tfloor_us\tratio\tstddev_pct\n")
        for i, r in enumerate(ok):
            f.write(
                f"{i+1}\t{r['m']}\t{r['n']}\t{r['topk']}\t{r['us']:.2f}\t{r['floor_us']:.2f}\t{r['ratio']:.3f}\t{r.get('stddev_pct',0):.2f}\n"
            )

    payload = {"rows": rows, "sorted_ok": ok}
    if args.propose_scoring:
        ok_sorted = [r for r in rows if r.get("status") == "ok"]
        ok_sorted.sort(key=lambda r: r.get("ratio", 0.0), reverse=True)
        seen = set()
        proposed = []
        for r in ok_sorted:
            if r["n"] <= 8192:
                reg = "small_n"
            elif r["n"] >= 524288:
                reg = "large_n"
            elif r["m"] <= 16:
                reg = "decode"
            else:
                reg = "prefill"
            if reg in seen and len(proposed) >= 6:
                continue
            proposed.append({"m": r["m"], "n": r["n"], "topk": r["topk"], "regime": reg})
            seen.add(reg)
            if len(proposed) >= 12:
                break
        payload["proposed_scoring_set"] = proposed
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"WROTE {OUT_TSV}")
    print(f"WROTE {OUT_JSON}")
    if ok:
        print("TOP ratio gaps:")
        for r in ok[:10]:
            print(f"  M={r['m']} N={r['n']} ratio={r['ratio']:.2f}x us={r['us']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
