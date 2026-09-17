#!/usr/bin/env python3
"""Build achievable-BW curve + launch-overhead floor model for topk-prefill."""

import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLOOR_BENCH = ROOT / "floor_bench"
OUT_JSON = ROOT / "knowledge" / "g0_floor_model.json"


def run_floor_bench():
    subprocess.check_call(["make", "-C", str(ROOT), "floor_bench"], stdout=subprocess.DEVNULL)
    proc = subprocess.run([str(FLOOR_BENCH)], cwd=ROOT, universal_newlines=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)
    return proc.stdout


def parse_bench(stdout):
    empty = {}
    bw = []
    phaseb = None
    anchors = []
    for line in stdout.splitlines():
        m = re.match(r"EMPTY_LAUNCH n=4 median_us=([0-9.]+).*per_launch_us=([0-9.]+)", line)
        if m:
            empty["four_launch_us"] = float(m.group(1))
            empty["per_launch_us"] = float(m.group(2))
            continue
        m = re.match(r"EMPTY_LAUNCH grid=(\d+) median_us=([0-9.]+)", line)
        if m:
            empty[f"grid_{m.group(1)}_us"] = float(m.group(2))
            continue
        m = re.match(
            r"BW_SWEEP bytes=(\d+) median_us=([0-9.]+) stddev_pct=([0-9.]+) bandwidth_TB_s=([0-9.]+)",
            line,
        )
        if m:
            b = int(m.group(1))
            us = float(m.group(2))
            bw.append(
                {
                    "bytes": b,
                    "median_us": us,
                    "stddev_pct": float(m.group(3)),
                    "bandwidth_tb_s": float(m.group(4)),
                }
            )
            continue
        m = re.match(r"PHASEB_FLOOR M=(\d+) N=(\d+) traffic_GB=([0-9.]+) median_ms=([0-9.]+)", line)
        if m:
            rec = {
                "m": int(m.group(1)),
                "n": int(m.group(2)),
                "traffic_gb": float(m.group(3)),
                "median_ms": float(m.group(4)),
            }
            anchors.append(rec)
            # The N=131072 anchor stays the single `phaseb_floor` for anything
            # that still expects one.
            if rec["n"] == 131072:
                phaseb = rec
    return {"empty_launch": empty, "bw_sweep": bw, "phaseb_floor": phaseb,
            "phaseb_anchors": anchors}


def fit_bw_curve(points):
    """Piecewise: small payloads are latency-bound; large ones plateau."""
    if not points:
        return {"peak_tb_s": 0.0, "latency_us": 0.0}

    peak = max(p["bandwidth_tb_s"] for p in points if p["bytes"] >= 64 * 1024)
    # Latency floor from smallest measurable point.
    p0 = min(points, key=lambda p: p["bytes"])
    latency_us = p0["median_us"]
    return {"peak_tb_s": peak, "latency_us": latency_us}


def read_time_us(m, n, k, bw_model, phaseb):
    bytes_in = m * n * 4
    # Candidate write traffic ~ margin*K*8 per row; use measured mean at main shape.
    w_per_row = 2867
    traffic = bytes_in + m * w_per_row * 8
    peak = bw_model["peak_tb_s"]
    lat = bw_model["latency_us"]
    if peak <= 0:
        return lat
    bw_us = traffic / (peak * 1e6)  # TB/s -> bytes/us: peak TB/s * 1e6 = bytes/us... 
    # peak TB/s => bytes/ns = peak * 1e3; bytes/us = peak * 1e3 / 1000 = peak * 1e6? 
    # TB/s = 1e12 B/s => us for B bytes = B / (peak*1e12) * 1e6 = B / (peak*1e6)
    bw_us = traffic / (peak * 1e6)
    est = max(lat, bw_us)
    if phaseb and m == phaseb["m"] and n == phaseb["n"]:
        # Anchor to measured read+write floor at reference shape.
        measured_us = phaseb["median_ms"] * 1000.0
        scale = measured_us / max(est, 1e-6)
        return est * scale
    return est


def launch_cost_us(m, empty):
    per = empty.get("per_launch_us", 8.0)
    return 4.0 * per  # four-kernel pipeline


def floor_us(m, n, k, bw_model, empty, phaseb):
    return max(read_time_us(m, n, k, bw_model, phaseb), launch_cost_us(m, empty))


def propose_scoring_set(shapes):
    """Pick up to 12 shapes spanning regimes with highest ratio gaps."""
    ok = [s for s in shapes if s.get("status") == "ok"]
    ok.sort(key=lambda s: s.get("ratio", 0.0), reverse=True)
    chosen = []
    seen_regime = set()

    def regime(m: int, n: int) -> str:
        if n <= 8192:
            return "small_n"
        if n >= 524288:
            return "large_n"
        if m <= 16:
            return "small_m"
        return "prefill"

    for s in ok:
        r = regime(s["m"], s["n"])
        if r in seen_regime and len(chosen) >= 6:
            continue
        chosen.append(s)
        seen_regime.add(r)
        if len(chosen) >= 12:
            break
    # Always include primary benchmark shape.
    primary = {"m": 4096, "n": 131072, "topk": 2048}
    if not any(x["m"] == primary["m"] and x["n"] == primary["n"] for x in chosen):
        chosen.insert(0, primary)
    return chosen[:12]


def main():
    stdout = run_floor_bench()
    parsed = parse_bench(stdout)
    bw_model = fit_bw_curve(parsed["bw_sweep"])
    empty = parsed["empty_launch"]
    phaseb = parsed["phaseb_floor"]

    if phaseb:
        ref_ms = phaseb["median_ms"]
        if abs(ref_ms - 0.4361) > 0.05:
            print(f"WARN: phaseb floor {ref_ms:.4f} ms deviates from expected 0.4361", file=sys.stderr)

    # Keep the raw curve and EVERY anchor. main() used to rebuild this dict by
    # hand and silently dropped both, so the achievable-vs-peak factor could
    # only ever be a single constant -- which is what let the floor model
    # over-state the floor at large N.
    model = {
        "bw_model": bw_model,
        "empty_launch": empty,
        "phaseb_floor": phaseb,
        "phaseb_anchors": parsed.get("phaseb_anchors", []),
        "bw_sweep": parsed.get("bw_sweep", []),
        "n_launches": 3,
        "formula": "floor_us = max(traffic/peak_bw * achievable_scale(N), latency_us, "
                   "n_launches*per_launch_us); achievable_scale is interpolated in log2(N) "
                   "between the measured read+write anchors",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(model, indent=2) + "\n")
    print(f"WROTE {OUT_JSON}")
    print(json.dumps(model, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
