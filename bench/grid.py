#!/usr/bin/env python3
"""Single source of truth for the pow2 scoring grid, its floor model, and the
measurement settings each shape needs.

Imported by sweep_matrix.py, score_grid.py and verify_grid.py so the grid, the
floor formula and the per-shape argv cannot drift between them.
"""

import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark_topk"
MODEL_JSON = ROOT / "knowledge" / "g0_floor_model.json"

# ---------------------------------------------------------------------------
# Grid: the customer matrix, restricted to powers of two. K is fixed at 2048,
# which by itself excludes N < 2048 (the contract requires K <= N).
# ---------------------------------------------------------------------------
TOPK = 2048
MS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
NS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

# Largest shape is M=4096 N=1048576 = 16 GB of input against 309 GB of VRAM
# (measured with rocm-smi). An earlier 14 GB guard was arbitrary and wrongly
# marked that point oom_skip; the real limit is the device, so query it.
VRAM_SAFETY_FRACTION = 0.55   # input + output + candidate buffers + fragmentation


def device_vram_bytes():
    try:
        out = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, universal_newlines=True).stdout
        vals = [int(m.group(1)) for m in re.finditer(r"VRAM Total Memory \(B\):\s*(\d+)", out)]
        if vals:
            return min(vals)
    except OSError:
        pass
    return 0


_VRAM = None


def fits_in_vram(m, n, k=TOPK):
    global _VRAM
    if _VRAM is None:
        _VRAM = device_vram_bytes()
    if _VRAM <= 0:
        return m * n * 4 <= 14 * 1024 ** 3     # conservative fallback only
    need = m * n * 4 + m * k * 4               # input + indices
    return need <= _VRAM * VRAM_SAFETY_FRACTION


def all_shapes():
    """Every (m, n, topk) in the grid that fits on the device."""
    out = []
    for n in NS:
        for m in MS:
            if fits_in_vram(m, n):
                out.append((m, n, TOPK))
    return out


# ---------------------------------------------------------------------------
# Inner-loop subset: 24 points, stratified over the three dispatch paths and
# over M, and deliberately loaded with the worst-ratio points of each path so a
# change that helps only the easy shapes cannot look like a win. prefill_main is
# included because it is the regression anchor.
# ---------------------------------------------------------------------------
INNER = [
    # small_n (N <= 8192)
    (1, 2048), (64, 4096), (512, 8192), (1024, 4096),
    (2048, 4096), (2048, 8192), (4096, 2048), (4096, 8192),
    # decode (small M, large N)
    (1, 1048576), (2, 1048576), (8, 524288), (16, 1048576),
    (32, 262144), (64, 262144), (128, 65536), (256, 32768),
    # prefill (large M)
    (4096, 131072), (512, 16384), (1024, 32768), (2048, 65536),
    (4096, 16384), (4096, 262144), (1024, 1048576), (2048, 262144),
]

ANCHOR = (4096, 131072, TOPK)
ANCHOR_LIMIT_US = 620.0     # v1 best measured here is 615.5-616.1 us
POINT_REGRESS_PCT = 5.0     # no single point may be slower than this

# A per-path geomean must not get worse, but "worse" needs a noise band or the
# gate cannot distinguish a regression from the same number measured twice: a
# change that only touched the decode path was rejected for moving the prefill
# geomean 211.49 -> 211.56 us (+0.03%). Per-shape stddev is 0.1-0.9% and the
# geomean over 8 points averages that down, so 0.5% sits well above the noise
# of the statistic and well below any meaningful regime sacrifice.
PATH_NOISE_BAND_PCT = 0.5


def inner_shapes():
    return [(m, n, TOPK) for (m, n) in INNER]


# ---------------------------------------------------------------------------
# Measurement settings. The sub-150 us shapes cannot hold stddev < 2% at
# warmup 20 / iters 100 (measured 2.7-3.1%), which is above the contract's own
# reject threshold, so a 2% change there would be indistinguishable from noise.
# ---------------------------------------------------------------------------
SMALL_ARGV = (100, 500, 7)      # warmup, iters, repeats
LARGE_ARGV = (20, 100, 5)
SMALL_THRESHOLD_US = 150.0
MAX_STDDEV_PCT = 2.0


def settings_for(expected_us):
    return SMALL_ARGV if (expected_us is None or expected_us < SMALL_THRESHOLD_US) else LARGE_ARGV


# ---------------------------------------------------------------------------
# Floor model. Regime-aware: charging every shape the sampled path's launch
# count and candidate-write traffic produced ratios below 1.0 for the
# single-kernel path, which is proof a model is wrong rather than that a kernel
# beat its floor.
# ---------------------------------------------------------------------------
N_LDS_MAX = 8192            # must track csrc/topk_shape.hip.hpp
NOMINAL_MARGIN = 1.4        # candidates per K; 1.4*2048 = 2867 at the anchor


def traffic_bytes(m, n, k):
    """Returns (bytes moved, kernel launches) for the path this shape takes."""
    read = m * n * 4
    idx_write = m * k * 4
    if n <= N_LDS_MAX:
        return read + idx_write, 1
    return read + int(m * NOMINAL_MARGIN * k) * 8 + idx_write, 3


def load_model():
    if not MODEL_JSON.exists():
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "floor_model.py")])
    return json.loads(MODEL_JSON.read_text())


def _anchor_scales(model):
    """[(N, scale)] from every measured read+write anchor, sorted by N.

    scale = measured / ideal-at-peak for that anchor, i.e. how far the real
    one-block-per-row access pattern sits from peak streaming bandwidth.
    """
    peak = model["bw_model"].get("peak_tb_s", 0.0)
    if peak <= 0:
        return []
    src = model.get("phaseb_anchors") or ([model["phaseb_floor"]]
                                          if model.get("phaseb_floor") else [])
    out = []
    for a in src:
        t, _ = traffic_bytes(a["m"], a["n"], TOPK)
        ideal = t / (peak * 1e6)
        if ideal > 0:
            out.append((a["n"], (a["median_ms"] * 1000.0) / ideal))
    return sorted(out)


def achievable_scale(model, n=None):
    """How far the real access pattern is from peak, as a function of N.

    A SINGLE anchor is not enough. The fraction of peak a shape reaches depends
    on how much contiguous data one block streams, which is N*4 bytes. Anchored
    only at N=131072 (512 KB per block, scale 1.294) the model over-stated the
    floor at N=1048576 (4 MB per block) enough that M=4096 N=1048576 scored
    0.94x -- below its own floor, which is impossible and therefore proof the
    model was wrong, not that the kernel was fast.

    Interpolated in log2(N) between the measured anchors and held flat outside
    them, so the curve is continuous and never extrapolates.
    """
    sc = _anchor_scales(model)
    if not sc:
        return 1.0
    if n is None or len(sc) == 1:
        # Legacy callers with no N: use the N=131072 anchor.
        for nn, s in sc:
            if nn == 131072:
                return s
        return sc[0][1]
    if n <= sc[0][0]:
        return sc[0][1]
    if n >= sc[-1][0]:
        return sc[-1][1]
    for i in range(len(sc) - 1):
        n0, s0 = sc[i]
        n1, s1 = sc[i + 1]
        if n0 <= n <= n1:
            w = (math.log(float(n)) - math.log(float(n0))) / (math.log(float(n1)) - math.log(float(n0)))
            return s0 + w * (s1 - s0)
    return sc[-1][1]


def floor_us(m, n, k, model, scale=None):
    """`scale` is ignored and kept only for call compatibility: the achievable
    fraction of peak is a function of N, so it must be looked up per shape."""
    del scale
    bwm = model["bw_model"]
    peak = bwm.get("peak_tb_s", 1.0)
    lat = bwm.get("latency_us", 5.0)
    per = model["empty_launch"].get("per_launch_us", 8.0)
    t, n_launch = traffic_bytes(m, n, k)
    bw_us = (t / (peak * 1e6)) * achievable_scale(model, n) if peak > 0 else lat
    return max(bw_us, lat, n_launch * per)


# ---------------------------------------------------------------------------
# Running the benchmark
# ---------------------------------------------------------------------------
def regime_of(m, n):
    """STATIC scoring stratum, a function of the shape alone.

    Scoring must not group by the path the binary reports, because path
    selection is itself tunable: when the coop_g table returned 1 for
    M=128 N=16384, that point's reported path flipped from decode to prefill,
    it left the decode geomean and joined the prefill geomean, and prefill
    appeared to improve 5.5% without a single instruction changing. A candidate
    could buy a path's geomean just by moving its slow points elsewhere.

    The observed path is still recorded per point, as information.
    """
    if n <= N_LDS_MAX:
        return "small_n"
    return "decode" if m < 256 else "prefill"


# Kept for callers that want the old name; identical function.
path_of = regime_of


def time_shape(m, n, k, warmup, iters, repeats, extra=()):
    """Returns (us, stddev_pct, path) or raises RuntimeError."""
    cmd = [str(BENCH), "--mode", "time", "--m", str(m), "--n", str(n), "--topk", str(k),
           "--warmup", str(warmup), "--iters", str(iters), "--repeats", str(repeats)]
    cmd += list(extra)
    p = subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True)
    m_us = re.search(r"wall_ms_median=([0-9.]+)", p.stdout)
    if not m_us:
        raise RuntimeError("M=%d N=%d K=%d failed: %s" % (m, n, k, (p.stderr or p.stdout)[:200]))
    sd = re.search(r"stddev_pct=([0-9.]+)", p.stdout)
    pa = re.search(r"path=(\w+)", p.stdout)
    return (float(m_us.group(1)) * 1000.0,
            float(sd.group(1)) if sd else 0.0,
            pa.group(1) if pa else path_of(m, n))


def time_shape_auto(m, n, k, extra=(), enforce_stddev=True):
    """Two-step: a cheap run picks the settings, then the real measurement."""
    us, _, _ = time_shape(m, n, k, 5, 20, 2, extra)
    w, i, r = settings_for(us)
    us, sd, path = time_shape(m, n, k, w, i, r, extra)
    if enforce_stddev and sd >= MAX_STDDEV_PCT:
        # One retry at the strictest settings before declaring the point unusable.
        us, sd, path = time_shape(m, n, k, *SMALL_ARGV, extra=extra)
    return us, sd, path, (w, i, r)


def geomean(values):
    vals = [v for v in values if v and v > 0]
    if not vals:
        return 0.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def per_regime_geomean(records):
    """Group by the static regime, never by the observed path (see regime_of)."""
    by = {}
    for r in records:
        key = r.get("regime") or regime_of(r["m"], r["n"])
        by.setdefault(key, []).append(r["us"])
    return {p: geomean(v) for p, v in by.items()}


per_path_geomean = per_regime_geomean
