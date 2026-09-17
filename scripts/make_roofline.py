#!/usr/bin/env python3
"""Roofline for the pow2 top-k grid.

  python3 scripts/make_roofline.py [-i reports/grid_report.json] [-o reports/grid_roofline]

A FLOP/byte roofline is meaningless for this kernel: top-k does comparisons, not
arithmetic, so every fp32 shape has the same ~0.25 elements/byte and all 130
points collapse onto one vertical line. The structure that IS informative for a
pure-memory kernel is the same picture with the axes changed:

    classic                        here
    ------------------------------ --------------------------------------
    y = attainable FLOP/s          y = attainable effective bandwidth
    x = arithmetic intensity       x = bytes the call must move
    diagonal roof = peak BW * AI   diagonal roof = bytes / fixed dispatch cost
    flat roof     = peak compute    flat roof     = achievable stream bandwidth
    ridge point                    ridge point (dispatch-bound -> BW-bound)

Both ceilings are measured on this machine, not theoretical: the flat roof is the
read bandwidth sweep from scripts/floor_bench.hip, and the diagonal comes from
the measured empty-kernel launch cost.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import grid  # noqa: E402

ROOT = grid.ROOT

REGIME_STYLE = {
    "small_n": ("#2e7d32", "o", "small_n  (one kernel, whole row in LDS)"),
    "decode": ("#1565c0", "s", "decode   (small M, blocks cooperate on a row)"),
    "prefill": ("#e07000", "^", "prefill  (large M, one block per row)"),
}


def _kn(n):
    return "%dK" % (n // 1024) if n >= 1024 else str(n)


def interp_loglog(xs, ys, x):
    """Piecewise log-log interpolation, held flat outside the measured range."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            lx0, lx1 = math.log(xs[i]), math.log(xs[i + 1])
            ly0, ly1 = math.log(ys[i]), math.log(ys[i + 1])
            w = (math.log(x) - lx0) / (lx1 - lx0)
            return math.exp(ly0 + w * (ly1 - ly0))
    return ys[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inp", default=str(ROOT / "reports" / "grid_report.json"))
    ap.add_argument("-o", "--out", default=str(ROOT / "reports" / "grid_roofline"))
    args = ap.parse_args()

    src = Path(args.inp)
    if not src.exists():
        print("ERROR: %s missing; run scripts/make_html_report.py first" % src, file=sys.stderr)
        return 2
    pts = json.loads(src.read_text())["points"]
    model = grid.load_model()

    per_launch = model["empty_launch"]["per_launch_us"]
    sweep = sorted(model["bw_sweep"], key=lambda p: p["bytes"])
    sw_x = [p["bytes"] for p in sweep]
    sw_y = [p["bandwidth_tb_s"] for p in sweep]
    anchors = sorted(model["phaseb_anchors"], key=lambda a: a["n"])

    # Each point: bytes it must move, and the bandwidth it actually achieved.
    for r in pts:
        t, nl = grid.traffic_bytes(r["m"], r["n"], r["topk"])
        r["bytes"] = t
        r["launches"] = nl
        r["tb_s"] = t / (r["us"] * 1e-6) / 1e12
        r["roof_tb_s"] = min(interp_loglog(sw_x, sw_y, t), t / (nl * per_launch * 1e-6) / 1e12)
        r["of_roof"] = r["tb_s"] / r["roof_tb_s"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.6),
                                  gridspec_kw={"width_ratios": [1.32, 1]})

    # ---------------- panel 1: the roofline ----------------
    xs = [10 ** (e / 24.0) for e in range(int(24 * math.log10(2e5)), int(24 * math.log10(4e10)))]
    ax.plot(sw_x, sw_y, color="#444", lw=2.0, zorder=4,
            label="measured read bandwidth (machine roof)")
    ax.plot(sw_x, sw_y, "o", color="#444", ms=3.0, zorder=5)

    for nl, style, lbl in ((1, ":", "1 launch (small_n)"), (3, "--", "3 launches (sampled paths)")):
        ax.plot(xs, [x / (nl * per_launch * 1e-6) / 1e12 for x in xs], style, color="#b42318",
                lw=1.6, zorder=3, label="dispatch roof, %s" % lbl)

    for a in anchors:
        t, _ = grid.traffic_bytes(a["m"], a["n"], grid.TOPK)
        ax.plot([t], [t / (a["median_ms"] * 1e-3) / 1e12], "*", color="#6a1b9a", ms=15, zorder=6)
    ax.plot([], [], "*", color="#6a1b9a", ms=13,
            label="measured read+write floor kernel\n(same one-block-per-row pattern)")

    for reg, (c, mk, lbl) in REGIME_STYLE.items():
        sel = [r for r in pts if r["regime"] == reg]
        ax.scatter([r["bytes"] for r in sel], [r["tb_s"] for r in sel], s=27, c=c, marker=mk,
                   alpha=0.85, edgecolors="white", linewidths=0.4, zorder=7, label=lbl)

    # Ridge point for the 3-launch pipeline: where the dispatch roof stops being
    # the binding one. The 1-launch ridge is far left and not annotated to keep
    # the label out of the legend.
    for i in range(len(sw_x) - 1):
        d0 = sw_x[i] / (3 * per_launch * 1e-6) / 1e12
        d1 = sw_x[i + 1] / (3 * per_launch * 1e-6) / 1e12
        if d0 < sw_y[i] and d1 >= sw_y[i + 1]:
            ax.plot([sw_x[i + 1]], [sw_y[i + 1]], "o", mfc="none", mec="#b42318", ms=14,
                    mew=1.8, zorder=8)
            ax.annotate("ridge (3 launches)\n%.0f MB  %.1f TB/s\nleft of here, dispatch binds"
                        % (sw_x[i + 1] / 1e6, sw_y[i + 1]),
                        (sw_x[i + 1], sw_y[i + 1]), textcoords="offset points",
                        xytext=(14, -52), fontsize=7.6, color="#b42318", ha="left",
                        arrowprops=dict(arrowstyle="->", color="#b42318", lw=0.8))
            break

    # Annotate genuinely different shapes: nearest the roof and furthest below.
    near = max(pts, key=lambda r: r["of_roof"])
    far = min(pts, key=lambda r: r["of_roof"])
    for r, txt, dx, dy, ha in (
            (near, "closest to the roof\nM=%d N=%s  %.0f%% of roof"
             % (near["m"], _kn(near["n"]), 100 * near["of_roof"]), -14, 34, "right"),
            (far, "furthest below the roof\nM=%d N=%s  %.0f%% of roof"
             % (far["m"], _kn(far["n"]), 100 * far["of_roof"]), 10, -44, "left")):
        if (r["m"], r["n"]) == (near["m"], near["n"]) and r is far:
            continue
        ax.annotate(txt, (r["bytes"], r["tb_s"]), textcoords="offset points", xytext=(dx, dy),
                    fontsize=8, color="#333", ha=ha,
                    arrowprops=dict(arrowstyle="->", color="#777", lw=0.8))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("bytes the call must move  (row reads + candidates + indices)")
    ax.set_ylabel("effective bandwidth achieved  (TB/s)")
    ax.set_title("Roofline: where each of the %d shapes sits\n"
                 "left of the ridge every shape is dispatch-bound, so no amount of\n"
                 "bandwidth work can help there" % len(pts), fontsize=10.5, loc="left")
    ax.grid(True, which="both", ls=":", lw=0.5, color="#ccc")
    ax.set_xlim(3e5, 9e10)
    ax.set_ylim(0.02, 22)
    ax.legend(fontsize=7.6, loc="lower right", framealpha=0.95)

    # ---------------- panel 2: why the roof itself moves ----------------
    # Log y, and only the per-N envelope of the 130 points: plotting all of them
    # here buries the message, because most sit low for dispatch reasons that
    # have nothing to do with this axis.
    for reg, (c, mk, _) in REGIME_STYLE.items():
        sel = [r for r in pts if r["regime"] == reg]
        ax2.scatter([r["n"] * 4 for r in sel], [r["tb_s"] for r in sel], s=16, c=c, marker=mk,
                    alpha=0.30, edgecolors="none", zorder=4)
    env_x, env_y = [], []
    for n in grid.NS:
        sel = [r["tb_s"] for r in pts if r["n"] == n]
        if sel:
            env_x.append(n * 4)
            env_y.append(max(sel))
    ax2.plot(env_x, env_y, "-", color="#333", lw=1.8, marker="o", ms=5, zorder=6,
             label="best of the %d shapes at each N" % len(pts))
    ax2.plot([a["n"] * 4 for a in anchors],
             [grid.traffic_bytes(a["m"], a["n"], grid.TOPK)[0] / (a["median_ms"] * 1e-3) / 1e12
              for a in anchors], "*-", color="#6a1b9a", ms=16, lw=1.8, zorder=7,
             label="read+write floor kernel (the roof\nthis access pattern can reach)")
    for a in anchors:
        t, _ = grid.traffic_bytes(a["m"], a["n"], grid.TOPK)
        y = t / (a["median_ms"] * 1e-3) / 1e12
        ax2.annotate("%.2f TB/s" % y, (a["n"] * 4, y), textcoords="offset points",
                     xytext=(0, -20), fontsize=8, color="#6a1b9a", ha="center")
    ax2.axhline(max(sw_y), color="#b42318", lw=1.5, ls="--", zorder=3,
                label="peak READ bandwidth, %.2f TB/s (no writes)" % max(sw_y))
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylim(0.02, 22)
    ax2.set_xlabel("bytes ONE block streams contiguously  (N x 4)")
    ax2.set_ylabel("effective bandwidth achieved  (TB/s)")
    ax2.set_title("Why the roof is not one number\n"
                  "the achievable fraction of peak tracks how much contiguous data a\n"
                  "single block streams, which is why the floor model needs 3 anchors",
                  fontsize=10.5, loc="left")
    ax2.grid(True, which="both", ls=":", lw=0.5, color="#ccc")
    ax2.legend(fontsize=7.6, loc="lower right", framealpha=0.95)

    fig.suptitle("fp32 per-row top-k on MI355X / gfx950 — %d pow2 shapes, K=2048"
                 % len(pts), fontsize=13, x=0.012, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    png, svg = args.out + ".png", args.out + ".svg"
    fig.savefig(png, dpi=155)
    fig.savefig(svg)
    print("WROTE %s" % png)
    print("WROTE %s" % svg)

    dispatch_bound = [r for r in pts if r["roof_tb_s"] < interp_loglog(sw_x, sw_y, r["bytes"])]
    print("\ndispatch-bound shapes (dispatch roof below the bandwidth roof): %d / %d"
          % (len(dispatch_bound), len(pts)))
    print("fraction of the attainable roof reached:")
    for reg in ("small_n", "decode", "prefill"):
        sel = [r["of_roof"] for r in pts if r["regime"] == reg]
        if sel:
            print("  %-8s n=%-3d  median %5.1f%%   best %5.1f%%   worst %5.1f%%"
                  % (reg, len(sel), 100 * sorted(sel)[len(sel) // 2], 100 * max(sel),
                     100 * min(sel)))
    allf = [r["of_roof"] for r in pts]
    print("  %-8s n=%-3d  median %5.1f%%" % ("ALL", len(allf), 100 * sorted(allf)[len(allf) // 2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
