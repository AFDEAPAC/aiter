#!/usr/bin/env python3
"""Measure the whole pow2 grid and emit a self-contained HTML report.

  python3 scripts/make_html_report.py [-o reports/grid_report.html] [--from-baseline]

--from-baseline reuses knowledge/grid_baseline_outer.json instead of re-measuring,
which is faster but reports whatever that file last recorded.

The output has no external dependencies (inline CSS/JS) so it can be opened over
a forwarded port or emailed as one file.
"""

import argparse
import base64
import datetime
import json
import math
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import grid  # noqa: E402

ROOT = grid.ROOT


def shape_params(m, n, k):
    """The parameters the dispatcher actually derives for this shape."""
    o = subprocess.run([str(grid.BENCH), "--mode", "time", "--m", str(m), "--n", str(n),
                        "--topk", str(k), "--warmup", "1", "--iters", "2", "--repeats", "1"],
                       cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True).stdout
    g = lambda pat: (re.search(pat, o).group(1) if re.search(pat, o) else None)
    return {"path": g(r"path=(\w+)"), "S": g(r"S=(\d+)"), "margin": g(r"margin=([0-9.]+)"),
            "cap": g(r"cap=(\d+)"), "coop_g": g(r"coop_g=(\d+)")}


def measure_all(model):
    scale = grid.achievable_scale(model)
    pts = []
    shapes = grid.all_shapes()
    for i, (m, n, k) in enumerate(shapes):
        us, sd, path, argv = grid.time_shape_auto(m, n, k)
        f = grid.floor_us(m, n, k, model, scale)
        p = shape_params(m, n, k)
        pts.append({"m": m, "n": n, "topk": k, "regime": grid.regime_of(m, n), "path": path,
                    "us": round(us, 2), "stddev_pct": sd, "floor_us": round(f, 2),
                    "ratio": round(us / f, 3), "argv": list(argv),
                    "S": p["S"], "margin": p["margin"], "cap": p["cap"], "coop_g": p["coop_g"]})
        print("  [%3d/%d] M=%-5d N=%-8d %8.2f us  %5.2fx  %s"
              % (i + 1, len(shapes), m, n, us, us / f, path), file=sys.stderr)
    return pts


def summarize(pts):
    pp = grid.per_regime_geomean(pts)
    counts = {}
    for r in pts:
        counts[r["regime"]] = counts.get(r["regime"], 0) + 1
    am, an, ak = grid.ANCHOR
    anchor = next((r for r in pts if r["m"] == am and r["n"] == an), None)
    return {
        "n_points": len(pts),
        "geomean_us": round(grid.geomean([r["us"] for r in pts]), 2),
        "geomean_ratio": round(grid.geomean([r["ratio"] for r in pts]), 3),
        "per_regime": {k: round(v, 2) for k, v in sorted(pp.items())},
        "regime_counts": counts,
        "anchor_us": anchor["us"] if anchor else None,
        "worst": max(pts, key=lambda r: r["ratio"]),
        "best": min(pts, key=lambda r: r["ratio"]),
        "max_stddev_pct": max(r["stddev_pct"] for r in pts),
    }


def gpu_name():
    o = subprocess.run([str(grid.BENCH), "--mode", "time", "--m", "1", "--n", "2048",
                        "--topk", "2048", "--warmup", "1", "--iters", "2", "--repeats", "1"],
                       cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True).stdout
    mm = re.search(r"GPU: (.+?) CUs=(\d+)", o)
    return (mm.group(1), int(mm.group(2))) if mm else ("unknown", 0)


def ratio_color(r):
    """1.0 (at floor) -> green, 5.0+ -> red."""
    t = max(0.0, min(1.0, (r - 1.0) / 4.0))
    h = (1.0 - t) * 130.0          # 130deg green -> 0deg red
    return "hsl(%.0f, 62%%, %.0f%%)" % (h, 30 + 14 * (1 - t))


def fmt_n(n):
    return "%dK" % (n // 1024) if n >= 1024 else str(n)


def matrix_html(pts, value_key, label, fmt, color_by_ratio=True):
    by = {(r["m"], r["n"]): r for r in pts}
    out = ['<table class="mx"><caption>%s</caption><thead><tr><th>M \\ N</th>' % label]
    for n in grid.NS:
        out.append("<th>%s</th>" % fmt_n(n))
    out.append("</tr></thead><tbody>")
    for m in grid.MS:
        out.append("<tr><th>%d</th>" % m)
        for n in grid.NS:
            r = by.get((m, n))
            if not r:
                out.append('<td class="na">n/a</td>')
                continue
            style = ' style="background:%s"' % ratio_color(r["ratio"]) if color_by_ratio else ""
            out.append('<td%s title="M=%d N=%d  %.2f us  floor %.2f us  %.2fx  %s">%s</td>'
                       % (style, m, n, r["us"], r["floor_us"], r["ratio"], r["path"],
                          fmt % r[value_key]))
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def path_matrix_html(pts):
    by = {(r["m"], r["n"]): r for r in pts}
    cls = {"small_n": "p-sn", "decode": "p-dec", "prefill": "p-pre"}
    out = ['<table class="mx"><caption>Dispatch path actually taken</caption>'
           '<thead><tr><th>M \\ N</th>']
    for n in grid.NS:
        out.append("<th>%s</th>" % fmt_n(n))
    out.append("</tr></thead><tbody>")
    for m in grid.MS:
        out.append("<tr><th>%d</th>" % m)
        for n in grid.NS:
            r = by.get((m, n))
            if not r:
                out.append('<td class="na">n/a</td>')
                continue
            extra = ""
            if r["path"] != "small_n" and r.get("coop_g"):
                extra = "<br><span class=sub>G=%s S=%s cap=%s</span>" % (
                    r["coop_g"], r["S"], r["cap"])
            out.append('<td class="%s">%s%s</td>' % (cls.get(r["path"], ""), r["path"], extra))
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def rows_html(pts):
    out = []
    for r in sorted(pts, key=lambda x: -x["ratio"]):
        out.append(
            "<tr><td>%d</td><td>%d</td><td>%d</td><td>%s</td><td>%s</td>"
            "<td class=num>%.2f</td><td class=num>%.2f</td>"
            '<td class=num style="background:%s">%.2f</td>'
            "<td class=num>%.2f</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
            "<td class=num>%d/%d/%d</td></tr>"
            % (r["m"], r["n"], r["topk"], r["regime"], r["path"], r["us"], r["floor_us"],
               ratio_color(r["ratio"]), r["ratio"], r["stddev_pct"],
               r["S"] or "-", r["margin"] or "-", r["cap"] or "-", r["coop_g"] or "-",
               r["argv"][0], r["argv"][1], r["argv"][2]))
    return "".join(out)


CSS = """
:root{--fg:#1b1f23;--mut:#5b6572;--line:#d8dee4;--bg:#fff;--head:#f4f6f8}
*{box-sizing:border-box}
body{font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
 color:var(--fg);background:var(--bg);margin:0;padding:28px 34px 60px}
h1{font-size:23px;margin:0 0 4px}h2{font-size:17px;margin:34px 0 10px;padding-bottom:5px;
 border-bottom:1px solid var(--line)}h3{font-size:14px;margin:22px 0 8px}
.meta{color:var(--mut);font-size:12.5px;margin-bottom:22px}
.cards{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0 6px}
.card{border:1px solid var(--line);border-radius:7px;padding:10px 14px;min-width:132px}
.card .k{color:var(--mut);font-size:11.5px;text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:19px;font-weight:600;margin-top:3px}
.card .s{color:var(--mut);font-size:11.5px}
table{border-collapse:collapse;font-size:12.5px}
caption{text-align:left;font-weight:600;padding:8px 0;font-size:13px}
th,td{border:1px solid var(--line);padding:4px 7px;text-align:left}
thead th{background:var(--head);font-weight:600;position:sticky;top:0}
.mx td{text-align:right;font-variant-numeric:tabular-nums;color:#fff;font-weight:500}
.mx th{background:var(--head);text-align:right;font-variant-numeric:tabular-nums}
.mx td.na{background:#f0f2f4;color:var(--mut);font-weight:400}
.mx td.p-sn{background:#e3f0e6;color:var(--fg);font-weight:400;text-align:left}
.mx td.p-dec{background:#e4ecf7;color:var(--fg);font-weight:400;text-align:left}
.mx td.p-pre{background:#f7eee2;color:var(--fg);font-weight:400;text-align:left}
.sub{color:var(--mut);font-size:10.5px}
td.num{text-align:right;font-variant-numeric:tabular-nums}
#all th{cursor:pointer;user-select:none}#all th:hover{background:#e9edf1}
.note{color:var(--mut);font-size:12.5px;max-width:105ch}
.legend{display:flex;gap:3px;align-items:center;margin:6px 0 2px;font-size:11.5px;color:var(--mut)}
.legend i{width:26px;height:11px;display:inline-block}
.ok{color:#1a7f37;font-weight:600}.bad{color:#b42318;font-weight:600}
code{background:#f4f6f8;padding:1px 4px;border-radius:3px;font-size:12px}
.wrap{overflow:auto;max-height:78vh;border:1px solid var(--line);border-radius:7px}
.wrap table{border:0}
"""

JS = """
document.querySelectorAll('#all th').forEach(function(th,i){
  th.addEventListener('click',function(){
    var tb=th.closest('table').tBodies[0];
    var rows=Array.prototype.slice.call(tb.rows);
    var asc=th.dataset.asc!=='1';
    th.dataset.asc=asc?'1':'0';
    rows.sort(function(a,b){
      var x=a.cells[i].textContent.trim(), y=b.cells[i].textContent.trim();
      var nx=parseFloat(x), ny=parseFloat(y);
      if(!isNaN(nx)&&!isNaN(ny)) return asc?nx-ny:ny-nx;
      return asc?x.localeCompare(y):y.localeCompare(x);
    });
    rows.forEach(function(r){tb.appendChild(r);});
  });
});
"""


def floor_model_notes(model, pts):
    """The measured constants, and an honest statement about sub-1.0x points."""
    bwm = model["bw_model"]
    per = model["empty_launch"].get("per_launch_us", 0.0)
    consts = ("Peak read %.2f TB/s, per-launch %.2f us, small-payload latency floor %.1f us."
              % (bwm.get("peak_tb_s", 0.0), per, bwm.get("latency_us", 0.0)))
    sc = grid._anchor_scales(model)
    anchors = ", ".join("N=%s &rarr; %.3f" % (fmt_n(n), s) for n, s in sc)
    below = sorted([r for r in pts if r["ratio"] < 1.0], key=lambda r: r["ratio"])
    if not below:
        tail = ("no point reads below 1.00x. A point that did would mean the model is wrong, "
                "not that the kernel beat its floor.")
    else:
        lst = ", ".join("M=%d N=%s at %.2fx" % (r["m"], fmt_n(r["n"]), r["ratio"])
                        for r in below)
        tail = ("<b>%d point%s reads slightly below 1.00x (%s).</b> That is the model's residual "
                "error, not a kernel beating its own floor: read those as <i>at</i> the floor. "
                "Everything else is an upper bound on remaining headroom."
                % (len(below), "" if len(below) == 1 else "s", lst))
    return consts, anchors, tail


def roofline_section(pts, model, png):
    """Embeds the roofline as base64 so the report stays one file."""
    if not png.exists():
        return ('<h2>Roofline</h2><div class="note">Not generated. Run '
                '<code>python3 scripts/make_roofline.py</code>.</div>')
    b64 = base64.b64encode(png.read_bytes()).decode("ascii")

    per_launch = model["empty_launch"]["per_launch_us"]
    sweep = sorted(model["bw_sweep"], key=lambda p: p["bytes"])
    sx = [p["bytes"] for p in sweep]
    sy = [p["bandwidth_tb_s"] for p in sweep]

    def interp(x):
        if x <= sx[0]:
            return sy[0]
        if x >= sx[-1]:
            return sy[-1]
        for i in range(len(sx) - 1):
            if sx[i] <= x <= sx[i + 1]:
                w = ((math.log(x) - math.log(sx[i]))
                     / (math.log(sx[i + 1]) - math.log(sx[i])))
                return math.exp(math.log(sy[i]) + w * (math.log(sy[i + 1]) - math.log(sy[i])))
        return sy[-1]

    rows, nd = [], 0
    by = {}
    for r in pts:
        t, nl = grid.traffic_bytes(r["m"], r["n"], r["topk"])
        bw = t / (r["us"] * 1e-6) / 1e12
        bwroof = interp(t)
        disproof = t / (nl * per_launch * 1e-6) / 1e12
        if disproof < bwroof:
            nd += 1
        by.setdefault(r["regime"], []).append(bw / min(bwroof, disproof))
    for reg in ("small_n", "decode", "prefill"):
        v = sorted(by.get(reg, []))
        if v:
            rows.append("<tr><td>%s</td><td class=num>%d</td><td class=num>%.0f%%</td>"
                        "<td class=num>%.0f%%</td><td class=num>%.0f%%</td></tr>"
                        % (reg, len(v), 100 * v[len(v) // 2], 100 * v[-1], 100 * v[0]))
    allv = sorted(x for v in by.values() for x in v)
    rows.append("<tr><td><b>all</b></td><td class=num><b>%d</b></td>"
                "<td class=num><b>%.0f%%</b></td><td class=num>%.0f%%</td>"
                "<td class=num>%.0f%%</td></tr>"
                % (len(allv), 100 * allv[len(allv) // 2], 100 * allv[-1], 100 * allv[0]))

    return """<h2>Roofline</h2>
<div class="note"><b>This is not a FLOP/byte roofline, and it should not be.</b> Top-k does
comparisons, not arithmetic: every fp32 shape moves 4 bytes per element regardless of M, N or K,
so on a classic roofline all %d points collapse onto one vertical line at the same arithmetic
intensity and the picture says nothing. The informative version for a pure-memory kernel keeps
the structure and changes the axes:
<table style="margin:10px 0">
<thead><tr><th>classic roofline</th><th>this plot</th></tr></thead><tbody>
<tr><td>y = attainable FLOP/s</td><td>y = effective bandwidth achieved</td></tr>
<tr><td>x = arithmetic intensity (FLOP/byte)</td><td>x = bytes the call must move</td></tr>
<tr><td>diagonal roof = peak BW &times; AI</td><td>diagonal roof = bytes / fixed dispatch cost</td></tr>
<tr><td>flat roof = peak compute</td><td>flat roof = achievable stream bandwidth</td></tr>
<tr><td>ridge point</td><td>ridge point: dispatch-bound &rarr; bandwidth-bound</td></tr>
</tbody></table>
Both ceilings are measured on this machine, not theoretical. The flat roof is the read-bandwidth
sweep from <code>scripts/floor_bench.hip</code>; the diagonal is <code>bytes / (launches &times;
%.2f us)</code> from the measured empty-kernel cost, drawn once for the 1-launch small_n path and
once for the 3-launch sampled paths. The stars are a read+write streaming kernel using the same
one-block-per-row access pattern as Phase B, so they show the roof this <i>pattern</i> can reach
rather than what a grid-stride read can.</div>

<img alt="roofline" style="width:100%%;max-width:1560px;border:1px solid var(--line);
border-radius:7px;margin:6px 0" src="data:image/png;base64,%s">

<h3>What it says</h3>
<div class="note"><b>%d of the %d shapes are dispatch-bound</b>, meaning their dispatch roof sits
below the bandwidth roof, so no amount of bandwidth work can move them &mdash; only removing
launches can. That is why the pipeline was cut from 6 (prefill) and 9 (decode) dispatches to 3,
and why a cooperative single-kernel version was rejected: a grid-wide barrier measures 7.4 us at
grid=64 and 101-196 us at the largest co-resident grid, against an 11 us target.
<br><br>Fraction of the attainable roof (the lower of the two ceilings) each regime reaches:</div>
<table><thead><tr><th>regime</th><th>points</th><th>median</th><th>best</th><th>worst</th></tr>
</thead><tbody>%s</tbody></table>
<div class="note" style="margin-top:8px">The right-hand panel is the evidence for the floor model
needing three anchors rather than one constant: the achievable fraction of peak tracks how much
contiguous data a <i>single block</i> streams (N&middot;4 bytes), rising from 4.87 TB/s at 64 KB
per block to 5.33 TB/s at 4 MB. Peak grid-stride read on the same card is 6.61 TB/s, so even a
perfect one-block-per-row kernel cannot reach it.</div>
""" % (len(pts), per_launch, b64, nd, len(pts), "".join(rows))


def build_html(pts, summ, gpu, cus, correctness, model):
    legend = "".join('<i style="background:%s"></i>' % ratio_color(v)
                     for v in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0))
    cards = [
        ("points measured", "%d" % summ["n_points"], "all of M x N, none skipped"),
        ("geomean", "%.2f us" % summ["geomean_us"], "over all %d points" % summ["n_points"]),
        ("geomean vs floor", "%.2fx" % summ["geomean_ratio"], "1.00x would be at the floor"),
        ("anchor M=4096 N=131072", "%.1f us" % summ["anchor_us"], "limit 620.0 us"),
        ("worst point", "%.2fx" % summ["worst"]["ratio"],
         "M=%d N=%d (%s)" % (summ["worst"]["m"], summ["worst"]["n"], summ["worst"]["regime"])),
        ("best point", "%.2fx" % summ["best"]["ratio"],
         "M=%d N=%d (%s)" % (summ["best"]["m"], summ["best"]["n"], summ["best"]["regime"])),
        ("max stddev", "%.2f%%" % summ["max_stddev_pct"], "reject threshold 2.0%"),
    ]
    for r, g in sorted(summ["per_regime"].items()):
        cards.append(("%s geomean" % r, "%.2f us" % g,
                      "%d points" % summ["regime_counts"][r]))

    card_html = "".join(
        '<div class="card"><div class="k">%s</div><div class="v">%s</div><div class="s">%s</div></div>'
        % c for c in cards)

    ok = correctness.get("passed") == correctness.get("total") and correctness.get("total")
    corr = ('<p class="note">Correctness is a hard gate and is judged before any timing here. '
            'Full grid x 5 distributions: <span class="%s">%s / %s verified, %s failed</span>. '
            'The <code>torch.topk</code> gates and <code>bench/gate_selftest.py</code> '
            '(which proves those gates can go red) ran in the ROCm PyTorch image: '
            '<span class="%s">%s</span>.</p>'
            % ("ok" if ok else "bad", correctness.get("passed", "?"),
               correctness.get("total", "?"), correctness.get("failed", "?"),
               "ok" if correctness.get("torch_ok") else "bad",
               "PASS" if correctness.get("torch_ok") else "NOT RUN"))

    consts, anchors, tail = floor_model_notes(model, pts)
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>fp32 per-row top-k: pow2 grid report</title><style>%s</style></head><body>
<h1>fp32 per-row top-k &mdash; full pow2 grid</h1>
<div class="meta">%s &middot; %d CUs &middot; ROCm 10.0 / gfx950 &middot; measured %s<br>
Contract: fp32 <code>[M,N]</code> &rarr; int32 <code>indices[M,K]</code>, K = 2048 fixed.
M and N are powers of two; K = 2048 already excludes N &lt; 2048 because the contract
requires K &le; N. Grid, floor model and measurement settings all come from
<code>bench/grid.py</code>; scoring contract in <code>.evo/config-v3.yaml</code>.</div>

<h2>Summary</h2>
<div class="cards">%s</div>
%s

<h2>Latency (us)</h2>
<div class="legend">at floor %s far from floor &nbsp;&mdash;&nbsp; cell colour is distance to
this shape's own floor, not absolute speed. Hover any cell for the floor and the path.</div>
%s

<h2>Distance to floor (x)</h2>
<div class="note">The floor is regime-aware:
<code>max(traffic/peak_bw x achievable_scale(N), latency, launches x per_launch)</code>. Both the
traffic and the launch count depend on the path a shape takes &mdash; small_n moves no candidates
and launches one kernel, the sampled paths move margin&middot;K&middot;8 B per row and launch
three. %s
<br><br><b>Accuracy of this column: about &plusmn;3%%.</b> <code>achievable_scale</code> is
measured with a read+write streaming kernel at three values of N and interpolated in log2(N),
because the fraction of peak a shape reaches depends on how much contiguous data ONE block
streams (N&middot;4 bytes): %s. A single anchor at N=131072 over-stated the floor at N=1048576
badly enough to produce 0.94x. It still depends weakly on the grid size, which the model does
not capture, so %s
</div>
%s

%s

<h2>Dispatch and derived parameters</h2>
<div class="note">Every parameter below is derived from the shape at run time; none is passed in.
<code>S</code> is the Phase A sample count, <code>cap</code> the Phase C candidate capacity,
<code>G</code> the number of blocks cooperating on one row.</div>
%s

<h2>All %d points</h2>
<div class="note">Sorted by distance to floor, worst first. Click any header to re-sort.
<code>w/i/r</code> is warmup / iters / repeats: shapes under 150 us use 100/500/7 because at
20/100/5 they measure 2.7-3.1%% stddev, which is above the 2%% reject threshold and would make a
2%% change indistinguishable from noise.</div>
<div class="wrap"><table id="all"><thead><tr>
<th>M</th><th>N</th><th>K</th><th>regime</th><th>path</th>
<th>us</th><th>floor us</th><th>x floor</th><th>stddev %%</th>
<th>S</th><th>margin</th><th>cap</th><th>G</th><th>w/i/r</th>
</tr></thead><tbody>%s</tbody></table></div>

<h2>How to reproduce</h2>
<pre class="note"><code>make benchmark_topk
python3 bench/verify_grid.py --dist all      # 650 verifies, correctness hard gate
python3 bench/score_grid.py --tier outer     # these 130 numbers
python3 scripts/make_html_report.py          # measures the grid, writes this file + .json
python3 scripts/make_roofline.py             # roofline png/svg from that .json
python3 scripts/make_html_report.py --from-baseline   # re-render without re-measuring</code></pre>
<div class="note">Falsified directions, each with the measurement that killed it, are in
<code>knowledge/known_bad.md</code>. Accepted lineage is in
<code>log/grid_evolution.tsv</code>.</div>
<script>%s</script></body></html>
""" % (CSS, gpu, cus, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
       card_html, corr, legend,
       matrix_html(pts, "us", "Wall time per call (us), coloured by distance to floor", "%.1f"),
       consts, anchors, tail,
       matrix_html(pts, "ratio", "Distance to this shape's own floor", "%.2f"),
       roofline_section(pts, model, ROOT / "reports" / "grid_roofline.png"),
       path_matrix_html(pts), summ["n_points"], rows_html(pts), JS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=str(ROOT / "reports" / "grid_report.html"))
    ap.add_argument("--from-baseline", action="store_true")
    ap.add_argument("--from-json", default=None,
                    help="re-render from a previous run's report JSON, which keeps the "
                         "per-shape derived parameters that --from-baseline does not have")
    ap.add_argument("--correctness-json", default=None,
                    help="JSON with passed/total/failed/torch_ok; omit to report as not run")
    args = ap.parse_args()

    if not grid.BENCH.exists():
        print("ERROR: missing %s; run make first" % grid.BENCH, file=sys.stderr)
        return 2

    model = grid.load_model()
    if args.from_json:
        pts = json.loads(Path(args.from_json).read_text())["points"]
        print("re-rendering from %s (%d points)" % (args.from_json, len(pts)), file=sys.stderr)
    elif args.from_baseline:
        src = ROOT / "knowledge" / "grid_baseline_outer.json"
        pts = json.loads(src.read_text())["points"]
        for r in pts:
            r.setdefault("regime", grid.regime_of(r["m"], r["n"]))
            for k in ("S", "margin", "cap", "coop_g"):
                r.setdefault(k, None)
        print("using %s (%d points)" % (src, len(pts)), file=sys.stderr)
    else:
        print("measuring %d points..." % len(grid.all_shapes()), file=sys.stderr)
        pts = measure_all(model)

    summ = summarize(pts)
    gpu, cus = gpu_name()
    corr = {"passed": None, "total": None, "failed": None, "torch_ok": False}
    if args.correctness_json:
        corr.update(json.loads(Path(args.correctness_json).read_text()))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(pts, summ, gpu, cus, corr, model))
    json.dump({"summary": summ, "points": pts},
              open(str(out.with_suffix(".json")), "w"), indent=1, default=str)
    print("\nWROTE %s" % out)
    print("WROTE %s" % out.with_suffix(".json"))
    print("geomean %.2f us over %d points, worst %.2fx at M=%d N=%d"
          % (summ["geomean_us"], summ["n_points"], summ["worst"]["ratio"],
             summ["worst"]["m"], summ["worst"]["n"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
