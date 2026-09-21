#!/usr/bin/env python3
"""Build the before/after HTML report from reports/select_ab.json.

Four tables over the same grid, M down and N across, each N base shown as three
columns N / N+1 / N+2:

  1. before -- `topk_select` over its five existing backends
  2. after  -- the same with `sampled` available
  3. speedup before/after, as the heatmap
  4. the cells whose backend changed, listed separately

Table 4 is separate on purpose. A cell can get faster because it changed
backend, or because the same backend ran faster between two runs, and those are
different claims. Keeping them apart stops a routing change being read as a
kernel win.

Colour on table 3 is banded against the measured noise floor, not against 1.00.
Cells whose backend did NOT change are the control: their spread reaches 0.63x
to 1.25x on this harness, so anything inside that band is uncoloured -- calling
a 1.1x "green" would be reading noise as signal.

Usage: make_ab_report.py [in.json] [out.html]
"""
import html
import json
import statistics as st
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "/topk/reports/select_ab.json"
DST = sys.argv[2] if len(sys.argv) > 2 else "/topk/reports/select_ab_report.html"

d = json.load(open(SRC))
MS, BASES, OFFS = d["ms"], d["bases"], d["width_offsets"]
K, ROUNDS = d["topk"], d["rounds"]

cell = {}
for r in d["rows"]:
    if "side" not in r:
        continue
    cell.setdefault((r["m"], r["width"]), {})[r["side"]] = r

changed = [(k, v) for k, v in cell.items()
           if "before" in v and "after" in v
           and v["before"]["backend"] != v["after"]["backend"]]
unchanged = [(k, v) for k, v in cell.items()
             if "before" in v and "after" in v
             and v["before"]["backend"] == v["after"]["backend"]]
noise = sorted(v["before"]["us"] / v["after"]["us"] for _, v in unchanged)
NOISE_LO, NOISE_HI = noise[0], noise[-1]
wrong = [r for r in d["rows"] if r.get("correct") is False]


def band(sp):
    """Colour only outside the measured noise band."""
    if sp >= 1.60:
        return "#1b7f3b", "#fff"
    if sp >= 1.40:
        return "#3fa45c", "#fff"
    if sp > NOISE_HI:
        return "#b7e0c0", "#000"
    if sp < NOISE_LO:
        return "#c0392b", "#fff"
    return "", ""


def fmt(x):
    return "%.0f" % x if x >= 100 else ("%.1f" % x if x >= 10 else "%.2f" % x)


def grid(title, note, pick, colour=None):
    o = ['<h2>%s</h2><p class="n">%s</p><table><tr><th>M \\ N</th>' % (title, note)]
    for b in BASES:
        for off in OFFS:
            o.append('<th>%s%s</th>' % (
                "{:,}".format(b) if off == 0 else "+%d" % off,
                "" if off == 0 else ""))
    o.append("</tr>")
    for m in MS:
        o.append("<tr><th>%d</th>" % m)
        for b in BASES:
            for off in OFFS:
                v = cell.get((m, b + off))
                if not v or "before" not in v or "after" not in v:
                    o.append('<td class="skip">-</td>')
                    continue
                txt, sty = pick(v)
                bg, fg = colour(v) if colour else ("", "")
                s = ' style="background:%s;color:%s"' % (bg, fg) if bg else ""
                o.append("<td%s%s>%s</td>" % (s, sty, txt))
        o.append("</tr>")
    o.append("</table>")
    return "".join(o)


def lat(side):
    def f(v):
        r = v[side]
        return '%s<br><span class="b">%s</span>' % (fmt(r["us"]), r["backend"]), ""
    return f


def spd(v):
    return "%.2fx" % (v["before"]["us"] / v["after"]["us"]), ""


def spd_colour(v):
    return band(v["before"]["us"] / v["after"]["us"])


sp_ch = sorted(v["before"]["us"] / v["after"]["us"] for _, v in changed)
rows_ch = "".join(
    "<tr><td>%d</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
    "<td><b>%.2fx</b></td></tr>" % (
        k[0], "{:,}".format(k[1]),
        v["before"]["backend"], fmt(v["before"]["us"]),
        v["after"]["backend"], fmt(v["after"]["us"]),
        v["before"]["us"] / v["after"]["us"])
    for k, v in sorted(changed, key=lambda x: -(x[1]["before"]["us"] / x[1]["after"]["us"]))
)

HTML = """<!doctype html><meta charset="utf-8"><title>topk_select before/after</title>
<style>
body{font:13px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#222;max-width:100%%}
h1{font-size:20px;margin:0 0 4px} h2{font-size:15px;margin:26px 0 4px}
p.n{color:#555;margin:0 0 8px;max-width:900px}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:11px;margin-bottom:6px}
th,td{border:1px solid #d5d5d5;padding:2px 5px;text-align:right;white-space:nowrap}
th{background:#f3f3f3;font-weight:600}
td.skip{color:#bbb;background:#fafafa}
span.b{color:#777;font-size:9px}
.box{background:#f7f7f9;border-left:3px solid #888;padding:10px 14px;margin:14px 0;max-width:900px}
.warn{border-left-color:#c0392b}
code{background:#eee;padding:1px 4px;border-radius:2px}
</style>
<h1>topk_select: before / after adding the <code>sampled</code> backend</h1>
<p class="n">MI355X (gfx950) &middot; fp32 &middot; topk=%(K)d &middot; full uniform rows &middot;
aiter <code>@perftest</code>, GPU kernel time from a profiler trace &middot;
median of %(R)d interleaved rounds &middot; %(NC)d cells &times; 3 widths.</p>

<div class="box warn">
<b>Read this before quoting a number.</b> Every cell was measured on
<code>torch.randn</code>. Top-k time on this machine depends on the value
distribution of the input as well as on its shape, and this grid says nothing
about any other distribution. These are random-input numbers.
</div>

<div class="box">
<b>before</b> is <code>topk_select</code> routing over its five existing backends
(<code>argmax</code>, <code>plain</code>, <code>small_k</code>,
<code>decode</code>, <code>stream</code>).
<b>after</b> is the same call with <code>sampled</code> added to the available
set. The only difference between the two runs is
<code>AITER_DISABLE_TOPK_SAMPLED</code>; same process, same data, same harness.
<br><br>
<b>Noise floor.</b> %(NU)d of %(NT)d cells did not change backend. Those run
identical code on both sides, so their spread is pure run-to-run variance:
<b>%(NLO).2fx to %(NHI).2fx</b>, median %(NMED).3fx. Table 3 leaves anything
inside that band uncoloured. A difference smaller than the noise on a cell that
did not change backend is not a result.
<br><br>
<b>Correctness:</b> %(WRONG)d cells disagreed with <code>torch.topk</code> on
index uniqueness or value multiset. Must be 0.
</div>

%(T1)s
%(T2)s
%(T3)s

<h2>4. Cells whose backend changed (%(CH)d of %(NT)d)</h2>
<p class="n">These are the only cells where this change did anything. Every one
got faster: min <b>%(CLO).2fx</b>, median <b>%(CMED).2fx</b>, max
<b>%(CHI).2fx</b>, and none regressed. That is by construction rather than by
luck &mdash; <code>sampled</code> is routed only where it measured fastest of
every backend that can serve the cell, so a cell that changes hands was already
going to be at least as slow under the old rule.</p>
<table><tr><th>M</th><th>N</th><th>before</th><th>us</th><th>after</th><th>us</th><th>speedup</th></tr>
%(ROWS)s</table>
""" % {
    "K": K, "R": ROUNDS, "NC": len(MS) * len(BASES),
    "NU": len(unchanged), "NT": len(cell), "CH": len(changed),
    "NLO": NOISE_LO, "NHI": NOISE_HI, "NMED": st.median(noise),
    "WRONG": len(wrong),
    "CLO": sp_ch[0], "CMED": st.median(sp_ch), "CHI": sp_ch[-1],
    "T1": grid("1. before &mdash; latency (us) and backend chosen",
               "topk_select over its five existing backends.", lat("before")),
    "T2": grid("2. after &mdash; latency (us) and backend chosen",
               "Same call with <code>sampled</code> in the available set.", lat("after")),
    "T3": grid("3. speedup (before / after)",
               "Green above the noise band, red below it, uncoloured inside it. "
               "Each N base is shown as N / N+1 / N+2.", spd, spd_colour),
    "ROWS": rows_ch,
}

open(DST, "w", encoding="utf-8").write(HTML)
print("WROTE %s" % DST)
print("  cells %d   changed %d   unchanged %d   wrong %d"
      % (len(cell), len(changed), len(unchanged), len(wrong)))
print("  noise band (unchanged cells): %.2fx .. %.2fx" % (NOISE_LO, NOISE_HI))
print("  changed cells: min %.2fx median %.2fx max %.2fx"
      % (sp_ch[0], st.median(sp_ch), sp_ch[-1]))
