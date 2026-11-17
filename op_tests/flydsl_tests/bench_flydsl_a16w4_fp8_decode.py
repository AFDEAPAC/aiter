# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Driver: gfx942 a16w4 fp8-mode decode variants, PR #7153 default vs each variant.

Runs ``op_tests/test_flydsl_moe_a16w4_fp8_decode.py`` once per variant, each in its own
process with ``AITER_FLYDSL_A16W4_FP8_DECODE=<variant>`` and a freshly emptied
``FLYDSL_RUNTIME_CACHE_DIR=<out>/fcache/<variant>``. ``base`` runs first and saves its
outputs; every other variant is compared against them.

With ``--rocprof`` each run is wrapped in ``rocprofv3 --kernel-trace``; per shape the
gemm1 and gemm2 metric is the median kernel-trace duration over the timed ``fused_moe``
calls (each preceded by a 1 GiB overwrite), sliced with the launch manifest. The
``fused_moe`` CUDA-event median (sorting + two quant kernels + both GEMMs) is reported
alongside. Speedup columns are base / variant.

Run (MI300X/MI325X, one idle GPU):
    python op_tests/flydsl_tests/bench_flydsl_a16w4_fp8_decode.py --rocprof \\
        --out /tmp/a16w4_fp8_decode
    # forward op-test args after --, e.g. a quick subset:
    python op_tests/flydsl_tests/bench_flydsl_a16w4_fp8_decode.py --out /tmp/q \\
        -v base aperm -- -t 1 8 256 -i 384 --iters 30
    # re-parse an existing --out without rerunning:
    python op_tests/flydsl_tests/bench_flydsl_a16w4_fp8_decode.py --out /tmp/q --parse-only
"""

import argparse
import csv
import glob
import json
import os
import shutil
import statistics
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OP_TEST = os.path.join(REPO, "op_tests", "test_flydsl_moe_a16w4_fp8_decode.py")
ENV = "AITER_FLYDSL_A16W4_FP8_DECODE"
VARIANTS = ["base", "aperm", "tab", "aperm+tab", "noslp", "aperm+tab+noslp"]
# fp8-mode kernel names: gemm1_a16w4_port_a16w4_afp8_..., gemm2_a16w4_afp8_port_...
KERNEL_KEY = {1: "gemm1_a16w4_port_a16w4_afp8", 2: "gemm2_a16w4_afp8_port"}


def run_variant(out, variant, rocprof, extra):
    cache = os.path.join(out, "fcache", variant)
    shutil.rmtree(cache, ignore_errors=True)
    os.makedirs(cache)
    os.makedirs(os.path.join(out, "results"), exist_ok=True)
    env = dict(os.environ, FLYDSL_RUNTIME_CACHE_DIR=cache)
    env[ENV] = "" if variant == "base" else variant
    base_dir = os.path.join(out, "outputs", "base")
    cmd = [
        sys.executable,
        OP_TEST,
        "--json",
        os.path.join(out, "results", f"{variant}.json"),
    ]
    cmd += (
        ["--save-dir", base_dir] if variant == "base" else ["--compare-dir", base_dir]
    )
    cmd += extra
    if rocprof:
        prof = os.path.join(out, "prof", variant)
        shutil.rmtree(prof, ignore_errors=True)
        cmd = [
            "rocprofv3",
            "--kernel-trace",
            "-d",
            prof,
            "-o",
            "run",
            "--output-format",
            "csv",
            "--",
        ] + cmd
    print(
        f"[{variant}] {ENV}={env[ENV]!r} FLYDSL_RUNTIME_CACHE_DIR={cache}", flush=True
    )
    print("  " + " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)


def trace_medians(prof_dir, manifest):
    """{(inter_dim, token, gemm): median timed kernel duration in us} from rocprofv3."""
    files = glob.glob(os.path.join(prof_dir, "**", "*kernel_trace.csv"), recursive=True)
    if not files:
        return {}
    disp = {1: [], 2: []}
    for path in files:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                for g, key in KERNEL_KEY.items():
                    if key in row["Kernel_Name"]:
                        disp[g].append(
                            (int(row["Start_Timestamp"]), int(row["End_Timestamp"]))
                        )
    res = {}
    for g in (1, 2):
        seq = sorted(disp[g])
        expected = sum(m["count"] for m in manifest)
        if len(seq) != expected:
            raise RuntimeError(
                f"{prof_dir}: {len(seq)} fp8 gemm{g} dispatches in the trace, manifest "
                f"expects {expected} (a stage fell back to bf16, or the names changed)"
            )
        i = 0
        for m in manifest:
            chunk = seq[i : i + m["count"]]
            i += m["count"]
            if m["phase"] == "timed":
                res[(m["inter_dim"], m["token"], g)] = statistics.median(
                    (e - s) / 1e3 for s, e in chunk
                )
    return res


_CHECKS = [
    ("gemm1 bitwise vs base", "g1 bitwise"),
    ("gemm1 max abs vs base", "g1 max abs"),
    ("out bitwise vs base (top1)", "top1 bitwise"),
    ("out max abs vs base (top1)", "top1 max abs"),
    ("out max abs vs base (top16)", "top16 max abs"),
    ("cos-diff vs fp8-aware ref", "cos-diff"),
]


def summarize(out, variants):
    import pandas as pd

    perf, checks, tiles = {}, [], {}
    for v in variants:
        path = os.path.join(out, "results", f"{v}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            res = json.load(f)
        trace = trace_medians(os.path.join(out, "prof", v), res["manifest"])
        for row in res["rows"]:
            key = (row["inter_dim"], row["token"])
            tiles[key] = (row["gemm1 tile"], row["gemm2 tile"])
            t = perf.setdefault(key, {"inter_dim": key[0], "tokens": key[1]})
            t[f"{v} fused us"] = row["fused_moe us"]
            for g in (1, 2):
                if (key[0], key[1], g) in trace:
                    t[f"{v} g{g} us"] = trace[(key[0], key[1], g)]
            c = {"variant": v, "inter_dim": key[0], "tokens": key[1]}
            c.update({short: row.get(long) for long, short in _CHECKS})
            checks.append(c)
    for t in perf.values():
        for v in variants[1:]:
            for m in ("fused", "g1", "g2"):
                b, x = t.get(f"base {m} us"), t.get(f"{v} {m} us")
                if b and x:
                    t[f"{v} {m} x"] = b / x
        t["tiles (g1 | g2)"] = " | ".join(tiles[(t["inter_dim"], t["tokens"])])
    df = pd.DataFrame(
        sorted(perf.values(), key=lambda r: (r["inter_dim"], r["tokens"]))
    )
    dc = pd.DataFrame(checks)
    md = (
        df.to_markdown(index=False, floatfmt=".3f")
        + "\n\n"
        + dc.to_markdown(index=False)
    )
    print(md)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(md + "\n")
    df.to_csv(os.path.join(out, "summary.csv"), index=False)
    dc.to_csv(os.path.join(out, "checks.csv"), index=False)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--out", default="/tmp/a16w4_fp8_decode")
    p.add_argument("-v", "--variants", nargs="*", default=VARIANTS)
    p.add_argument(
        "--rocprof", action="store_true", help="wrap runs in rocprofv3 --kernel-trace"
    )
    p.add_argument("--parse-only", action="store_true")
    args, extra = p.parse_known_args()
    extra = [a for a in extra if a != "--"]
    variants = ["base"] + [v for v in args.variants if v != "base"]
    if not args.parse_only:
        for v in variants:
            run_variant(args.out, v, args.rocprof, extra)
    summarize(args.out, variants)


if __name__ == "__main__":
    main()
