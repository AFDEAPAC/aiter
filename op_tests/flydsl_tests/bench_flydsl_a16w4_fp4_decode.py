# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Driver: gfx942 a16w4 MXFP4 decode variants, baseline vs each variant vs combined.

Runs ``op_tests/test_flydsl_moe_a16w4_fp4_decode.py`` once per variant, each in its
own process with ``AITER_FLYDSL_A16W4_FP4_DECODE=<variant>`` and a freshly emptied
``FLYDSL_RUNTIME_CACHE_DIR=<out>/fcache/<variant>`` (FlyDSL's cache key ignores the
decode helper bodies, so a shared cache could serve another variant's kernel).
``base`` runs first and saves its outputs; every other variant is checked against them
(gemm1 bitwise, gemm2 within --max-ulp bf16 ulps).

With ``--rocprof`` each run is wrapped in ``rocprofv3 --kernel-trace``; the per-shape
metric is then the median kernel-trace duration of the timed launches (each preceded
by a 1 GiB cache flush), sliced with the launch manifest the op test writes. CUDA-event
medians are reported alongside.

Run (MI300X/MI325X, one idle GPU):
    python op_tests/flydsl_tests/bench_flydsl_a16w4_fp4_decode.py --rocprof \\
        --out /tmp/a16w4_fp4_decode
    # forward op-test args after --, e.g. a quick subset:
    python op_tests/flydsl_tests/bench_flydsl_a16w4_fp4_decode.py --out /tmp/q -- \\
        -t 1 256 -i 512 --iters 50
    # re-parse an existing --out without rerunning:
    python op_tests/flydsl_tests/bench_flydsl_a16w4_fp4_decode.py --out /tmp/q --parse-only
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
OP_TEST = os.path.join(REPO, "op_tests", "test_flydsl_moe_a16w4_fp4_decode.py")
ENV = "AITER_FLYDSL_A16W4_FP4_DECODE"
VARIANTS = ["base", "sign", "perm", "pkmul", "all"]
KERNEL_PREFIX = {1: "gemm1_a16w4_port_", 2: "gemm2_a16w4_port_"}


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
                for g, prefix in KERNEL_PREFIX.items():
                    if prefix in row["Kernel_Name"]:
                        disp[g].append(
                            (int(row["Start_Timestamp"]), int(row["End_Timestamp"]))
                        )
    res = {}
    for g in (1, 2):
        seq = sorted(disp[g])
        segs = [m for m in manifest if m["gemm"] == g]
        expected = sum(m["count"] for m in segs)
        if len(seq) != expected:
            raise RuntimeError(
                f"{prof_dir}: {len(seq)} gemm{g} dispatches in the trace, manifest "
                f"expects {expected}"
            )
        i = 0
        for m in segs:
            chunk = seq[i : i + m["count"]]
            i += m["count"]
            if m["phase"] == "timed":
                us = [(e - s) / 1e3 for s, e in chunk]
                res[(m["inter_dim"], m["token"], g)] = statistics.median(us)
    return res


def summarize(out, variants):
    import pandas as pd

    table = {}
    for v in variants:
        path = os.path.join(out, "results", f"{v}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            res = json.load(f)
        trace = trace_medians(os.path.join(out, "prof", v), res["manifest"])
        for row in res["rows"]:
            for g in (1, 2):
                if f"gemm{g} us" not in row:
                    continue
                key = (row["inter_dim"], row["token"], g)
                t = table.setdefault(
                    key, {"inter_dim": key[0], "tokens": key[1], "gemm": g}
                )
                if key in trace:
                    t[f"{v} trace us"] = trace[key]
                t[f"{v} event us"] = row[f"gemm{g} us"]
                check = (
                    row.get("gemm1 bitwise vs base")
                    if g == 1
                    else row.get("gemm2 max ulp vs base")
                )
                if check is not None:
                    t[f"{v} {'bitwise' if g == 1 else 'max ulp'}"] = check
    df = pd.DataFrame(
        sorted(table.values(), key=lambda r: (r["gemm"], -r["inter_dim"], r["tokens"]))
    )
    md = df.to_markdown(index=False)
    print(md)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(md + "\n")
    df.to_csv(os.path.join(out, "summary.csv"), index=False)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--out", default="/tmp/a16w4_fp4_decode")
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
