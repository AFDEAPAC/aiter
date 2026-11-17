# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compile-only gfx942 ISA dump + VALU/VGPR count for the a16w4 fp8-mode decode variants.

No GPU needed (FlyDSL wheel + torch + ld.lld; ROCm optional). Every (variant, kernel)
is compiled in its own subprocess with ``COMPILE_ONLY=1``, ``FLYDSL_GPU_ARCH=gfx942``,
the disk cache disabled and a fresh ``FLYDSL_RUNTIME_CACHE_DIR``.

Kernels default to the Kimi-K3 (model_dim 3584, E=896, topk 16, SiTUv2) tiles that
``fused_moe`` runs in fp8 mode on a 304-CU gfx942: the tuned rows of
``kimik3_a16w4_tuned_fmoe.csv`` (fp8 mode reuses their tile names) plus the PR's fp8
fallback tiles. A spec is ``gemm{1,2}:BMxTNxTK:inter[:k_wave[:waves_per_eu[:b_nt]]]``.

Run:
    python op_tests/flydsl_tests/dump_flydsl_a16w4_fp8_decode_isa.py --out /tmp/fp8isa \\
        -v base aperm tab aperm+tab noslp aperm+tab+noslp
Each ISA lands in ``<out>/<variant>/<spec>/<kernel name>/22_final_isa.s``; rows go to
``<out>/isa-counts.jsonl`` and the table to ``<out>/isa-counts.md``.
"""

import argparse
import collections
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VARIANTS = ["base", "aperm", "tab", "aperm+tab", "noslp", "aperm+tab+noslp"]
KERNELS = [
    # tuned 304-CU rows: 1-4 tokens (k_wave 2/4), 8 tokens, 16-512, 1024+
    "gemm1:16x32x128:384:2:0:0",
    "gemm1:16x32x128:384:2:2:0",
    "gemm1:16x32x128:384:2:3:0",
    "gemm1:16x32x128:512:4:2:0",
    "gemm1:16x32x128:512:2:4:0",
    "gemm1:16x64x256:384:1:4:0",
    "gemm1:16x64x256:512:1:4:0",
    "gemm1:16x64x128:384:1:4:2",
    "gemm1:16x64x128:512:1:4:2",
    "gemm1:32x64x128:384:1:4:2",
    "gemm1:32x128x128:384:1:4:2",
    "gemm1:32x192x256:384:1:0:2",
    # PR #7153 fp8 fallback tiers (untuned shapes)
    "gemm1:16x32x256:384:2:0:2",
    "gemm1:16x128x256:384:1:0:2",
    "gemm1:32x128x256:384:1:0:0",
    "gemm2:16x256x128:384:1:0:2",
    "gemm2:16x128x128:384:1:0:2",
    "gemm2:32x128x128:384:1:0:2",
    "gemm2:16x256x128:512:1:0:0",
    "gemm2:16x128x256:512:1:0:2",
    "gemm2:32x128x256:512:1:0:0",
]
TRACKED = [
    "v_perm_b32",
    "v_and_or_b32",
    "v_mul_lo_u32",
    "v_med3_i32",
    "v_mov_b32",
    "v_mfma",
    "s_nop",
]

_CHILD = r"""
import os, sys, types
sys.path.insert(0, {repo!r})
try:
    import aiter  # noqa: F401
except Exception:
    for k in [k for k in sys.modules if k == "aiter" or k.startswith("aiter.")]:
        del sys.modules[k]
    for name, path in (("aiter", "aiter"), ("aiter.ops", "aiter/ops"),
                       ("aiter.ops.flydsl", "aiter/ops/flydsl")):
        m = types.ModuleType(name)
        m.__path__ = [os.path.join({repo!r}, path)]
        sys.modules[name] = m
from aiter.ops.flydsl.kernels.tensor_shim import _preload_compiled

kernel, bm, tn, tk, inter, kw, wpe, bnt = {spec!r}
extra = {{"fp8_decode": {variant!r}}} if {variant!r} else {{}}
if kernel == "gemm1":
    from aiter.ops.flydsl.kernels.moe_2stage_a16wmix.gemm1 import compile_gemm1_a16w4_port
    launch = compile_gemm1_a16w4_port(
        BM=bm, D_HIDDEN=3584, D_INTER=inter, NE=896, TOPK=16, TILE_N=tn, TILE_K=tk,
        act="situv2", b_cache_mod=2 if bnt < 0 else bnt, rocm_arch="gfx942",
        k_wave=kw, waves_per_eu=wpe or None, a_dtype="fp8", **extra)
    args = [0x1000] * 6 + [1, 1] + [1.0] * 5 + [0x1000, 0x1000, 0x1000, None]
else:
    from aiter.ops.flydsl.kernels.moe_2stage_a16wmix.gemm2 import compile_gemm2_a16w4_port
    launch = compile_gemm2_a16w4_port(
        BM=bm, NE=896, N_OUT=3584, D_INTER=inter, TILE_N=tn, TILE_K=tk, xcd_swizzle=1,
        b_cache_mod=0 if bnt < 0 else bnt, waves_per_eu=wpe or None, rocm_arch="gfx942",
        epilog="atomic", topk=16, a_dtype="fp8", **extra)
    args = [0x1000] * 7 + [1, 1, 1, 0x1000, 0x1000, 0x1000, None]
_preload_compiled(launch, *args)
"""


def parse_spec(s):
    parts = s.split(":")
    kernel, tile, inter = parts[0], parts[1], int(parts[2])
    kw, wpe, bnt = ([int(x) for x in parts[3:]] + [1, 0, -1][len(parts) - 3 :])[:3]
    bm, tn, tk = (int(x) for x in tile.split("x"))
    return kernel, bm, tn, tk, inter, kw, wpe, bnt


def kernel_body(text):
    ops, inside = [], False
    for line in text.splitlines():
        s = line.strip()
        if re.match(r"^[A-Za-z_][\w.]*:$", s) and not s.startswith("."):
            inside = True
            continue
        if not inside or not s or s.startswith((";", ".")) or s.endswith(":"):
            continue
        ops.append(s)
        if s.split()[0] == "s_endpgm":
            break
    return ops


def count_isa(path):
    with open(path) as f:
        text = f.read()
    lines = kernel_body(text)
    hist = collections.Counter(s.split()[0] for s in lines)
    row = {
        "VALU": sum(
            n for op, n in hist.items() if op.startswith("v_") and "mfma" not in op
        )
    }
    for t in TRACKED:
        row[t] = sum(n for op, n in hist.items() if op.startswith(t))
    row["scratch ops"] = sum(
        1 for s in lines if s.startswith(("scratch_", "buffer_")) and "s[0:3]" in s
    )
    for key in ("vgpr_count", "agpr_count", "sgpr_count", "vgpr_spill_count"):
        m = re.search(rf"\.{key}:\s+(\d+)", text)
        row[key] = int(m.group(1)) if m else None
    m = re.search(r"\.group_segment_fixed_size:\s+(\d+)", text)
    row["lds"] = int(m.group(1)) if m else 0
    # gfx942: 512 VGPR+AGPR per SIMD lane in granules of 8, 64 KiB LDS per CU, and a
    # 256-thread workgroup is one wave per SIMD.
    vg = ((row["vgpr_count"] or 0) + 7) // 8 * 8
    occ = min(8, 512 // max(vg, 8))
    if row["lds"]:
        occ = min(occ, 65536 // row["lds"])
    row["waves/SIMD"] = occ
    return row


def compile_one(out_dir, variant, spec_str):
    spec = parse_spec(spec_str)
    v = "" if variant == "base" else variant
    dump = os.path.join(out_dir, variant, spec_str.replace(":", "_"))
    with tempfile.TemporaryDirectory(prefix="flydsl_cache_") as cache:
        env = dict(
            os.environ,
            COMPILE_ONLY="1",
            FLYDSL_GPU_ARCH="gfx942",
            FLYDSL_DUMP_IR="1",
            FLYDSL_DUMP_DIR=dump,
            FLYDSL_RUNTIME_ENABLE_CACHE="0",
            FLYDSL_RUNTIME_CACHE_DIR=cache,
        )
        # FlyDSL's AST rewriter reads kernel source with inspect, so run from a file.
        child = os.path.join(cache, "compile_child.py")
        with open(child, "w") as f:
            f.write(_CHILD.format(repo=REPO, spec=spec, variant=v))
        res = subprocess.run(
            [sys.executable, child],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    if res.returncode != 0:
        raise RuntimeError(f"{spec_str}/{variant} failed:\n{res.stderr[-4000:]}")
    isas = [
        os.path.join(r, f)
        for r, _, fs in os.walk(dump)
        for f in fs
        if f == "22_final_isa.s"
    ]
    assert len(isas) == 1, isas
    row = {
        "spec": spec_str,
        "variant": variant,
        "kernel": os.path.basename(os.path.dirname(isas[0])),
    }
    row.update(count_isa(isas[0]))
    row["isa"] = isas[0]
    return row


def table(rows, variants):
    by = {(r["spec"], r["variant"]): r for r in rows}
    specs = list(dict.fromkeys(r["spec"] for r in rows))
    head = "| spec | MFMA | " + " | ".join(variants) + " |"
    out = [head, "|" + "---|" * (2 + len(variants))]
    for s in specs:
        b = by.get((s, "base"))
        cells = []
        for v in variants:
            r = by.get((s, v))
            if r is None:
                cells.append("-")
                continue
            d = f" ({(r['VALU'] - b['VALU']) / b['VALU'] * 100:+.1f}%)" if b else ""
            sp = f", spill {r['vgpr_spill_count']}" if r["vgpr_spill_count"] else ""
            cells.append(f"{r['VALU']}{d}, {r['vgpr_count']}v{sp}, {r['waves/SIMD']}w")
        mfma = (b or by[(s, variants[0])])["v_mfma"]
        out.append(f"| {s} | {mfma} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--out", default="/tmp/a16w4_fp8_decode_isa")
    p.add_argument("-v", "--variants", nargs="*", default=VARIANTS)
    p.add_argument("-k", "--kernels", nargs="*", default=KERNELS)
    p.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 4)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    jobs = [(v, s) for v in args.variants for s in args.kernels]
    rows = []
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as ex:
        futs = {ex.submit(compile_one, args.out, v, s): (v, s) for v, s in jobs}
        for fut in concurrent.futures.as_completed(futs):
            row = fut.result()
            rows.append(row)
            print(json.dumps(row), flush=True)
    order = {j: i for i, j in enumerate(jobs)}
    rows.sort(key=lambda r: order[(r["variant"], r["spec"])])
    with open(os.path.join(args.out, "isa-counts.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    md = table(rows, args.variants)
    with open(os.path.join(args.out, "isa-counts.md"), "w") as f:
        f.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
