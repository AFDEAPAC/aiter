# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compile-only gfx942 ISA dump + VALU count for the a16w4 MXFP4 decode variants.

No GPU needed. Every (variant, kernel) is compiled in its own subprocess with
``COMPILE_ONLY=1``, ``FLYDSL_GPU_ARCH=gfx942``, the disk cache disabled and a fresh
``FLYDSL_RUNTIME_CACHE_DIR``: FlyDSL's cache key does not hash the decode helpers, so
a reused cache can hand back another variant's kernel.

Kernels:
  * ``micro``  one decode (8 weights) per thread: load raw dword + f32 scale, decode,
               store 4 dwords. ``decode VALU`` counts only the decode window.
  * ``gemm1``/``gemm2``  the real stage kernels, Kimi-K3 shape (model_dim 3584,
               E=896, topk 16, tile t32x128x128, SiTUv2, atomic gemm2).

Run:
    python op_tests/flydsl_tests/dump_flydsl_a16w4_fp4_decode_isa.py \
        --out /tmp/a16w4_fp4_decode_isa
Each ISA lands in ``<out>/<variant>/<kernel>[_i<inter_dim>]/<kernel name>/22_final_isa.s``.
"""

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VARIANTS = ["", "sign", "perm", "pkmul", "sign+perm+pkmul"]
KERNELS = ["micro", "gemm1", "gemm2"]
TRACKED = [
    "v_perm_b32",
    "v_cvt_pk_f32_fp8",
    "v_pk_mul_f32",
    "v_mul_f32",
    "v_and_or_b32",
    "v_lshl_or_b32",
    "v_lshrrev_b32",
    "v_lshlrev_b32",
    "v_and_b32",
    "v_or_b32",
    "v_add_u32",
    "v_mfma",
]

_CHILD = r"""
import os, sys, types
sys.path.insert(0, {repo!r})
try:
    import aiter  # noqa: F401
except Exception:
    # Hosts without ROCm cannot import the aiter package root; the FlyDSL kernel
    # modules themselves only need flydsl + torch.
    for k in [k for k in sys.modules if k == "aiter" or k.startswith("aiter.")]:
        del sys.modules[k]
    for name, path in (("aiter", "aiter"), ("aiter.ops", "aiter/ops"),
                       ("aiter.ops.flydsl", "aiter/ops/flydsl")):
        m = types.ModuleType(name)
        m.__path__ = [os.path.join({repo!r}, path)]
        sys.modules[name] = m
from aiter.ops.flydsl.kernels.tensor_shim import _preload_compiled

kernel, variant = {kernel!r}, {variant!r}
if kernel == "gemm1":
    from aiter.ops.flydsl.kernels.moe_2stage_a16wmix.gemm1 import compile_gemm1_a16w4_port
    launch = compile_gemm1_a16w4_port(
        BM=32, D_HIDDEN=3584, D_INTER={inter}, NE=896, TOPK=16, TILE_N=128,
        TILE_K=128, act="situv2", b_cache_mod=2, rocm_arch="gfx942",
        fp4_decode=variant)
    args = [0x1000] * 6 + [1, 1] + [1.0] * 5 + [0x1000, None]
elif kernel == "gemm2":
    from aiter.ops.flydsl.kernels.moe_2stage_a16wmix.gemm2 import compile_gemm2_a16w4_port
    launch = compile_gemm2_a16w4_port(
        BM=32, NE=896, N_OUT=3584, D_INTER={inter}, TILE_N=128, TILE_K=128,
        xcd_swizzle=1, b_cache_mod=0, rocm_arch="gfx942", epilog="atomic", topk=16,
        fp4_decode=variant)
    args = [0x1000] * 7 + [1, 1, 1, 0x1000, None]
else:
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import gpu, range_constexpr
    from flydsl.expr.typing import T
    from aiter.ops.flydsl.kernels.mxfp4_gemm_common import global_typed_ptr
    from aiter.ops.flydsl.kernels.moe_2stage_a16wmix.utils import (
        _fp4_nibble_to_bf16x8, fp4_decode_spec)

    variant = fp4_decode_spec(variant)

    @flyc.kernel(name=f"fp4_decode_{{kernel}}", known_block_size=[256, 1, 1])
    def micro_kernel(arg_w: fx.Int64, arg_s: fx.Int64, arg_o: fx.Int64):
        tid = fx.Int32(gpu.thread_id("x")) + fx.Int32(gpu.block_id("x")) * fx.Int32(256)
        w = global_typed_ptr(arg_w, T.i32)
        s = global_typed_ptr(arg_s, T.f32)
        o = global_typed_ptr(arg_o, T.i32)
        v8 = _fp4_nibble_to_bf16x8(fx.Int32(w[tid]), fx.Float32(s[tid]), decode=variant)
        words = fx.Vector(v8).bitcast(fx.Int32)
        for j in range_constexpr(4):
            o[tid * fx.Int32(4) + fx.Int32(j)] = fx.Int32(words[j])

    @flyc.jit
    def launch(arg_w: fx.Int64, arg_s: fx.Int64, arg_o: fx.Int64, stream: fx.Stream):
        micro_kernel(arg_w, arg_s, arg_o).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)

    args = [0x1000, 0x1000, 0x1000, None]
_preload_compiled(launch, *args)
"""


def _kernel_body(isa_text, full=False):
    """Instructions between the kernel entry label and s_endpgm (mnemonics or lines)."""
    ops, inside = [], False
    for line in isa_text.splitlines():
        s = line.strip()
        if re.match(r"^[A-Za-z_][\w.]*:$", s) and not s.startswith("."):
            inside = True
            continue
        if not inside or not s or s.startswith((";", ".")) or s.endswith(":"):
            continue
        op = s.split()[0]
        ops.append(s if full else op)
        if op == "s_endpgm":
            break
    return ops


def _is_valu(op):
    return op.startswith("v_") and not op.startswith("v_mfma")


def _decode_window_valu(isa_text):
    """micro only: VALU after the loads land (first vmcnt wait) and before the store.

    Addressing and the LUT/constant materialization are all issued before the wait,
    so this window is exactly one decode.
    """
    lines = _kernel_body(isa_text, full=True)
    start = next(i for i, s in enumerate(lines) if s.startswith("s_waitcnt vmcnt"))
    end = next(i for i, s in enumerate(lines) if s.startswith("global_store"))
    return sum(1 for s in lines[start:end] if _is_valu(s.split()[0]))


def count_isa(path, kernel=""):
    with open(path) as f:
        text = f.read()
    ops = _kernel_body(text)
    hist = collections.Counter(ops)
    valu = sum(n for op, n in hist.items() if _is_valu(op))
    row = {"VALU": valu}
    if kernel == "micro":
        row["decode VALU"] = _decode_window_valu(text)
    for t in TRACKED:
        row[t] = sum(n for op, n in hist.items() if op.startswith(t))
    row["decodes"] = row["v_cvt_pk_f32_fp8"] // 4
    for key in ("vgpr_count", "sgpr_count", "vgpr_spill_count"):
        m = re.search(rf"\.{key}:\s+(\d+)", text)
        row[key] = int(m.group(1)) if m else None
    return row


def compile_one(out_dir, kernel, variant, inter):
    tag = kernel if inter is None else f"{kernel}_i{inter}"
    dump = os.path.join(out_dir, variant or "base", tag)
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
            f.write(
                _CHILD.format(repo=REPO, kernel=kernel, variant=variant, inter=inter)
            )
        res = subprocess.run(
            [sys.executable, child],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    if res.returncode != 0:
        raise RuntimeError(
            f"{kernel}/{variant or 'base'} failed:\n{res.stderr[-4000:]}"
        )
    isas = [
        os.path.join(r, f)
        for r, _, fs in os.walk(dump)
        for f in fs
        if f == "22_final_isa.s"
    ]
    assert len(isas) == 1, isas
    return isas[0]


def main():
    import pandas as pd

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--out", default="/tmp/a16w4_fp4_decode_isa")
    p.add_argument(
        "-v",
        "--variants",
        nargs="*",
        default=VARIANTS,
        help='decode specs; "" (or "base") is the default decode',
    )
    p.add_argument("-k", "--kernels", nargs="*", default=KERNELS, choices=KERNELS)
    p.add_argument("-i", "--inter-dim", type=int, nargs="*", default=[512])
    args = p.parse_args()
    variants = ["" if v in ("base", "baseline") else v for v in args.variants]

    rows = []
    for kernel in args.kernels:
        for inter in args.inter_dim if kernel.startswith("gemm") else [None]:
            for v in variants:
                isa = compile_one(args.out, kernel, v, inter)
                row = {"kernel": kernel, "inter_dim": inter, "variant": v or "base"}
                row.update(count_isa(isa, kernel))
                row["isa"] = isa
                rows.append(row)
                print(json.dumps(row), flush=True)
    df = pd.DataFrame(rows)
    print(df.drop(columns=["isa"]).to_markdown(index=False))


if __name__ == "__main__":
    main()
