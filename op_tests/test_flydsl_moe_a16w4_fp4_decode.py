# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942 a16w4 MoE: one MXFP4 decode variant, timed + checked against a saved base.

Runs ``flydsl_a16w4_gemm1`` (gate+up, SiTUv2) and ``flydsl_a16w4_gemm2`` (down, atomic)
with the decode variant selected by ``$AITER_FLYDSL_A16W4_FP4_DECODE`` (the same switch
production ``fused_moe`` honours):

  unset / base   the #6144 decode (default)
  sign           negative fp8 LUT + AND-mask sign (Triton upcast8xMxfp4_SW)
  perm           v_perm_b32 bf16 pack
  pkmul          explicit v_pk_mul_f32 scale multiply
  all            sign+perm+pkmul (any '+'-joined subset also works)

One variant per process: FlyDSL's cache key does not hash the decode helper bodies, so
give every run its own empty ``FLYDSL_RUNTIME_CACHE_DIR``. The driver
``op_tests/flydsl_tests/bench_flydsl_a16w4_fp4_decode.py`` does that for all variants
(optionally under rocprofv3) and builds the comparison table.

Defaults are the Kimi-K3 shapes of PR #6144: model_dim 3584, inter_dim 512/384, E=896,
topk 16, tile t32x128x128 for both stages, wrapper-default knobs.

Timing: every timed iteration first overwrites a 1 GiB buffer (L2/MALL flush), and
CUDA events bracket only the kernel; the table reports their median. Under rocprofv3
the kernel-trace durations are the metric: ``--json`` writes the launch manifest the
driver uses to slice the trace per shape.

Checks: ``--save-dir`` stores gemm1/gemm2 outputs (run it with the base decode);
``--compare-dir`` checks gemm1 bit for bit (it is bitwise reproducible) and gemm2 in
bf16 ulps (its atomic epilogue adds in a nondeterministic order: ~1 ulp run to run).
Both stages are also checked against the torch reference for tokens <= --ref-max-token.

Run (MI300X/MI325X, one variant):
    FLYDSL_RUNTIME_CACHE_DIR=$(mktemp -d) AITER_FLYDSL_A16W4_FP4_DECODE=sign \\
        python op_tests/test_flydsl_moe_a16w4_fp4_decode.py --compare-dir out/base
"""

import argparse
import functools
import itertools
import json
import os

import pandas as pd
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_topk, moe_sorting, torch_moe_stage1, torch_moe_stage2
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import (
    FP4_DECODE_ENV,
    flydsl_a16w4_gemm1,
    flydsl_a16w4_gemm2,
    fp4_decode_spec,
)
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.test_common import benchmark, checkAllclose

torch.set_default_device("cuda")

# The variants only change the gfx942 decode; gfx950 uses v_cvt_scalef32_pk_bf16_fp4.
SUPPORTED_GFX = ["gfx942"]
SITU_BETA, SITU_LINEAR_BETA = 4.0, 25.0  # Kimi-K3 production SiTUv2 betas
FLUSH_BYTES = 1 << 30
_cfg = {
    "iters": 200,
    "warmup": 10,
    "save_dir": None,
    "compare_dir": None,
    "max_ulp": 2,
    "ref_max_token": 256,
}
# Every gemm1/gemm2 launch segment in issue order, for slicing a rocprofv3 trace.
_manifest = []


def _log_launches(token, inter_dim, gemm, phase, count):
    _manifest.append(
        {
            "token": token,
            "inter_dim": inter_dim,
            "gemm": gemm,
            "phase": phase,
            "count": count,
        }
    )


def _tile(tile):
    m, n, k = (int(x) for x in tile.lstrip("t").split("x"))
    return m, n, k


@functools.lru_cache(maxsize=1)
def _weights(model_dim, inter_dim, E):
    torch.manual_seed(0)
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtypes.bf16) / 10
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtypes.bf16) / 10
    tq = aiter.get_torch_quant(QuantType.per_1x32)
    w1_qt, w1_scale = tq(w1, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = tq(w2, quant_dtype=dtypes.fp4x2)
    del w1, w2
    w1_qt = w1_qt.view(E, inter_dim * 2, model_dim // 2)
    w2_qt = w2_qt.view(E, model_dim, inter_dim // 2)
    return {
        "w1_qt": w1_qt,
        "w2_qt": w2_qt,
        "w1_scale_e": w1_scale.view(E, inter_dim * 2, model_dim // 32),
        "w2_scale_e": w2_scale.view(E, model_dim, inter_dim // 32),
        # Standard GGUU (separated gate/up) caller layout, as fused_moe passes it.
        "w1": shuffle_weight_a16w4(w1_qt, 16, False).view(torch.uint8),
        "w2": shuffle_weight_a16w4(w2_qt, 16, False).view(torch.uint8),
        "w1_scale": shuffle_scale_a16w4(w1_scale, E, False).view(torch.uint8).view(-1),
        "w2_scale": shuffle_scale_a16w4(w2_scale, E, False).view(torch.uint8).view(-1),
    }


def _routing(token, model_dim, E, topk, tile_m):
    torch.manual_seed(1)
    inp = torch.randn((token, model_dim), dtype=dtypes.bf16)
    score = torch.randn((token, E), dtype=dtypes.bf16)
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)
    sorted_ids, sorted_w, sorted_eids, num_valid, _moe_buf = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtypes.bf16, block_size=tile_m
    )
    return inp, topk_weights, topk_ids, sorted_ids, sorted_w, sorted_eids, num_valid


def run_torch(inp, w, topk_weights, topk_ids, inter_dim):
    """bf16 SiTUv2 reference: stage1 [token, topk, inter], final [token, model_dim]."""
    o1 = torch_moe_stage1(
        inp,
        w["w1_qt"].view(dtypes.fp4x2),
        w["w2_qt"].view(dtypes.fp4x2),
        topk_weights,
        topk_ids,
        dtype=dtypes.bf16,
        activation=ActivationType.Situv2,
        quant_type=QuantType.per_1x32,
        w1_scale=w["w1_scale_e"],
        doweight=False,
        situ_beta=SITU_BETA,
        situ_linear_beta=SITU_LINEAR_BETA,
    )
    o1 = o1.view(inp.shape[0], topk_ids.shape[1], inter_dim)
    out = torch_moe_stage2(
        o1,
        w["w1_qt"].view(dtypes.fp4x2),
        w["w2_qt"].view(dtypes.fp4x2),
        topk_weights,
        topk_ids,
        dtype=dtypes.bf16,
        quant_type=QuantType.per_1x32,
        w2_scale=w["w2_scale_e"],
        doweight=True,
    )
    return o1, out


def _gemm1(r, w, inter, token, model_dim, inter_dim, E, topk, tile):
    tile_m, tile_n, tile_k = _tile(tile)
    inp, _, _, sorted_ids, _, sorted_eids, num_valid = r
    flydsl_a16w4_gemm1(
        a_bf16=inp,
        w1_u8=w["w1"],
        w1_scale_u8=w["w1_scale"],
        sorted_expert_ids=sorted_eids,
        cumsum_tensor=num_valid.to(torch.int32),
        m_indices=sorted_ids.to(torch.int32),
        inter_sorted_bf16=inter,
        n_tokens=token,
        NE=E,
        D_HIDDEN=model_dim,
        D_INTER=inter_dim,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        act="situv2",
        situ_beta=SITU_BETA,
        situ_linear_beta=SITU_LINEAR_BETA,
    )


def _gemm2(r, w, inter, out, token, model_dim, inter_dim, E, topk, tile):
    tile_m, tile_n, tile_k = _tile(tile)
    _, _, _, sorted_ids, sorted_w, sorted_eids, num_valid = r
    flydsl_a16w4_gemm2(
        inter_sorted_bf16=inter,
        w2_u8=w["w2"],
        w2_scale_u8=w["w2_scale"],
        sorted_expert_ids=sorted_eids,
        cumsum_tensor=num_valid.to(torch.int32),
        sorted_token_ids=sorted_ids,
        sorted_weights=sorted_w,
        flat_out=out.view(-1),
        M_logical=token,
        max_sorted=int(inter.shape[0]),
        NE=E,
        D_HIDDEN=model_dim,
        D_INTER=inter_dim,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        epilog="atomic",
    )


@functools.lru_cache(maxsize=1)
def _flush_buffer():
    return torch.empty(FLUSH_BYTES, dtype=torch.uint8)


def time_flushed(fn, iters, warmup):
    """Median kernel time (us): 1 GiB overwrite before every timed launch, events on fn only."""
    flush = _flush_buffer()
    for _ in range(warmup):
        fn()
    ev = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    for start, end in ev:
        flush.fill_(1)
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return float(torch.tensor([s.elapsed_time(e) for s, e in ev]).median()) * 1e3


def bf16_ulp_diff(a, b):
    """|a - b| in bf16 ulps (sign-magnitude ordering, +0 == -0)."""

    def key(x):
        u = x.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        return torch.where(u >= 0x8000, 0x8000 - u, u)

    return (key(a) - key(b)).abs()


def _cos_diff(x, y):
    x, y = x.double(), y.double()
    return float(1 - 2 * (x * y).sum() / (x * x + y * y).sum())


def _gather_stage1(inter, sorted_ids, num_valid, token, topk):
    """Sorted-position rows -> [token, topk, inter] (packed id = slot << 24 | token)."""
    n = int(num_valid[0])
    ids = sorted_ids[:n].to(torch.int64)
    tok, slot = ids & 0xFFFFFF, ids >> 24
    keep = tok < token
    out = torch.zeros((token, topk, inter.shape[1]), dtype=inter.dtype)
    out[tok[keep], slot[keep]] = inter[:n][keep]
    return out


@benchmark()
def test_a16w4_fp4_decode(token, model_dim, inter_dim, E, topk, tile, gemms):
    variant = fp4_decode_spec() or "base"
    tile_m = _tile(tile)[0]
    w = _weights(model_dim, inter_dim, E)
    r = _routing(token, model_dim, E, topk, tile_m)
    inp, topk_weights, topk_ids, sorted_ids, _, sorted_eids, num_valid = r
    shape = (token, model_dim, inter_dim, E, topk, tile)
    sorted_size = int(sorted_eids.shape[0]) * tile_m

    inter = torch.zeros((sorted_size, inter_dim), dtype=dtypes.bf16)
    out = torch.zeros((token, model_dim), dtype=dtypes.bf16)
    _gemm1(r, w, inter, *shape)
    _gemm2(r, w, inter, out, *shape)
    torch.cuda.synchronize()
    for g in (1, 2):
        _log_launches(token, inter_dim, g, "check", 1)

    ret = {"gfx": get_gfx(), "variant": variant}
    fname = f"inter{inter_dim}_t{token}.pt"
    if _cfg["save_dir"]:
        os.makedirs(_cfg["save_dir"], exist_ok=True)
        torch.save(
            {"gemm1": inter.cpu(), "gemm2": out.cpu()},
            os.path.join(_cfg["save_dir"], fname),
        )
    if _cfg["compare_dir"]:
        base = torch.load(os.path.join(_cfg["compare_dir"], fname))
        g1_exact = torch.equal(
            inter.cpu().view(torch.int16), base["gemm1"].view(torch.int16)
        )
        ulp = bf16_ulp_diff(out.cpu(), base["gemm2"])
        ret["gemm1 bitwise vs base"] = g1_exact
        ret["gemm2 max ulp vs base"] = int(ulp.max())
        ret["gemm2 >1ulp vs base"] = int((ulp > 1).sum())
        assert g1_exact, f"gemm1 fp4_decode={variant!r} is not bit-exact to base"
        assert int(ulp.max()) <= _cfg["max_ulp"], (
            f"gemm2 fp4_decode={variant!r}: {int(ulp.max())} ulp vs base "
            f"(> {_cfg['max_ulp']})"
        )
    if token <= _cfg["ref_max_token"]:
        o1_ref, out_ref = run_torch(inp, w, topk_weights, topk_ids, inter_dim)
        got1 = _gather_stage1(inter, sorted_ids, num_valid, token, topk)
        ret["gemm1 cos-diff vs torch"] = _cos_diff(o1_ref, got1)
        ret["gemm2 cos-diff vs torch"] = _cos_diff(out_ref, out)
        ret["gemm1 err"] = checkAllclose(
            o1_ref.float(), got1.float(), rtol=5e-2, atol=5e-2, msg="gemm1 vs torch"
        )
        ret["gemm2 err"] = checkAllclose(
            out_ref.float(), out.float(), rtol=5e-2, atol=5e-2, msg="gemm2 vs torch"
        )

    n_rows, n_exp = token * topk, int(torch.unique(topk_ids).numel())
    for g in gemms:
        if g == 1:
            fn = functools.partial(_gemm1, r, w, inter, *shape)
            flops = 2 * n_rows * (2 * inter_dim) * model_dim
            w_bytes = n_exp * 2 * inter_dim * model_dim * (0.5 + 1 / 32)
            nbytes = token * model_dim * 2 + w_bytes + n_rows * inter_dim * 2
        else:
            # Timed runs keep accumulating into `out`; it is not checked afterwards.
            fn = functools.partial(_gemm2, r, w, inter, out, *shape)
            flops = 2 * n_rows * model_dim * inter_dim
            w_bytes = n_exp * model_dim * inter_dim * (0.5 + 1 / 32)
            nbytes = n_rows * inter_dim * 2 + w_bytes + token * model_dim * 2
        us = time_flushed(fn, _cfg["iters"], _cfg["warmup"])
        _log_launches(token, inter_dim, g, "warmup", _cfg["warmup"])
        _log_launches(token, inter_dim, g, "timed", _cfg["iters"])
        ret[f"gemm{g} us"] = us
        ret[f"gemm{g} TFLOPS"] = flops / us / 1e6
        ret[f"gemm{g} TB/s"] = nbytes / us / 1e6
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "fp4_decode variants are gfx942-only; skipping on %s", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    parser.add_argument("-g", "--gemm", type=int, nargs="*", default=[1, 2])
    parser.add_argument(
        "-t", "--token", type=int, nargs="*", default=[1, 4, 16, 64, 256, 1024, 4096]
    )
    parser.add_argument("-m", "--model_dim", type=int, nargs="*", default=[3584])
    parser.add_argument("-i", "--inter_dim", type=int, nargs="*", default=[512, 384])
    parser.add_argument("-e", "--expert", type=int, nargs="*", default=[896])
    parser.add_argument("-k", "--topk", type=int, nargs="*", default=[16])
    parser.add_argument("--tile", type=str, nargs="*", default=["t32x128x128"])
    parser.add_argument("--iters", type=int, default=_cfg["iters"])
    parser.add_argument("--warmup", type=int, default=_cfg["warmup"])
    parser.add_argument("--save-dir", help="store outputs (run with the base decode)")
    parser.add_argument(
        "--compare-dir", help="--save-dir of the base run to check against"
    )
    parser.add_argument("--max-ulp", type=int, default=_cfg["max_ulp"])
    parser.add_argument("--ref-max-token", type=int, default=_cfg["ref_max_token"])
    parser.add_argument("--json", help="write rows + launch manifest here")
    args = parser.parse_args()
    _cfg.update(
        iters=args.iters,
        warmup=args.warmup,
        save_dir=args.save_dir,
        compare_dir=args.compare_dir,
        max_ulp=args.max_ulp,
        ref_max_token=args.ref_max_token,
    )
    aiter.logger.info(
        "%s=%r -> fp4_decode=%r, FLYDSL_RUNTIME_CACHE_DIR=%s",
        FP4_DECODE_ENV,
        os.environ.get(FP4_DECODE_ENV),
        fp4_decode_spec() or "base",
        os.environ.get("FLYDSL_RUNTIME_CACHE_DIR", "<default ~/.flydsl/cache>"),
    )

    rows = []
    for model_dim, inter_dim, E, topk, tile, token in itertools.product(
        args.model_dim, args.inter_dim, args.expert, args.topk, args.tile, args.token
    ):
        rows.append(
            test_a16w4_fp4_decode(
                token, model_dim, inter_dim, E, topk, tile, tuple(args.gemm)
            )
        )
    df = pd.DataFrame(rows).drop(columns=["gemms"])
    aiter.logger.info(
        "a16w4 fp4_decode summary (markdown):\n%s", df.to_markdown(index=False)
    )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "variant": fp4_decode_spec() or "base",
                    "rows": rows,
                    "manifest": _manifest,
                },
                f,
                indent=1,
                default=str,
            )


if __name__ == "__main__":
    main()
