# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942 a16w4 fp8 MFMA mode: one fp8 decode variant through ``fused_moe``, timed + checked.

Forces ``AITER_FLYDSL_A16W4_FP8_MFMA=1`` and runs production ``fused_moe`` (tuned rows,
the two per-row fp8 quant kernels, fp8 gemm1 + gemm2) with the decode variant selected
by ``$AITER_FLYDSL_A16W4_FP8_DECODE``:

  unset / base   the PR #7153 decode (default)
  aperm          no B interleave; A staged to LDS in the same K order
  tab            residual table from a v_perm broadcast (no v_mul_lo_u32)
  noslp          gemm1 built with SLP off for SiTUv2 too
  any '+'-joined subset, e.g. aperm+tab+noslp

One variant per process, each with its own empty ``FLYDSL_RUNTIME_CACHE_DIR``. The driver
``op_tests/flydsl_tests/bench_flydsl_a16w4_fp8_decode.py`` does that for every variant
(optionally under rocprofv3) and builds the comparison table.

Defaults are the Kimi-K3 shapes: model_dim 3584, inter_dim 384/512, E=896, topk 16,
SiTUv2 (beta 4, linear 25); weights N(0,1)*2/sqrt(K) quantized to MXFP4.

Per shape:
  * check: one ``fused_moe`` call with spies on ``flydsl_a16w4_gemm{1,2}``. Both must run
    a_dtype="fp8"; the spy keeps the gemm1 intermediate rows that hold a real token (gemm1
    is deterministic). ``--save-dir`` stores them and the final output, ``--compare-dir``
    reports gemm1 bitwise / max abs diff and the final output's bf16 ulp spread against
    the saved run (top-16 gemm2 adds atomically in a nondeterministic order).
  * top-1 check (``--topk1``): the same weights with top-1 routing, which makes gemm2
    deterministic; bitwise vs the saved run.
  * reference: cos-diff against the PR's fp8-aware torch reference (activation and
    stage-1 output quantized per row) for tokens <= ``--ref-max-token``.
  * timing: ``fused_moe`` with a 1 GiB overwrite before every timed call, CUDA events on
    the call only (median). Under rocprofv3 the per-kernel durations come from the trace;
    ``--json`` writes the launch manifest the driver uses to slice it.

Run (MI300X/MI325X, one variant):
    FLYDSL_RUNTIME_CACHE_DIR=$(mktemp -d) AITER_FLYDSL_A16W4_FP8_DECODE=aperm \\
        python op_tests/test_flydsl_moe_a16w4_fp8_decode.py --compare-dir out/base
"""

import argparse
import functools
import itertools
import json
import os

os.environ["AITER_FLYDSL_A16W4_FP8_MFMA"] = "1"

import pandas as pd
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    fused_moe,
    fused_topk,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.flydsl.kernels import moe_2stage_a16wmix
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import (
    FP8_DECODE_ENV,
    fp8_decode_spec,
)
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

torch.set_default_device("cuda")

SITU_BETA, SITU_LINEAR_BETA = 4.0, 25.0  # Kimi-K3 production SiTUv2 betas
FLUSH_BYTES = 1 << 30
_cfg = {
    "iters": 100,
    "warmup": 10,
    "save_dir": None,
    "compare_dir": None,
    "ref_max_token": 128,
    "topk1": True,
}
# Every fused_moe call segment in issue order (one gemm1 + one gemm2 dispatch each).
_manifest = []


def _variant():
    return fp8_decode_spec(os.environ.get(FP8_DECODE_ENV, "")) or "base"


def _log(token, inter_dim, phase, count):
    _manifest.append(
        {"token": token, "inter_dim": inter_dim, "phase": phase, "count": count}
    )


@functools.lru_cache(maxsize=1)
def _weights(model_dim, inter_dim, E):
    torch.manual_seed(0)
    tq = aiter.get_torch_quant(QuantType.per_1x32)
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtypes.bf16)
    w1 *= 2 / model_dim**0.5
    w1_qt, w1_scale = tq(w1, quant_dtype=dtypes.fp4x2)
    del w1
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtypes.bf16)
    w2 *= 2 / inter_dim**0.5
    w2_qt, w2_scale = tq(w2, quant_dtype=dtypes.fp4x2)
    del w2
    w1_qt = w1_qt.view(E, inter_dim * 2, model_dim // 2)
    w2_qt = w2_qt.view(E, model_dim, inter_dim // 2)
    return {
        "w1_qt": w1_qt,
        "w2_qt": w2_qt,
        "w1_scale_e": w1_scale.view(E, inter_dim * 2, model_dim // 32),
        "w2_scale_e": w2_scale.view(E, model_dim, inter_dim // 32),
        # Standard GGUU (separated gate/up) caller layout, as fused_moe takes it.
        "w1": shuffle_weight_a16w4(w1_qt, 16, False),
        "w2": shuffle_weight_a16w4(w2_qt, 16, False),
        "w1_scale": shuffle_scale_a16w4(w1_scale, E, False),
        "w2_scale": shuffle_scale_a16w4(w2_scale, E, False),
    }


def _routing(token, model_dim, E, topk):
    torch.manual_seed(1)
    inp = torch.randn((token, model_dim), dtype=dtypes.bf16)
    score = torch.randn((token, E), dtype=dtypes.bf16)
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)
    return inp, topk_weights, topk_ids


def _moe(w, inp, topk_weights, topk_ids):
    return fused_moe(
        inp,
        w["w1"],
        w["w2"],
        topk_weights,
        topk_ids,
        w1_scale=w["w1_scale"],
        w2_scale=w["w2_scale"],
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Situv2,
        doweight_stage1=False,
        gate_mode=GateMode.SEPARATED.value,
        beta=SITU_BETA,
        linear_beta=SITU_LINEAR_BETA,
    )


def _fp8_rows(x):
    """Per-row e4m3fnuz quantize -> f32 dequantize, with the torch quantizer."""
    q, s = aiter.get_torch_quant(QuantType.per_Token)(
        x.reshape(-1, x.shape[-1]), quant_dtype=dtypes.fp8
    )
    return (q.float() * s).view(x.shape)


def _reference(w, inp, topk_weights, topk_ids, inter_dim):
    """The PR's fp8-aware SiTUv2 reference (op_tests/test_flydsl_moe_a16wfp4.py)."""
    token, topk = topk_ids.shape
    o1 = torch_moe_stage1(
        _fp8_rows(inp),
        w["w1_qt"].view(dtypes.fp4x2),
        w["w2_qt"].view(dtypes.fp4x2),
        topk_weights,
        topk_ids,
        dtype=dtypes.bf16,
        activation=ActivationType.Situv2,
        quant_type=QuantType.per_1x32,
        a1_scale=None,
        w1_scale=w["w1_scale_e"],
        doweight=False,
        situ_beta=SITU_BETA,
        situ_linear_beta=SITU_LINEAR_BETA,
    )
    o1 = _fp8_rows(o1.view(token, topk, inter_dim))
    return torch_moe_stage2(
        o1,
        w["w1_qt"].view(dtypes.fp4x2),
        w["w2_qt"].view(dtypes.fp4x2),
        topk_weights,
        topk_ids,
        dtype=dtypes.bf16,
        quant_type=QuantType.per_1x32,
        w2_scale=w["w2_scale_e"],
        a2_scale=None,
        doweight=True,
    )


def _spied_moe(w, inp, topk_weights, topk_ids):
    """fused_moe once, recording each a16w4 launch's a_dtype + kernel tiles and the
    gemm1 rows that hold a real token."""
    ran, rows = [], {}
    originals = {}

    def spy(name):
        fn = getattr(moe_2stage_a16wmix, name)
        originals[name] = fn

        def wrapped(**kw):
            out = fn(**kw)
            tiles = "t{}x{}x{}_kw{}_w{}_bnt{}_xcd{}".format(
                *(
                    kw.get(k)
                    for k in (
                        "tile_m",
                        "tile_n",
                        "tile_k",
                        "k_wave",
                        "waves_per_eu",
                        "b_nt",
                        "xcd_swizzle",
                    )
                )
            )
            ran.append((name, kw.get("a_dtype", "bf16"), tiles))
            if name == "flydsl_a16w4_gemm1":
                torch.cuda.synchronize()
                inter = kw["inter_sorted_bf16"]
                ids = kw["m_indices"][: inter.shape[0]].to(torch.int64)
                n = int(kw["cumsum_tensor"][0])
                keep = (ids & 0xFFFFFF) < int(kw["n_tokens"])
                keep[n:] = False
                rows["gemm1"] = inter[keep].clone()
            return out

        return wrapped

    for name in ("flydsl_a16w4_gemm1", "flydsl_a16w4_gemm2"):
        setattr(moe_2stage_a16wmix, name, spy(name))
    try:
        out = _moe(w, inp, topk_weights, topk_ids)
        torch.cuda.synchronize()
    finally:
        for name, fn in originals.items():
            setattr(moe_2stage_a16wmix, name, fn)
    return out, ran, rows["gemm1"]


@functools.lru_cache(maxsize=1)
def _flush_buffer():
    return torch.empty(FLUSH_BYTES, dtype=torch.uint8)


def time_flushed(fn, iters, warmup):
    """Median call time (us): 1 GiB overwrite before every timed call, events on fn only."""
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


def run_shape(token, model_dim, inter_dim, E, topk):
    variant = _variant()
    w = _weights(model_dim, inter_dim, E)
    inp, topk_weights, topk_ids = _routing(token, model_dim, E, topk)
    ret = {
        "token": token,
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "E": E,
        "topk": topk,
        "gfx": get_gfx(),
        "cu": get_cu_num(),
        "variant": variant,
    }

    out, ran, g1 = _spied_moe(w, inp, topk_weights, topk_ids)
    _log(token, inter_dim, "check", 1)
    assert [r[:2] for r in ran] == [
        ("flydsl_a16w4_gemm1", "fp8"),
        ("flydsl_a16w4_gemm2", "fp8"),
    ], f"fp8 MFMA mode did not run on both stages: {ran}"
    ret["gemm1 tile"], ret["gemm2 tile"] = ran[0][2], ran[1][2]
    assert not out.isnan().any().item(), "fused_moe fp8 output has NaN"

    out1 = None
    if _cfg["topk1"]:
        tw1, ti1 = fused_topk(inp, torch.randn((token, E), dtype=dtypes.bf16), 1, True)
        out1 = _moe(w, inp, tw1, ti1)
        torch.cuda.synchronize()
        _log(token, inter_dim, "topk1", 1)

    fname = f"inter{inter_dim}_t{token}.pt"
    if _cfg["save_dir"]:
        os.makedirs(_cfg["save_dir"], exist_ok=True)
        torch.save(
            {
                "gemm1": g1.cpu(),
                "out": out.cpu(),
                "out_topk1": None if out1 is None else out1.cpu(),
            },
            os.path.join(_cfg["save_dir"], fname),
        )
    if _cfg["compare_dir"]:
        base = torch.load(os.path.join(_cfg["compare_dir"], fname))
        bg1 = base["gemm1"]
        same_shape = bg1.shape == g1.shape
        ret["gemm1 bitwise vs base"] = same_shape and torch.equal(
            g1.cpu().view(torch.int16), bg1.view(torch.int16)
        )
        ret["gemm1 max abs vs base"] = (
            float((g1.cpu().float() - bg1.float()).abs().max()) if same_shape else None
        )
        ulp = bf16_ulp_diff(out.cpu(), base["out"])
        ret["out max ulp vs base (top16, atomic noise)"] = int(ulp.max())
        ret["out max abs vs base (top16)"] = float(
            (out.cpu().float() - base["out"].float()).abs().max()
        )
        if out1 is not None and base.get("out_topk1") is not None:
            ret["out bitwise vs base (top1)"] = torch.equal(
                out1.cpu().view(torch.int16), base["out_topk1"].view(torch.int16)
            )
            ret["out max abs vs base (top1)"] = float(
                (out1.cpu().float() - base["out_topk1"].float()).abs().max()
            )
    if token <= _cfg["ref_max_token"]:
        ref = _reference(w, inp, topk_weights, topk_ids, inter_dim)
        ret["cos-diff vs fp8-aware ref"] = _cos_diff(ref.float(), out.float())

    fn = functools.partial(_moe, w, inp, topk_weights, topk_ids)
    ret["fused_moe us"] = time_flushed(fn, _cfg["iters"], _cfg["warmup"])
    _log(token, inter_dim, "warmup", _cfg["warmup"])
    _log(token, inter_dim, "timed", _cfg["iters"])
    return ret


def main():
    if get_gfx() != "gfx942":
        aiter.logger.warning("the a16w4 fp8 MFMA mode is gfx942-only; skipping")
        return
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    p.add_argument(
        "-t",
        "--token",
        type=int,
        nargs="*",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
    )
    p.add_argument("-m", "--model_dim", type=int, nargs="*", default=[3584])
    p.add_argument("-i", "--inter_dim", type=int, nargs="*", default=[384, 512])
    p.add_argument("-e", "--expert", type=int, nargs="*", default=[896])
    p.add_argument("-k", "--topk", type=int, nargs="*", default=[16])
    p.add_argument("--iters", type=int, default=_cfg["iters"])
    p.add_argument("--warmup", type=int, default=_cfg["warmup"])
    p.add_argument("--save-dir", help="store outputs (run with the base decode)")
    p.add_argument("--compare-dir", help="--save-dir of the base run to check against")
    p.add_argument("--ref-max-token", type=int, default=_cfg["ref_max_token"])
    p.add_argument("--no-topk1", action="store_true", help="skip the top-1 exact check")
    p.add_argument("--json", help="write rows + launch manifest here")
    args = p.parse_args()
    _cfg.update(
        iters=args.iters,
        warmup=args.warmup,
        save_dir=args.save_dir,
        compare_dir=args.compare_dir,
        ref_max_token=args.ref_max_token,
        topk1=not args.no_topk1,
    )
    aiter.logger.info(
        "%s=%r -> fp8_decode=%r, FLYDSL_RUNTIME_CACHE_DIR=%s",
        FP8_DECODE_ENV,
        os.environ.get(FP8_DECODE_ENV),
        _variant(),
        os.environ.get("FLYDSL_RUNTIME_CACHE_DIR", "<default ~/.flydsl/cache>"),
    )
    rows = []
    for model_dim, inter_dim, E, topk, token in itertools.product(
        args.model_dim, args.inter_dim, args.expert, args.topk, args.token
    ):
        rows.append(run_shape(token, model_dim, inter_dim, E, topk))
        aiter.logger.info("%s", json.dumps(rows[-1], default=str))
    aiter.logger.info(
        "a16w4 fp8_decode summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {"variant": _variant(), "rows": rows, "manifest": _manifest},
                f,
                indent=1,
                default=str,
            )


if __name__ == "__main__":
    main()
