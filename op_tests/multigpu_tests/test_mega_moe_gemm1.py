# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone MegaMoE GEMM1 (gate/up + SwiGLU + FP8 1x32 requant) vs torch on one GPU.

Sweeps the Stage1 tile configurations and runs a negative control (A scales
shifted by one K group), which must fail, so a pass means the check can see a
broken K mapping. Shapes default to DeepSeek-V4.1-Flash (5120 / 2304).
"""

import argparse
import sys

import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.ops.flydsl.kernels.mega_moe.gemm1 import gemm1_kernel
from aiter.ops.flydsl.kernels.mega_moe.mega_moe_v2 import per_token_fp8_quant, w1_sref_kernel_order
from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.utility import fp4_utils

SWEEP = (
    dict(sort_block_m=32, tile_n=256, num_waves=4),
    dict(sort_block_m=32, tile_n=256, num_waves=4, mfma_amajor=True),
    dict(sort_block_m=32, tile_n=256, num_waves=4, pipe_weights=False),
    dict(sort_block_m=32, tile_n=512, num_waves=8, mfma_amajor=True),
    dict(sort_block_m=64, tile_n=256, num_waves=4),
    dict(sort_block_m=64, tile_n=256, num_waves=8, mfma_amajor=True),
)
# SBM16 needs per-token A (no paired 16-row A scales).
SWEEP_PER_TOKEN = (
    dict(sort_block_m=16, tile_n=256, num_waves=4),
    dict(sort_block_m=16, tile_n=256, num_waves=4, pipe_weights=False),
)


def fp8_dtype():
    arch = torch.cuda.get_device_properties(0).gcnArchName
    return torch.float8_e4m3fnuz if "gfx94" in arch else torch.float8_e4m3fn


def run_case(args, cfg, x_q, x_s, w_k, ws_k, xd, w1d, shift_scale=False, w_sref=None):
    dev = x_q.device
    sbm = cfg["sort_block_m"]
    rows = args.experts * args.rows_per_expert
    n_tiles = rows // sbm
    tile_row_base = (torch.arange(n_tiles, dtype=torch.int32, device=dev) * sbm).contiguous()
    expert_ids = (torch.arange(n_tiles, device=dev) // (args.rows_per_expert // sbm)).to(torch.int32)
    scale_cols = (args.inter_dim // 32 + 7) // 8 * 8
    out = torch.zeros(rows, args.inter_dim, dtype=torch.uint8, device=dev)
    out_scale = torch.zeros(rows * scale_cols + 8192, dtype=torch.uint8, device=dev)
    # Negative control: per-1x32 scales shift one K group; per-token scales shift one row.
    xs_in = torch.roll(x_s, 1, dims=0 if w_sref is not None else 1).contiguous() if shift_scale else x_s
    gemm1_kernel(out, x_q, w_k, xs_in, ws_k, tile_row_base, expert_ids, out_scale, rows,
                 torch.cuda.current_stream(), model_dim=args.model_dim, inter_dim=args.inter_dim,
                 swiglu_limit=args.swiglu_limit, num_cu=torch.cuda.get_device_properties(0).multi_processor_count,
                 async_a_copy=False, w_sref=w_sref, **cfg)
    torch.cuda.synchronize()
    r = torch.arange(rows, device=dev)[:, None]
    c = torch.arange(args.inter_dim // 32, device=dev)[None, :]
    off = ((r >> 5) * scale_cols * 32 + (c >> 3) * 256 + (c & 3) * 64 + (r & 15) * 4
           + ((c >> 2) & 1) * 2 + ((r >> 4) & 1))
    o_s = fp4_utils.e8m0_to_f32(out_scale[off].view(dtypes.fp8_e8m0))
    got = out.view(fp8_dtype()).float() * o_s.repeat_interleave(32, dim=-1)
    ref = torch.empty(rows, args.inter_dim, device=dev)
    inter = args.inter_dim
    for e in range(args.experts):
        sl = slice(e * args.rows_per_expert, (e + 1) * args.rows_per_expert)
        g = xd[sl] @ w1d[e, :inter].T
        u = xd[sl] @ w1d[e, inter:].T
        if args.swiglu_limit > 0:
            g, u = g.clamp(max=args.swiglu_limit), u.clamp(-args.swiglu_limit, args.swiglu_limit)
        ref[sl] = F.silu(g) * u
    rel = ((got - ref).norm() / ref.norm()).item()
    return rel, bool(torch.isnan(got).any().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dim", type=int, default=5120)
    p.add_argument("--inter-dim", type=int, default=2304)
    p.add_argument("--experts", type=int, default=2)
    p.add_argument("--rows-per-expert", type=int, default=128)
    p.add_argument("--swiglu-limit", type=float, default=10.0)
    p.add_argument("--rtol", type=float, default=0.05)
    p.add_argument("--a-per-token", action="store_true",
                   help="gfx942 per-token FP8 A (FP8 MFMA) instead of per-1x32 scales")
    args = p.parse_args()
    dev = torch.device("cuda", 0)
    torch.manual_seed(0)
    k, inter, e = args.model_dim, args.inter_dim, args.experts
    rows = e * args.rows_per_expert
    # Rows span 2^-3..2^3 so a misplaced per-token scale is visible to relL2.
    row_mag = 2.0 ** (torch.arange(rows, device=dev) % 7 - 3).float()
    x = (torch.randn(rows, k, device=dev) * row_mag[:, None]).to(torch.bfloat16)
    if args.a_per_token:
        x_q = torch.empty(rows, k, dtype=fp8_dtype(), device=dev)
        x_s = torch.zeros(rows, k // 32, dtype=torch.uint8, device=dev)
        tok_scale = per_token_fp8_quant(x, x_q, x_s)
        xd = x_q.float() * tok_scale
        x_q = x_q.view(torch.uint8)
    else:
        x_q, x_s = per_1x32_mx_quant(x, quant_mode="fp8")
        x_q, x_s = x_q.view(torch.uint8).contiguous(), x_s.view(torch.uint8).contiguous()
        xd = x_q.view(fp8_dtype()).float() * fp4_utils.e8m0_to_f32(
            x_s.view(dtypes.fp8_e8m0)).repeat_interleave(32, dim=-1)
    quant = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    w1 = torch.randn(e, 2 * inter, k, dtype=torch.bfloat16, device=dev) * k**-0.25
    w1_q, w1_s = quant(w1, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(e, 2 * inter, k // 2)
    w1d = fp4_utils.mxfp4_to_f32(w1_q).view(e, 2 * inter, k) * fp4_utils.e8m0_to_f32(
        w1_s.view(e, 2 * inter, k // 32)).repeat_interleave(32, dim=-1)
    w_k = shuffle_weight_a16w4(w1_q, 16, True).contiguous()
    ws_k = shuffle_scale_a16w4(w1_s, e, True).contiguous()
    w_sref = w1_sref_kernel_order(ws_k.view(torch.uint8), e, inter, k) if args.a_per_token else None

    failed = 0
    for cfg in SWEEP + (SWEEP_PER_TOKEN if args.a_per_token else ()):
        rel, nan = run_case(args, cfg, x_q, x_s, w_k, ws_k, xd, w1d, w_sref=w_sref)
        ok = rel < args.rtol and not nan
        failed += not ok
        print(f"[GEMM1] {'PASS' if ok else 'FAIL'} relL2={rel:.5f} nan={nan} cfg={cfg}", flush=True)
    rel, _ = run_case(args, SWEEP[0], x_q, x_s, w_k, ws_k, xd, w1d, shift_scale=True, w_sref=w_sref)
    neg_ok = rel > 10 * args.rtol
    failed += not neg_ok
    print(f"[GEMM1] negative control (A scales shifted): relL2={rel:.5f} "
          f"{'detected (PASS)' if neg_ok else 'NOT detected (FAIL)'}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
