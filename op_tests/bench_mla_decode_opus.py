# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Standalone benchmark for mla_decode_opus (gfx950).

Reports per-(QLEN,H) latency, achieved matrix TFLOPS and HBM read GB/s, and
% of MI355X peaks (bf16 matrix 2.5 PFLOPS, HBM 8 TB/s). Realistic decode
shapes: B requests, dense KV length L, D=512 absorbed.
"""

from __future__ import annotations

import argparse
import math
import torch

import aiter  # noqa: F401
from aiter.ops.mla_decode_opus import mla_decode_opus

PEAK_MATRIX_TFLOPS = 2500.0  # MI355X bf16 matrix peak (case-study)
PEAK_HBM_TBs = 8.0           # MI355X HBM BW peak


def _dense_csr(bsz, total_pages, device):
    indptr = torch.arange(0, (bsz + 1) * total_pages, total_pages, dtype=torch.int32, device=device)
    indices = torch.arange(total_pages, dtype=torch.int32, device=device).repeat(bsz)
    return indptr, indices


def _bench_one(bsz, qlen, h, d, L, dtype, use_sink, iters, warmup):
    device = torch.device("cuda")
    q = (torch.randn(bsz, qlen, h, d, device=device, dtype=torch.float32) * 0.5).to(dtype)
    unified_kv = (torch.randn(L, d, device=device, dtype=torch.float32) * 0.5).to(dtype)
    kv_indptr, kv_indices = _dense_csr(bsz, L, device)
    attn_sink = (torch.randn(h, device=device, dtype=torch.float32) * 0.25) if use_sink else None
    softmax_scale = 1.0 / math.sqrt(d)
    out = torch.empty_like(q)

    for _ in range(warmup):
        mla_decode_opus(q, unified_kv, kv_indices, kv_indptr, attn_sink, softmax_scale, out=out)
    torch.cuda.synchronize()

    # min over many repeat windows: DVFS/contention only slow a kernel down, so
    # the min window is the reproducible peak-clock latency.
    def _window():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            mla_decode_opus(q, unified_kv, kv_indices, kv_indptr, attn_sink, softmax_scale, out=out)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters * 1e3
    lat_us = min(_window() for _ in range(20))

    # Causal visible tokens summed over positions: sum_p max(0, L - qlen + p)
    vis = sum(max(0, L - qlen + p) for p in range(qlen))
    flops = 4.0 * h * vis * d * bsz  # QK + PV, 2 GEMMs * 2 flop/MAC
    tflops = flops / (lat_us * 1e-6) / 1e12
    # Strategy A re-reads KV per position from global into LDS
    kv_bytes = bsz * vis * d * (2 if dtype != torch.float8_e4m3fnuz else 1)
    gbs = kv_bytes / (lat_us * 1e-6) / 1e9
    return dict(
        B=bsz, QLEN=qlen, H=h, L=L, dtype=str(dtype).split(".")[-1], sink=use_sink,
        lat_us=round(lat_us, 2), TFLOPS=round(tflops, 1),
        pct_mat=round(100 * tflops / PEAK_MATRIX_TFLOPS, 1),
        GBs=round(gbs, 1), pct_hbm=round(100 * gbs / (PEAK_HBM_TBs * 1e3), 1),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bsz", type=int, default=256)
    p.add_argument("--qlen", type=int, nargs="*", default=[1, 2, 4, 8])
    p.add_argument("--h_q", type=int, nargs="*", default=[16, 128])
    p.add_argument("--L", type=int, default=4096)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--no-sink", action="store_true")
    args = p.parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    rows = []
    for h in args.h_q:
        for ql in args.qlen:
            rows.append(_bench_one(args.bsz, ql, h, 512, args.L, dtype, not args.no_sink, args.iters, args.warmup))
    hdr = ["B", "QLEN", "H", "L", "dtype", "sink", "lat_us", "TFLOPS", "pct_mat", "GBs", "pct_hbm"]
    print("\t".join(hdr))
    for r in rows:
        print("\t".join(str(r[k]) for k in hdr))
