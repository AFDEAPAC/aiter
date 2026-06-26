# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``mla_decode_opus`` (gfx950 OPUS MLA absorbed decode, D=512).

Reference: per-batch CSR prefix of ``unified_kv`` rows, causal visible length
``valid_len(p) = max(0, L - QLEN + p)`` for query position ``p`` in ``[0, QLEN)``,
optional per-head sink in the denominator only (same convention as
``test_pa_sparse_prefill_opus.py``).
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
from typing import Optional, Tuple

import pytest
import torch

import aiter  # noqa: F401
from aiter.ops.mla_decode_opus import mla_decode_opus, mla_decode_opus_splitkv
from aiter.test_common import benchmark, checkAllclose, perftest


def _skip(reason: str) -> bool:
    if "PYTEST_CURRENT_TEST" in os.environ:
        pytest.skip(reason)
    print(f"SKIP: {reason}")
    return True


def _get_gpu_arch() -> Optional[str]:
    if not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(0)
        if hasattr(props, "gcnArchName"):
            arch_name = props.gcnArchName
            return arch_name.split(":")[0] if ":" in arch_name else arch_name
    except (AttributeError, RuntimeError):
        pass
    return None


def _skip_if_unsupported(d: int) -> bool:
    if not torch.cuda.is_available():
        return _skip("CUDA/HIP device not available")
    arch = _get_gpu_arch()
    if arch != "gfx950":
        return _skip(f"mla_decode_opus requires gfx950, found {arch}")
    if d != 512:
        return _skip(f"Only D=512 is compiled, requested D={d}")
    return False


def _ref_mla_decode_opus(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
    softmax_scale: float,
    *,
    use_sink: bool,
) -> torch.Tensor:
    """PyTorch reference in fp32 (matches fp32 accum + sink convention)."""
    bsz, qlen, h, d = q.shape
    out = torch.zeros_like(q)
    q_f32 = q.to(torch.float32)
    ukv_f32 = unified_kv.to(torch.float32)
    indptr = kv_indptr.to(torch.int64).cpu().tolist()
    idx = kv_indices.to(torch.int64)

    for b in range(bsz):
        ps, pe = indptr[b], indptr[b + 1]
        row_len = pe - ps
        if row_len <= 0:
            continue
        kv_rows = ukv_f32.index_select(0, idx[ps:pe])
        for p in range(qlen):
            valid_len = max(0, row_len - qlen + p)
            if valid_len == 0:
                continue
            prefix = kv_rows[:valid_len]
            scores = q_f32[b, p] @ prefix.t() * softmax_scale
            if use_sink and attn_sink is not None:
                sink_f32 = attn_sink.to(torch.float32)
                sink_col = sink_f32.unsqueeze(1)
                scores_with_sink = torch.cat([scores, sink_col], dim=1)
                max_score = scores_with_sink.amax(dim=1, keepdim=True)
                exp_scores = torch.exp(scores - max_score)
                exp_sink = torch.exp(sink_col - max_score)
                denom = exp_scores.sum(dim=1, keepdim=True) + exp_sink
                pmat = exp_scores / denom
                out[b, p] = (pmat @ prefix).to(q.dtype)
            else:
                max_score = scores.amax(dim=1, keepdim=True)
                exp_scores = torch.exp(scores - max_score)
                denom = exp_scores.sum(dim=1, keepdim=True)
                pmat = exp_scores / denom
                out[b, p] = (pmat @ prefix).to(q.dtype)
    return out


def _dense_csr_kv(bsz: int, total_pages: int, *, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    indptr = torch.arange(0, (bsz + 1) * total_pages, total_pages, dtype=torch.int32, device=device)
    indices = torch.arange(total_pages, dtype=torch.int32, device=device).repeat(bsz)
    return indptr, indices


def _make_inputs(
    bsz: int,
    qlen: int,
    h: int,
    d: int,
    total_pages: int,
    dtype: torch.dtype,
    *,
    device: torch.device | str = "cuda",
    seed: int = 0,
    use_sink: bool = True,
) -> dict:
    torch.manual_seed(seed)
    device = torch.device(device)
    q = (torch.randn(bsz, qlen, h, d, device=device, dtype=torch.float32) * 0.5).to(dtype)
    unified_kv = (torch.randn(total_pages, d, device=device, dtype=torch.float32) * 0.5).to(dtype)
    kv_indptr, kv_indices = _dense_csr_kv(bsz, total_pages, device=device)
    attn_sink = (
        (torch.randn(h, device=device, dtype=torch.float32) * 0.25) if use_sink else None
    )
    return dict(
        q=q,
        unified_kv=unified_kv,
        kv_indices=kv_indices,
        kv_indptr=kv_indptr,
        attn_sink=attn_sink,
    )


def _get_tolerances(dtype: torch.dtype) -> Tuple[float, float]:
    if dtype == torch.float16:
        return 1e-2, 1e-2
    return 2e-2, 2e-2


@perftest()
def _profile_func(target_func, *args, **kwargs):
    return target_func(*args, **kwargs)


@benchmark()
def run_mla_decode_opus(
    bsz: int,
    qlen: int,
    h: int,
    d: int,
    total_pages: int,
    dtype: torch.dtype,
    *,
    use_sink: bool = True,
    seed: int = 0,
    verify: bool = True,
    bench: bool = True,
) -> Optional[dict]:
    if _skip_if_unsupported(d=d):
        return None
    inputs = _make_inputs(bsz, qlen, h, d, total_pages, dtype, seed=seed, use_sink=use_sink)
    softmax_scale = 1.0 / math.sqrt(d)

    row: dict = {}

    if verify:
        ref = _ref_mla_decode_opus(
            **inputs,
            softmax_scale=softmax_scale,
            use_sink=use_sink,
        )
        got = mla_decode_opus(
            inputs["q"],
            inputs["unified_kv"],
            inputs["kv_indices"],
            inputs["kv_indptr"],
            inputs["attn_sink"],
            softmax_scale,
        )
        rtol, atol = _get_tolerances(dtype)
        checkAllclose(
            got,
            ref,
            rtol=rtol,
            atol=atol,
            msg=f"[B={bsz} QLEN={qlen} H={h} D={d} pages={total_pages} dtype={dtype} sink={use_sink}]",
        )

    if bench:
        _, lat_us = _profile_func(
            mla_decode_opus,
            inputs["q"],
            inputs["unified_kv"],
            inputs["kv_indices"],
            inputs["kv_indptr"],
            inputs["attn_sink"],
            softmax_scale,
        )
        row["latency_us"] = round(float(lat_us), 2)
    return row


_PYTEST_QLENS = [1, 2, 3, 4, 5, 8, 16, 17]
_PYTEST_HS = [8, 16, 128]
_PYTEST_DTYPES = [torch.bfloat16, torch.float16]
_PYTEST_SINK = [True, False]


@pytest.mark.parametrize("dtype", _PYTEST_DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("use_sink", _PYTEST_SINK)
@pytest.mark.parametrize("h", _PYTEST_HS)
@pytest.mark.parametrize("qlen", _PYTEST_QLENS)
def test_mla_decode_opus(dtype, use_sink, h, qlen):
    run_mla_decode_opus(
        bsz=4,
        qlen=qlen,
        h=h,
        d=512,
        total_pages=256,
        dtype=dtype,
        use_sink=use_sink,
        seed=(hash((qlen, h, str(dtype), use_sink)) & 0xFFFF),
        verify=True,
        bench=False,
    )


@pytest.mark.parametrize("dtype", _PYTEST_DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("use_sink", _PYTEST_SINK)
@pytest.mark.parametrize("h", _PYTEST_HS)
@pytest.mark.parametrize("qlen", [1, 2, 4])
@pytest.mark.parametrize("num_splits", [2, 4, 8])
def test_mla_decode_opus_splitkv(dtype, use_sink, h, qlen, num_splits):
    if _skip_if_unsupported(d=512):
        return
    seed = hash((qlen, h, str(dtype), use_sink, num_splits)) & 0xFFFF
    inputs = _make_inputs(4, qlen, h, 512, 256, dtype, seed=seed, use_sink=use_sink)
    softmax_scale = 1.0 / math.sqrt(512)
    ref = _ref_mla_decode_opus(**inputs, softmax_scale=softmax_scale, use_sink=use_sink)
    got = mla_decode_opus_splitkv(
        inputs["q"],
        inputs["unified_kv"],
        inputs["kv_indices"],
        inputs["kv_indptr"],
        inputs["attn_sink"],
        softmax_scale,
        num_splits=num_splits,
    )
    rtol, atol = _get_tolerances(dtype)
    checkAllclose(got, ref, rtol=rtol, atol=atol,
                  msg=f"[splitkv B=4 QLEN={qlen} H={h} splits={num_splits} dtype={dtype} sink={use_sink}]")


@pytest.mark.parametrize("dtype", _PYTEST_DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("use_sink", _PYTEST_SINK)
@pytest.mark.parametrize("qlen", [4, 8])
@pytest.mark.parametrize("num_splits", [1, 2, 4])
def test_mla_decode_opus_qpack(dtype, use_sink, qlen, num_splits):
    if _skip_if_unsupported(d=512):
        return
    from aiter.ops.mla_decode_opus import mla_decode_opus_qpack
    seed = hash((qlen, str(dtype), use_sink, num_splits)) & 0xFFFF
    inputs = _make_inputs(8, qlen, 16, 512, 512, dtype, seed=seed, use_sink=use_sink)
    softmax_scale = 1.0 / math.sqrt(512)
    ref = _ref_mla_decode_opus(**inputs, softmax_scale=softmax_scale, use_sink=use_sink)
    got = mla_decode_opus_qpack(
        inputs["q"], inputs["unified_kv"], inputs["kv_indices"], inputs["kv_indptr"],
        inputs["attn_sink"], softmax_scale, num_splits=num_splits,
    )
    rtol, atol = _get_tolerances(dtype)
    checkAllclose(got, ref, rtol=rtol, atol=atol,
                  msg=f"[qpack B=8 QLEN={qlen} H=16 splits={num_splits} dtype={dtype} sink={use_sink}]")


@pytest.mark.parametrize("dtype", _PYTEST_DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("use_sink", _PYTEST_SINK)
@pytest.mark.parametrize("ctx", [300, 1024, 4096])
def test_mla_decode_opus_qpack_h8(dtype, use_sink, ctx):
    if _skip_if_unsupported(d=512):
        return
    from aiter.ops.mla_decode_opus import mla_decode_opus_qpack_h8
    seed = hash((ctx, str(dtype), use_sink)) & 0xFFFF
    inputs = _make_inputs(6, 8, 8, 512, ctx, dtype, seed=seed, use_sink=use_sink)
    softmax_scale = 1.0 / math.sqrt(512)
    ref = _ref_mla_decode_opus(**inputs, softmax_scale=softmax_scale, use_sink=use_sink)
    got = mla_decode_opus_qpack_h8(
        inputs["q"], inputs["unified_kv"], inputs["kv_indices"], inputs["kv_indptr"],
        inputs["attn_sink"], softmax_scale,
    )
    rtol, atol = _get_tolerances(dtype)
    checkAllclose(got, ref, rtol=rtol, atol=atol,
                  msg=f"[qpack_h8 B=6 QLEN=8 H=8 ctx={ctx} dtype={dtype} sink={use_sink}]")


parser = argparse.ArgumentParser(description="mla_decode_opus correctness + benchmark")
parser.add_argument("--bsz", type=int, nargs="*", default=[4])
parser.add_argument("--qlen", type=int, nargs="*", default=[1, 2, 4, 8])
parser.add_argument("--h_q", type=int, nargs="*", default=[16, 128])
parser.add_argument("--dtype", type=str, nargs="*", default=["bf16"], choices=["bf16", "fp16"])
parser.add_argument("--no-verify", action="store_true")
parser.add_argument("--no-bench", action="store_true")
parser.add_argument("--no-sink", action="store_true")
parser.add_argument("--seed", type=int, default=0)

_DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16}

if __name__ == "__main__":
    args = parser.parse_args()
    rows = []
    for bsz, qlen, h, dtype_str in itertools.product(
        args.bsz, args.qlen, args.h_q, args.dtype
    ):
        row = run_mla_decode_opus(
            bsz=bsz,
            qlen=qlen,
            h=h,
            d=512,
            total_pages=256,
            dtype=_DTYPE_MAP[dtype_str],
            use_sink=not args.no_sink,
            seed=args.seed,
            verify=not args.no_verify,
            bench=not args.no_bench,
        )
        if row:
            rows.append(row)
    if rows:
        print(rows)
    sys.exit(0)
