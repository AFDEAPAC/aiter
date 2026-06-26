# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Reference + harness for the RoPE (D=576) MLA decode variant.

Contract (matches aiter asm mla_decode_fwd):
  q  : [B, QLEN, H, 576]  bf16/fp16  (512 latent/nope || 64 rope, pre-rotated)
  kv : [total_pages, 576] same dtype (512 latent || 64 k_rope, pre-rotated)
  score = q . k over 576 (latent + rope)
  V     = kv[:, :512] (the latent); O = P . V over 512 -> out [B,QLEN,H,512]
  causal: position p attends first max(0, full_kv_len - QLEN + p) kv rows; optional per-head sink.

The kernel does NOT compute the rotation (applied upstream); it only handles the
asymmetric QK(576)/PV(512) contraction. This file is the executable contract; the
kernel (mla_decode_opus_rope) is verified against _ref_mla_decode_rope.
"""

from __future__ import annotations
import math
import torch

KV_LORA = 512
QK_ROPE = 64
QK_DIM = KV_LORA + QK_ROPE  # 576
V_DIM = KV_LORA             # 512


def _ref_mla_decode_rope(
    q: torch.Tensor,            # [B, QLEN, H, 576]
    kv: torch.Tensor,           # [pages, 576]
    kv_indices: torch.Tensor,   # [nnz] int32
    kv_indptr: torch.Tensor,    # [B+1] int32
    attn_sink,                  # [H] fp32 or None
    softmax_scale: float,
    *,
    use_sink: bool,
) -> torch.Tensor:
    bsz, qlen, h, d = q.shape
    assert d == QK_DIM, f"q last dim must be {QK_DIM}, got {d}"
    out = torch.zeros(bsz, qlen, h, V_DIM, dtype=q.dtype, device=q.device)
    q_f32 = q.to(torch.float32)
    kv_f32 = kv.to(torch.float32)
    indptr = kv_indptr.to(torch.int64).cpu().tolist()
    idx = kv_indices.to(torch.int64)

    for b in range(bsz):
        ps, pe = indptr[b], indptr[b + 1]
        row_len = pe - ps
        if row_len <= 0:
            continue
        kv_rows = kv_f32.index_select(0, idx[ps:pe])      # [row_len, 576]
        k_full = kv_rows                                  # QK uses all 576
        v_lat = kv_rows[:, :V_DIM]                        # V = latent 512
        for p in range(qlen):
            valid = max(0, row_len - qlen + p)
            if valid == 0:
                continue
            kf = k_full[:valid]
            vf = v_lat[:valid]
            scores = q_f32[b, p] @ kf.t() * softmax_scale  # [H, valid] over 576
            if use_sink and attn_sink is not None:
                sink_col = attn_sink.to(torch.float32).unsqueeze(1)
                sw = torch.cat([scores, sink_col], dim=1)
                m = sw.amax(dim=1, keepdim=True)
                e = torch.exp(scores - m)
                es = torch.exp(sink_col - m)
                denom = e.sum(dim=1, keepdim=True) + es
                out[b, p] = ((e / denom) @ vf).to(q.dtype)
            else:
                m = scores.amax(dim=1, keepdim=True)
                e = torch.exp(scores - m)
                out[b, p] = ((e / e.sum(dim=1, keepdim=True)) @ vf).to(q.dtype)
    return out


def make_rope_inputs(bsz, qlen, h, total_pages, dtype, *, device="cuda", seed=0, use_sink=True):
    torch.manual_seed(seed)
    dev = torch.device(device)
    q = (torch.randn(bsz, qlen, h, QK_DIM, device=dev, dtype=torch.float32) * 0.5).to(dtype)
    kv = (torch.randn(total_pages, QK_DIM, device=dev, dtype=torch.float32) * 0.5).to(dtype)
    kv_indptr = torch.arange(0, (bsz + 1) * total_pages, total_pages, dtype=torch.int32, device=dev)
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=dev).repeat(bsz)
    attn_sink = (torch.randn(h, device=dev, dtype=torch.float32) * 0.25) if use_sink else None
    return dict(q=q, kv=kv, kv_indices=kv_indices, kv_indptr=kv_indptr, attn_sink=attn_sink)


def _run_and_check(B, qlen, H, ctx, dtype, use_sink, num_splits=None, use_qpack=None, *, seed=0, atol=2e-2, rtol=2e-2):
    """Run the rope kernel and compare against the fp32 reference. Returns max|diff|."""
    import aiter  # noqa: F401
    from aiter.ops.mla_decode_opus import mla_decode_opus_rope

    inp = make_rope_inputs(B, qlen, H, ctx, dtype, device="cuda", seed=seed, use_sink=use_sink)
    scale = 1.0 / math.sqrt(QK_DIM)
    ref = _ref_mla_decode_rope(
        inp["q"], inp["kv"], inp["kv_indices"], inp["kv_indptr"], inp["attn_sink"],
        scale, use_sink=use_sink,
    )
    out = mla_decode_opus_rope(
        inp["q"], inp["kv"], inp["kv_indices"], inp["kv_indptr"], inp["attn_sink"],
        scale, num_splits=num_splits, use_qpack=use_qpack,
    )
    assert out.shape == ref.shape, f"shape {tuple(out.shape)} != {tuple(ref.shape)}"
    d = (out.float() - ref.float()).abs()
    return float(d.max()), float(d.mean())


try:
    import pytest

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("use_sink", [False, True])
    @pytest.mark.parametrize("qlen", [1, 2, 4])
    @pytest.mark.parametrize("H", [16, 8, 64, 128])
    @pytest.mark.parametrize("ctx", [48, 64, 200, 4096])
    @pytest.mark.parametrize("num_splits", [None, 1, 4])
    def test_rope_decode(dtype, use_sink, qlen, H, ctx, num_splits):
        if not torch.cuda.is_available():
            pytest.skip("no GPU")
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        if arch != "gfx950":
            pytest.skip(f"needs gfx950, got {arch}")
        mx, mn = _run_and_check(2, qlen, H, ctx, dtype, use_sink, num_splits=num_splits)
        assert mx < 3e-2, f"max|diff|={mx:.4g} mean={mn:.4g}"

    # qpack (H==16, qlen==4): force use_qpack to exercise the qlen-into-warps path,
    # both single (ns=1) and split (ns=2,4); short ctx -> le2, long -> pipelined.
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("use_sink", [False, True])
    @pytest.mark.parametrize("ctx", [40, 64, 200, 4096, 8192])
    @pytest.mark.parametrize("ns", [1, 2, 4])
    @pytest.mark.parametrize("B", [4, 64])
    def test_rope_qpack(dtype, use_sink, ctx, ns, B):
        if not torch.cuda.is_available():
            pytest.skip("no GPU")
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        if arch != "gfx950":
            pytest.skip(f"needs gfx950, got {arch}")
        mx, mn = _run_and_check(B, 4, 16, ctx, dtype, use_sink, num_splits=ns, use_qpack=True)
        assert mx < 3e-2, f"qpack max|diff|={mx:.4g} mean={mn:.4g}"
except ImportError:
    pass


if __name__ == "__main__":
    if not torch.cuda.is_available():
        inp = make_rope_inputs(2, 4, 16, 64, torch.float32, device="cpu", seed=0, use_sink=True)
        out = _ref_mla_decode_rope(**inp, softmax_scale=1.0 / math.sqrt(QK_DIM), use_sink=True)
        print("ref out shape:", tuple(out.shape), "mean|o|:", float(out.abs().mean()))
        raise SystemExit(0)

    print("=== RoPE MLA decode: kernel vs reference ===")
    cases = [
        # (B, qlen, H, ctx, dtype, use_sink, num_splits) — start single-tile then grow
        (2, 1, 16, 48, torch.bfloat16, False, 1),
        (2, 1, 16, 64, torch.bfloat16, True, 1),
        (2, 1, 16, 200, torch.bfloat16, True, 1),
        (2, 4, 16, 4096, torch.bfloat16, True, 1),
        (2, 2, 16, 300, torch.float16, True, 1),
        (2, 1, 8, 200, torch.bfloat16, True, 1),
        # H>32 (16mx8, NUM_WARPS=4): single-tile, multi-tile, split
        (2, 1, 128, 64, torch.bfloat16, True, 1),
        (2, 1, 128, 200, torch.bfloat16, True, 1),
        (2, 1, 128, 4096, torch.bfloat16, True, 1),
        (2, 4, 128, 4096, torch.bfloat16, True, 1),
        (2, 2, 64, 300, torch.float16, True, 1),
        (2, 1, 128, 4096, torch.bfloat16, True, 4),
        # split-KV
        (2, 1, 16, 4096, torch.bfloat16, True, 4),
        (4, 4, 16, 4096, torch.bfloat16, True, 8),
        # auto num_splits
        (8, 1, 16, 4096, torch.bfloat16, True, None),
    ]
    worst = 0.0
    for (B, qlen, H, ctx, dt, sink, ns) in cases:
        mx, mn = _run_and_check(B, qlen, H, ctx, dt, sink, num_splits=ns)
        worst = max(worst, mx)
        tag = "OK " if mx < 3e-2 else "FAIL"
        print(f"[{tag}] B={B} qlen={qlen} H={H} ctx={ctx} {str(dt).split('.')[-1]:8s} "
              f"sink={int(sink)} ns={ns}: max|d|={mx:.4g} mean={mn:.4g}")
    print(f"worst max|diff| = {worst:.4g}  ->  {'PASS' if worst < 3e-2 else 'FAIL'}")
