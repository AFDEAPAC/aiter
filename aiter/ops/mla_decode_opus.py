# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""OPUS-based MLA absorbed decode (D=512) on gfx950.

Single paged ``unified_kv`` source with CSR ``kv_indptr`` / ``kv_indices``,
``qlen`` speculative query positions per batch row (1..17), causal prefix
length ``valid_kv(p) = max(0, full_kv_len - qlen + p)`` per position ``p``.

See ``aiter/csrc/include/mla_decode_opus.h`` for the C++ API.
"""

import torch
from typing import Optional

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx_runtime
from ..jit.utils.torch_guard import torch_compile_guard

MD_NAME = "module_mla_decode_opus"


@compile_ops("module_mla_decode_opus", develop=True)
def mla_decode_opus_fwd(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
    qlen: int,
) -> None: ...


def _mla_decode_opus_fake(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    qlen: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return out if out is not None else torch.empty_like(q)


@compile_ops("module_mla_decode_opus", develop=True)
def mla_decode_opus_splitkv_fwd(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
    partial_o: torch.Tensor,
    partial_ml: torch.Tensor,
    softmax_scale: float,
    qlen: int,
    num_splits: int,
) -> None: ...


@compile_ops("module_mla_decode_opus", develop=True)
def mla_decode_opus_qpack_fwd(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
    qlen: int,
) -> None: ...


@compile_ops("module_mla_decode_opus", develop=True)
def mla_decode_opus_qpack_splitkv_fwd(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
    partial_o: torch.Tensor,
    partial_ml: torch.Tensor,
    softmax_scale: float,
    qlen: int,
    num_splits: int,
) -> None: ...


@compile_ops("module_mla_decode_opus", develop=True)
def mla_decode_opus_rope_fwd(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
    qlen: int,
) -> None: ...


@compile_ops("module_mla_decode_opus", develop=True)
def mla_decode_opus_rope_splitkv_fwd(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
    partial_o: torch.Tensor,
    partial_ml: torch.Tensor,
    softmax_scale: float,
    qlen: int,
    num_splits: int,
) -> None: ...


@compile_ops("module_mla_decode_opus", develop=True)
def mla_decode_opus_rope_qpack_fwd(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
    qlen: int,
) -> None: ...


@compile_ops("module_mla_decode_opus", develop=True)
def mla_decode_opus_rope_qpack_splitkv_fwd(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
    partial_o: torch.Tensor,
    partial_ml: torch.Tensor,
    softmax_scale: float,
    qlen: int,
    num_splits: int,
) -> None: ...


def mla_decode_opus_workspace(B, qlen, H, D, num_splits, device, dtype=torch.float32):
    """Allocate the split-KV / qpack-split partial-O and partial-(m,l) workspace.

    Allocate once for the worst-case (max B, max num_splits) and pass it back into
    ``mla_decode_opus_splitkv`` / ``mla_decode_opus_qpack`` on the hot path -- the
    fwd accepts any buffer with ``>=`` the needed rows, so a cached buffer avoids a
    per-call ``torch.empty`` and the ``num_splits`` heuristic's GPU sync.
    """
    rows = B * qlen * num_splits
    partial_o = torch.empty((rows, H, D), dtype=dtype, device=device)
    partial_ml = torch.empty((rows, H, 2), dtype=dtype, device=device)
    return partial_o, partial_ml


@compile_ops("module_mla_decode_opus", develop=True)
def mla_decode_opus_qpack_h8_fwd(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
    qlen: int,
) -> None: ...


def mla_decode_opus_qpack_h8(
    q, unified_kv, kv_indices, kv_indptr, attn_sink, softmax_scale, out=None
):
    """qpack for H=8 (TP16): warp owns 2 positions (16 rows = 2 pos x 8 heads,
    no padding). qlen==8 only. Shares KV across positions like H=16 qpack."""
    if out is None:
        out = torch.empty_like(q)
    sink = attn_sink if attn_sink is not None else torch.empty(0, dtype=torch.float32, device=q.device)
    mla_decode_opus_qpack_h8_fwd(
        q, unified_kv, kv_indices, kv_indptr, sink, out, float(softmax_scale), int(q.size(1))
    )
    return out


def mla_decode_opus_qpack(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
    softmax_scale: float,
    out: Optional[torch.Tensor] = None,
    num_splits: Optional[int] = None,
    partial_o: Optional[torch.Tensor] = None,
    partial_ml: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """qlen-packed MLA decode (H==16, qlen in {4,8}): each warp owns one
    speculative position's 16 heads and shares the KV LDS tile, reusing the
    efficient 16mx8 layout to amortize the KV load across qlen positions.
    Adds split-KV (num_splits) to fill the GPU at small/mid batch.

    Hot path (zero alloc / zero sync): pass an explicit ``num_splits`` and a cached
    ``partial_o``/``partial_ml`` (see ``mla_decode_opus_workspace``)."""
    if out is None:
        out = torch.empty_like(q)
    B, qlen, H, D = q.shape
    sink = attn_sink if attn_sink is not None else torch.empty(0, dtype=torch.float32, device=q.device)
    if num_splits is None:
        # qpack block count = B; one warp per position holds the qlen parallelism,
        # so split the KV dim to fill the CUs. qlen=4 uses the single-buffer kernel
        # (1x LDS -> 2 blocks/CU); qlen=8 uses the pipelined 4x (1 block/CU).
        # This branch does a GPU sync (.item()); pass num_splits to skip it.
        kv_len = int((kv_indptr[1:] - kv_indptr[:-1]).max().item()) if kv_indptr.numel() > 1 else 0
        num_splits = _pick_num_splits(B, 1, 1, kv_len, blocks_per_cu=1)
    if num_splits <= 1:
        mla_decode_opus_qpack_fwd(q, unified_kv, kv_indices, kv_indptr, sink, out, float(softmax_scale), int(qlen))
        return out
    if partial_o is None or partial_ml is None:
        partial_o, partial_ml = mla_decode_opus_workspace(B, qlen, H, D, num_splits, q.device)
    mla_decode_opus_qpack_splitkv_fwd(
        q, unified_kv, kv_indices, kv_indptr, sink, out, partial_o, partial_ml,
        float(softmax_scale), int(qlen), int(num_splits),
    )
    return out


def _qpack_min_b(qlen: int, kv_len: int) -> int:
    """Min batch at which qpack beats split-KV (H==16). The crossover drops as KV
    grows (qpack shares 1 KV read across qlen positions, so its saving scales with
    KV length). Coarse buckets fit the measured crossovers on gfx950 (ctx 1024 /
    4096 / 8192 / >=16384): qlen=4 -> 48/24/24/16; qlen=8 -> 64/48/16."""
    if qlen == 4:
        if kv_len < 2048:
            return 48
        if kv_len < 16384:
            return 24
        return 16  # long ctx: qpack's 1-KV-read saving wins at B=16 (split->qpack 1.3-1.5x, gfx950)
    if qlen == 8:
        if kv_len < 2048:
            return 64
        if kv_len < 8192:
            return 48
        return 16
    return 1 << 30  # qpack only defined for qlen in {4,8}


def _pick_num_splits(
    B: int, qlen: int, num_h_blocks: int, kv_len: int, num_cu: int = 256,
    blocks_per_cu: int = 1,
) -> int:
    """Heuristic: split KV to fill the GPU, bounded by a per-split work floor.

    From a dense gfx950 sweep (nhead x B x qlen x ctx), the optimal split count is the
    min of: (a) blocks needed to fill the device, (b) a per-split work floor, (c) the
    kernel's hard max of 64. Both (a) and (b) scale with ``W = num_cu * blocks_per_cu``
    -- the heavier 16mx8 (blocks_per_cu=1 -> W=256) amortizes the stage-2 reduction
    ~2x better than the light 16mx1 (blocks_per_cu=2 -> W=512), so it tolerates ~2x
    more splits, and longer ctx affords proportionally more:

        ns = min(floor(W / base), kv_len // W, 64),   base = B * qlen * num_h_blocks

    ``base`` folds qlen/heads in cleanly (verified: same optimum at equal ``base``
    across qlen in {1,2,4}). This single trend replaces the earlier ad-hoc caps.

    ``fill`` uses *floor* (not ceil): keeping ``base*ns <= W`` confines the launch to
    one resident wave. Overshooting W by even a few blocks spills a near-empty 2nd
    wave whose tail ~doubles latency (measured: nh128 base10 L=64K ns24=315us vs
    ns26=tail; ceil would pick 26). floor==ceil whenever base divides W (the clean
    cases), so this only helps the non-dividing (e.g. odd-batch) ones.
    """
    W = num_cu * max(1, blocks_per_cu)
    base = max(1, B * qlen * num_h_blocks)
    fill = max(1, W // base)            # blocks/ns to stay within one resident wave
    work_floor = max(1, kv_len // W)     # keep >= ~W KV tokens of work per split
    return max(1, min(fill, work_floor, 64))


@torch_compile_guard(mutates_args=["out"], gen_fake=_mla_decode_opus_fake)
def mla_decode_opus(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
    softmax_scale: float,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MLA decode with absorbed head dim D=512 (gfx950 OPUS).

    Args:
      q: ``[B, QLEN, H, D]`` bf16/fp16.
      unified_kv: ``[total_pages, D]`` same dtype as ``q``.
      kv_indices: ``[nnz]`` int32 row indices into ``unified_kv``.
      kv_indptr: ``[B+1]`` int32 CSR row pointers.
      attn_sink: ``[H]`` fp32 per-head softmax denominator bias, or ``None`` to disable.
      softmax_scale: Scalar (no implicit ``1/sqrt(D)``).
      out: Optional ``[B, QLEN, H, D]`` buffer.

    Returns:
      ``out`` (same shape/dtype as ``q``).
    """
    gfx = get_gfx_runtime()
    if gfx != "gfx950":
        raise RuntimeError(f"mla_decode_opus requires gfx950, got {gfx}")

    if q.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"mla_decode_opus expects fp16/bf16 q, got {q.dtype}")
    if unified_kv.dtype != q.dtype:
        raise RuntimeError(
            f"unified_kv dtype mismatch: unified_kv={unified_kv.dtype}, q={q.dtype}"
        )

    qlen = int(q.size(1))
    if qlen < 1 or qlen > 17:
        raise RuntimeError(f"mla_decode_opus requires 1 <= QLEN <= 17, got {qlen}")

    if out is None:
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.dtype != q.dtype:
        raise RuntimeError(
            f"out shape/dtype mismatch: got shape={tuple(out.shape)} dtype={out.dtype}, "
            f"expected shape={tuple(q.shape)} dtype={q.dtype}"
        )

    sink = attn_sink if attn_sink is not None else torch.empty(0, dtype=torch.float32, device=q.device)

    mla_decode_opus_fwd(
        q,
        unified_kv,
        kv_indices,
        kv_indptr,
        sink,
        out,
        float(softmax_scale),
        qlen,
    )
    return out


def mla_decode_opus_splitkv(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
    softmax_scale: float,
    out: Optional[torch.Tensor] = None,
    num_splits: Optional[int] = None,
    partial_o: Optional[torch.Tensor] = None,
    partial_ml: Optional[torch.Tensor] = None,
    use_qpack: Optional[bool] = None,
) -> torch.Tensor:
    """Split-KV (flash-decode) MLA decode: parallelizes the per-block serial KV
    walk across ``num_splits`` blocks + a stage-2 reduction. Big win when the
    batch under-fills the GPU. ``num_splits=None`` auto-picks from B/qlen/kv_len.

    Hot path (zero alloc / zero sync): pass explicit ``num_splits`` + cached
    ``partial_o``/``partial_ml`` (``mla_decode_opus_workspace``), and ``use_qpack``
    to pin the kernel choice so the auto-dispatch's ``B*qlen`` test still applies
    without the heuristic's GPU sync.
    """
    gfx = get_gfx_runtime()
    if gfx != "gfx950":
        raise RuntimeError(f"mla_decode_opus requires gfx950, got {gfx}")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"mla_decode_opus expects fp16/bf16 q, got {q.dtype}")

    B, qlen, H, D = q.shape
    if out is None:
        out = torch.empty_like(q)

    # Size dispatch between the two complementary kernels (H==16, qlen in {4,8}):
    #   - split-KV: qlen on the grid (blocks = B*qlen*splits) -> best parallelism
    #     when the GPU is starved (very small batch).
    #   - qpack:    qlen in the warps (1 KV read shared by all qlen positions) ->
    #     best once there are enough batch rows that cutting KV traffic dominates.
    # The crossover B is per-qlen AND ctx-dependent (longer KV -> qpack's traffic
    # saving wins at smaller B). Measured on gfx950 (see _qpack_min_b). With this
    # routing qpack ties/beats the asm decode at large batch.
    # On the auto path we reuse the single .item() (also needed by the split
    # heuristic) for the ctx-aware threshold -> no extra sync; the hot path passes
    # use_qpack explicitly so it stays sync-free.
    # H=8 qlen=8: qpack-h8 (2 positions/warp, no padding) beats split-KV at large
    # batch (measured crossover ~B>=96, ctx=4096); small batch stays on split-KV.
    if num_splits is None and H == 8 and qlen == 8 and B >= 96:
        return mla_decode_opus_qpack_h8(
            q, unified_kv, kv_indices, kv_indptr, attn_sink, softmax_scale, out=out
        )

    num_h_blocks = max(1, H // 128) if H > 32 else 1
    blocks_per_cu = 2 if H <= 32 else 1  # 16mx1 fits 2 blocks/CU; 16mx8 LDS-capped at 1
    kv_len = None
    if (use_qpack is None or num_splits is None):
        kv_len = int((kv_indptr[1:] - kv_indptr[:-1]).max().item()) if kv_indptr.numel() > 1 else 0
    if use_qpack is None:
        use_qpack = (H == 16 and qlen in (4, 8) and B >= _qpack_min_b(qlen, kv_len))
    if use_qpack:
        qp_splits = num_splits
        if qp_splits is None:
            qp_splits = _pick_num_splits(B, 1, 1, kv_len, blocks_per_cu=1)
        return mla_decode_opus_qpack(
            q, unified_kv, kv_indices, kv_indptr, attn_sink, softmax_scale,
            out=out, num_splits=qp_splits, partial_o=partial_o, partial_ml=partial_ml,
        )

    if num_splits is None:
        num_splits = _pick_num_splits(B, qlen, num_h_blocks, kv_len, blocks_per_cu=blocks_per_cu)

    if num_splits <= 1:
        return mla_decode_opus(
            q, unified_kv, kv_indices, kv_indptr, attn_sink, softmax_scale, out=out
        )

    if partial_o is None or partial_ml is None:
        partial_o, partial_ml = mla_decode_opus_workspace(B, qlen, H, D, num_splits, q.device)
    sink = attn_sink if attn_sink is not None else torch.empty(0, dtype=torch.float32, device=q.device)

    mla_decode_opus_splitkv_fwd(
        q,
        unified_kv,
        kv_indices,
        kv_indptr,
        sink,
        out,
        partial_o,
        partial_ml,
        float(softmax_scale),
        int(qlen),
        int(num_splits),
    )
    return out


_ROPE_QK_DIM = 576  # latent 512 + rope 64
_ROPE_V_DIM = 512   # latent (V / output)


def mla_decode_opus_rope(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
    softmax_scale: float,
    out: Optional[torch.Tensor] = None,
    num_splits: Optional[int] = None,
    partial_o: Optional[torch.Tensor] = None,
    partial_ml: Optional[torch.Tensor] = None,
    use_qpack: Optional[bool] = None,
) -> torch.Tensor:
    """RoPE (D=576) MLA decode (H<=32 path).

    ``q``/``unified_kv`` carry the full 576 dims (latent 512 + rope 64, already
    rotated upstream); the score contracts over all 576 while V/output use the
    latent 512 -> ``out`` is ``[B, QLEN, H, 512]``.

    Routing (H==16, qlen==4): split-KV at small batch, **qpack** (qlen-into-warps,
    one shared KV read across the 4 positions) once batch fills the CUs -- this
    closes the nhead=16 qlen=4 large-batch loss vs the asm ``m16x4`` decode.

    ``num_splits=None`` auto-picks a flash-decode split (one GPU sync via
    ``.item()``); pass an explicit ``num_splits`` + cached ``partial_o``/
    ``partial_ml`` (see ``mla_decode_opus_workspace``) + ``use_qpack`` for a
    sync-free hot path.
    """
    gfx = get_gfx_runtime()
    if gfx != "gfx950":
        raise RuntimeError(f"mla_decode_opus requires gfx950, got {gfx}")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"mla_decode_opus expects fp16/bf16 q, got {q.dtype}")
    if q.size(-1) != _ROPE_QK_DIM or unified_kv.size(-1) != _ROPE_QK_DIM:
        raise RuntimeError(
            f"RoPE path needs q/unified_kv last dim {_ROPE_QK_DIM}, got "
            f"q={q.size(-1)}, kv={unified_kv.size(-1)}"
        )

    B, qlen, H, _ = q.shape
    if out is None:
        out = torch.empty((B, qlen, H, _ROPE_V_DIM), dtype=q.dtype, device=q.device)
    sink = attn_sink if attn_sink is not None else torch.empty(0, dtype=torch.float32, device=q.device)
    # H<=32 -> 16mx1 (2 blocks/CU); H>32 -> 16mx8 (LDS-capped at 1 block/CU).
    _blocks_per_cu = 2 if H <= 32 else 1

    # qpack (shares 1 KV read across qlen positions) is built for: H==16 qlen==4, and
    # H==32 qlen in {2,4} (2 warps/position; qlen=4 -> NW8 le2). H=32 qpack beats the
    # 16mx1 re-read path at every batch for qlen>=2, so route it whenever eligible.
    qpack_eligible = (H == 16 and qlen == 4) or (H == 32 and qlen in (2, 4))
    kv_len = None
    if qpack_eligible and (use_qpack is None or num_splits is None):
        kv_len = int((kv_indptr[1:] - kv_indptr[:-1]).max().item()) if kv_indptr.numel() > 1 else 0
    if use_qpack is None:
        if H == 32:
            use_qpack = qpack_eligible           # always wins for qlen>=2
        else:
            use_qpack = qpack_eligible and B >= _qpack_min_b(qlen, kv_len if kv_len is not None else 0)

    if use_qpack:
        qp_splits = num_splits
        if qp_splits is None:
            qp_splits = _pick_num_splits(B, 1, 1, kv_len, blocks_per_cu=1)
        if qp_splits <= 1:
            mla_decode_opus_rope_qpack_fwd(
                q, unified_kv, kv_indices, kv_indptr, sink, out, float(softmax_scale), int(qlen)
            )
            return out
        if partial_o is None or partial_ml is None:
            partial_o, partial_ml = mla_decode_opus_workspace(
                B, qlen, H, _ROPE_V_DIM, qp_splits, q.device
            )
        mla_decode_opus_rope_qpack_splitkv_fwd(
            q, unified_kv, kv_indices, kv_indptr, sink, out, partial_o, partial_ml,
            float(softmax_scale), int(qlen), int(qp_splits),
        )
        return out

    if num_splits is None:
        if kv_len is None:
            kv_len = int((kv_indptr[1:] - kv_indptr[:-1]).max().item()) if kv_indptr.numel() > 1 else 0
        # heads/block depends on the variant: 16mx1 packs 16 (H<=32), 16mx8 NW8 packs
        # 128 (H%128==0), 16mx8 NW4 packs 64 (otherwise).
        heads_per_block = 16 if H <= 32 else (128 if H % 128 == 0 else 64)
        num_h_blocks = max(1, -(-H // heads_per_block))
        num_splits = _pick_num_splits(B, qlen, num_h_blocks, kv_len, blocks_per_cu=_blocks_per_cu)

    if num_splits <= 1:
        mla_decode_opus_rope_fwd(
            q, unified_kv, kv_indices, kv_indptr, sink, out, float(softmax_scale), int(qlen)
        )
        return out

    if partial_o is None or partial_ml is None:
        partial_o, partial_ml = mla_decode_opus_workspace(
            B, qlen, H, _ROPE_V_DIM, num_splits, q.device
        )
    mla_decode_opus_rope_splitkv_fwd(
        q, unified_kv, kv_indices, kv_indptr, sink, out, partial_o, partial_ml,
        float(softmax_scale), int(qlen), int(num_splits),
    )
    return out


__all__ = [
    "mla_decode_opus_fwd",
    "mla_decode_opus_splitkv_fwd",
    "mla_decode_opus",
    "mla_decode_opus_splitkv",
    "mla_decode_opus_qpack",
    "mla_decode_opus_qpack_fwd",
    "mla_decode_opus_qpack_splitkv_fwd",
    "mla_decode_opus_qpack_h8",
    "mla_decode_opus_qpack_h8_fwd",
    "mla_decode_opus_workspace",
    "mla_decode_opus_rope",
    "mla_decode_opus_rope_fwd",
    "mla_decode_opus_rope_splitkv_fwd",
    "mla_decode_opus_rope_qpack_fwd",
    "mla_decode_opus_rope_qpack_splitkv_fwd",
]
