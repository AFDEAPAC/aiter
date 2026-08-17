# SPDX-License-Identifier: Apache-2.0

"""Monolithic a16w4 (bf16 A x MXFP4 W) SiTUv2 fused MoE for gfx942.

Runs the a16wmix gemm1/gemm2 pair as one unit instead of through aiter's
two-stage plumbing. That plumbing hands stage1 -> stage2 a ``(token, slot)``
indexed intermediate of ``token_num * topk`` rows, while this kernel pair both
writes and reads a sorted-position ``[sorted_size, D_INTER]`` buffer, and
``sorted_size`` exceeds ``token_num * topk`` because each expert's run is padded
up to ``block_m``. Reshaping either side would mean re-addressing the gemm1
epilogue and the gemm2 A-gather, so the pair is kept intact and driven directly.

gfx942 only: on gfx950 the stock mixed_moe a16w4 path already applies.
"""

import logging

import torch

from .moe_a16wmix_host import flydsl_a16w4_gemm1, flydsl_a16w4_gemm2

logger = logging.getLogger(__name__)
_ANNOUNCED = False


def _pick_tile(dim, candidates=(256, 128, 64)):
    """Largest supported tile that divides ``dim``.

    Both stages assert ``dim % TILE == 0``. Kimi-K3 at TP=8 shards inter_dim to 384,
    which 256 does not divide, so the tile cannot be fixed at the 256 default.
    """
    for t in candidates:
        if dim % t == 0:
            return t
    raise ValueError(f"no tile in {candidates} divides {dim}")


def fused_moe_a16wmix(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    situ_beta: float = 1.0,
    situ_linear_beta: float = 1.0,
    act: str = "situv2",
    block_m: int = 32,
    w1_layout: str = "standard",
    tile_m: int | None = None,
    out: torch.Tensor | None = None,
):
    """bf16 hidden_states x MXFP4 experts -> bf16 ``[tokens, model_dim]``.

    ``w1``/``w2`` and their e8m0 scales must already be preshuffled by the caller
    (vLLM does this once at load time).     ``w1_layout`` selects which preshuffle the
    stage1 kernel decodes: ``"standard"`` for the separated GGUU layout,
    ``"guinterleave"`` for the gate/up-interleaved GUGU one. Stage2 takes no such
    mode: its gate_up=False layout is byte-identical to standard whenever
    ``experts * model_dim % 256 == 0``.
    """
    from aiter.fused_moe import moe_sorting

    tokens, model_dim = hidden_states.shape
    experts = w1.shape[0]
    topk = topk_ids.shape[1]
    # Preshuffling reorders w2's dims, and fp4 packs two values per byte, so recover
    # inter_dim from the byte count rather than from a shape that the layout controls.
    inter_dim = int(w2.view(torch.uint8).numel() * 2 // (experts * model_dim))
    bm = int(tile_m if tile_m is not None else block_m)

    global _ANNOUNCED
    if not _ANNOUNCED:
        _ANNOUNCED = True
        logger.warning(
            "A16WMIX_ACTIVE gfx942 a16w4 SiTUv2 fused MoE: E=%d model_dim=%d "
            "inter_dim=%d topk=%d block_m=%d act=%s w1_layout=%s",
            experts, model_dim, inter_dim, topk, bm, act, w1_layout,
        )

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids.to(torch.int32),
        topk_weights.to(torch.float32),
        experts,
        model_dim,
        hidden_states.dtype,
        bm,
    )
    # Some builds return num_valid_ids as [2] (padded total, logical total).
    if num_valid_ids.numel() > 1:
        num_valid_ids = num_valid_ids[:1].contiguous()

    sorted_size = int(sorted_ids.numel())
    inter_sorted = torch.zeros(
        sorted_size, inter_dim, dtype=torch.bfloat16, device=hidden_states.device
    )

    flydsl_a16w4_gemm1(
        a_bf16=hidden_states.to(torch.bfloat16).contiguous(),
        w1_u8=w1.view(torch.uint8),
        w1_scale_u8=w1_scale.view(torch.uint8),
        sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=num_valid_ids.to(torch.int32).contiguous(),
        m_indices=sorted_ids.to(torch.int32).contiguous(),
        inter_sorted_bf16=inter_sorted,
        n_tokens=tokens,
        NE=experts,
        D_HIDDEN=model_dim,
        D_INTER=inter_dim,
        topk=topk,
        tile_m=bm,
        tile_n=None,
        tile_k=_pick_tile(model_dim),
        act=act,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        w_dtype="mxfp4",
        w_layout=w1_layout,
        use_csv_config=True,
    )

    flat_out = (
        moe_buf.view(-1)
        if out is None
        else out.view(-1)
    )
    flat_out.zero_()
    flydsl_a16w4_gemm2(
        inter_sorted_bf16=inter_sorted,
        w2_u8=w2.view(torch.uint8),
        w2_scale_u8=w2_scale.view(torch.uint8),
        sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=num_valid_ids.to(torch.int32).contiguous(),
        sorted_token_ids=sorted_ids.to(torch.int32).contiguous(),
        sorted_weights=sorted_weights,
        flat_out=flat_out,
        M_logical=tokens,
        max_sorted=sorted_size,
        NE=experts,
        D_HIDDEN=model_dim,
        D_INTER=inter_dim,
        topk=topk,
        tile_m=bm,
        tile_n=_pick_tile(model_dim),
        tile_k=_pick_tile(inter_dim),
        w_dtype="mxfp4",
        use_csv_config=True,
    )
    return flat_out.view(tokens, model_dim)
