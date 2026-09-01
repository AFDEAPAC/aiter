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
import os

import torch

from .moe_a16wmix_host import flydsl_a16w4_gemm1, flydsl_a16w4_gemm2, _resolve_ours_tuned

logger = logging.getLogger(__name__)
_ANNOUNCED = False
_FP8_FALLBACK_WARNED = False
_FP8_S2_FALLBACK_WARNED = False


def _fp8_enabled():
    """``AITER_A16WMIX_FP8=1``: run stage1 on fp8 MFMA instead of bf16.

    Off by default. Read per call rather than at import so a test can flip arms in one
    process; the kernel is cached per (a_dtype, tiles) so each arm costs one JIT
    compile. NOTE: an A/B in one process also needs FLYDSL_RUNTIME_ENABLE_CACHE=0 only
    if the kernel *name* collides -- it does not here, a_dtype is in the name.
    """
    return os.environ.get("AITER_A16WMIX_FP8", "0") not in ("0", "", "false", "False")


def _fp8_s2_enabled():
    """``AITER_A16WMIX_FP8_S2=1``: also run stage2 on fp8 MFMA.

    Separate from the stage1 gate on purpose. Stage2 quantizes the *intermediate*, so it
    stacks a second quantization on top of the activation's and is the more likely of
    the two to cost accuracy -- keeping the gates independent makes "stage1 only" and
    "both stages" separately measurable instead of one all-or-nothing switch.
    """
    return os.environ.get("AITER_A16WMIX_FP8_S2", "0") not in ("0", "", "false", "False")


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
    expert_mask: torch.Tensor | None = None,
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
    _csv = _resolve_ours_tuned(
        w_dtype="mxfp4",
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tokens=tokens,
        stage=1,
    )
    if tile_m is None and _csv is not None:
        bm = int(_csv["tile_m"])

    global _ANNOUNCED
    if not _ANNOUNCED:
        _ANNOUNCED = True
        # a_dtype is in this line so an A/B can prove which arm it measured from the
        # server log alone, instead of trusting that the env var reached the workers.
        logger.warning(
            "A16WMIX_ACTIVE gfx942 a16w4 SiTUv2 fused MoE: E=%d model_dim=%d "
            "inter_dim=%d topk=%d block_m=%d act=%s w1_layout=%s stage1_a=%s stage2_a=%s",
            experts, model_dim, inter_dim, topk, bm, act, w1_layout,
            "fp8" if _fp8_enabled() else "bf16",
            "fp8" if _fp8_s2_enabled() else "bf16",
        )

    # EP weights contain only local experts, while topk_ids and expert_mask use
    # global expert IDs.  The sorter iterates over global IDs and uses the mask
    # prefix sum to emit local weight indices in sorted_expert_ids.
    sorting_experts = experts if expert_mask is None else int(expert_mask.numel())
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids.to(torch.int32),
        topk_weights.to(torch.float32),
        sorting_experts,
        model_dim,
        hidden_states.dtype,
        bm,
        expert_mask=expert_mask,
    )
    # Some builds return num_valid_ids as [2] (padded total, logical total).
    if num_valid_ids.numel() > 1:
        num_valid_ids = num_valid_ids[:1].contiguous()

    if expert_mask is not None:
        # Sort globally but launch GEMMs only over local expert capacity.
        local_max_sorted = int(topk_ids.numel() + experts * bm - topk)
        local_max_blocks = (local_max_sorted + bm - 1) // bm
        local_max_sorted = local_max_blocks * bm
        sorted_ids = sorted_ids[:local_max_sorted]
        sorted_weights = sorted_weights[:local_max_sorted]
        sorted_expert_ids = sorted_expert_ids[:local_max_blocks]

    if expert_mask is not None:
        # The global-E sorter allocates for every global expert.  The GEMM
        # launchers compile with local E and derive their grid from these tensor
        # lengths; handing them the global-E capacity launches hundreds of
        # uninitialised expert blocks and can read beyond local weight tensors.
        # Masking guarantees the padded valid count fits the local-E bound.
        local_max_sorted = int(topk_ids.numel() + experts * bm - topk)
        local_max_blocks = (local_max_sorted + bm - 1) // bm
        local_max_sorted = local_max_blocks * bm
        sorted_ids = sorted_ids[:local_max_sorted]
        sorted_weights = sorted_weights[:local_max_sorted]
        sorted_expert_ids = sorted_expert_ids[:local_max_blocks]

    if os.environ.get("AITER_A16WMIX_DEBUG", "0") == "1" and not globals().get(
        "_A16WMIX_DEBUG_DUMP", False
    ):
        globals()["_A16WMIX_DEBUG_DUMP"] = True
        logger.warning(
            "A16WMIX_ARGS tokens=%s model_dim=%s inter_dim=%s E=%s topk=%s bm=%s "
            "hs=%s/%s w1=%s/%s w2=%s/%s w1s=%s/%s w2s=%s/%s "
            "topk_ids=%s/%s min=%s max=%s topk_w=%s/%s "
            "sorted_ids=%s num_valid=%s(%s) sorted_expert_ids=%s max_eid=%s "
            "moe_buf=%s expert_mask=%s",
            tokens, model_dim, inter_dim, experts, topk, bm,
            tuple(hidden_states.shape), hidden_states.dtype,
            tuple(w1.shape), w1.dtype, tuple(w2.shape), w2.dtype,
            tuple(w1_scale.shape), w1_scale.dtype,
            tuple(w2_scale.shape), w2_scale.dtype,
            tuple(topk_ids.shape), topk_ids.dtype,
            int(topk_ids.min().item()), int(topk_ids.max().item()),
            tuple(topk_weights.shape), topk_weights.dtype,
            tuple(sorted_ids.shape),
            num_valid_ids.tolist(), tuple(num_valid_ids.shape),
            tuple(sorted_expert_ids.shape),
            int(sorted_expert_ids.max().item()),
            tuple(moe_buf.shape),
            None if expert_mask is None else tuple(expert_mask.shape),
        )

    sorted_size = int(sorted_ids.numel())
    # empty, not zeros: no reader can act on an unwritten row. gemm1 stores under
    # ``mask=valid`` so it does leave padding rows untouched, but gemm2 clamps its block
    # count to ``cumsum0 // BM`` and its epilogue drops rows on ``token_id >= i32_M``, and
    # the stage2 quant is bounded by the same ``num_valid_ids``. So an untouched row can
    # only reach its own accumulator, which is then dropped. Zeroing the whole sorted_size
    # buffer cost one ~6.6 us fill per layer per rank at decode.
    inter_sorted = torch.empty(
        sorted_size, inter_dim, dtype=torch.bfloat16, device=hidden_states.device
    )

    # fp8 stage1 (AITER_A16WMIX_FP8=1): quantize the activation per token here rather
    # than upstream, so the vLLM/SGLang dispatch and the weight-prep contract are both
    # unchanged -- switching arms is one env var plus a restart. The weight and its
    # e8m0 scale are untouched; only the per-column reference exponent is derived.
    _a_dtype, _a_fp8, _a_scale, _w1_sref = "bf16", None, None, None
    _s1_fp8, _s2_fp8 = _fp8_enabled(), _fp8_s2_enabled()
    if _s1_fp8 or _s2_fp8:
        # Imported lazily: the bf16 path must not pay for, or depend on, any of this.
        from aiter import dtypes
        from aiter.ops.flydsl.a16wmix_fp8_prep import get_sref_u8
        from aiter.ops.quant import per_token_quant_hip
        from aiter.ops.shuffle import shuffle_scale_a16w4
        from aiter.utility.fp4_utils import e8m0_shuffle
    if _s1_fp8:
        # Kimi-K3's own e8m0 residuals stop at -3, which is exactly the last table the
        # fp8 decode carries (0.0001% of a billion groups reach it), so a differently
        # trained checkpoint could step past the end. Degrade to bf16 rather than take
        # the server down, and say so once and loudly -- a silent 2.2x loss on stage1
        # is the kind of thing that goes unnoticed for a month.
        try:
            _w1_sref = get_sref_u8(
                w1_scale.view(torch.uint8),
                experts=experts,
                n_out=2 * inter_dim,
                k_groups=model_dim // 32,
                shuffle_fn=lambda t, e: shuffle_scale_a16w4(
                    t, e, w1_layout == "guinterleave"
                ),
            )
        except ValueError as exc:
            global _FP8_FALLBACK_WARNED
            if not _FP8_FALLBACK_WARNED:
                _FP8_FALLBACK_WARNED = True
                logger.error(
                    "A16WMIX_FP8 disabled, falling back to the bf16 stage1: %s", exc
                )
        else:
            _a_fp8, _a_scale = per_token_quant_hip(
                hidden_states.to(torch.bfloat16).contiguous(), quant_dtype=dtypes.fp8
            )
            _a_dtype = "fp8"

    flydsl_a16w4_gemm1(
        a_bf16=(
            _a_fp8.view(torch.uint8)
            if _a_dtype == "fp8"
            else hidden_states.to(torch.bfloat16).contiguous()
        ),
        a_dtype=_a_dtype,
        a_scale=_a_scale,
        w1_sref=_w1_sref,
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
        tile_k=256,
        act=act,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        w_dtype="mxfp4",
        w_layout=w1_layout,
        use_csv_config=False,
    )

    flat_out = (
        moe_buf.view(-1)
        if out is None
        else out.view(-1)
    )
    flat_out.zero_()

    # fp8 stage2 (AITER_A16WMIX_FP8_S2=1): quantize the intermediate per SORTED ROW.
    # A row of inter_sorted is one route and stage2 contracts over inter_dim, so the
    # per-row scale per_token_quant_hip produces is exactly the per-A-operand scale the
    # fp8 MFMA needs -- no new kernel, and stage1 keeps writing bf16.
    # Quantize only the rows gemm2 can read. It clamps its block count to
    # ``cumsum0 // BM`` from the same ``num_valid_ids`` (gemm2.py, _gemm2 prologue), so the
    # tail of the ``sorted_size`` buffer is dead: at decode that is 14576 allocated rows
    # against 3680 live ones, and quantizing all of them measured 30 us against 18 us.
    # Intra-expert padding *inside* the live region still gets quantized, and since
    # inter_sorted is torch.empty those rows hold whatever the allocator handed over. That
    # is safe but only because of the epilogue's ``token_id >= i32_M`` drop, not because
    # they are zero: verified by filling them with NaN, which moved the output by 1.0x the
    # kernel's own atomic-fadd noise while the same NaN over live rows moved it 95x.
    _s2_dtype, _s2_scale, _w2_sref = "bf16", None, None
    if _s2_fp8:
        _inter_q, _s2_scale = per_token_quant_hip(
            inter_sorted, quant_dtype=dtypes.fp8, num_rows=num_valid_ids
        )
        try:
            _w2_sref = get_sref_u8(
                w2_scale.view(torch.uint8),
                experts=experts,
                n_out=model_dim,
                k_groups=inter_dim // 32,
                shuffle_fn=lambda t, e: e8m0_shuffle(t),
            )
        except ValueError as exc:
            global _FP8_S2_FALLBACK_WARNED
            if not _FP8_S2_FALLBACK_WARNED:
                _FP8_S2_FALLBACK_WARNED = True
                logger.error(
                    "A16WMIX_FP8_S2 disabled, stage2 stays bf16: %s", exc
                )
        else:
            inter_sorted = _inter_q.view(torch.uint8)
            _s2_dtype = "fp8"

    flydsl_a16w4_gemm2(
        inter_sorted_bf16=inter_sorted,
        a_dtype=_s2_dtype,
        a_scale=_s2_scale,
        w2_sref=_w2_sref,
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
        tile_n=None,
        tile_k=256,
        w_dtype="mxfp4",
        use_csv_config=False,
    )
    return flat_out.view(tokens, model_dim)
