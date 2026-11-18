# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from ..jit.core import compile_ops

MD_NAME = "module_moe_sorting_opus"


@compile_ops("module_moe_sorting_opus", develop=True)
def moe_sorting_opus_get_workspace_size(
    tokens: int,
    num_experts: int,
    topk: int,
    dispatch_policy: int = 0,
    device_id: int = -1,
) -> int: ...


@compile_ops("module_moe_sorting_opus", develop=True)
def moe_sorting_opus_fwd(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    moe_buf: torch.Tensor,
    num_experts: int,
    unit_size: int,
    local_expert_mask: torch.Tensor | None = None,
    num_local_tokens: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    dispatch_policy: int = 0,
    local_topk_ids: torch.Tensor | None = None,
    m_indices: torch.Tensor | None = None,
    reverse_sorted: torch.Tensor | None = None,
) -> None: ...


@compile_ops("module_moe_sorting_opus", develop=True)
def mxfp4_moe_sort_quant_fwd(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    moe_buf: torch.Tensor,
    activation_quant: torch.Tensor,
    activation_scale_token: torch.Tensor,
    num_experts: int,
    num_local_tokens: torch.Tensor | None = None,
) -> None:
    """Fused route sort + compact MXFP4 activation quantization.

    gfx950 only, ``model_dim=4096``, ``block_m=32``, ``M`` in ``[1, 128]``.
    ``(num_experts, topk)`` must be one of ``(512, 10)`` or ``(513, 11)``
    -- the expert histogram is a kernel template parameter.
    ``num_local_tokens``: as for ``moe_sorting_opus_fwd`` -- only that many
    leading rows are sorted and quantized; ``moe_buf`` is zeroed whole.
    """


@compile_ops("module_moe_sorting_opus", develop=True)
def a16w4_moe_sort_quant_fwd(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    moe_buf: torch.Tensor,
    num_experts: int,
    block_m: int,
    activation_quant: torch.Tensor | None = None,
    activation_scale: torch.Tensor | None = None,
    num_local_tokens: torch.Tensor | None = None,
) -> None:
    """Decode route sort + moe_buf zeroing, optionally fused with per-token fp8 quant.

    The sorted metadata matches ``moe_sorting_opus_fwd`` for the same ``block_m``;
    ``activation_quant`` [M, model_dim] fp8 and ``activation_scale`` [M, 1] fp32 match
    ``dynamic_per_token_scaled_quant``. ``M`` in ``[1, 128]``, ``block_m`` 16 or 32,
    ``(num_experts, topk, model_dim)`` = (896, 16, 3584). ``num_local_tokens``: only
    that many leading rows are sorted and quantized; ``moe_buf`` is zeroed whole.
    A token's top-k must not repeat an expert (as for the multi-phase Opus sort).
    """
