# SPDX-License-Identifier: MIT
"""Kimi-K3 decode MoE preamble: opus moe_sorting (accumulate) + fp8 per_token_quant_hip in one launch."""
import os

import torch
from torch import Tensor

from ..jit.core import compile_ops

ENABLED = os.environ.get("AITER_K3_FUSED_ROUTE_PREP", "0") not in ("", "0", "false", "False")
MAX_TOKENS = int(os.environ.get("AITER_K3_FUSED_ROUTE_PREP_MAX_TOKENS", "16"))
_announced = False


@compile_ops("module_k3_route_prep", ffi_type="ctypes")
def k3_moe_route_prep(
    topk_ids: Tensor,
    topk_weights: Tensor,
    hidden: Tensor,
    sorted_ids: Tensor,
    sorted_weights: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    moe_buf: Tensor,
    a_q: Tensor,
    a_scale: Tensor,
    num_experts: int,
    unit_size: int,
) -> None: ...


def covered(topk_ids: Tensor, hidden: Tensor, num_experts: int, expert_mask) -> bool:
    return (
        ENABLED
        and expert_mask is None
        and topk_ids.dim() == 2
        and 1 <= topk_ids.shape[0] <= min(MAX_TOKENS, 32)
        and num_experts <= 1024
        and hidden.dtype == torch.bfloat16
        and 2048 < hidden.shape[-1] <= 4096
    )


def route_prep(topk_ids, topk_weights, hidden, num_experts, block_size, moebuf_dtype, quant_dtype,
               moe_buf=None):
    """Returns (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf, a_q, a_scale),
    allocated exactly as _moe_sorting_impl and per_token_quant_hip allocate them. A caller-owned
    [M, model_dim] moe_buf is zeroed in place of the allocated one."""
    global _announced
    device = topk_ids.device
    M, topk = topk_ids.shape
    model_dim = hidden.shape[-1]
    max_num_tokens_padded = int(topk_ids.numel() + num_experts * block_size - topk)
    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
    sorted_ids = torch.empty(max_num_tokens_padded, dtype=torch.int32, device=device)
    sorted_weights = torch.empty(max_num_tokens_padded, dtype=torch.float32, device=device)
    sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=torch.int32, device=device)
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
    if moe_buf is None:
        moe_buf = torch.empty((M, model_dim), dtype=moebuf_dtype, device=device)
    a_q = torch.empty(hidden.shape, dtype=quant_dtype, device=device)
    a_scale = torch.empty((*hidden.shape[:-1], 1), dtype=torch.float32, device=device)
    k3_moe_route_prep(
        topk_ids, topk_weights, hidden, sorted_ids, sorted_weights, sorted_expert_ids,
        num_valid_ids, moe_buf, a_q, a_scale, int(num_experts), int(block_size),
    )
    if not _announced:
        _announced = True
        print(f"[k3-fused-route-prep] active tokens={M} topk={topk} E={num_experts} unit={block_size}", flush=True)
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf, a_q, a_scale
