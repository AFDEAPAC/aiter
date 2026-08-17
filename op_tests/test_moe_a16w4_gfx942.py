# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness gate for the gfx942 a16w4 (bf16 A x MXFP4 W) SiTUv2 fused MoE.

Drives `aiter.fused_moe.fused_moe` — the same entry vLLM's AiterExperts uses —
with weights prepared exactly the way vLLM's Kimi-K3 path prepares them, and
checks the result against an independent torch reference.

Self-contained: quantization comes from aiter's own `per_1x32_f4_quant`, and the
dequant reference is an explicit E2M1 table, so nothing outside aiter is needed.

Run:  python3 op_tests/test_moe_a16w4_gfx942.py
"""

import pytest
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
from aiter.ops.quant import per_1x32_f4_quant
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.utility import fp4_utils

# OCP E2M1: index = 4-bit code, value = what that code decodes to.
FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
              -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

# Kimi-K3's SiTUv2 constants.
BETA, LINEAR_BETA = 2.0, 1.5


def _is_gfx942():
    return "gfx942" in torch.cuda.get_device_properties(0).gcnArchName


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not _is_gfx942(),
    reason="a16w4 SiTUv2 fused MoE gate is gfx942-only",
)


def _dequant_mxfp4(packed_u8, scale_u8, rows, k, device):
    """Packed MXFP4 + E8M0 -> fp32, independent of the kernel under test."""
    lut = torch.tensor(FP4_VALUES, dtype=torch.float32, device=device)
    b = packed_u8.reshape(rows, k // 2).to(torch.int16)
    out = torch.empty((rows, k), dtype=torch.float32, device=device)
    out[:, 0::2] = lut[(b & 0xF).long()]
    out[:, 1::2] = lut[((b >> 4) & 0xF).long()]
    exp = scale_u8.reshape(rows, k // 32).to(torch.int32) - 127
    return out * torch.pow(2.0, exp.float()).repeat_interleave(32, dim=1)


def _situ_mul(gate, up):
    """SiTUv2: situ(g) = beta*tanh(g/beta)*sigmoid(g); situ_up(u) = lb*tanh(u/lb)."""
    return (BETA * torch.tanh(gate / BETA) * torch.sigmoid(gate)) * (
        LINEAR_BETA * torch.tanh(up / LINEAR_BETA)
    )


def _silu_mul(gate, up):
    return (gate * torch.sigmoid(gate)) * up


def _torch_moe(x, w1_deq, w2_deq, topk_ids, topk_weights, inter_dim, act):
    out = torch.zeros((x.shape[0], w2_deq.shape[1]), dtype=torch.float32, device=x.device)
    for slot in range(topk_ids.shape[1]):
        e = topk_ids[:, slot].long()
        weight = topk_weights[:, slot].float().unsqueeze(1)
        gate = torch.einsum("tk,tnk->tn", x, w1_deq[e, :inter_dim, :])
        up = torch.einsum("tk,tnk->tn", x, w1_deq[e, inter_dim:, :])
        hidden = act(gate, up)
        out += weight * torch.einsum("tn,tmn->tm", hidden, w2_deq[e])
    return out


def _metrics(got, ref):
    g, r = got.flatten().float(), ref.flatten().float()
    return (
        torch.nn.functional.cosine_similarity(g, r, dim=0).item(),
        ((g - r).norm() / r.norm()).item(),
        (torch.dot(g, r) / torch.dot(r, r)).item(),   # best-fit uniform gain
    )


@pytest.mark.parametrize(
    "tokens, model_dim, inter_dim, experts, topk",
    [
        (128, 1024, 256, 8, 2),
        # Kimi-K3 per-shard geometry at TP=8: inter_dim=384 is not a multiple of
        # 256, which the stage tiles must adapt to.
        (128, 4096, 384, 8, 2),
    ],
    ids=["small", "kimi_k3_tp8_shard"],
)
def test_a16w4_situv2_fused_moe(tokens, model_dim, inter_dim, experts, topk):
    device = torch.device("cuda")
    torch.manual_seed(0)
    scale = 0.2
    n_out = 2 * inter_dim

    x = torch.randn((tokens, model_dim), device=device, dtype=torch.float32) * scale
    w1 = torch.randn((experts, n_out, model_dim), device=device, dtype=torch.float32) * scale
    w2 = torch.randn((experts, model_dim, inter_dim), device=device, dtype=torch.float32) * (
        scale / inter_dim**0.5
    )
    score = torch.rand((tokens, experts), device=device, dtype=torch.float32)
    topk_vals, topk_ids = torch.topk(score, k=topk, dim=1)
    topk_weights = torch.softmax(topk_vals, dim=1).to(torch.float32)

    w1_q, w1_scale = per_1x32_f4_quant(w1.reshape(experts * n_out, model_dim))
    w2_q, w2_scale = per_1x32_f4_quant(w2.reshape(experts * model_dim, inter_dim))

    w1_deq = _dequant_mxfp4(
        w1_q.view(torch.uint8), w1_scale.view(torch.uint8), experts * n_out, model_dim, device
    ).view(experts, n_out, model_dim)
    w2_deq = _dequant_mxfp4(
        w2_q.view(torch.uint8), w2_scale.view(torch.uint8), experts * model_dim, inter_dim, device
    ).view(experts, model_dim, inter_dim)

    x_bf16 = x.to(torch.bfloat16).contiguous()
    ref_situ = _torch_moe(x_bf16.float(), w1_deq, w2_deq, topk_ids, topk_weights, inter_dim, _situ_mul)

    # vLLM's Kimi-K3 weight prep (quantization/mxfp4.py::_setup_kernel_k3_situ):
    # gate_up=False for both weights; w1 scale via shuffle_scale_a16w4, w2 via e8m0_shuffle.
    fp4 = dtypes.fp4x2
    w1_shuffled = shuffle_weight_a16w4(w1_q.view(fp4).reshape(experts, n_out, -1), 16, False)
    w2_shuffled = shuffle_weight_a16w4(w2_q.view(fp4).reshape(experts, model_dim, -1), 16, False)
    w1_scale_shuffled = shuffle_scale_a16w4(
        w1_scale.view(experts * n_out, model_dim // 32), experts, False
    )
    w2_scale_shuffled = fp4_utils.e8m0_shuffle(
        w2_scale.view(experts * model_dim, inter_dim // 32)
    )
    w1_shuffled.is_shuffled = True
    w2_shuffled.is_shuffled = True

    # A pass is only meaningful if the a16wmix path is what ran; a silent fall-back
    # to another backend must not be able to masquerade as one.
    import aiter.ops.flydsl.a16wmix_fused_moe as a16wmix

    fired = {"hit": False}
    original = a16wmix.fused_moe_a16wmix

    def _spy(*args, **kwargs):
        fired["hit"] = True
        return original(*args, **kwargs)

    a16wmix.fused_moe_a16wmix = _spy
    aiter.fused_moe.fused_moe_a16wmix = _spy
    try:
        got = fused_moe(
            x_bf16, w1_shuffled, w2_shuffled, topk_weights, topk_ids,
            activation=ActivationType.Situv2,
            quant_type=QuantType.per_1x32,
            w1_scale=w1_scale_shuffled,
            w2_scale=w2_scale_shuffled,
            beta=BETA,
            linear_beta=LINEAR_BETA,
        ).float()
    finally:
        a16wmix.fused_moe_a16wmix = original
        aiter.fused_moe.fused_moe_a16wmix = original

    assert fired["hit"], "a16wmix path did not run; fused_moe chose another backend"

    cos, rel, gain = _metrics(got, ref_situ)
    print(f"  a16w4 SiTUv2: cos={cos:.6f} rel_fro={rel:.3e} best_fit_gain={gain:.6f}")

    # The gain check is the one that catches a systematic scale error; cosine alone
    # does not. Cross-checking against the wrong activation shows why: it still
    # scores cos ~0.96 while the gain collapses to ~0.48.
    assert cos > 0.999, f"cosine {cos:.6f} below the fp4 bar"
    assert rel < 5e-2, f"relative Frobenius error {rel:.3e} too large"
    assert abs(gain - 1.0) < 2e-2, f"systematic gain error {gain:.6f}"

    ref_silu = _torch_moe(x_bf16.float(), w1_deq, w2_deq, topk_ids, topk_weights, inter_dim, _silu_mul)
    _, _, gain_wrong = _metrics(got, ref_silu)
    assert abs(gain_wrong - 1.0) > 0.1, (
        f"gate cannot distinguish SiTUv2 from SiLU (gain {gain_wrong:.6f}); "
        "the tolerances above prove nothing"
    )


if __name__ == "__main__":
    for shape in [(128, 1024, 256, 8, 2), (128, 4096, 384, 8, 2)]:
        print(f"### tokens={shape[0]} model_dim={shape[1]} inter_dim={shape[2]} "
              f"E={shape[3]} topk={shape[4]}")
        test_a16w4_situv2_fused_moe(*shape)
        print("  PASS")
