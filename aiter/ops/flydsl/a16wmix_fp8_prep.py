# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-output-column E8M0 reference exponents for the fp8 a16wmix MoE path.

An fp8 weight operand cannot carry the MXFP4 groupwise scale -- E8M0 spans
2^-127..2^127 while e4m3fnuz reaches only about 2^-10..240 -- and the accumulator
cannot carry a per-K-group scale either, because the K-group index in the kernel's
``adj_ku`` depends on ``lane_div_16``, so one MFMA contracts four different groups and
an accumulator element mixes four scales. (That is what broke the earlier
accumulator-scale experiment; see ``moe_2stage_a16wmix.utils.acc_scale_for``.)

What the accumulator *can* carry is anything that depends only on the output column,
since a lane's four accumulator elements all share one column. So split the exponent:

    s(n, kg) = s_ref(n) + r(n, kg),    s_ref(n) = max over kg,   hence r <= 0

``2^s_ref(n)`` goes in the epilogue, once per column. ``2^r`` rides in the weight, and
measurement makes that nearly free: on the real checkpoint r takes only the values
0, -1, -2 (and -3 twice in 5.5M groups), so the kernel just picks one of four constant
v_perm_b32 magnitude pools instead of doing any exponent arithmetic. See
``prof/e8m0_residual_scan.json``.

The kernel computes the residual itself as ``s_ref_byte - scale_byte`` while it is
already loading the scale byte, so **the weight and its scale tensor are never
rewritten** -- this module only produces the per-column reference.

Deriving s_ref has one wrinkle: the scale tensor arrives already preshuffled by
``shuffle_scale_a16w4``, so a plain ``max`` over the last axis would maximise over the
wrong thing. Rather than invert the permutation analytically, we push an ``arange``
through the same shuffle to learn where every logical (row, group) landed, then
scatter-reduce. One-time per weight tensor, then cached.
"""
from __future__ import annotations

from collections import OrderedDict

import torch

E8M0_BIAS = 127
# Must match FP8_MAX_NEG_RESIDUAL in moe_2stage_a16wmix/utils.py, which is the largest
# residual the SWAR magnitude adjustment stays exact for. The down projection reaches
# -4 on the real Kimi-K3 checkpoint (10 groups in 4.13 billion), so 3 was not enough.
MAX_NEG_RESIDUAL = 6

# Keyed by (data_ptr, shape, rows, groups) -- but a data_ptr alone is NOT an identity.
# PyTorch's caching allocator hands a freed tensor's address straight to the next
# allocation of the same size, so two different weight-scale tensors collide on that key
# and the second silently receives the first's reference exponents. Verified: a tensor of
# scale byte 120 got 127 back after the 127 tensor was freed.
#
# The entry therefore holds a STRONG reference to the tensor it was computed from. While
# that reference is alive the address cannot be reused, so (data_ptr, shape) really does
# identify it; and once an entry is evicted there is nothing left to return staleley.
# Keeping model weights alive is what the server does anyway, but a harness that churns
# weight tensors would grow without bound, hence the LRU bound.
#
# Remaining assumption, documented rather than defended: nobody mutates a weight-scale
# tensor in place. Quantized scales are produced once at load time here.
_SREF_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_SREF_CACHE_MAX = 512  # a 92-layer model needs 2 per layer; this covers it several times


def _shuffled_positions(rows: int, groups: int, experts: int, shuffle_fn, device):
    """Which logical (row, group) each shuffled slot holds.

    ``shuffle_scale`` is a pure pad/view/permute/contiguous on a 2-D tensor and keeps the
    input dtype, so an int32 ``arange`` survives it unchanged and comes back telling us
    where every element went. That beats re-deriving the index math, which would
    silently drift if the shuffle ever changes.

    ``groups`` must be the group count of the tensor the kernel actually reads, i.e.
    already padded. Probing at the padded width matters: the shuffle pads with
    ``torch.empty``, so probing at the logical width would hand back uninitialized
    garbage in the pad columns and there would be no way to tell it from a position.
    Probing at the padded width makes every output slot a position we chose.

    Returns a flat int32 tensor: entry ``i`` is the logical index ``row*groups + group``
    whose scale now lives at flat position ``i``.
    """
    src = torch.arange(rows * groups, device=device, dtype=torch.int32).view(rows, groups)
    out = shuffle_fn(src, experts).reshape(-1)
    if out.numel() != rows * groups:
        raise ValueError(
            f"shuffle changed the element count ({rows * groups} -> {out.numel()}); "
            f"the position probe assumes a pure permutation of the padded tensor"
        )
    return out


def compute_sref_u8(
    scale_u8: torch.Tensor,
    *,
    experts: int,
    n_out: int,
    k_groups: int,
    shuffle_fn=None,
) -> torch.Tensor:
    """Per-(expert, output column) maximum E8M0 byte, as ``[experts, n_out] uint8``.

    ``scale_u8`` is the **preshuffled** scale tensor the kernel reads. ``shuffle_fn``
    takes ``(tensor, experts)`` and applies the same permutation that produced it; pass
    it so the position map is learned rather than assumed.
    """
    flat = scale_u8.reshape(-1)
    rows = experts * n_out
    if flat.numel() % rows:
        raise ValueError(
            f"scale tensor has {flat.numel()} bytes, not a multiple of "
            f"{rows} rows (experts={experts} n_out={n_out})"
        )
    # shuffle_scale pads K/32 up to a multiple of 8, so trust the tensor over the
    # caller's k_groups; the padding bytes are 0x7F (scale 1.0) and never exceed a
    # real column maximum.
    groups = flat.numel() // rows
    if groups < k_groups:
        raise ValueError(f"scale tensor holds {groups} groups, fewer than {k_groups}")

    if shuffle_fn is None:
        # No permutation to undo: the tensor is in logical (row, group) order.
        s = flat.view(rows, groups).to(torch.int32)
        ref = s.max(dim=1).values
    else:
        pos = _shuffled_positions(rows, groups, experts, shuffle_fn, scale_u8.device)
        logical_row = (pos // groups).long()
        # Drop the pad columns. K/32 is rounded up to a multiple of 8 and the pad is
        # uninitialized, so for w2 at Kimi-K3 TP=8 (inter_dim 384 -> 12 groups padded to
        # 16) a quarter of the entries are garbage. Including them would inflate the
        # column max and scale that whole column wrong. w1 (112 groups) needs no mask,
        # which is why this only shows up on stage2.
        keep = (pos % groups) < k_groups
        ref = torch.zeros(rows, dtype=torch.int32, device=scale_u8.device)
        ref.scatter_reduce_(
            0,
            logical_row[keep],
            flat.to(torch.int32)[keep],
            reduce="amax",
            include_self=False,
        )
    return ref.to(torch.uint8).view(experts, n_out).contiguous()


def check_residual_range(scale_u8, sref_u8, *, experts, n_out, k_groups, shuffle_fn=None):
    """Largest ``|r|`` implied by this (scale, s_ref) pair.

    The kernel clamps the residual to ``MAX_NEG_RESIDUAL``, so anything beyond that
    would be silently scaled wrong. Callers assert on this rather than trusting the
    checkpoint to keep looking like the one we measured.
    """
    flat = scale_u8.reshape(-1)
    rows = experts * n_out
    groups = flat.numel() // rows
    ref_flat = sref_u8.reshape(-1).to(torch.int32)
    if shuffle_fn is None:
        s = flat.view(rows, groups).to(torch.int32)
        r = s - ref_flat.unsqueeze(1)
    else:
        pos = _shuffled_positions(rows, groups, experts, shuffle_fn, scale_u8.device)
        logical_row = (pos // groups).long()
        keep = (pos % groups) < k_groups
        r = flat.to(torch.int32)[keep] - ref_flat[logical_row[keep]]
    return int((-r).max().item())


def get_sref_u8(scale_u8, *, experts, n_out, k_groups, shuffle_fn=None, strict=True):
    """Cached :func:`compute_sref_u8`, with the residual-range assertion.

    ``strict=False`` downgrades an out-of-range residual to a warning, for measuring
    how bad it would be rather than refusing to run.
    """
    key = (scale_u8.data_ptr(), tuple(scale_u8.shape), experts, n_out, k_groups)
    hit = _SREF_CACHE.get(key)
    if hit is not None:
        src, sref = hit
        # The stored tensor pins the address, so a live entry cannot be a different
        # tensor. Compare storage rather than object identity: callers reach this
        # through w1_scale.view(torch.uint8), which builds a fresh Tensor every call, and an
        # identity check would miss every time and recompute per layer per forward.
        if src.data_ptr() == scale_u8.data_ptr() and src.shape == scale_u8.shape:
            _SREF_CACHE.move_to_end(key)
            return sref
        del _SREF_CACHE[key]
    sref = compute_sref_u8(
        scale_u8, experts=experts, n_out=n_out, k_groups=k_groups, shuffle_fn=shuffle_fn
    )
    worst = check_residual_range(
        scale_u8, sref, experts=experts, n_out=n_out, k_groups=k_groups,
        shuffle_fn=shuffle_fn,
    )
    if worst > MAX_NEG_RESIDUAL:
        msg = (
            f"e8m0 residual reaches -{worst}, past -{MAX_NEG_RESIDUAL}, the deepest the "
            f"fp8 magnitude adjustment stays exact for (below that the smallest E2M1 "
            f"magnitude leaves e4m3's normal range). Groups that far below their column "
            f"maximum would be scaled wrong, so these weights need the bf16 path."
        )
        if strict:
            raise ValueError(msg)
        import warnings

        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    _SREF_CACHE[key] = (scale_u8, sref)
    _SREF_CACHE.move_to_end(key)
    while len(_SREF_CACHE) > _SREF_CACHE_MAX:
        _SREF_CACHE.popitem(last=False)
    return sref


def clear_cache():
    _SREF_CACHE.clear()
