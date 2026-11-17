# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU proof that the a16w4 fp8-mode decode variants compute what the PR #7153 decode does.

No GPU. Emulates, instruction by instruction, what gfx942 runs for the fp8 mode
(``a_dtype="fp8"``) of ``moe_2stage_a16wmix``:

  * ``v_perm_b32``: selector bytes 0-7 pick a byte of {S0:S1} (S1 = bytes 0-3), 8-11
    replicate the sign of byte 1/3/5/7, 12 gives 0x00, 13 and up give 0xFF.
  * E4M3FNUZ decode (bias 8, 0x80 is NaN, no infinities, no negative zero).

Checks:
  * ``tab``: the broadcast residual table equals the PR table for every (s_ref, s) byte
    pair, so the decoded bytes, and the kernel outputs, are unchanged bit for bit.
  * ``aperm``: B decoded without the interleave and A written with the aperm byte map
    are the PR's B and A under one shared permutation of the 8 K slots, for every
    nibble pattern, every residual t in 0..6 and random A bytes; so each MFMA K32 step
    multiplies the same (A, B) pairs. The fp32 dot products agree exactly here; the
    MFMA's internal summation order is not modelled (see the handoff notes).
  * the PR decode itself matches E2M1 * 2^-t (zero -> the 0x01 placeholder), and never
    produces the FNUZ NaN.

Run:
    python -m pytest op_tests/flydsl_tests/test_flydsl_a16w4_fp8_decode_emu.py -q
"""

import os
import sys
import types

import numpy as np
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _utils():
    """The kernel constants, importable without ROCm (stubs the aiter package roots)."""
    sys.path.insert(0, _REPO)
    try:
        import aiter  # noqa: F401
    except (ImportError, OSError, RuntimeError):
        for k in [k for k in sys.modules if k == "aiter" or k.startswith("aiter.")]:
            del sys.modules[k]
        for name, path in (
            ("aiter", "aiter"),
            ("aiter.ops", "aiter/ops"),
            ("aiter.ops.flydsl", "aiter/ops/flydsl"),
        ):
            m = types.ModuleType(name)
            m.__path__ = [os.path.join(_REPO, path)]
            sys.modules[name] = m
    from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import utils

    return utils


U = _utils()
LO, HI = U._FP4_E2M1_FP8_RES_LO, U._FP4_E2M1_FP8_RES_HI
TMAX = U.A16W4_FP8_MAX_RESIDUAL
SEL_E, SEL_O = U._FP8_KPERM_SEL_EVEN, U._FP8_KPERM_SEL_ODD
M32 = np.uint64(0xFFFFFFFF)


def u32(x):
    return np.asarray(x, dtype=np.uint64) & M32


def perm_b32(s0, s1, sel):
    """v_perm_b32 D = perm({S0, S1}, sel), vectorized over uint64 arrays of dwords."""
    s0, s1, sel = np.broadcast_arrays(u32(s0), u32(s1), u32(sel))
    data = (s0 << np.uint64(32)) | s1
    out = np.zeros(s0.shape, dtype=np.uint64)
    for j in range(4):
        b = (sel >> np.uint64(8 * j)) & np.uint64(0xFF)
        pick = (data >> (np.minimum(b, 7) * np.uint64(8))) & np.uint64(0xFF)
        sign_idx = np.clip(b.astype(np.int64) - 8, 0, 3)
        sign_bit = np.array([15, 31, 47, 63], dtype=np.uint64)[sign_idx]
        rep = np.where((data >> sign_bit) & np.uint64(1), np.uint64(0xFF), np.uint64(0))
        r = np.where(b <= 7, pick, np.where(b <= 11, rep, np.where(b == 12, 0, 0xFF)))
        out |= r.astype(np.uint64) << np.uint64(8 * j)
    return out


def table_pr(t):
    """``_fp4_fp8_residual_table`` (t = s_ref - s, any int)."""
    t = np.clip(np.asarray(t, dtype=np.int64), 0, TMAX).astype(np.uint64)
    sub = u32(t * np.uint64(0x08080808))
    lo = (
        u32(np.uint64(LO) - sub + (np.uint64(1) << np.uint64(32)))
        & np.uint64(0xFFFFFF00)
    ) | np.uint64(1)
    hi = u32(np.uint64(HI) - sub + (np.uint64(1) << np.uint64(32)))
    return lo, hi


def table_tab(sref, s):
    """``_fp4_fp8_residual_table_bcast`` (the ``tab`` variant), in 32-bit wraparound."""
    t8 = (np.asarray(sref, dtype=np.int64) - np.asarray(s, dtype=np.int64)) * 8
    t8 = np.clip(t8, 0, 8 * TMAX).astype(np.uint64)
    sub = perm_b32(t8, t8, 0)
    hi = u32(np.uint64(HI) + (np.uint64(1) << np.uint64(32)) - sub)
    lo_minus_hi = ((LO & 0xFFFFFF00) | 1) - HI
    lo = u32(hi + t8 + u32(np.int64(lo_minus_hi)))
    return lo, hi


def decode(raw, lo, hi, interleave=True):
    """``_fp4_nibble_to_fp8x8`` -> (dword0, dword1)."""
    raw = u32(raw)
    even = raw & np.uint64(0x07070707)
    odd = (raw >> np.uint64(4)) & np.uint64(0x07070707)
    mag_e = perm_b32(hi, lo, even) | ((raw & np.uint64(0x08080808)) << np.uint64(4))
    mag_o = perm_b32(hi, lo, odd) | (raw & np.uint64(0x80808080))
    mag_e, mag_o = u32(mag_e), u32(mag_o)
    if not interleave:
        return mag_e, mag_o
    return perm_b32(mag_o, mag_e, 0x05010400), perm_b32(mag_o, mag_e, 0x07030602)


def kperm_a(a0, a1):
    """aperm A staging: (K0..K3, K4..K7) -> (K0,K2,K4,K6 ; K1,K3,K5,K7)."""
    return perm_b32(a1, a0, SEL_E), perm_b32(a1, a0, SEL_O)


def bytes8(d0, d1):
    """Two dword arrays -> [..., 8] uint8 in MFMA K-slot order (byte j of d0|d1<<32)."""
    v = u32(d0) | (u32(d1) << np.uint64(32))
    return (
        (v[..., None] >> (np.arange(8, dtype=np.uint64) * np.uint64(8)))
        & np.uint64(0xFF)
    ).astype(np.uint8)


def e4m3fnuz(b):
    b = np.asarray(b, dtype=np.int64)
    assert not np.any(b == 0x80), "FNUZ NaN (0x80) produced"
    sign = np.where(b & 0x80, -1.0, 1.0)
    e = (b >> 3) & 0xF
    m = b & 7
    val = np.where(e == 0, m / 8.0 * 2.0**-7, (1 + m / 8.0) * 2.0 ** (e - 8))
    return sign * val


_E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)
# Every nibble pair in every byte lane at once: raw = b * 0x01010101.
_RAW_ALL = np.arange(256, dtype=np.uint64) * np.uint64(0x01010101)
_RNG = np.random.default_rng(7153)
_RAW_RND = _RNG.integers(0, 2**32, size=200_000, dtype=np.uint64)
_PI = [0, 2, 4, 6, 1, 3, 5, 7]  # aperm K-slot j holds original K _PI[j]


def test_tab_table_equals_pr_table_for_every_byte_pair():
    sref, s = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    lo_p, hi_p = table_pr(sref - s)
    lo_t, hi_t = table_tab(sref, s)
    assert np.array_equal(lo_p, lo_t) and np.array_equal(hi_p, hi_t)


def test_tab_table_bytes_have_no_borrow():
    # Byte-wise view of the claim in the docstring: lo = [0x01, 0x38-8t, 0x40-8t, 0x44-8t].
    for t in range(TMAX + 1):
        lo, hi = table_tab(t, 0)
        want_lo = (
            0x01 | (0x38 - 8 * t) << 8 | (0x40 - 8 * t) << 16 | (0x44 - 8 * t) << 24
        )
        want_hi = (
            (0x48 - 8 * t)
            | (0x4C - 8 * t) << 8
            | (0x50 - 8 * t) << 16
            | (0x54 - 8 * t) << 24
        )
        assert int(lo) == want_lo and int(hi) == want_hi, t


@pytest.mark.parametrize(
    "raw", [_RAW_ALL, _RAW_RND], ids=["all-nibble-pairs", "random"]
)
@pytest.mark.parametrize("t", range(TMAX + 1))
def test_pr_decode_matches_e2m1_times_2_pow_minus_t(raw, t):
    lo, hi = table_pr(t)
    got = e4m3fnuz(bytes8(*decode(raw, lo, hi)))
    nib = (raw[:, None] >> (np.arange(8, dtype=np.uint64) * np.uint64(4))) & np.uint64(
        0xF
    )
    mag = _E2M1[(nib & np.uint64(7)).astype(np.int64)] * 2.0**-t
    mag = np.where(mag == 0, 2.0**-10, mag)  # 0x01 placeholder for both zeros
    want = np.where(nib & np.uint64(8), -mag, mag)
    assert np.array_equal(got, want)


@pytest.mark.parametrize(
    "raw", [_RAW_ALL, _RAW_RND], ids=["all-nibble-pairs", "random"]
)
@pytest.mark.parametrize("t", range(TMAX + 1))
def test_aperm_b_is_pr_b_under_shared_k_permutation(raw, t):
    lo, hi = table_pr(t)
    b_pr = bytes8(*decode(raw, lo, hi))
    b_ap = bytes8(*decode(raw, lo, hi, interleave=False))
    assert np.array_equal(b_ap, b_pr[:, _PI])


def test_aperm_a_staging_applies_the_same_permutation():
    a0 = _RNG.integers(0, 2**32, size=200_000, dtype=np.uint64)
    a1 = _RNG.integers(0, 2**32, size=200_000, dtype=np.uint64)
    a_pr = bytes8(a0, a1)
    a_ap = bytes8(*kperm_a(a0, a1))
    assert np.array_equal(a_ap, a_pr[:, _PI])


@pytest.mark.parametrize("t", range(TMAX + 1))
def test_aperm_k32_dot_products_unchanged(t):
    # One lane's 8-slot slice of a K32 MFMA step: sum_j A[j] * B[j] over the same pairs.
    raw = _RAW_RND[:50_000]
    a = _RNG.integers(0, 256, size=(raw.size, 8), dtype=np.int64)
    a = np.where(a == 0x80, 0, a).astype(np.uint8)  # random fp8 A without the NaN code
    a0 = u32(
        np.sum(
            a[:, :4].astype(np.uint64)
            << (np.arange(4, dtype=np.uint64) * np.uint64(8)),
            axis=1,
        )
    )
    a1 = u32(
        np.sum(
            a[:, 4:].astype(np.uint64)
            << (np.arange(4, dtype=np.uint64) * np.uint64(8)),
            axis=1,
        )
    )
    lo, hi = table_pr(t)
    dot_pr = (e4m3fnuz(bytes8(a0, a1)) * e4m3fnuz(bytes8(*decode(raw, lo, hi)))).sum(1)
    a_ap = bytes8(*kperm_a(a0, a1))
    b_ap = bytes8(*decode(raw, lo, hi, interleave=False))
    dot_ap = (e4m3fnuz(a_ap) * e4m3fnuz(b_ap)).sum(1)
    # fp8 x fp8 products are exact in f64 and every sum of 8 is too.
    assert np.array_equal(dot_pr, dot_ap)


def test_perm_model_negative_control():
    # The interleave the PR emits is not the identity: dropping it on B alone (without
    # the A staging map) must change the per-slot bytes.
    lo, hi = table_pr(0)
    b_pr = bytes8(*decode(_RAW_ALL, lo, hi))
    b_ap = bytes8(*decode(_RAW_ALL, lo, hi, interleave=False))
    assert not np.array_equal(b_pr, b_ap)


def test_flag_spec():
    assert U.fp8_decode_spec(None) == ""
    assert U.fp8_decode_spec("base") == ""
    assert U.fp8_decode_spec("noslp,aperm") == "aperm+noslp"
    assert U.fp8_decode_spec("tab+aperm+noslp") == "aperm+tab+noslp"
    assert U.fp8_decode_name_tag("tab+aperm") == "_d8_aperm_tab"
    with pytest.raises(ValueError):
        U.fp8_decode_spec("sign")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
