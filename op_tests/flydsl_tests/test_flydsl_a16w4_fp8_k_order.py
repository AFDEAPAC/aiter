# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU check of the gfx942 a16w4 fp8 mode's K order: B decode vs A staging.

No GPU. The fp8 B decode (``_fp4_nibble_to_fp8x8``) returns each MFMA fragment as
[K0,K2,K4,K6, K1,K3,K5,K7] and the fp8 A loader stores A in the same order, so each K32
step multiplies the same (A, B) pairs as in K order. This emulates the gfx942
instructions involved:

  * ``v_perm_b32``: selector bytes 0-7 pick a byte of {S0:S1} (S1 = bytes 0-3), 8-11
    replicate the sign of byte 1/3/5/7, 12 gives 0x00, 13 and up give 0xFF.
  * E4M3FNUZ (bias 8, 0x80 is NaN, no negative zero).

for every nibble pair in every byte lane, 200k random words, every residual t in 0..6
and random A bytes. The MFMA's internal summation order is not modelled.

Run:
    python -m pytest op_tests/flydsl_tests/test_flydsl_a16w4_fp8_k_order.py -q
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
PI = [0, 2, 4, 6, 1, 3, 5, 7]  # fragment slot j holds K PI[j]


def u32(x):
    return np.asarray(x, dtype=np.uint64) & M32


def perm_b32(s0, s1, sel):
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


def residual_table(t):
    """``_fp4_fp8_residual_table``."""
    t = np.uint64(min(max(t, 0), TMAX))
    sub = u32(t * np.uint64(0x08080808))
    lo = (
        u32(np.uint64(LO) + (np.uint64(1) << np.uint64(32)) - sub)
        & np.uint64(0xFFFFFF00)
    ) | np.uint64(1)
    hi = u32(np.uint64(HI) + (np.uint64(1) << np.uint64(32)) - sub)
    return lo, hi


def decode_b(raw, lo, hi):
    """``_fp4_nibble_to_fp8x8`` -> (dword0, dword1)."""
    raw = u32(raw)
    even = raw & np.uint64(0x07070707)
    odd = (raw >> np.uint64(4)) & np.uint64(0x07070707)
    mag_e = perm_b32(hi, lo, even) | ((raw & np.uint64(0x08080808)) << np.uint64(4))
    mag_o = perm_b32(hi, lo, odd) | (raw & np.uint64(0x80808080))
    return u32(mag_e), u32(mag_o)


def stage_a(a0, a1):
    """``_fp8_kperm_dwords`` (fp8 A loader, before the LDS write)."""
    return perm_b32(a1, a0, SEL_E), perm_b32(a1, a0, SEL_O)


def bytes8(d0, d1):
    v = u32(d0) | (u32(d1) << np.uint64(32))
    shifts = np.arange(8, dtype=np.uint64) * np.uint64(8)
    return ((v[..., None] >> shifts) & np.uint64(0xFF)).astype(np.uint8)


def e4m3fnuz(b):
    b = np.asarray(b, dtype=np.int64)
    assert not np.any(b == 0x80), "FNUZ NaN (0x80) produced"
    sign = np.where(b & 0x80, -1.0, 1.0)
    e, m = (b >> 3) & 0xF, b & 7
    return sign * np.where(e == 0, m / 8.0 * 2.0**-7, (1 + m / 8.0) * 2.0 ** (e - 8))


_E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)
_RNG = np.random.default_rng(7153)
_RAW_ALL = np.arange(256, dtype=np.uint64) * np.uint64(0x01010101)
_RAW_RND = _RNG.integers(0, 2**32, size=200_000, dtype=np.uint64)


def _nibble_values(raw, t):
    """E2M1 * 2^-t per nibble in K order (both zeros -> the 0x01 placeholder 2^-10)."""
    nib = (raw[:, None] >> (np.arange(8, dtype=np.uint64) * np.uint64(4))) & np.uint64(
        0xF
    )
    mag = _E2M1[(nib & np.uint64(7)).astype(np.int64)] * 2.0**-t
    mag = np.where(mag == 0, 2.0**-10, mag)
    return np.where(nib & np.uint64(8), -mag, mag)


@pytest.mark.parametrize(
    "raw", [_RAW_ALL, _RAW_RND], ids=["all-nibble-pairs", "random"]
)
@pytest.mark.parametrize("t", range(TMAX + 1))
def test_b_decode_is_e2m1_in_fp8_k_order(raw, t):
    got = e4m3fnuz(bytes8(*decode_b(raw, *residual_table(t))))
    assert np.array_equal(got, _nibble_values(raw, t)[:, PI])


def test_a_staging_uses_the_same_k_order():
    a0 = _RNG.integers(0, 2**32, size=200_000, dtype=np.uint64)
    a1 = _RNG.integers(0, 2**32, size=200_000, dtype=np.uint64)
    assert np.array_equal(bytes8(*stage_a(a0, a1)), bytes8(a0, a1)[:, PI])


def _random_a(n):
    a = _RNG.integers(0, 256, size=(n, 8), dtype=np.int64)
    a = np.where(a == 0x80, 0, a)  # random fp8 A without the NaN code
    sh = np.arange(4, dtype=np.uint64) * np.uint64(8)
    a0 = u32(np.sum(a[:, :4].astype(np.uint64) << sh, axis=1))
    a1 = u32(np.sum(a[:, 4:].astype(np.uint64) << sh, axis=1))
    return a0, a1


@pytest.mark.parametrize("t", range(TMAX + 1))
def test_k32_dot_products_match_k_order(t):
    raw = _RAW_RND[:50_000]
    a0, a1 = _random_a(raw.size)
    want = (e4m3fnuz(bytes8(a0, a1)) * _nibble_values(raw, t)).sum(1)
    got = (
        e4m3fnuz(bytes8(*stage_a(a0, a1)))
        * e4m3fnuz(bytes8(*decode_b(raw, *residual_table(t))))
    ).sum(1)
    # fp8 x fp8 products and their sums of 8 are exact in f64.
    assert np.array_equal(got, want)


def test_negative_control_unstaged_a_mismatches():
    raw = _RAW_RND[:50_000]
    a0, a1 = _random_a(raw.size)
    b = e4m3fnuz(bytes8(*decode_b(raw, *residual_table(0))))
    want = (e4m3fnuz(bytes8(a0, a1)) * _nibble_values(raw, 0)).sum(1)
    assert not np.array_equal((e4m3fnuz(bytes8(a0, a1)) * b).sum(1), want)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
