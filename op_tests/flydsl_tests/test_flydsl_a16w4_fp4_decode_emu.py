# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU bit-exactness proof for the gfx942 a16w4 MXFP4 decode variants (no GPU).

Emulates, instruction by instruction, the sequence each ``fp4_decode`` variant of
``moe_2stage_a16wmix.utils._fp4_nibble_to_bf16x8`` emits on gfx942, using the ISA
semantics of ``v_perm_b32`` (all selector codes), ``v_cvt_pk_f32_fp8`` (E4M3FNUZ),
IEEE f32 multiply (``v_mul_f32`` and each lane of ``v_pk_mul_f32``) and the integer
ops, and checks every variant's 4 output dwords against the default decode.

Each of the 4 byte lanes of the raw dword is decoded independently, so
``raw = b * 0x01010101`` for all 256 ``b`` covers every (lo, hi) nibble pair in every
lane; random dwords add mixed lanes. Every E8M0 scale byte 0..255 is crossed with
them (``_e8m0_byte_to_f32``: 0 -> +0.0, 255 -> +inf).

Run:
    pytest op_tests/flydsl_tests/test_flydsl_a16w4_fp4_decode_emu.py -q
    python op_tests/flydsl_tests/test_flydsl_a16w4_fp4_decode_emu.py
"""

import itertools

import numpy as np
import pytest

U32 = np.uint32
FLAGS = ("sign", "perm", "pkmul")
# Every subset, "" first: the three single ideas, the pairs, and all three combined.
VARIANTS = [
    "+".join(c) for r in range(len(FLAGS) + 1) for c in itertools.combinations(FLAGS, r)
]

LUT_LO, LUT_HI = 0x44403800, 0x54504C48
NEG_LUT_LO, NEG_LUT_HI = 0xC4C0B800, 0xD4D0CCC8
PERM_SEL_BF16_HI_PAIR = 0x07060302
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def v_perm_b32(src0, src1, sel):
    """D.b[i] = byte_permute({src0, src1}, sel.b[i]) for all 256 selector codes."""
    src0, src1, sel = (np.asarray(x, dtype=U32) for x in (src0, src1, sel))
    pool = (src0.astype(np.uint64) << np.uint64(32)) | src1.astype(np.uint64)
    out = np.zeros(np.broadcast(src0, src1, sel).shape, dtype=U32)
    for i in range(4):
        s = (sel >> U32(8 * i)) & U32(0xFF)
        byte = (
            (pool >> (np.minimum(s, 7).astype(np.uint64) * np.uint64(8))) & 0xFF
        ).astype(U32)
        sign = [
            (src1 >> U32(15)) & U32(1),
            (src1 >> U32(31)) & U32(1),
            (src0 >> U32(15)) & U32(1),
            (src0 >> U32(31)) & U32(1),
        ]
        for code in range(8, 12):
            byte = np.where(s == code, sign[code - 8] * U32(0xFF), byte)
        byte = np.where(s == 12, U32(0), byte)
        byte = np.where(s >= 13, U32(0xFF), byte)
        out |= byte.astype(U32) << U32(8 * i)
    return out


def _fnuz_table():
    """E4M3FNUZ byte -> f32 bits (bias 8, no inf, 0x80 is the only NaN)."""
    t = np.zeros(256, dtype=U32)
    for b in range(256):
        if b == 0x80:
            t[b] = 0x7FC00000
            continue
        e, m = (b >> 3) & 15, b & 7
        v = 2.0 ** (e - 8) * (1 + m / 8) if e else 2.0**-7 * (m / 8)
        t[b] = np.float32(-v if b & 0x80 else v).view(U32)
    return t


_FNUZ = _fnuz_table()


def v_cvt_pk_f32_fp8(src, word_sel):
    """-> (f32 bits of byte 2*ws, f32 bits of byte 2*ws+1); ws = op_sel / SDWA WORD_1."""
    sh = 16 if word_sel else 0
    b0 = (src >> U32(sh)) & U32(0xFF)
    b1 = (src >> U32(sh + 8)) & U32(0xFF)
    assert not np.any(b0 == 0x80) and not np.any(b1 == 0x80), "FNUZ NaN produced"
    return _FNUZ[b0], _FNUZ[b1]


def v_mul_f32(a_bits, b_bits):
    """IEEE f32 multiply, RNE. v_pk_mul_f32 is this op on each 32-bit lane."""
    with np.errstate(all="ignore"):
        p = a_bits.view(np.float32) * b_bits.view(np.float32)
    bits = p.view(U32).copy()
    # x86 yields the negative default NaN for 0*inf; AMDGPU yields 0x7FC00000.
    bits[np.isnan(p)] = U32(0x7FC00000)
    return bits


def v_pk_mul_f32(a_pair, s_bits):
    """v_pk_mul_f32 with op_sel_hi broadcasting the scalar scale to both lanes."""
    return v_mul_f32(a_pair[0], s_bits), v_mul_f32(a_pair[1], s_bits)


def e8m0_byte_to_f32_bits(b):
    return (np.asarray(b, dtype=U32) & U32(0xFF)) << U32(23)


def fp4x4_to_fp8x4(nib, sign_and):
    mag = nib & U32(0x07070707)
    if sign_and:
        neg = v_perm_b32(NEG_LUT_HI, NEG_LUT_LO, mag)
        return neg & ((nib << U32(4)) | U32(0x7F7F7F7F))
    magbyte = v_perm_b32(LUT_HI, LUT_LO, mag)
    nz = ((mag + U32(0x7F7F7F7F)) >> U32(7)) & U32(0x01010101)
    signbit = (nib & U32(0x08080808)) << U32(4)
    return magbyte | (signbit & (nz << U32(7)))


def decode(raw, scale_bits, variant):
    """Emulated ``_fp4_nibble_to_bf16x8(raw, scale, decode=variant)`` -> 4 dwords."""
    flags = variant.split("+") if variant else ()
    sign_and = "sign" in flags
    raw = np.asarray(raw, dtype=U32)
    even = raw & U32(0x0F0F0F0F)
    odd = (raw >> U32(4)) & U32(0x0F0F0F0F)
    halves = []
    for nib in (even, odd):
        fp8 = fp4x4_to_fp8x4(nib, sign_and)
        halves.append((v_cvt_pk_f32_fp8(fp8, False), v_cvt_pk_f32_fp8(fp8, True)))
    (e_lo, e_hi), (o_lo, o_hi) = halves
    if "pkmul" in flags:
        e_lo, e_hi, o_lo, o_hi = (
            v_pk_mul_f32(v, scale_bits) for v in (e_lo, e_hi, o_lo, o_hi)
        )
    else:
        e_lo, e_hi, o_lo, o_hi = (
            tuple(v_mul_f32(x, scale_bits) for x in v) for v in (e_lo, e_hi, o_lo, o_hi)
        )
    f32s = [e_lo[0], o_lo[0], e_lo[1], o_lo[1], e_hi[0], o_hi[0], e_hi[1], o_hi[1]]
    out = []
    for i in range(4):
        b0, b1 = f32s[2 * i], f32s[2 * i + 1]
        if "perm" in flags:
            out.append(v_perm_b32(b1, b0, PERM_SEL_BF16_HI_PAIR))
        else:
            out.append((b0 >> U32(16)) | (b1 & U32(0xFFFF0000)))
    return np.stack(out, axis=-1)


def reference_bf16_bits(raw, scale_byte):
    """Math reference: E2M1 value x 2^(e-127), -0 -> +0, truncated to bf16 bits."""
    raw = np.asarray(raw, dtype=U32)
    s = e8m0_byte_to_f32_bits(scale_byte)
    words = []
    for i in range(4):
        halves = []
        for k in (2 * i, 2 * i + 1):
            n = (raw >> U32(4 * k)) & U32(0xF)
            v = np.array(E2M1, dtype=np.float32)[n & U32(7)]
            negative = ((n & U32(8)) != 0) & ((n & U32(7)) != 0)
            v = np.where(negative, -v, v).astype(np.float32)
            halves.append(v_mul_f32(v.view(U32), s) >> U32(16))
        words.append(halves[0] | (halves[1] << U32(16)))
    return np.stack(words, axis=-1)


def _cases(n_random=1 << 20, seed=0):
    rng = np.random.default_rng(seed)
    same_lane = (np.arange(256, dtype=U32) * U32(0x01010101)).repeat(256)
    same_lane_scale = np.tile(np.arange(256, dtype=U32), 256)
    rnd = rng.integers(0, 1 << 32, n_random, dtype=np.uint64).astype(U32)
    rnd_scale = rng.integers(0, 256, n_random, dtype=np.uint64).astype(U32)
    return np.concatenate([same_lane, rnd]), np.concatenate(
        [same_lane_scale, rnd_scale]
    )


def check_variant(variant, raw, scale_byte):
    s = e8m0_byte_to_f32_bits(scale_byte)
    base = decode(raw, s, "")
    got = decode(raw, s, variant)
    return int(np.any(base != got, axis=-1).sum())


def test_perm_semantics():
    # LUT gathers, the bf16 pack selector, and the special selector codes.
    assert int(v_perm_b32(LUT_HI, LUT_LO, 0x07060100)) == 0x54503800
    assert int(v_perm_b32(0xAABBCCDD, 0x11223344, PERM_SEL_BF16_HI_PAIR)) == 0xAABB1122
    # 8..11 replicate bit 15/31 of src1, then bit 15/31 of src0.
    assert int(v_perm_b32(0x80000000, 0x00008000, 0x0B0A0908)) == 0xFF0000FF
    assert int(v_perm_b32(0, 0, 0x0D0C0D0C)) == 0xFF00FF00


def test_baseline_matches_math_reference():
    raw, sb = _cases(n_random=1 << 16)
    got = decode(raw, e8m0_byte_to_f32_bits(sb), "")
    ref = reference_bf16_bits(raw, sb)
    finite = np.isfinite(e8m0_byte_to_f32_bits(sb).view(np.float32))
    # Scale byte 255 (+inf): 0*inf is NaN in both, NaN bits compare equal after canonicalization.
    assert np.array_equal(got[finite], ref[finite])


@pytest.mark.parametrize("variant", VARIANTS[1:])
def test_variant_bit_exact(variant):
    raw, sb = _cases()
    assert check_variant(variant, raw, sb) == 0


@pytest.mark.parametrize("variant", VARIANTS)
def test_all_16_nibbles_every_scale(variant):
    # One nibble value n in every position: raw = n * 0x11111111.
    nib = np.arange(16, dtype=U32).repeat(256)
    sb = np.tile(np.arange(256, dtype=U32), 16)
    raw = nib * U32(0x11111111)
    got = decode(raw, e8m0_byte_to_f32_bits(sb), variant)
    ref = decode(raw, e8m0_byte_to_f32_bits(sb), "")
    assert np.array_equal(got, ref)
    # E2M1 -0 (nibble 8) decodes to +0 at every finite scale.
    neg0 = (nib == 8) & (sb < 255)
    assert np.all(got[neg0] == 0)


def test_sign_trick_fp8_bytes_identical():
    # Stronger than the post-scale check: the fp8 codes themselves agree for all 16 nibbles.
    nib = np.arange(16, dtype=U32)
    lanes = nib * U32(0x01010101)
    assert np.array_equal(fp4x4_to_fp8x4(lanes, False), fp4x4_to_fp8x4(lanes, True))


def test_perm_pack_equals_shift_pack_any_dword():
    rng = np.random.default_rng(1)
    b0, b1 = rng.integers(0, 1 << 32, (2, 1 << 20), dtype=np.uint64).astype(U32)
    shift_pack = (b0 >> U32(16)) | (b1 & U32(0xFFFF0000))
    assert np.array_equal(v_perm_b32(b1, b0, PERM_SEL_BF16_HI_PAIR), shift_pack)


def test_negative_control_unmasked_sign_is_caught():
    # Without the AND mask's mag==0 guarantee (sign OR'd unconditionally) E2M1 -0 becomes
    # FNUZ 0x80 (NaN); the emulator must flag it.
    nib = U32(0x08080808)
    bad = v_perm_b32(LUT_HI, LUT_LO, nib & U32(0x07070707)) | ((nib & U32(8)) << U32(4))
    with pytest.raises(AssertionError, match="FNUZ NaN"):
        v_cvt_pk_f32_fp8(np.asarray([bad], dtype=U32), False)


def main():
    import pandas as pd

    raw, sb = _cases()
    rows = []
    for v in VARIANTS:
        rows.append(
            {
                "variant": v or "base",
                "cases": raw.size,
                "e8m0 scales": 256,
                "mismatching outputs": check_variant(v, raw, sb),
            }
        )
    print(pd.DataFrame(rows).to_markdown(index=False))


if __name__ == "__main__":
    main()
