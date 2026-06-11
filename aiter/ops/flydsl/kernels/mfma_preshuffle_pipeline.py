# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Shared MFMA preshuffle helpers for preshuffle GEMM kernels.

Key primitives:
- B preshuffle layout builder (supports byte-packed element types, incl. packed int4)
- B pack load for MFMA K32 micro-steps (8B output pack; optional int4->int8 unpack)
"""

from __future__ import annotations
from dataclasses import dataclass
from flydsl._mlir import ir
from flydsl.expr.typing import T
from flydsl.expr import arith as _arith
import flydsl.expr as fx


def crd2idx(crd, layout):
    """crd2idx returning an index-type scalar (unwraps fly.int_tuple)."""
    result = fx.crd2idx(crd, layout)
    scalar = fx.get_scalar(result)
    if isinstance(scalar, ir.Value) and not isinstance(scalar.type, ir.IndexType):
        scalar = _arith.IndexCastOp(T.index, scalar).result
    return scalar


def swizzle_xor16(row, col, k_blocks16):
    """XOR-with-row swizzle on the K dimension at 16B granularity.

    Computes: col XOR ((row & (k_blocks16 - 1)) * 16)

    k_blocks16 is always a power of 2 (tile_k_bytes / 16), so use
    bitwise AND instead of remui to save ~10 VALU cycles on CDNA.
    """
    from flydsl.expr import arith as _swz_arith

    mask = k_blocks16 - _swz_arith.index(1)
    rem = _swz_arith.andi(row, mask)
    return col ^ (rem * 16)


def lds_row_major_idx(row, col, row_stride, base=None):
    """Linearize a 2D LDS coordinate with explicit index arithmetic."""
    idx = row * row_stride + col
    return idx if base is None else idx + base


def split_row_major_2d(index, minor_extent):
    """Split a linear row-major index into (major, minor)."""
    return index // minor_extent, index % minor_extent


def _buffer_load_vec(
    buffer_ops,
    vector,
    rsrc,
    idx,
    *,
    elem_type,
    vec_elems,
    elem_bytes,
    offset_in_bytes,
    cache_modifier=0,
):
    """Load vec_elems elements via buffer_load dwordx[1,2,4] + bitcast."""
    from flydsl.expr import arith as _ld_arith

    elem_size = int(elem_bytes)
    load_bytes = int(vec_elems) * elem_size
    vec_width = load_bytes // 4

    if offset_in_bytes:
        idx_i32 = _ld_arith.shrui(idx, _ld_arith.index(2))
    elif elem_bytes == 2:
        idx_i32 = _ld_arith.shrui(idx, _ld_arith.index(1))
    else:
        idx_i32 = idx

    i32_val = buffer_ops.buffer_load(
        rsrc,
        idx_i32,
        vec_width=vec_width,
        dtype=T.i32,
        cache_modifier=cache_modifier,
    )
    if vec_width == 1:
        i32_vec = vector.from_elements(T.vec(1, T.i32), [i32_val])
    else:
        i32_vec = i32_val
    return vector.bitcast(T.vec(int(vec_elems), elem_type), i32_vec)


@dataclass(frozen=True)
class PreshuffleScaleLayout:
    """Container returned by `make_preshuffle_scale_layout`.

    The scale layout is ``(c_mn1, c_k1, 4, 16) : (stride_n0, stride_k0, stride_klane, 1)``.
    Callers compute flat index directly with plain arith::

        idx = mni * stride_n0 + ku * stride_k0 + k_lane * stride_klane + n_lane
    """

    layout_scale: object  # fly layout value (same as PreshuffleBLayout.layout_b)
    stride_n0: object  # index-typed MLIR value (dynamic)
    stride_k0: object  # index-typed MLIR value (= 64)
    stride_klane: object  # index-typed MLIR value (= 16)


def make_preshuffle_scale_layout(
    arith,
    *,
    c_mn: ir.Value,
    c_k: ir.Value,
    mn_pack: int = 2,
    k_pack: int = 2,
    elem_bytes: int = 4,
    scale_block_size: int = 32,
) -> PreshuffleScaleLayout:
    """Build scale layout matching aiter/CK preshuffle for FP4/FP8 microscale.

    Layout shape: ``(c_mn1, c_k1, 4, 16)`` where
    ``c_mn1 = c_mn / 16 / mn_pack`` and ``c_k1 = (c_k / scale_block_size) / 4 / k_pack``.
    """
    from .layout_utils import _div_pow2

    c16 = arith.constant(16, index=True)
    c4 = arith.constant(4, index=True)
    c_k_scale = _div_pow2(c_k, scale_block_size)

    c_mn1 = _div_pow2(_div_pow2(c_mn, 16), mn_pack)
    c_k1 = _div_pow2(_div_pow2(c_k_scale, 4), k_pack)
    if elem_bytes != mn_pack * k_pack:
        raise ValueError(
            f"elem_bytes of scale must be {mn_pack} * {k_pack}, got {elem_bytes!r}"
        )

    stride_klane = c16
    stride_k0 = c4 * stride_klane
    stride_n0 = c_k1 * stride_k0

    # Build fly layout (i32 strides for fx.make_layout).
    c_mn1_i32 = arith.index_cast(T.i32, c_mn1)
    c_k1_i32 = arith.index_cast(T.i32, c_k1)
    stride_n0_i32 = arith.index_cast(T.i32, stride_n0)
    stride_k0_i32 = arith.index_cast(T.i32, stride_k0)
    stride_klane_i32 = arith.index_cast(T.i32, stride_klane)

    layout_scale = fx.make_layout(
        (c_mn1_i32, c_k1_i32, 4, 16),
        stride=(stride_n0_i32, stride_k0_i32, stride_klane_i32, 1),
    )

    return PreshuffleScaleLayout(
        layout_scale=layout_scale,
        stride_n0=stride_n0,
        stride_k0=stride_k0,
        stride_klane=stride_klane,
    )


@dataclass(frozen=True)
class PreshuffleBLayout:
    """Container returned by `make_preshuffle_b_layout`."""

    layout_b: object
    kpack_bytes: int


def make_preshuffle_b_layout(
    arith,
    *,
    c_n: ir.Value,
    c_k: ir.Value,
    kpack_bytes: int = 16,
    elem_bytes: int = 1,
    k_major: bool = False,
) -> PreshuffleBLayout:
    """Build B layout matching aiter/CK preshuffle for A8 MFMA kernels.

    When *k_major* is True the block-level order is K-major (``k_blk`` outermost),
    matching the ``(0,3,1,4,2,5)`` shuffle permutation.  The default N-major
    order (``k_major=False``) matches the legacy ``(0,1,3,4,2,5)`` permutation.
    """
    if kpack_bytes not in (8, 16):
        raise ValueError(f"kpack_bytes must be 8 or 16, got {kpack_bytes!r}")

    c16 = arith.constant(16, index=True)
    c_kpack = arith.constant(kpack_bytes, index=True)

    from .layout_utils import _div_pow2

    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    c_k_bytes = c_k * arith.constant(int(elem_bytes), index=True)
    c_k0 = _div_pow2(c_k_bytes, 64)
    n0 = _div_pow2(c_n, 16)

    c_kpack_elems = c_kpack if elem_bytes == 1 else _div_pow2(c_kpack, int(elem_bytes))

    stride_nlane = c_kpack_elems

    if k_major:
        c32 = arith.constant(32, index=True)
        c2 = arith.constant(2, index=True)
        c_k0 = c_k_bytes // c32
        klane_dim = 2
        stride_klane = c16 * stride_nlane
        stride_n0 = c2 * stride_klane
        stride_k0 = n0 * stride_n0
    else:
        c64 = arith.constant(64, index=True)
        c4 = arith.constant(4, index=True)
        c_k0 = c_k_bytes // c64
        klane_dim = 4
        stride_klane = c16 * stride_nlane
        stride_k0 = c4 * stride_klane
        stride_n0 = c_k0 * stride_k0

    # fly.make_shape requires i32/i64 for dynamic operands (not index).
    # Convert dynamic index values to i32; use Python ints for static constants.
    kpack_elems_static = kpack_bytes if elem_bytes == 1 else kpack_bytes // elem_bytes
    n0_i32 = arith.index_cast(T.i32, n0)
    c_k0_i32 = arith.index_cast(T.i32, c_k0)
    stride_n0_i32 = arith.index_cast(T.i32, stride_n0)
    stride_k0_i32 = arith.index_cast(T.i32, stride_k0)
    stride_klane_i32 = arith.index_cast(T.i32, stride_klane)
    stride_nlane_i32 = arith.index_cast(T.i32, stride_nlane)

    stride_b = (stride_n0_i32, stride_k0_i32, stride_klane_i32, stride_nlane_i32, 1)
    layout_b = fx.make_layout(
        (n0_i32, c_k0_i32, klane_dim, 16, kpack_elems_static), stride_b
    )
    return PreshuffleBLayout(layout_b=layout_b, kpack_bytes=kpack_bytes)


def _unpack_int4_to_int8_pair(packed32):
    """Split packed int4 dword into two int8 dwords (even/odd nibbles).

    7-op bit manipulation shared by all int4 unpack paths (W4A8, W4A16, W4A_FP8).
    """
    c_08 = fx.Int32(0x08080808)
    c_0f = fx.Int32(0x0F0F0F0F)
    c_1e = fx.Int32(0x1E)
    c_4 = fx.Int32(4)
    s0 = (packed32 & c_08) * c_1e
    even = (packed32 & c_0f) | s0
    t = packed32 >> c_4
    s1 = (t & c_08) * c_1e
    odd = (t & c_0f) | s1
    return even, odd


def _map4_e2m1_to_e4m3fnuz(nibs, exp_add_i32=None):
    """4 E2M1 (FP4) nibbles (one in the low 4 bits of each byte of `nibs`) ->
    4 e4m3fnuz (FP8) bytes, packed into one i32.

    Branchless SIMD over the 4 bytes, same multiply-by-mask style as the int4
    `*0x1E` sign-extend trick: no per-lane select, no vector.extract.

    Nibble bits = [s | e1 e0 | m0]; mag = e1e0m0 (0..7). Single closed form
    (verified vs all 16 codes): magnitude_byte = 0x34 + 4*(mag + [mag>=2]),
    then forced to 0x00 at mag==0; sign bit (bit3->bit7) is OR'd in, zero-guarded
    so -0 -> +0 (gfx942 fp8 = e4m3fnuz, where 0x80 == NaN).
      mag : 0    .5    1     1.5   2     3     4     6
      out : 0x00 0x38  0x40  0x44  0x48  0x4C  0x50  0x54
    All ops are per-byte carry-free (mag+ge2 <= 8, <<2 <= 0x20, +0x34 <= 0x54).

    exp_add_i32: optional packed-per-byte exponent delta (already <<3, i.e. in the
    e4m3fnuz exponent-field position bits 3..6), used to FOLD the MXFP4 block-32
    E8M0 weight scale into the e4m3fnuz exponent (W4A8 block-32 path). Bit-exact for
    in-range weights (verified 0% saturate/underflow on DSV4); no saturation yet.
    """
    M = nibs & fx.Int32(0x07070707)                       # magnitude bits per byte
    M1 = M >> fx.Int32(1)
    M2 = M >> fx.Int32(2)
    ge1 = (M | M1 | M2) & fx.Int32(0x01010101)            # per byte: 1 if mag>=1
    ge2 = (M1 | M2) & fx.Int32(0x01010101)                # per byte: 1 if mag>=2
    mag_byte = fx.Int32(0x34343434) + ((M + ge2) << fx.Int32(2))  # 0x34 + 4*(mag + [mag>=2])
    if exp_add_i32 is not None:
        # Fold MXFP4 block-32 E8M0 scale: per-byte (carry-suppressed SWAR) add of the
        # exponent delta (= delta<<3, broadcast x4) into the e4m3fnuz exponent field.
        # per-byte (a+b) mod 256 = ((a&0x7F..)+(b&0x7F..)) ^ ((a^b)&0x80..).
        _a = mag_byte
        _b = exp_add_i32
        _swar = ((_a & fx.Int32(0x7F7F7F7F)) + (_b & fx.Int32(0x7F7F7F7F))) ^ (
            (_a ^ _b) & fx.Int32(0x80808080)
        )
        # Clamp per-byte out-of-range exponents (bit7 set in the carry-free result):
        # delta>=0 (exp_add bit7=0) -> overflow -> saturate 0x7F; delta<0 -> underflow
        # -> flush 0x00. Real MXFP4 weights have ~1% blocks whose exp+delta leaves
        # e4m3fnuz's range; without this they wrap to 0x80(NaN)/garbage -> model NaN.
        _oor = ((_swar >> fx.Int32(7)) & fx.Int32(0x01010101)) * fx.Int32(0xFF)
        _neg = ((_b >> fx.Int32(7)) & fx.Int32(0x01010101)) * fx.Int32(0xFF)
        _clampv = fx.Int32(0x7F7F7F7F) ^ (_neg & fx.Int32(0x7F7F7F7F))
        mag_byte = _swar ^ (_oor & (_swar ^ _clampv))
    mag_byte = mag_byte & (ge1 * fx.Int32(0xFF))          # -> 0x00 where mag==0
    # Sign only where the FINAL mag_byte != 0. Covers mag==0, underflow-flush AND the
    # exp+delta==0 edge (SWAR yields 0x00 with bit7 clear, so an out-of-range mask would
    # MISS it -> 0x00|0x80=NaN). Per-byte "byte!=0" via carry-free SWAR hasvalue:
    # (((b&0x7F..)+0x7F..) | b) & 0x80.. sets bit7 per byte iff that byte is nonzero.
    _nz80 = (((mag_byte & fx.Int32(0x7F7F7F7F)) + fx.Int32(0x7F7F7F7F)) | mag_byte) & fx.Int32(
        0x80808080
    )
    sign = ((nibs & fx.Int32(0x08080808)) << fx.Int32(4)) & _nz80
    return mag_byte | sign


def _unpack_fp4_to_fp8_pair(packed32, exp_add_i32=None):
    """Split packed FP4 (E2M1) dword into two FP8 (e4m3fnuz) dwords (even/odd nibbles).

    Mirrors `_unpack_int4_to_int8_pair`: even = low nibble of each byte, odd = high nibble.
    Feeds rocdl.mfma_f32_16x16x32_fp8_fp8 (W4A8 FP8-act x FP4-weight on gfx942/CDNA3).

    exp_add_i32: optional MXFP4 block-32 E8M0 fold (see _map4_e2m1_to_e4m3fnuz). Both
    nibbles of a byte are adjacent K -> same 32-block -> one delta for even and odd.
    """
    c_0f = fx.Int32(0x0F0F0F0F)
    even = _map4_e2m1_to_e4m3fnuz(packed32 & c_0f, exp_add_i32)
    odd = _map4_e2m1_to_e4m3fnuz((packed32 >> fx.Int32(4)) & c_0f, exp_add_i32)
    return even, odd


def _pack_i32_pair_to_i64(lo, hi, vector):
    """Pack two i32 values into one i64 via vector bitcast."""
    v2 = vector.from_elements(T.vec(2, T.i32), [lo, hi])
    v64 = vector.bitcast(T.vec(1, T.i64), v2)
    return vector.extract(v64, static_position=[0], dynamic_position=[])


def _i8x4_in_i32_to_bf16x4_i64(val_i32, arith, vector, scale_val=None):
    """Convert one i32 (4 signed int8 bytes) to 4 bf16 packed as i64.

    Uses shift-based f32->bf16 truncation (lshr 16) instead of arith.truncf
    which on gfx942 expands to ~5 VALU per element. The shift is exact for
    unscaled int8 values and introduces <0.5 ULP error for scaled values.
    """
    vec1_i32_t = T.vec(1, T.i32)
    vec2_i32 = T.i32x2
    vec4_i8 = T.i8x4
    vec1_i64 = T.vec(1, T.i64)

    v1 = vector.from_elements(vec1_i32_t, [val_i32])
    i8x4 = vector.bitcast(vec4_i8, v1)

    f32_vals = []
    for i in range(4):
        val_i8 = vector.extract(i8x4, static_position=[i], dynamic_position=[])
        v = arith.sitofp(T.f32, val_i8)
        if scale_val is not None:
            v = v * scale_val
        f32_vals.append(v)

    c16 = fx.Int32(16)
    c_ffff0000 = fx.Int32(0xFFFF0000)
    bits0 = arith.bitcast(T.i32, f32_vals[0])
    bits1 = arith.bitcast(T.i32, f32_vals[1])
    bits2 = arith.bitcast(T.i32, f32_vals[2])
    bits3 = arith.bitcast(T.i32, f32_vals[3])
    i32_lo = (bits0 >> c16) | (bits1 & c_ffff0000)
    i32_hi = (bits2 >> c16) | (bits3 & c_ffff0000)

    v2 = vector.from_elements(vec2_i32, [i32_lo, i32_hi])
    v64 = vector.bitcast(vec1_i64, v2)
    return vector.extract(v64, static_position=[0], dynamic_position=[])


def _e4m3x4_in_i32_to_bf16x4_i64(val_i32, arith, vector, scale_val=None):
    """4 e4m3fnuz bytes (NATURAL E2M1 magnitudes from _map4, all normal exp 7..10 or 0x00)
    packed in an i32 -> 4 bf16 packed as i64, optionally x scale_val (f32). W4A16 FP4 weight
    dequant: e4m3fnuz(bias 8) -> f32(bias 127) is exp += 119, mant <<= 20, plus sign; then
    x scale (E8M0 = 2^(e8m0-127), supplied as f32). Operates on the i32 (byte i shifted to
    low 8 bits) to avoid i8 signedness. (Natural E2M1 has no subnormals; the rare code-0
    weight maps to ~2^-8 x scale ~= 2^-15, negligible.)"""
    from flydsl._mlir.dialects._arith_ops_gen import MulFOp as _MulFOp

    _uw = _arith._to_raw
    _av = _arith.ArithValue
    f32_vals = []
    for i in range(4):
        bsh = val_i32 >> fx.Int32(i * 8)
        bexp = (bsh >> fx.Int32(3)) & fx.Int32(0xF)
        bmant = bsh & fx.Int32(0x7)
        bsign = (bsh & fx.Int32(0x80)) << fx.Int32(24)
        fbits = bsign | ((bexp + fx.Int32(119)) << fx.Int32(23)) | (bmant << fx.Int32(20))
        # Zero the code-0 weight (E2M1 0.0 -> e4m3 byte 0x00 -> bexp 0). Without this it
        # maps to 2^-8 x scale: a SYSTEMATIC positive bias on every ~zero weight (~10% of
        # weights) that is tiny per-layer (cosine 0.99999) but compounds across 60 layers
        # and drifts GSM8K. Natural E2M1 nonzero codes have bexp in 7..10, so bexp!=0 marks
        # nonzero. nzmask = 0xFFFFFFFF iff bexp!=0 else 0.
        _bnz = (
            bexp | (bexp >> fx.Int32(1)) | (bexp >> fx.Int32(2)) | (bexp >> fx.Int32(3))
        ) & fx.Int32(1)
        fbits = fbits & (fx.Int32(0) - _bnz)
        # bitcast needs a raw mlir Value; _uw unwraps the ArithValue operator result.
        v = arith.bitcast(T.f32, _uw(fbits))
        f32_vals.append(v)
    if scale_val is not None:
        raw_scale = _uw(scale_val)
        f32_vals = [_MulFOp(v, raw_scale).result for v in f32_vals]
    c16_shift = fx.Int32(16)
    c_ffff0000 = fx.Int32(0xFFFF0000)
    bf16_vals = [arith.bitcast(T.i32, _av(v)) for v in f32_vals]
    i32_lo = (bf16_vals[0] >> c16_shift) | (bf16_vals[1] & c_ffff0000)
    i32_hi = (bf16_vals[2] >> c16_shift) | (bf16_vals[3] & c_ffff0000)
    v2 = vector.from_elements(T.vec(2, T.i32), [i32_lo, i32_hi])
    v64 = vector.bitcast(T.vec(1, T.i64), v2)
    return vector.extract(v64, static_position=[0], dynamic_position=[])


def unpack_b_w4a16_fp4(packed32, scale_val, arith, vector):
    """W4A16 FP4 (E2M1) groupwise unpack: packed FP4 dword -> (b0, b1) two i64 of 4 bf16
    each, x scale. Mirrors unpack_b_w4a16 (int4) but with E2M1->bf16 dequant + E8M0 scale
    (2^delta as f32). Feeds rocdl.mfma_f32_16x16x16_bf16 (bf16-act x FP4-weight W4A16)."""
    even, odd = _unpack_fp4_to_fp8_pair(packed32)  # natural E2M1 -> e4m3 (no fold)
    b0 = _e4m3x4_in_i32_to_bf16x4_i64(even, arith, vector, scale_val=scale_val)
    b1 = _e4m3x4_in_i32_to_bf16x4_i64(odd, arith, vector, scale_val=scale_val)
    return (b0, b1)


def load_b_raw_w4a16(
    buffer_ops,
    arith,
    vector,
    *,
    arg_b,
    b_rsrc,
    layout_b,
    base_k: ir.Value,
    ku: int,
    n_blk: ir.Value,
    n_intra: ir.Value,
    lane_div_16: ir.Value,
    elem_type: ir.Type,
    kpack_bytes: int = 8,
):
    """Phase 1 of W4A16 B load: issue buffer_load_dword, return raw packed i32.

    Same address calculation as the int4 unpack path in load_b_pack_k32
    but using ku-based indexing for 2-phase latency hiding.
    """
    if kpack_bytes != 8:
        raise ValueError(f"W4A16 requires kpack_bytes=8, got {kpack_bytes!r}")

    c64 = fx.Index(64)
    half_bytes = kpack_bytes // 2
    c2_idx = fx.Index(2)
    c4_idx = fx.Index(4)

    k0_base = base_k // c64
    k1_layout_offset = ku * 2
    lane_div_32 = lane_div_16 // c2_idx
    total_k1 = fx.Index(k1_layout_offset) + lane_div_32
    k0 = k0_base + (total_k1 // c4_idx)
    k1_local = total_k1 % c4_idx
    lane_odd = lane_div_16 % c2_idx
    k2_base = lane_odd * fx.Index(half_bytes)

    coord_pack = (n_blk, k0, k1_local, n_intra, fx.Index(0))
    idx_pack = crd2idx(coord_pack, layout_b)
    idx_bytes = idx_pack + k2_base

    b4 = _buffer_load_vec(
        buffer_ops,
        vector,
        b_rsrc,
        idx_bytes,
        elem_type=elem_type,
        vec_elems=4,
        elem_bytes=1,
        offset_in_bytes=True,
    )
    packed32 = vector.extract(
        vector.bitcast(T.vec(1, T.i32), b4),
        static_position=[0],
        dynamic_position=[],
    )
    return packed32


def _int4_to_bf16x4_i64_gfx950(
    packed32, nibble_offsets, arith, vector, scale_val=None, defer_scale16=False
):
    """Convert 4 int4 nibbles to 4 bf16 packed as i64 using gfx950 instructions.

    Uses v_cvt_off_f32_i4_sdwa with byte_sel to avoid per-nibble shifts.
    Even nibbles (0,2,4,6) → SDWA BYTE_0/1/2/3 on original src.
    Odd nibbles (1,3,5,7)  → SDWA BYTE_0/1/2/3 on (src >> 4).
    Only 1 shift total instead of 7.

    When defer_scale16=True, the ×16 correction factor for v_cvt_off_f32_i4 is
    omitted and must be applied later (e.g. in the epilogue).  This saves VALU
    in the hot loop and uses v_cvt_pk_bf16_f32 for proper f32→bf16 conversion.
    """
    from flydsl.expr import rocdl
    from flydsl._mlir.dialects._arith_ops_gen import MulFOp as _MulFOp

    _uw = _arith._to_raw
    _av = _arith.ArithValue

    src_even = packed32
    src_odd = packed32 >> fx.Int32(4)

    f32_vals = []
    for nib in nibble_offsets:
        byte_idx = nib // 2
        src = src_odd if (nib % 2) else src_even
        v = rocdl.cvt_off_f32_i4(src, byte_sel=byte_idx)
        f32_vals.append(v)

    if defer_scale16:
        # Skip ×16; multiply by scale_val only if groupwise.
        if scale_val is not None:
            raw_scale = _uw(scale_val)
            f32_vals = [_MulFOp(v, raw_scale).result for v in f32_vals]
        # Use v_cvt_pk_bf16_f32 for proper f32→bf16 (no bit-shift trick needed).
        i32_lo = rocdl.cvt_pk_bf16_f32(f32_vals[0], f32_vals[1])
        i32_hi = rocdl.cvt_pk_bf16_f32(f32_vals[2], f32_vals[3])
    else:
        c16 = fx.Float32(16.0)
        if scale_val is not None:
            effective_scale = scale_val * c16
        else:
            effective_scale = c16
        raw_scale = _uw(effective_scale)
        f32_vals = [_MulFOp(v, raw_scale).result for v in f32_vals]
        # Truncate f32→bf16 via bit-shift (exact for scaled int values).
        c16_shift = fx.Int32(16)
        c_ffff0000 = fx.Int32(0xFFFF0000)
        bf16_vals = [arith.bitcast(T.i32, _av(v)) for v in f32_vals]
        i32_lo = (bf16_vals[0] >> c16_shift) | (bf16_vals[1] & c_ffff0000)
        i32_hi = (bf16_vals[2] >> c16_shift) | (bf16_vals[3] & c_ffff0000)

    v2 = vector.from_elements(T.vec(2, T.i32), [i32_lo, i32_hi])
    v64 = vector.bitcast(T.vec(1, T.i64), v2)
    return vector.extract(v64, static_position=[0], dynamic_position=[])


def unpack_b_w4a16(
    packed32, arith, vector, scale_val=None, use_gfx950_cvt=False, defer_scale16=False
):
    """Phase 2 of W4A16 B load: unpack int4->int8 + convert int8->bf16.

    Takes raw packed32 from load_b_raw_w4a16 and produces (b0, b1) --
    two i64 values each containing 4 bf16 for one MFMA.

    When use_gfx950_cvt=True, uses v_cvt_off_f32_i4 + v_cvt_pk_bf16_f32
    for ~2x fewer VALU instructions.

    When defer_scale16=True (requires use_gfx950_cvt=True), the ×16
    correction for v_cvt_off_f32_i4 is omitted; caller must apply it
    in the epilogue.
    """
    if use_gfx950_cvt:
        b0 = _int4_to_bf16x4_i64_gfx950(
            packed32,
            [0, 2, 4, 6],
            arith,
            vector,
            scale_val,
            defer_scale16=defer_scale16,
        )
        b1 = _int4_to_bf16x4_i64_gfx950(
            packed32,
            [1, 3, 5, 7],
            arith,
            vector,
            scale_val,
            defer_scale16=defer_scale16,
        )
        return (b0, b1)
    even, odd = _unpack_int4_to_int8_pair(packed32)
    b0 = _i8x4_in_i32_to_bf16x4_i64(even, arith, vector, scale_val=scale_val)
    b1 = _i8x4_in_i32_to_bf16x4_i64(odd, arith, vector, scale_val=scale_val)
    return (b0, b1)


def load_b_pack_k32(
    buffer_ops,
    arith,
    vector,
    *,
    arg_b,
    b_rsrc,
    layout_b,
    base_k: ir.Value,
    ki_step: int,
    n_blk: ir.Value,
    n_intra: ir.Value,
    lane_div_16: ir.Value,
    elem_type: ir.Type,
    kpack_bytes: int = 16,
    elem_bytes: int = 1,
    unpack_int4: bool = False,
    unpack_fp4: bool = False,
    fp4_scale_rsrc=None,
    fp4_expert_offset=None,
    fp4_num_groups: int = 1,
    fp4_group_size: int = 32,
    fp4_n_per_expert: int = 0,
) -> ir.Value:
    """Load one B pack for one MFMA(x32) micro-step.

    Returns an i64 Value containing 8 bytes consumed by MFMA.

    unpack_int4: packed int4 -> int8 (W4A8 int8 MFMA path).
    unpack_fp4:  packed E2M1 -> e4m3fnuz (W4A8 FP8 MFMA / afp8_wfp4 path).
    """
    if kpack_bytes not in (8, 16):
        raise ValueError(f"kpack_bytes must be 8 or 16, got {kpack_bytes!r}")
    if (unpack_int4 or unpack_fp4) and kpack_bytes != 8:
        raise ValueError("unpack_int4/unpack_fp4 requires kpack_bytes=8 (packed 4-bit layout)")
    if unpack_int4 and unpack_fp4:
        raise ValueError("unpack_int4 and unpack_fp4 are mutually exclusive")
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")

    c64 = fx.Index(64)
    base_k_bytes = base_k * arith.constant(int(elem_bytes), index=True)
    k0_base = base_k_bytes // c64
    k0 = k0_base + arith.constant(ki_step // 2, index=True)
    k1 = lane_div_16
    half_bytes = kpack_bytes // 2
    k2_base = arith.constant((ki_step % 2) * half_bytes, index=True)

    coord_pack = (n_blk, k0, k1, n_intra, fx.Index(0))
    idx_pack = crd2idx(coord_pack, layout_b)

    if unpack_int4 or unpack_fp4:
        idx_bytes = idx_pack + k2_base
        b4 = _buffer_load_vec(
            buffer_ops,
            vector,
            b_rsrc,
            idx_bytes,
            elem_type=elem_type,
            vec_elems=4,
            elem_bytes=1,
            offset_in_bytes=True,
        )
        packed32 = vector.extract(
            vector.bitcast(T.vec(1, T.i32), b4),
            static_position=[0],
            dynamic_position=[],
        )
        if unpack_fp4:
            exp_add_i32 = None
            if fp4_scale_rsrc is not None:
                # block-32 FP4 W4A8 group mapping (verified by probe):
                #   ku = ki_step//2 (K64 micro-step); within a ku the 64 K = two 32-blocks
                #   (2*ku, 2*ku+1) and the 4 K-lanes split lanes{0,1}->block 2ku,
                #   lanes{2,3}->block 2ku+1; ki_step%2 is the 16-K half (same block).
                #   => group = base_k//32 + 2*(ki_step//2) + lane_div_16//2.
                lane_blk = lane_div_16 // fx.Index(2)
                k_pos = (
                    base_k
                    + fx.Index(2 * (ki_step // 2) * fp4_group_size)
                    + lane_blk * fx.Index(fp4_group_size)
                )
                exp_add_i32 = _load_groupwise_scale(
                    buffer_ops,
                    arith,
                    scale_rsrc=fp4_scale_rsrc,
                    expert_offset=fp4_expert_offset,
                    n_blk=n_blk,
                    n_intra=n_intra,
                    k_pos=k_pos,
                    num_groups=fp4_num_groups,
                    group_size=fp4_group_size,
                    n_per_expert=fp4_n_per_expert,
                    scale_dtype=T.i32,
                )
            even, odd = _unpack_fp4_to_fp8_pair(packed32, exp_add_i32)
        else:
            even, odd = _unpack_int4_to_int8_pair(packed32)
        return _pack_i32_pair_to_i64(even, odd, vector)

    vec_elems = kpack_bytes // int(elem_bytes)
    b16 = _buffer_load_vec(
        buffer_ops,
        vector,
        b_rsrc,
        idx_pack,
        elem_type=elem_type,
        vec_elems=vec_elems,
        elem_bytes=elem_bytes,
        offset_in_bytes=(elem_bytes == 1),
    )

    b_i32x4 = vector.bitcast(T.i32x4, b16)

    half = ki_step % 2
    if half == 0:
        d0 = vector.extract(b_i32x4, static_position=[0], dynamic_position=[])
        d1 = vector.extract(b_i32x4, static_position=[1], dynamic_position=[])
    else:
        d0 = vector.extract(b_i32x4, static_position=[2], dynamic_position=[])
        d1 = vector.extract(b_i32x4, static_position=[3], dynamic_position=[])

    v2 = vector.from_elements(T.vec(2, T.i32), [d0, d1])
    v64 = vector.bitcast(T.vec(1, T.i64), v2)
    return vector.extract(v64, static_position=[0], dynamic_position=[])


def tile_chunk_coord_i32(
    arith,
    *,
    tx_i32_base: ir.Value,
    i: int,
    total_threads: int,
    layout_tile_div4,
    chunk_i32: int = 4,
):
    """Map (thread, chunk_id) -> (row_local, col_local_i32) for X/A loads."""
    if chunk_i32 not in (1, 2, 4):
        raise ValueError(f"chunk_i32 must be one of (1,2,4), got {chunk_i32!r}")
    chunk_off_i32 = arith.constant(i * total_threads * chunk_i32, index=True)
    tile_idx_i32 = tx_i32_base + chunk_off_i32
    coord_local = fx.idx2crd(tile_idx_i32, layout_tile_div4)
    row_local = fx.get(coord_local, 0)
    col_local_i32 = fx.get(coord_local, 1)
    return row_local, col_local_i32


def buffer_copy_gmem16_dwordx4(
    buffer_ops,
    vector,
    *,
    elem_type,
    idx_i32: ir.Value,
    rsrc,
    vec_elems: int = 16,
    elem_bytes: int = 1,
):
    """Copy 16 bytes from global memory into regs via buffer-load dwordx4 lowering."""
    if int(vec_elems) <= 0:
        raise ValueError(f"vec_elems must be > 0, got {vec_elems!r}")
    return _buffer_load_vec(
        buffer_ops,
        vector,
        rsrc,
        idx_i32,
        elem_type=elem_type,
        vec_elems=vec_elems,
        elem_bytes=elem_bytes,
        offset_in_bytes=False,
    )


def lds_store_16b_xor16(
    arith,
    vector,
    *,
    lds_memref,
    vec16_ty,
    layout_lds,
    row_local: ir.Value,
    col_local_i32: ir.Value,
    tx_c4: ir.Value,
    k_blocks16: ir.Value,
    lds_base: ir.Value,
    vec_part_i32x4: ir.Value,
    elem_bytes: int = 1,
):
    """Store one 16B chunk into LDS with CK-style XOR16 swizzle on the K dimension."""
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    col_local_bytes = col_local_i32 * tx_c4
    col_swz_bytes = swizzle_xor16(row_local, col_local_bytes, k_blocks16)
    col_swz = col_swz_bytes if elem_bytes == 1 else col_swz_bytes // 2
    coord_store = (row_local, col_swz)
    idx0 = crd2idx(coord_store, layout_lds) + lds_base
    v16 = vector.bitcast(vec16_ty, vec_part_i32x4)
    vector.store(v16, lds_memref, [idx0])


def lds_store_8b_xor16(
    arith,
    vector,
    *,
    lds_memref,
    vec8_ty,
    layout_lds,
    row_local: ir.Value,
    col_local_i32: ir.Value,
    tx_c4: ir.Value,
    k_blocks16: ir.Value,
    lds_base: ir.Value,
    vec_part_i32x2: ir.Value,
    elem_bytes: int = 1,
):
    """Store one 8B chunk into LDS with CK-style XOR16 swizzle on the K dimension."""
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    col_local_bytes = col_local_i32 * tx_c4
    col_swz_bytes = swizzle_xor16(row_local, col_local_bytes, k_blocks16)
    col_swz = col_swz_bytes if elem_bytes == 1 else col_swz_bytes // 2
    coord_store = (row_local, col_swz)
    idx0 = crd2idx(coord_store, layout_lds) + lds_base
    v8 = vector.bitcast(vec8_ty, vec_part_i32x2)
    vector.store(v8, lds_memref, [idx0])


def lds_store_4b_xor16(
    arith,
    vector,
    *,
    lds_memref,
    vec4_ty,
    layout_lds,
    row_local: ir.Value,
    col_local_i32: ir.Value,
    tx_c4: ir.Value,
    k_blocks16: ir.Value,
    lds_base: ir.Value,
    vec_part_i32x1: ir.Value,
    elem_bytes: int = 1,
):
    """Store one 4B chunk into LDS with CK-style XOR16 swizzle on the K dimension."""
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    col_local_bytes = col_local_i32 * tx_c4
    col_swz_bytes = swizzle_xor16(row_local, col_local_bytes, k_blocks16)
    col_swz = col_swz_bytes if elem_bytes == 1 else col_swz_bytes // 2
    coord_store = (row_local, col_swz)
    idx0 = crd2idx(coord_store, layout_lds) + lds_base
    v4 = vector.bitcast(vec4_ty, vec_part_i32x1)
    vector.store(v4, lds_memref, [idx0])


def lds_load_pack_k32(
    arith,
    vector,
    *,
    lds_memref,
    layout_lds,
    k_blocks16: ir.Value,
    curr_row_a_lds: ir.Value,
    col_base: ir.Value,
    half: int,
    lds_base: ir.Value,
    ck_lds128: bool,
    vec16_ty,
    vec8_ty,
    vec2_i64_ty,
    vec1_i64_ty,
):
    """Load one i64 A-pack for an MFMA K32 micro-step from LDS."""
    col_base_swz = swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
    if ck_lds128:
        coord_a16 = (curr_row_a_lds, col_base_swz)
        idx_a16 = crd2idx(coord_a16, layout_lds) + lds_base
        loaded_a16 = vector.load_op(vec16_ty, lds_memref, [idx_a16])
        a_vec128 = vector.bitcast(vec2_i64_ty, loaded_a16)
        return vector.extract(a_vec128, static_position=[half], dynamic_position=[])
    else:
        col_swizzled = col_base_swz + (half * 8)
        coord_a = (curr_row_a_lds, col_swizzled)
        idx_a = crd2idx(coord_a, layout_lds) + lds_base
        loaded_a8 = vector.load_op(vec8_ty, lds_memref, [idx_a])
        a_vec64 = vector.bitcast(vec1_i64_ty, loaded_a8)
        return vector.extract(a_vec64, static_position=[0], dynamic_position=[])


__all__ = [
    "PreshuffleBLayout",
    "PreshuffleScaleLayout",
    "buffer_copy_gmem16_dwordx4",
    "lds_load_pack_k32",
    "lds_row_major_idx",
    "lds_store_4b_xor16",
    "lds_store_8b_xor16",
    "lds_store_16b_xor16",
    "make_preshuffle_b_layout",
    "make_preshuffle_scale_layout",
    "load_b_pack_k32",
    "load_b_raw_w4a16",
    "unpack_b_w4a16",
    "load_b_raw_w4a16_groupwise",
    "unpack_b_w4a16_groupwise",
    "extract_bf16_scale",
    "split_row_major_2d",
    "swizzle_xor16",
    "tile_chunk_coord_i32",
]


# ---------------------------------------------------------------------------
# Groupwise scale load helper (shared by W4A16 and W4A8 groupwise paths)
# ---------------------------------------------------------------------------


def _load_groupwise_scale(
    buffer_ops,
    arith,
    *,
    scale_rsrc,
    expert_offset,
    n_blk,
    n_intra,
    k_pos,
    num_groups: int,
    group_size: int,
    n_per_expert: int,
    scale_dtype=None,
):
    """Load one per-group scale value from the scale buffer.

    Computes the linear index into the scale tensor from expert offset,
    N position, and group index derived from ``k_pos``.

    For bf16 scales the tensor uses ``(E, G//2, N, 2)`` layout — two
    adjacent groups for the same N position are packed into one dword.
    We load the raw i32 dword (no extraction) so it can be carried as
    loop state without register copies.  Use :func:`extract_bf16_scale`
    in the compute phase to obtain the f32 value.
    """
    c16 = fx.Index(16)
    n_global = n_blk * c16 + n_intra
    c_group_size = fx.Index(group_size)
    c_npe = fx.Index(n_per_expert)
    group_idx = k_pos // c_group_size
    if scale_dtype is None:
        scale_dtype = T.f32

    if scale_dtype == T.bf16:
        # (E, G//2, N, 2) layout: dword at [e, pair, n] holds bf16 scales
        # for groups 2*pair and 2*pair+1.
        pair_idx = group_idx >> fx.Index(1)  # group_idx // 2
        # Flat dword index: expert_offset * (num_pairs-1) + n_global
        # The (num_pairs-1) cancels the expert part of n_global:
        #   e*N*(G//2-1) + (e*N + n_local) = e*N*G//2 + n_local
        num_pairs = num_groups // 2
        c_npm1 = fx.Index(num_pairs - 1)
        dword_base = expert_offset * c_npm1 + n_global
        dword_elem = dword_base + pair_idx * c_npe
        dword_idx = arith.index_cast(T.i32, dword_elem)
        # Return raw i32 dword — extraction deferred to compute phase.
        scale_val = buffer_ops.buffer_load(
            scale_rsrc, dword_idx, vec_width=1, dtype=T.i32
        )
    elif scale_dtype == T.i32:
        # MXFP4 block-32 fold delta: (E, G, N) layout, one precomputed i32 dword
        # per (expert, group, n) = ((e8m0-127)<<3 & 0xFF) broadcast x4. Loaded raw
        # and SWAR-added into the e4m3fnuz exponent in _map4_e2m1_to_e4m3fnuz
        # (W4A8 block-32 FP4 path). Same flat index as the f32 (E, G, N) path.
        c_gm1 = fx.Index(num_groups - 1)
        base_scale = expert_offset * c_gm1 + n_global
        elem_idx = base_scale + group_idx * c_npe
        scale_idx_i32 = arith.index_cast(T.i32, elem_idx)
        scale_val = buffer_ops.buffer_load(
            scale_rsrc, scale_idx_i32, vec_width=1, dtype=T.i32
        )
    else:
        # (E, G, N) layout with f32 dtype
        # Flat index: expert_offset * (G-1) + n_global
        # The (G-1) cancels the expert part of n_global:
        #   e*N*(G-1) + (e*N + n_local) = e*N*G + n_local
        c_gm1 = fx.Index(num_groups - 1)
        base_scale = expert_offset * c_gm1 + n_global
        elem_idx = base_scale + group_idx * c_npe
        scale_idx_i32 = arith.index_cast(T.i32, elem_idx)
        scale_val = buffer_ops.buffer_load(
            scale_rsrc, scale_idx_i32, vec_width=1, dtype=T.f32
        )
    return scale_val


def extract_bf16_scale(arith, scale_raw_i32, ku: int):
    """Extract f32 scale from raw i32 dword loaded by bf16 groupwise path.

    In the ``(E, G//2, N, 2)`` layout two adjacent groups share one dword.
    ``ku`` determines which half: even ku → low bf16, odd ku → high bf16.
    """
    if ku % 2 == 0:
        # Low bf16: shift left by 16 to place in upper 16 bits → f32
        return arith.bitcast(T.f32, scale_raw_i32 << fx.Int32(16))
    else:
        # High bf16: mask upper 16 bits → f32
        return arith.bitcast(T.f32, scale_raw_i32 & fx.Int32(0xFFFF0000))


# ---------------------------------------------------------------------------
# W4A16 groupwise load / unpack helpers
# ---------------------------------------------------------------------------


def load_b_raw_w4a16_groupwise(
    buffer_ops,
    arith,
    vector,
    *,
    arg_b,
    b_rsrc,
    layout_b,
    base_k,
    ku: int,
    n_blk,
    n_intra,
    lane_div_16,
    elem_type,
    scale_rsrc,
    expert_offset,
    num_groups: int,
    group_size: int,
    n_per_expert: int,
    kpack_bytes: int = 8,
    scale_dtype=None,
):
    """Phase 1 of W4A16 groupwise B load: buffer_loads for weight + scale.

    Reuses :func:`load_b_raw_w4a16` for the weight load, then issues an
    additional ``buffer_load_dword`` for the per-group scale.

    Returns ``(packed32, scale_val)``.
    """
    packed32 = load_b_raw_w4a16(
        buffer_ops,
        arith,
        vector,
        arg_b=arg_b,
        b_rsrc=b_rsrc,
        layout_b=layout_b,
        base_k=base_k,
        ku=ku,
        n_blk=n_blk,
        n_intra=n_intra,
        lane_div_16=lane_div_16,
        elem_type=elem_type,
        kpack_bytes=kpack_bytes,
    )
    k_pos = base_k + fx.Index(ku * 32)
    scale_val = _load_groupwise_scale(
        buffer_ops,
        arith,
        scale_rsrc=scale_rsrc,
        expert_offset=expert_offset,
        n_blk=n_blk,
        n_intra=n_intra,
        k_pos=k_pos,
        num_groups=num_groups,
        group_size=group_size,
        n_per_expert=n_per_expert,
        scale_dtype=scale_dtype,
    )
    return (packed32, scale_val)


def unpack_b_w4a16_groupwise(packed32, scale_val, arith, vector, use_gfx950_cvt=False):
    """Phase 2 of W4A16 groupwise: unpack + scale + convert to bf16."""
    return unpack_b_w4a16(
        packed32, arith, vector, scale_val=scale_val, use_gfx950_cvt=use_gfx950_cvt
    )
