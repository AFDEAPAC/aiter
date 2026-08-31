# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Shared low-level helpers for the a16w4/a16wi4/a16w16 fused MoE kernels
(:mod:`gemm1` stage1 and :mod:`gemm2` stage2). Pointer/GEP builders, buffer-tensor
views, e8m0/int4 dequant, the A-LDS XOR swizzle, and the arch gate."""

import os

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, buffer_ops, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch

_PTR3 = "!llvm.ptr<3>"
LOG2E = 1.4426950408889634

# a16wi4 (int4 W) groupwise scale: group_size = 32 == one MFMA K32 step (one ku per
# K-group). Scale packed bf16 pairs (E, N, G//2, 2); even/odd ku selects lo/hi half.
A16WI4_GROUP_SIZE = 32


def a16wmix_use_k16(arch=None):
    """True for the gfx942 (CDNA3) codepath: K=16 MFMA + scalar int4 dequant.

    Arch-gate: gfx950 (CDNA4) has K=32 mfma_f32_16x16x32_bf16 + v_cvt_pk_bf16_f32;
    gfx942 has neither and falls back to K=16 MFMA + scalar-trunc dequant.
    ``FLYDSL_A16WMIX_FORCE_K16=1`` forces the gfx942 path (a strict ISA subset) for
    validation on a gfx950 box.
    """
    if os.environ.get("FLYDSL_A16WMIX_FORCE_K16", "0") not in ("0", "", "false", "False"):
        return True
    if arch is None:
        arch = get_rocm_arch() or ""
    return "gfx95" not in str(arch)


# s_waitcnt immediate for lgkmcnt(0) leaving vmcnt/expcnt unconstrained, so the global
# loads deliberately kept in flight are not waited on. gfx9 encoding: vmcnt [3:0]+[15:14]
# = 63, expcnt [6:4] = 7, lgkmcnt [11:8] = 0. Named counters are a flydsl 0.3.1 API; 0.2.4
# takes only this raw immediate, and llvm-mc assembles both to [0x7f,0xc0,0x8c,0xbf].
LGKMCNT_0 = 0xC07F


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _udiv(a, c):
    cc = fx.Int32(c) if isinstance(c, int) else c
    return fx.Int32(arith.divui(_raw(a), _raw(cc)))


def _umod(a, c):
    cc = fx.Int32(c) if isinstance(c, int) else c
    return fx.Int32(arith.remui(_raw(a), _raw(cc)))


def _global_i32_buffer_view(addr_i64, num_bytes):
    # fx.copy BufferCopy atoms take soffset as an element count (not bytes); the
    # make_layout dynamic-shape leaf must be i32/i64, not fx.Index.
    num_bytes_i64 = fx.Int64(num_bytes)
    ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
    ptr = fx.inttoptr(ptr_ty, fx.Int64(addr_i64))
    view = fx.Tensor(fx.make_view(ptr, fx.make_layout(num_bytes_i64 // fx.Int64(4), 1)))
    return fx.rocdl.make_buffer_tensor(view, max_size=False, num_records_bytes=num_bytes_i64)


def _global_i32_buffer_tiles(addr_i64, num_bytes, tile_elems):
    return fx.logical_divide(_global_i32_buffer_view(addr_i64, num_bytes), fx.make_layout(tile_elems, 1))


def _buffer_i32_scalar_read(tiles1, idx, atom):
    """Read one i32 dword at element ``idx`` from a ``_global_i32_buffer_tiles(..., 1)``
    view via the layout-API BufferCopy atom (buffer_load_dword; OOB-clamped by the
    buffer resource). ``tiles1`` is 1-dword tiles so the tile index == ``idx``.
    """
    r = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
    fx.copy(atom, fx.slice(tiles1, (None, idx)), r)
    return fx.Int32(fx.Vector(fx.memref_load_vec(r))[0])


def _lds_ptr3(base_i32, byte_off_i32):
    addr_i64 = fx.Int64(base_i32 + byte_off_i32)
    return llvm.inttoptr(ir.Type.parse(_PTR3), _raw(addr_i64))


def _gep3(base_ptr, byte_off_i32):
    return buffer_ops.get_element_ptr(base_ptr, byte_offset=_raw(byte_off_i32), elem_type=T.i8)


def _global_base_ptr1(addr_i64):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(addr_i64)))


def _gep1(base_ptr, byte_off_i32):
    return buffer_ops.get_element_ptr(base_ptr, byte_offset=_raw(byte_off_i32), elem_type=T.i8)


def _global_i32_ptr(addr_i64):
    ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def _global_i32_at(addr_i64, idx):
    return _global_i32_ptr(addr_i64)[idx]


def _global_f32_at(addr_i64, idx):
    ptr_ty = fx.PointerType.get(T.f32, address_space=fx.AddressSpace.Global, alignment=4)
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))[idx]


def _global_u8_at(addr_i64, idx):
    """One byte from a global u8 array, zero-extended to i32.

    The fp8 path's per-column reference exponent is a byte per (expert, column); a
    scalar byte load keeps it off the dword-aligned paths the rest of the kernel uses.
    """
    ptr_ty = fx.PointerType.get(T.i8, address_space=fx.AddressSpace.Global, alignment=1)
    b = fx.inttoptr(ptr_ty, fx.Int64(addr_i64))[idx]
    return fx.Int32(arith.extui(T.i32, _raw(b)))


def _e8m0_byte_to_f32(packed_i32, byte_pos):
    shift = byte_pos * fx.Int32(8)
    b = packed_i32.shrui(shift) & fx.Int32(0xFF)
    return fx.Float32(_raw(b << fx.Int32(23)).bitcast(T.f32))


def _e8m0_byte_raw(packed_i32, byte_pos):
    """The e8m0 byte as an integer, not the 2^(b-127) float.

    The fp8 path needs the exponent itself so it can form the per-column residual
    ``s_ref(n) - s(n, kg)`` and pick a magnitude table with 2^r already folded in;
    :func:`_e8m0_byte_to_f32` throws that away.
    """
    return packed_i32.shrui(byte_pos * fx.Int32(8)) & fx.Int32(0xFF)


def _cvt_pk_bf16_f32_se(src_a_f32, src_b_f32):
    # Side-effecting v_cvt_pk_bf16_f32 (pack 2 f32 -> 2xbf16 in i32). LOAD-BEARING:
    # the stateless rocdl.cvt_pk_bf16_f32 gets CSE-merged/reordered across K steps in
    # the a16wi4 gemm1 hot loop (garbage output); side_effects pins each call.
    return llvm.inline_asm(
        ir.IntegerType.get_signless(32),
        [_raw(src_a_f32), _raw(src_b_f32)],
        "v_cvt_pk_bf16_f32 $0, $1, $2",
        "=v,v,v",
        has_side_effects=True,
    )


def _int4_nibble_to_bf16x8(raw_i32, scale_f32, *, use_k16=False):
    """int4 (signed) -> bf16 upconvert for one MFMA K32 step (8 nibbles -> v8bf16).

    ``raw_i32`` holds 8 signed-int4 nibbles in bits[4n+3:4n] (same K order as the
    mxfp4 sel 0..3 path). ``v_cvt_off_f32_i4`` reads the nibble unsigned, subtracts 8,
    and scales the mantissa by 16, so the x16 is folded into eff = scale*16.
    ``use_k16`` (gfx942): v_cvt_pk_bf16_f32 is gfx950-only -> scalar .to(BFloat16).
    """
    eff = fx.Float32(scale_f32 * fx.Float32(16.0))
    raw_even = fx.Int32(raw_i32)
    raw_odd = raw_even.shrui(fx.Int32(4))
    if use_k16:
        # gfx942 fallback: scalar f32 -> bf16 truncation (no v_cvt_pk_bf16_f32).
        bf16s = []
        for j in range_constexpr(4):
            f_lo = fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_even), byte_sel=j)) * eff
            f_hi = fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_odd), byte_sel=j)) * eff
            bf16s.append(f_lo.to(fx.BFloat16))
            bf16s.append(f_hi.to(fx.BFloat16))
        return fx.Vector.from_elements([_raw(x) for x in bf16s], fx.BFloat16)  # v8bf16
    # byte_sel loads (1 shift total); side-effecting pk-convert.
    i32s = []
    for j in range_constexpr(4):
        f_lo = fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_even), byte_sel=j)) * eff
        f_hi = fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_odd), byte_sel=j)) * eff
        i32s.append(fx.Int32(_cvt_pk_bf16_f32_se(_raw(f_lo), _raw(f_hi))))
    v4i32 = fx.Vector.from_elements([_raw(x) for x in i32s], fx.Int32)
    return v4i32.bitcast(fx.BFloat16)  # v8bf16


def _int4_nibble_to_bf16x8_raw(raw_i32, *, use_k16=False):
    """int4 (signed) -> bf16 for one MFMA K32 step WITHOUT the groupwise scale.

    Same as :func:`_int4_nibble_to_bf16x8` but emits the raw dequant weights
    ``(nibble-8)/16`` (``v_cvt_off_f32_i4``'s native output -- no per-element
    ``v_mul_f32``). The groupwise scale (and the folded x16) is applied ONCE per
    K-group on the small MFMA accumulator instead (see the ``_acc_scale_int4`` path in
    the stage1 body): for BM16 (m_repeat=1) that trades 8 per-nibble muls for 4
    per-accumulator fmas and drops the long-lived scaled-f32 operand VGPRs.
    ``(nibble-8)/16`` is bf16-exact (values in ``{-7/16..7/16}``).
    ``use_k16`` (gfx942): v_cvt_pk_bf16_f32 is gfx950-only -> scalar .to(BFloat16).
    """
    raw_even = fx.Int32(raw_i32)
    raw_odd = raw_even.shrui(fx.Int32(4))
    if use_k16:
        bf16s = []
        for j in range_constexpr(4):
            f_lo = fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_even), byte_sel=j))
            f_hi = fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_odd), byte_sel=j))
            bf16s.append(f_lo.to(fx.BFloat16))
            bf16s.append(f_hi.to(fx.BFloat16))
        return fx.Vector.from_elements([_raw(x) for x in bf16s], fx.BFloat16)  # v8bf16
    i32s = []
    for j in range_constexpr(4):
        f_lo = fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_even), byte_sel=j))
        f_hi = fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_odd), byte_sel=j))
        i32s.append(fx.Int32(_cvt_pk_bf16_f32_se(_raw(f_lo), _raw(f_hi))))
    v4i32 = fx.Vector.from_elements([_raw(x) for x in i32s], fx.Int32)
    return v4i32.bitcast(fx.BFloat16)  # v8bf16


def _fp4_nibble_to_bf16x8_sw(raw_i32, scale_f32):
    """FP4 (E2M1) -> bf16 upconvert for one MFMA K32 step (8 nibbles -> v8bf16).

    Software path for gfx942, which has neither v_cvt_scalef32_pk_bf16_fp4 nor
    v_cvt_pk_bf16_f32. Each nibble is bits[4n+3:4n] in the SAME K order as the
    gfx950 ``sel 0..3`` path, so the MMA operand layout is unchanged.

    E2M1: bit3 = sign, bits[2:1] = exp, bit0 = mantissa.
      exp == 0 -> subnormal: 0.0 (m=0) or 0.5 (m=1)
      exp  > 0 -> 2^(exp-1) * (1 + 0.5*m), i.e. f32 exp field 126+exp, mant<<22
    """
    raw = fx.Int32(raw_i32)
    bf16s = []
    for j in range_constexpr(8):
        n = raw.shrui(fx.Int32(4 * j)) & fx.Int32(0xF)
        e = n.shrui(fx.Int32(1)) & fx.Int32(0x3)
        mant = n & fx.Int32(0x1)
        sign = n.shrui(fx.Int32(3)) & fx.Int32(0x1)
        normal_bits = ((e + fx.Int32(126)) << fx.Int32(23)) | (mant << fx.Int32(22))
        sub_bits = (mant == fx.Int32(1)).select(fx.Int32(0x3F000000), fx.Int32(0))
        mag_bits = (e == fx.Int32(0)).select(sub_bits, normal_bits)
        bits = mag_bits | (sign << fx.Int32(31))
        v = fx.Float32(_raw(bits).bitcast(T.f32)) * fx.Float32(scale_f32)
        bf16s.append(v.to(fx.BFloat16))
    return fx.Vector.from_elements([_raw(x) for x in bf16s], fx.BFloat16)  # v8bf16


# ---------------------------------------------------------------------------
# FP4 (E2M1) -> bf16 byte-lookup decode.
#
# Every E2M1 value is exactly representable in bf16 (2 mantissa bits against bf16's 8),
# so the decode carries no arithmetic -- only the bit pattern:
#
#   nibble & 7  0       1       2       3       4       5       6       7
#   value       0.0     0.5     1.0     1.5     2.0     3.0     4.0     6.0
#   bf16 bits   0x0000  0x3F00  0x3F80  0x3FC0  0x4000  0x4040  0x4080  0x40C0
#
# Three distinct high bytes and four distinct low bytes, so one v_perm_b32 pair decodes
# four nibbles. The sign is bit 7 of the bf16 high byte, which is where it already sits
# in the nibble, so it folds in with a mask and a shift rather than a select.
#
# This replaces the per-element reconstruction in _fp4_nibble_to_bf16x8_sw, which built
# an f32 through two compares and two selects and then round-tripped it to bf16 with a
# full round-to-nearest -- including a NaN check that E2M1 can never trigger
# (v_cmp_u_f32 appeared 1664 times in the gemm1 ISA). On MI308X the two stages are
# 92-94% VALU instructions, so removing that work is worth 2.5-2.8x on gemm1.
_FP4_MAG_HI_LO = 0x3F3F3F00  # bf16 high bytes for nibble&7 = 0,1,2,3
_FP4_MAG_HI_HI = 0x40404040  # bf16 high bytes for nibble&7 = 4,5,6,7
_FP4_MAG_LO_LO = 0xC0800000  # bf16 low bytes  for nibble&7 = 0,1,2,3
_FP4_MAG_LO_HI = 0xC0804000  # bf16 low bytes  for nibble&7 = 4,5,6,7


def _perm(src_hi, src_lo, sel):
    """v_perm_b32: result byte i = pool[sel.byte(i)], pool = {src_hi:src_lo}.

    Selector bytes 0..3 index src_lo bytes 0..3, selector bytes 4..7 index src_hi.
    """
    return fx.Int32(rocdl.perm_b32(_raw(fx.Int32(src_hi)), _raw(fx.Int32(src_lo)), _raw(sel)))


def _fp4_mag_dwords(raw_i32):
    """Decode 8 FP4 nibbles to 4 dwords, each holding 2 bf16, scale not applied.

    ``raw_i32`` holds 8 nibbles in bits[4n+3:4n], the same K order as the gfx950
    ``sel 0..3`` path, so element j lands in dword j//2 half j%2 and the MMA operand
    layout is unchanged.
    """
    raw = fx.Int32(raw_i32)
    # Magnitude selectors, one per byte. v_perm only honours selector values 0..7, so
    # the sign bit is masked off here and folded back into the high byte below.
    sel_even = raw & fx.Int32(0x07070707)  # low nibble of each byte -> elements 0,2,4,6
    sel_odd = raw.shrui(fx.Int32(4)) & fx.Int32(0x07070707)  # -> elements 1,3,5,7

    hb_even = _perm(_FP4_MAG_HI_HI, _FP4_MAG_HI_LO, sel_even)
    lb_even = _perm(_FP4_MAG_LO_HI, _FP4_MAG_LO_LO, sel_even)
    hb_odd = _perm(_FP4_MAG_HI_HI, _FP4_MAG_HI_LO, sel_odd)
    lb_odd = _perm(_FP4_MAG_LO_HI, _FP4_MAG_LO_LO, sel_odd)

    # Sign: nibble bit 3 -> bf16 bit 15, i.e. bit 7 of the high byte.
    hb_even = hb_even | ((raw & fx.Int32(0x08080808)) << fx.Int32(4))
    hb_odd = hb_odd | (raw & fx.Int32(0x80808080))

    # dword d = [lb_even[d], hb_even[d], lb_odd[d], hb_odd[d]].
    ev01 = _perm(hb_even, lb_even, fx.Int32(0x05010400))  # bf16 of elements 0 and 2
    od01 = _perm(hb_odd, lb_odd, fx.Int32(0x05010400))  # elements 1 and 3
    ev23 = _perm(hb_even, lb_even, fx.Int32(0x07030602))  # elements 4 and 6
    od23 = _perm(hb_odd, lb_odd, fx.Int32(0x07030602))  # elements 5 and 7
    return [
        _perm(od01, ev01, fx.Int32(0x05040100)),
        _perm(od01, ev01, fx.Int32(0x07060302)),
        _perm(od23, ev23, fx.Int32(0x05040100)),
        _perm(od23, ev23, fx.Int32(0x07060302)),
    ]


def _fp4_nibble_to_bf16x8_raw(raw_i32):
    """FP4 (E2M1) -> v8bf16 for one MFMA K32 step, WITHOUT the groupwise scale.

    Same magnitudes as :func:`_fp4_nibble_to_bf16x8_lut`; the e8m0 scale is left for
    the caller to apply once per K-group on the MFMA accumulator (``_mma_scaled_add``
    in the stage bodies). Both halves stay exact: E2M1 magnitudes are representable
    in bf16, and the scale is a power of two, so folding it into the f32 accumulator
    instead of into each weight changes nothing numerically.

    What it saves is the scale chain in ``_fp4_nibble_to_bf16x8_lut`` -- per dword an
    lshl, an and, a pk_mul, an lshr and an and_or. Measured on the prefill shape
    (bm32/tn192, prof/isa_summary_prefill_bm32.json) that chain is 21.5 of the 39.5
    VALU each 8-weight group costs, against 4 accumulator FMAs per (mi, ni) to put
    the scale back.
    """
    return fx.Vector.from_elements(
        [_raw(d) for d in _fp4_mag_dwords(raw_i32)], fx.Int32
    ).bitcast(fx.BFloat16)  # v8bf16


def _fp4_nibble_to_bf16x8_lut(raw_i32, scale_f32):
    """FP4 (E2M1) -> v8bf16 for one MFMA K32 step, with the groupwise scale applied.

    Drop-in replacement for :func:`_fp4_nibble_to_bf16x8_sw`. The magnitudes come from
    the byte lookup; the e8m0 scale is a power of two, so the product stays exactly
    representable in bf16 and the f32 -> bf16 step is a truncation, not a
    round-to-nearest. bf16 <-> f32 is just a 16-bit shift, so each half is scaled in
    place without unpacking to a vector.
    """
    scale = fx.Float32(scale_f32)
    out = []
    for d in _fp4_mag_dwords(raw_i32):
        lo_f = fx.Float32(_raw(d << fx.Int32(16)).bitcast(T.f32))
        hi_f = fx.Float32(_raw(d & fx.Int32(0xFFFF0000)).bitcast(T.f32))
        lo_b = fx.Int32(_raw(lo_f * scale).bitcast(T.i32)).shrui(fx.Int32(16))
        hi_b = fx.Int32(_raw(hi_f * scale).bitcast(T.i32)) & fx.Int32(0xFFFF0000)
        out.append(lo_b | hi_b)
    return fx.Vector.from_elements([_raw(x) for x in out], fx.Int32).bitcast(fx.BFloat16)


# ---------------------------------------------------------------------------
# FP4 (E2M1) -> e4m3fnuz byte-lookup decode, for the fp8 MFMA path.
#
# An fp8 weight operand cannot carry the MXFP4 groupwise scale: e8m0 spans 2^-127..
# 2^127 and e4m3fnuz only about 2^-10..240. Nor can the accumulator absorb it -- the
# K-group index depends on lane_div_16, so one MFMA contracts four groups and an
# accumulator element mixes four scales (see acc_scale_for).
#
# What works is splitting the scale by what each side can legally hold:
#
#   s(n, kg) = s_ref(n) + r(n, kg),   s_ref(n) = max over kg,  so r <= 0
#
# 2^s_ref(n) depends only on the output column, which is constant across a lane's
# accumulator, so the epilogue applies it once. 2^r rides in the weight -- and
# measurement makes that nearly free: on the real checkpoint r takes three values
# (0: 44.5%, -1: 49.4%, -2: 6.2%, -3 twice in 5.5M groups, never below), so instead of
# any per-element exponent arithmetic the decode just picks one of four constant
# v_perm_b32 pools. See prof/e8m0_residual_scan.json and tools/gen_fp8_lut.py.
#
# Every E2M1 magnitude times 2^r stays exactly representable in e4m3fnuz for those r,
# so the weight side of the fp8 path carries no error at all.
#
# Index 0 maps to 0x01 (smallest subnormal, 2^-10) rather than 0x00. e4m3fnuz has no
# negative zero -- 0x80 is NaN -- and 5.765% of the checkpoint's nibbles are 0x8
# (-0.0), so folding the sign onto a 0x00 magnitude would put a NaN in essentially
# every K-group (prof/fp4_negzero_scan.json). +-2^-10 against a group maximum of
# 6*scale is a relative 1.6e-4, far under fp8's own ~6% resolution.
#
# Tables are derived and round-trip checked by tools/gen_fp8_lut.py.
# Reference tables, kept for documentation and for test_fp8_lut to check the SWAR form
# against. r = 0 .. -6 is the whole exactly-representable range: every table is the
# previous one with 0x08 taken off each magnitude byte (one e4m3 exponent step), and at
# r = -7 the smallest magnitude (0.5) would leave the normal range, so the pattern
# stops being a plain subtract.
_FP8_MAG_TABLE = (
    (0x44403801, 0x54504C48),  # r =  0
    (0x3C383001, 0x4C484440),  # r = -1
    (0x34302801, 0x44403C38),  # r = -2
    (0x2C282001, 0x3C383430),  # r = -3
    (0x24201801, 0x34302C28),  # r = -4
    (0x1C181001, 0x2C282420),  # r = -5
    (0x14100801, 0x24201C18),  # r = -6
)
FP8_MAX_NEG_RESIDUAL = len(_FP8_MAG_TABLE) - 1

# SWAR form of the table above: subtract 0x08 per binade from all four magnitude bytes
# at once. Byte 0 of the low dword is the 0x01 zero placeholder, and subtracting from
# 0x01 would borrow into byte 1, so the base carries 0x40 there instead -- big enough
# that 8*6 never borrows out of it -- and byte 0 is overwritten afterwards.
_FP8_SWAR_LO_BASE = 0x44403840
_FP8_SWAR_HI_BASE = 0x54504C48


def _fp8_table_select(neg_r):
    """The (lo, hi) v_perm pool for residual r = -``neg_r``, clamped to the exact range.

    Computed rather than selected. A select chain costs two v_cndmask per table and
    caps the residual at however many tables are compiled in; this is a multiply and
    two subtracts for any r, so it is both slightly cheaper and good to -6. That
    matters: the down projection reaches -4 on the real checkpoint (10 groups in 4.13
    billion, see prof/w2_sharded_residual.json), which four tables did not cover.

    Hoisted per (lane, k0-block) -- a lane reads one e8m0 group per 4 K-steps, i.e. per
    32 weights -- so it is well under 0.2 instructions per weight either way.

    ``neg_r`` should be a non-negative i32; it is clamped on both sides because a
    negative value would mean the reference is not the column maximum, and a wrong
    magnitude is preferable to a borrow cascade across the packed bytes.
    """
    t = fx.Int32(arith.minsi(_raw(fx.Int32(neg_r)), _raw(fx.Int32(FP8_MAX_NEG_RESIDUAL))))
    t = fx.Int32(arith.maxsi(_raw(t), _raw(fx.Int32(0))))
    sub = t * fx.Int32(0x08080808)
    lo = ((fx.Int32(_FP8_SWAR_LO_BASE) - sub) & fx.Int32(0xFFFFFF00)) | fx.Int32(0x01)
    hi = fx.Int32(_FP8_SWAR_HI_BASE) - sub
    return lo, hi


def _fp4_nibble_to_fp8x8(raw_i32, tab_lo, tab_hi):
    """8 FP4 nibbles -> 8 e4m3fnuz bytes, as two dwords (one MFMA 16x16x32 B operand).

    ``raw_i32`` holds nibble j in bits[4j+3:4j], the same K order the bf16 path uses,
    so element j lands in byte j and the operand layout is unchanged.

    All eight magnitudes fit one v_perm_b32 pool (unlike bf16, which needs a high-byte
    and a low-byte pool), so four nibbles decode per perm instead of two perms plus a
    repack. Twelve instructions for eight weights, against 39.5 for the scaled bf16
    decode measured in prof/isa_summary_prefill_bm32.json.
    """
    raw = fx.Int32(raw_i32)
    # v_perm honours selector values 0..7 only, so mask the sign off and fold it back
    # into bit 7 afterwards -- which is where it already sits in the nibble.
    sel_even = raw & fx.Int32(0x07070707)  # low nibble of each byte -> elements 0,2,4,6
    sel_odd = raw.shrui(fx.Int32(4)) & fx.Int32(0x07070707)  # -> elements 1,3,5,7

    mag_even = _perm(tab_hi, tab_lo, sel_even)
    mag_odd = _perm(tab_hi, tab_lo, sel_odd)
    mag_even = mag_even | ((raw & fx.Int32(0x08080808)) << fx.Int32(4))
    mag_odd = mag_odd | (raw & fx.Int32(0x80808080))

    # mag_even = [e0,e2,e4,e6], mag_odd = [e1,e3,e5,e7]; interleave to K order.
    d0 = _perm(mag_odd, mag_even, fx.Int32(0x05010400))  # e0,e1,e2,e3
    d1 = _perm(mag_odd, mag_even, fx.Int32(0x07030602))  # e4,e5,e6,e7
    return d0, d1


def _fp4_nibble_to_fp8x8_vec(raw_i32, tab_lo, tab_hi):
    """:func:`_fp4_nibble_to_fp8x8` as a v8 fp8 vector, ready for an MMA fragment."""
    d0, d1 = _fp4_nibble_to_fp8x8(raw_i32, tab_lo, tab_hi)
    return fx.Vector.from_elements([_raw(d0), _raw(d1)], fx.Int32).bitcast(fp8_elem_type())


def fp8_elem_type():
    """gfx942 MFMA reads e4m3**fnuz**; gfx950 and gfx12 use OCP e4m3fn."""
    arch = str(get_rocm_arch() or "")
    return fx.Float8E4M3FN if ("gfx95" in arch or "gfx12" in arch) else fx.Float8E4M3FNUZ


def acc_scale_for(w_dtype, m_repeat):
    """Can the groupwise scale be applied on the MFMA accumulator, not per weight?

    **No, not with this weight layout.** Off by default for every dtype. The env
    override exists only to reproduce the analysis below.

    The idea is to emit unscaled weights and fold the groupwise scale into the small
    f32 accumulator once per K-group, which for mxfp4 would drop 21.5 of the 39.5 VALU
    an 8-weight group costs. It is worth real time -- forced on, gemm1 at 8 decode
    tokens goes 319.2 -> 224.4 us -- but it is **numerically wrong**, and the speed is
    the speed of computing the wrong answer:

        tile_m=16  cos=0.870992  rel_fro=5.671e-01   (against the op_test reference)

    The reason is the K-group index in ``load_b_scale`` / ``load_b_scale_int4``::

        adj_ku = base_k // 32 + (ku // 4) * 4 + lane_div_16

    It depends on ``lane_div_16``, so within one MFMA K32 step the four lane groups
    read four *different* e8m0 groups. Scaling each lane's own weights handles that
    correctly. The accumulator cannot: an accumulator element is a sum contributed by
    all 64 lanes, so it mixes all four scales, and there is no single scalar to pull
    out. Confirmed by construction -- force every e8m0 byte to a single value and the
    two arms agree bit for bit (gemm1 max|delta| = 0.0); let the scales vary and they
    diverge. See tools/verify_acc_scale.py --const-scale.

    This also means the pre-existing int4 path is affected: ``load_b_scale_int4`` uses
    the same ``adj_ku`` expression, so a16wi4 at BM16 has the same flaw. It was never
    caught because op_tests/test_moe_a16w4_gfx942.py only covers mxfp4. Hoisting the
    scale would need a weight preshuffle whose K-group index does not vary with the
    lane group, which ``shuffle_weight_a16w4`` does not provide.

    ``FLYDSL_A16WMIX_ACC_SCALE=0|1`` forces the arm. NOTE: flydsl caches compiled
    kernels in ~/.flydsl/cache and both arms share one kernel name, so an A/B also
    needs FLYDSL_RUNTIME_ENABLE_CACHE=0 or it silently measures one arm twice.
    """
    forced = os.environ.get("FLYDSL_A16WMIX_ACC_SCALE")
    if forced is not None and forced != "auto":
        return forced not in ("0", "", "false", "False")
    return False


def kmchunks_for(BM):
    return BM // 16


def lds_acc_bytes_for(rows, BN):
    return rows * BN * 4


def _a16w4_swizzle_xor16(row, col_bytes, k_blocks16, *, enable=False):
    """A-LDS bank-conflict XOR swizzle (aiter swizzle_xor16: col ^ ((row&(kb16-1))*16)).

    Both the DMA write and the LDS read go through this helper so the physical layout
    stays consistent. gemm1 keeps linear (enable=False); gemm2 enables it.
    """
    if not enable:
        return col_bytes
    rem = row & fx.Int32(k_blocks16 - 1)
    return col_bytes ^ (rem * fx.Int32(16))
