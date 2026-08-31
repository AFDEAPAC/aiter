# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from ..layout_utils import crd2idx

from .utils import (
    A16WI4_GROUP_SIZE,
    _a16w4_swizzle_xor16,
    _buffer_i32_scalar_read,
    _e8m0_byte_raw,
    _e8m0_byte_to_f32,
    _fp4_nibble_to_bf16x8_lut,
    _fp4_nibble_to_bf16x8_raw,
    _fp4_nibble_to_bf16x8_sw,
    _fp4_nibble_to_fp8x8_vec,
    _fp8_table_select,
    _gep1,
    _gep3,
    _global_base_ptr1,
    _global_f32_at,
    _global_i32_at,
    _global_i32_buffer_tiles,
    _global_i32_buffer_view,
    _global_u8_at,
    _int4_nibble_to_bf16x8,
    _lds_ptr3,
    _raw,
    _udiv,
    _umod,
    a16wmix_use_k16,
    acc_scale_for,
    fp8_elem_type,
    kmchunks_for,
    lds_acc_bytes_for,
)

# Caps the persistent gemm2 grid. Sized from the device rather than hardcoded: the
# previous value was gfx950's 256 CUs, which over-subscribes an 80-CU MI308X unevenly
# (12642 tiles over 256 CTAs is 49.4 each, and that cap measured slower at bs=8 than
# either 240 or 320).
NUM_CU = 256  # fallback when the device cannot be queried
LDS_LIMIT_BYTES = 64 * 1024  # gfx9 per-workgroup LDS


@functools.cache
def _persistent_cta_cap():
    """CTA cap for the persistent grid: 8 waves per CU, measured best on MI308X."""
    try:
        import torch

        return torch.cuda.get_device_properties(0).multi_processor_count * 8
    except Exception:
        return NUM_CU


# @flyc.jit is LOAD-BEARING: it AST-rewrites ``if token_id < i32_M`` into an scf.if.
# Without it the guard runs as a plain Python if (dropped at trace), so the atomic-fadd
# scatter fires on padded/OOB rows -- ~13x s2 regression (39us -> ~490us at E896).
@flyc.jit
def _atomic_bf16_epilog(
    lds_acc_base_i32,
    accm,
    arg_out,
    arg_stids,
    arg_sweights,
    m_row,
    n_block_idx,
    wave,
    lane,
    i32_M,
    BM,
    N_OUT,
    BN,
):
    _kMChunks = kmchunks_for(BM)
    M_REPS = BM // 8
    # 4 waves split the BN(=TILE_N) tile (generic over BN, e.g. int4 tile_n=128).
    _n_per_wave = BN // 4
    num_acc_n = _n_per_wave // 16
    _s_count = BN // 64  # readback: each s-iter covers 64 cols (32 lanes x vec2)
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lds_base = _lds_ptr3(lds_acc_base_i32, fx.Int32(0))

    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(32)
    n_lane = tx_i32 % fx.Int32(32)
    col_start = n_lane * fx.Int32(2)
    stids_base = _global_base_ptr1(arg_stids)
    sweights_base = _global_base_ptr1(arg_sweights)
    out_base = _global_base_ptr1(arg_out)

    packed = []
    weight = []
    for mr in range_constexpr(M_REPS):
        sorted_pos = m_row + fx.Int32(mr * 8) + m_lane
        packed.append(llvm.load(T.i32, _gep1(stids_base, sorted_pos * fx.Int32(4)), invariant=True))
        weight.append(llvm.load(T.f32, _gep1(sweights_base, sorted_pos * fx.Int32(4)), invariant=True))

    for i in range_constexpr(_kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(num_acc_n):
            col = wave * fx.Int32(_n_per_wave) + fx.Int32(J * 16) + lane_mod_16
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + col
                llvm.StoreOp(_raw(vec[v]), _gep3(lds_base, idx * fx.Int32(4)))

    gpu.barrier()

    for mr in range_constexpr(M_REPS):
        row_in_block = fx.Int32(mr * 8) + m_lane
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if token_id < i32_M:
            row_base_addr = token_id * fx.Int32(N_OUT) + n_block_idx * fx.Int32(BN) + col_start
            for s in range_constexpr(_s_count):
                idx0 = row_in_block * fx.Int32(BN) + col_start + fx.Int32(s * 64)
                v2 = Vec(llvm.load(T.vec(2, T.f32), _gep3(lds_base, idx0 * fx.Int32(4))))
                pk = Vec.from_elements([v2[0] * weight[mr], v2[1] * weight[mr]], fx.Float32).to(fx.BFloat16)
                off = (row_base_addr + fx.Int32(s * 64)) * fx.Int32(2)
                out_ptr = _gep1(out_base, off)
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.fadd,
                    out_ptr,
                    _raw(pk),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )


def _gemm2_body_a16w4(
    lds_raw_ptr,
    arg_a,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_stids,
    arg_sweights,
    arg_out,
    bx_i32,
    lane,
    wave,
    i32_M,
    arg_a_scale=None,
    arg_sref=None,
    *,
    BM,
    TILE_N,
    TILE_K,
    N_OUT,
    INTER,
    NE,
    b_cache_mod=2,
    w_dtype="mxfp4",
    a_dtype="bf16",
    use_k16=False,
):
    """a16w4/a16wi4/a16w16 stage2 body. K=inter_dim (contraction), N=model_dim (N_OUT).

    A = the stage1 intermediate by SORTED position, bf16 or (``a_dtype='fp8'``) fp8 with
    a per-row scale. W2 = mxfp4/int4/bf16 (see gemm1). Output = bf16 atomic-fadd
    (routing-weighted) scatter to [tokens, model_dim].

    The fp8 arm mirrors gemm1's: gfx942 has mfma_f32_16x16x32_fp8_fp8 but no bf16
    16x16x32, so one MFMA replaces two and the FP4 weight decodes to fp8 bytes instead
    of bf16. The e8m0 groupwise scale cannot ride in an fp8 weight, so it is split into
    a per-output-column reference (``arg_sref``, applied on the accumulator) and a
    residual folded into the magnitude table -- see a16wmix_fp8_prep.

    A's scale is per SORTED ROW here, not per token: stage2 contracts over inter_dim, so
    a row of the intermediate is one route, and that is exactly what per_token_quant_hip
    produces when handed inter_sorted.
    """
    _is_int4 = w_dtype == "int4"
    _is_bf16 = w_dtype == "bf16"  # a16w16: raw bf16 W (unpacked, no scale, no upconvert)
    _is_a_fp8 = a_dtype == "fp8"
    _a_elem = fp8_elem_type() if _is_a_fp8 else fx.BFloat16
    # A-side element width. The W side is unaffected: it is FP4 either way.
    elem_bytes = 1 if _is_a_fp8 else 2
    KH_TILE_BYTES = TILE_K * elem_bytes
    LDS_STRIDE = TILE_K
    K = INTER
    K_HALF = K // 2
    K_TILES_TOTAL = K // TILE_K
    m_repeat = BM // 16
    # Groupwise scale on the MFMA accumulator instead of per weight element; decode
    # only, see acc_scale_for. Measured 150.5 -> 115.9 us at 8 tokens (1.30x).
    _acc_scale = acc_scale_for(w_dtype, m_repeat)
    _acc_fold = 16.0 if _is_int4 else 1.0
    # K32 MFMA steps per K-tile. Must come from the ELEMENT count, not the byte count:
    # the old KH_TILE_BYTES//64 happens to equal this at 2 B/elem, but under fp8 it
    # halves and the kernel would contract only half of every K tile.
    k_unroll = TILE_K // 32
    _k0_count = TILE_K // 128
    # 4 waves split the TILE_N tile (matches the atomic-epilog wave-split).
    _n_per_wave = TILE_N // 4
    num_acc_n = _n_per_wave // 16
    k_blocks16 = KH_TILE_BYTES // 16
    _num_n_blocks = N_OUT // TILE_N

    # W2 (mxfp4) preshuffle layout (make_preshuffle_b_layout, N-major, fp4).
    bl_k0 = K_HALF // 64
    bl_stride_klane = 256
    bl_stride_k0 = 1024
    bl_stride_n0 = bl_k0 * bl_stride_k0
    layout_b = fx.make_layout(
        (N_OUT // 16, bl_k0, 4, 16, 16),
        (bl_stride_n0, bl_stride_k0, bl_stride_klane, 16, 1),
    )
    # W2 (raw bf16) preshuffle layout (N-major == shuffle_weight (16,16)), bf16-elem units:
    #   shape (N_OUT/16, K/32, 4, 16, 8). One kpack=8 bf16=one MFMA K32 fragment; K
    #   reindexed to the fp4 (klane_hw, ku)->K order (see load_b_raw_bf16).
    bfl_k0 = K // 32
    bfl_stride_klane = 128
    bfl_stride_k0 = 512
    bfl_stride_n0 = bfl_k0 * bfl_stride_k0
    layout_b_bf16 = fx.make_layout(
        (N_OUT // 16, bfl_k0, 4, 16, 8),
        (bfl_stride_n0, bfl_stride_k0, bfl_stride_klane, 8, 1),
    )
    scale_k_padded = ((K + 255) // 256) * 256
    sc_k1 = ((scale_k_padded // 32) // 4) // 2
    sc_stride_klane = 16
    sc_stride_k0 = 64
    sc_stride_n0 = sc_k1 * sc_stride_k0

    # a16wi4 groupwise scale: bf16 pairs (E, N_OUT, num_groups//2, 2), K = inter_dim.
    _num_groups = K // A16WI4_GROUP_SIZE
    _g_half = _num_groups // 2

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)

    m_block_idx = bx_i32 // fx.Int32(_num_n_blocks)
    n_block_idx = bx_i32 % fx.Int32(_num_n_blocks)
    e = rocdl.readfirstlane(T.i32, _raw(_global_i32_at(arg_eids, m_block_idx)))
    m_row = m_block_idx * fx.Int32(BM)  # first sorted row of this m-block
    by_n = n_block_idx * fx.Int32(TILE_N)
    expert_off = e * fx.Int32(N_OUT)

    # bf16 W overflows the 32-bit num_records / i32 byte-offset at large E; fold the
    # per-expert base into the i64 resource addr and index within the expert. mxfp4/int4
    # keep the whole-tensor path.
    if const_expr(_is_bf16):
        _w_per_expert_bytes = N_OUT * (K * 2)
        w_base_i64 = fx.Int64(arg_bq) + fx.Int64(e) * fx.Int64(_w_per_expert_bytes)
        w_tiles = _global_i32_buffer_tiles(w_base_i64, min(_w_per_expert_bytes, 0xFFFFFFFF), 4)
    else:
        _w_bytes = NE * N_OUT * K_HALF
        w_tiles = _global_i32_buffer_tiles(arg_bq, min(_w_bytes, 0xFFFFFFFF), 4)
    # W dwordx4 load via BufferCopy128b atom (cache modifier in the aux field).
    w_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(b_cache_mod), fx.Int32)
    w_reg_lay = fx.make_layout(4, 1)
    if _is_int4:
        _sw_bytes = NE * N_OUT * _g_half * 4
    else:
        _sw_bytes = NE * N_OUT * (scale_k_padded // 32)
    # Per-lane scalar scale gather via make_buffer_tensor 1-dword tiles + BufferCopy32b
    # scalar read (see gemm1._buffer_i32_scalar_read), replacing raw buffer_ops.
    sw_tiles = None if _is_bf16 else _global_i32_buffer_tiles(arg_bscale, min(_sw_bytes, 0xFFFFFFFF), 1)
    sw_read_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(0), fx.Int32)

    # ---- A gather (per-thread) -> LDS. A row = SORTED position m_row + row_local.
    total_threads = 256
    _a_tile_bytes = BM * TILE_K * elem_bytes
    # fp8 halves the tile; BM16/TILE_K128 then leaves 8 B per thread, and holding
    # x_load_bytes at 16 would floor num_x_loads to 0 and silently DMA nothing. Same
    # fallback as gemm1. The XOR swizzle below stays valid at 8 B because it only ever
    # flips bits >= 4, so an 8-B access cannot straddle a swizzle boundary.
    x_load_bytes = 16 if _a_tile_bytes >= total_threads * 16 else 8
    bytes_per_thread = _a_tile_bytes // total_threads
    num_x_loads = bytes_per_thread // x_load_bytes
    assert num_x_loads >= 1 and bytes_per_thread % x_load_bytes == 0, (
        f"A tile {BM}x{TILE_K}x{elem_bytes}B = {_a_tile_bytes} does not divide evenly "
        f"into {total_threads} threads x {x_load_bytes} B"
    )
    tile_k_dwords = (TILE_K * elem_bytes) // 4
    c_k_div4 = (K * elem_bytes) // 4
    tx_i32 = fx.Int32(gpu.thread_id("x"))
    chunk_i32 = x_load_bytes // 4
    tx_base = tx_i32 * fx.Int32(chunk_i32)

    x_row_local = []
    x_col_dw = []
    x_row_base_div4 = []
    for i in range_constexpr(num_x_loads):
        tile_idx = tx_base + fx.Int32(i * total_threads * chunk_i32)
        row_local = tile_idx // fx.Int32(tile_k_dwords)
        col_dw = tile_idx % fx.Int32(tile_k_dwords)
        x_row_local.append(row_local)
        x_col_dw.append(col_dw)
        sorted_row = m_row + row_local
        x_row_base_div4.append(sorted_row * fx.Int32(c_k_div4))

    x_buf = _global_i32_buffer_view(arg_a, fx.Int64(0xFFFFFFFF))
    x_dma_tiles4 = fx.logical_divide(x_buf, fx.make_layout(chunk_i32, 1))
    # gfx950 (K=32): BufferCopyLDS128b direct-to-LDS async copy. gfx942 (use_k16): CDNA3
    # direct-to-LDS is 4 B/lane only (the 16 B form fails LLVM ISA lowering), so stage via
    # VGPRs like the legacy kernel: buffer_load 16 B gmem->regs then ds_write 16 B regs->
    # LDS (both b128, valid on CDNA3), preserving the same swizzled-src / linear-LDS layout.
    if const_expr(x_load_bytes == 16):
        _buf_copy, _uni_copy, _buf_lds = (
            fx.rocdl.BufferCopy128b, fx.UniversalCopy128b, fx.rocdl.BufferCopyLDS128b,
        )
    else:
        _buf_copy, _uni_copy, _buf_lds = (
            fx.rocdl.BufferCopy64b, fx.UniversalCopy64b, fx.rocdl.BufferCopyLDS64b,
        )
    if const_expr(use_k16):
        x_dma_atom = fx.make_copy_atom(_buf_copy(b_cache_mod), fx.Int32)  # gmem->regs
        x_lds_store_atom = fx.make_copy_atom(_uni_copy(), fx.Int32)  # regs->LDS
    else:
        x_dma_atom = fx.make_copy_atom(_buf_lds(), fx.Int32)

    # A LDS read: 8 A elements per (mi, ku) = the K32 MFMA operand, so 16 B for bf16 and
    # 8 B for fp8. Element count and K meaning are identical; only the width differs.
    _a_read_bytes = 8 * elem_bytes
    _a_read_dwords = _a_read_bytes // 4
    s_x_i32_flat = fx.make_view(
        fx.recast_iter(fx.Int32, lds_raw_ptr),
        fx.make_layout(BM * LDS_STRIDE * elem_bytes // 4, 1),
    )
    # The DMA chunk and the MFMA read width need not agree; share one view when they do
    # so the bf16 path keeps its exact instruction sequence.
    s_x_dma_tiles = fx.logical_divide(s_x_i32_flat, fx.make_layout(chunk_i32, 1))
    s_x_i32x4_tiles = (
        s_x_dma_tiles
        if const_expr(_a_read_dwords == chunk_i32)
        else fx.logical_divide(s_x_i32_flat, fx.make_layout(_a_read_dwords, 1))
    )
    a_copy_atom = fx.make_copy_atom(
        fx.UniversalCopy128b() if _a_read_bytes == 16 else fx.UniversalCopy64b(), fx.Int32
    )

    def dma_a_tile_to_lds(base_k):
        base_k_div4 = (base_k * fx.Int32(elem_bytes)) // fx.Int32(4)
        for i in range_constexpr(num_x_loads):
            col_bytes = x_col_dw[i] * fx.Int32(4)
            # A-LDS bank-conflict XOR swizzle: LDS dest stays LINEAR (buffer_load_lds
            # ignores an arbitrary swizzled per-lane dest -> NaN); swizzle the GMEM
            # source col instead, and lds_load_a applies the SAME swizzle on read.
            col_sw = _a16w4_swizzle_xor16(x_row_local[i], col_bytes, fx.Int32(k_blocks16), enable=True)
            row_k_dw = x_row_base_div4[i] + base_k_div4
            global_byte = row_k_dw * fx.Int32(4) + col_sw
            lds_byte = x_row_local[i] * fx.Int32(KH_TILE_BYTES) + col_bytes
            g_tile = global_byte // fx.Int32(x_load_bytes)
            l_tile = lds_byte // fx.Int32(x_load_bytes)
            if const_expr(use_k16):
                # gfx942: buffer_load gmem->regs, then ds_write regs->LDS.
                r = fx.make_rmem_tensor(fx.make_layout(chunk_i32, 1), fx.Int32)
                fx.copy(x_dma_atom, fx.slice(x_dma_tiles4, (None, g_tile)), r)
                fx.copy(x_lds_store_atom, r, fx.slice(s_x_dma_tiles, (None, l_tile)))
            else:
                fx.copy(
                    x_dma_atom,
                    fx.slice(x_dma_tiles4, (None, g_tile)),
                    fx.slice(s_x_dma_tiles, (None, l_tile)),
                )

    row_a_lds = lane_mod_16
    # Lane L covers K[L*32 .. L*32+31]; 32 elements is 64 B in bf16, 32 B in fp8.
    col_base_bytes_L = lane_div_16 * fx.Int32(32 * elem_bytes)

    def _a_col_bytes_for_ku(ku):
        _k0_blk = ku // 4
        _ku_in = ku % 4
        # One ku is 8 elements; one k0-block is 128 elements.
        return col_base_bytes_L + fx.Int32(
            _ku_in * (8 * elem_bytes) + _k0_blk * (128 * elem_bytes)
        )

    def lds_load_a(mi, ku):
        row = row_a_lds + fx.Int32(mi * 16)
        # Same XOR swizzle as the DMA write. It only flips bits >= 4, so it preserves the
        # low 4 address bits and an 8 B fp8 read still lands inside one 16 B block.
        col_swz_bytes = _a16w4_swizzle_xor16(row, _a_col_bytes_for_ku(ku), fx.Int32(k_blocks16), enable=True)
        byte_off = row * fx.Int32(KH_TILE_BYTES) + col_swz_bytes
        r = fx.make_rmem_tensor(fx.make_layout(_a_read_dwords, 1), fx.Int32)
        fx.copy_atom_call(
            a_copy_atom, fx.slice(s_x_i32x4_tiles, (None, byte_off // fx.Int32(_a_read_bytes))), r
        )
        return fx.Vector(fx.memref_load_vec(r)).bitcast(_a_elem)

    def load_b_raw(base_k, n_blk, n_intra):
        raw = []
        for k0i in range_constexpr(_k0_count):
            k0 = (base_k + fx.Int32(k0i * 128)) // fx.Int32(128)
            idx_pack = fx.Int32(
                crd2idx(
                    [fx.Int64(n_blk), fx.Int64(k0), fx.Int64(lane_div_16), fx.Int64(n_intra), fx.Int64(0)],
                    layout_b,
                )
            )
            # idx_pack is a fp4-byte offset; the dwordx4 tile index = (idx_pack/4 dwords)/4.
            r = fx.make_rmem_tensor(w_reg_lay, fx.Int32)
            fx.copy(w_copy_atom, fx.slice(w_tiles, (None, idx_pack // fx.Int32(16))), r)
            v4 = fx.Vector(fx.memref_load_vec(r))
            raw.append([fx.Int32(v4[j]) for j in range(4)])
        return raw

    def load_b_raw_bf16(base_k, n_blk, n_intra):
        # Raw bf16 W: one dwordx4 (8 bf16) per ku = one MFMA K32 fragment. Index to
        # match the fp4 (klane_hw=lane_div_16, ku)->K order (see gemm1 counterpart).
        raw = []
        base_k0 = base_k // fx.Int32(32)
        for ku in range_constexpr(k_unroll):
            _k0_blk = ku // 4
            bf_k0 = base_k0 + fx.Int32(_k0_blk * 4) + lane_div_16
            bf_klane = fx.Int32(ku % 4)
            elem_idx = fx.Int32(
                crd2idx(
                    [fx.Int64(n_blk), fx.Int64(bf_k0), fx.Int64(bf_klane), fx.Int64(n_intra), fx.Int64(0)],
                    layout_b_bf16,
                )
            )
            # elem_idx is a bf16-elem offset; dword index = elem_idx*2/4, tile index = /4.
            r = fx.make_rmem_tensor(w_reg_lay, fx.Int32)
            fx.copy(w_copy_atom, fx.slice(w_tiles, (None, elem_idx // fx.Int32(8))), r)
            raw.append(fx.Vector(fx.memref_load_vec(r)).bitcast(fx.BFloat16))  # v8bf16
        return raw

    def load_b_scale(base_k, mni, n_pack):
        # Per-lane scalar e8m0 gather (dict-cached across ku).
        scales = []
        cache = {}
        for ku in range_constexpr(k_unroll):
            _k0_blk = ku // 4
            adj_ku = base_k // fx.Int32(32) + fx.Int32(_k0_blk * 4) + lane_div_16
            k_pack_sub = (adj_ku // fx.Int32(4)) % fx.Int32(2)
            s_ku = adj_ku // fx.Int32(8)
            if _k0_blk not in cache:
                idx = (
                    mni * fx.Int32(sc_stride_n0)
                    + s_ku * fx.Int32(sc_stride_k0)
                    + lane_div_16 * fx.Int32(sc_stride_klane)
                    + lane_mod_16
                )
                cache[_k0_blk] = _buffer_i32_scalar_read(sw_tiles, idx, sw_read_atom)
            packed = cache[_k0_blk]
            byte_even = k_pack_sub * fx.Int32(2)
            byte_odd = byte_even + fx.Int32(1)
            se = _e8m0_byte_to_f32(packed, byte_even)
            so = _e8m0_byte_to_f32(packed, byte_odd)
            scales.append((n_pack == fx.Int32(0)).select(se, so))
        return scales

    def load_b_scale_byte(base_k, mni, n_pack):
        """The raw e8m0 exponent bytes, one per k0-block rather than per ku.

        Same addressing as :func:`load_b_scale`, but ``adj_ku`` depends only on the
        k0-block and ``lane_div_16``, so all four ku in a block share one byte. The fp8
        path turns that byte into a magnitude table and has no reason to redo it per ku.
        """
        out = {}
        for ku in range_constexpr(k_unroll):
            _k0_blk = ku // 4
            if _k0_blk in out:
                continue
            adj_ku = base_k // fx.Int32(32) + fx.Int32(_k0_blk * 4) + lane_div_16
            k_pack_sub = (adj_ku // fx.Int32(4)) % fx.Int32(2)
            s_ku = adj_ku // fx.Int32(8)
            idx = (
                mni * fx.Int32(sc_stride_n0)
                + s_ku * fx.Int32(sc_stride_k0)
                + lane_div_16 * fx.Int32(sc_stride_klane)
                + lane_mod_16
            )
            packed = _buffer_i32_scalar_read(sw_tiles, idx, sw_read_atom)
            byte_even = k_pack_sub * fx.Int32(2)
            be = _e8m0_byte_raw(packed, byte_even)
            bo = _e8m0_byte_raw(packed, byte_even + fx.Int32(1))
            out[_k0_blk] = (n_pack == fx.Int32(0)).select(be, bo)
        return out

    def load_b_scale_int4(base_k, col_g):
        # int4 groupwise (bf16-pair) scale, per-lane N = col_g. See gemm1 counterpart.
        scales = []
        base_dword = col_g * fx.Int32(_g_half)
        for ku in range_constexpr(k_unroll):
            _k0_blk = ku // 4
            adj_ku = base_k // fx.Int32(32) + fx.Int32(_k0_blk * 4) + lane_div_16
            pair_idx = adj_ku // fx.Int32(2)
            packed = _buffer_i32_scalar_read(sw_tiles, base_dword + pair_idx, sw_read_atom)
            lo = fx.Float32(_raw(packed << fx.Int32(16)).bitcast(T.f32))
            hi = fx.Float32(_raw(packed & fx.Int32(0xFFFF0000)).bitcast(T.f32))
            scales.append((adj_ku % fx.Int32(2) == fx.Int32(0)).select(lo, hi))
        return scales

    vec2_bf16 = ir.Type.parse("vector<2xbf16>")

    def upconvert_b_fp8(raw, ku, tab):
        """FP4 -> 8 fp8 bytes with 2^r already folded in by the chosen table."""
        return _fp4_nibble_to_fp8x8_vec(fx.Int32(_raw(raw[ku // 4][ku % 4])), *tab)

    def fp8_tables_for(scale_bytes, sref_byte):
        """Magnitude-table pool per k0-block from the per-output-column reference.

        ``neg_r = s_ref(n) - s(n, kg)`` is non-negative because s_ref is the column max
        over K-groups; the host asserts the residual stays inside the table set.
        """
        return {blk: _fp8_table_select(sref_byte - b) for blk, b in scale_bytes.items()}

    def upconvert_b(raw, ku, scale_f32):
        if const_expr(_is_bf16):
            return raw[ku]  # already v8bf16 (no scale, no upconvert)
        i32_val = _raw(raw[ku // 4][ku % 4])
        if const_expr(_is_int4):
            return _int4_nibble_to_bf16x8(fx.Int32(i32_val), scale_f32, use_k16=use_k16)
        if const_expr(use_k16):
            # gfx942: v_cvt_scalef32_pk_bf16_fp4 is gfx950-only -> byte-lookup E2M1 decode.
            return _fp4_nibble_to_bf16x8_lut(fx.Int32(i32_val), scale_f32)
        s_raw = _raw(scale_f32)
        i32s = []
        for sel in range_constexpr(4):
            pp = rocdl.cvt_scalef32_pk_bf16_fp4(vec2_bf16, i32_val, s_raw, sel)
            i32s.append(fx.Int32(fx.Vector(pp).bitcast(fx.Int32)[0]))
        v4i32 = fx.Vector.from_elements([_raw(x) for x in i32s], fx.Int32)
        return v4i32.bitcast(fx.BFloat16)

    # ---- N-column addressing (W2 cols of model_dim; wave owns _n_per_wave) ------
    n_tile_base = wave * fx.Int32(_n_per_wave)
    col_g_list = []
    n_blk_list, n_intra_list, scale_mni_list, scale_np_list = [], [], [], []
    for ni in range_constexpr(num_acc_n):
        col_g = by_n + n_tile_base + fx.Int32(ni * 16) + lane_mod_16
        col_g_list.append(col_g)
        # bf16 W folds expert_off into the resource base (see w_tiles); mxfp4/int4 index it.
        _row_expert_off = fx.Int32(0) if const_expr(_is_bf16) else expert_off
        row_w = _row_expert_off + col_g
        n_blk_list.append(row_w // fx.Int32(16))
        n_intra_list.append(row_w % fx.Int32(16))
        ng = expert_off + by_n + n_tile_base + fx.Int32(ni * 16)
        scale_mni_list.append(ng // fx.Int32(32))
        scale_np_list.append((ng // fx.Int32(16)) % fx.Int32(2))
    # int4 groupwise scale: per-lane N = expert_off + col_g (row_w already computed).
    if const_expr(_is_int4):
        scale_n_list = [expert_off + col_g_list[ni] for ni in range_constexpr(num_acc_n)]
    if const_expr(_is_a_fp8):
        # Per-output-column reference exponent, [E, N_OUT] uint8. Feeds both the in-loop
        # residual (s_ref - s) and the 2^s_ref applied on the accumulator below.
        sref_col = [
            _global_u8_at(arg_sref, expert_off + col_g_list[ni])
            for ni in range_constexpr(num_acc_n)
        ]

    # ---- accumulators: accm[mi][ni] f32[4] (layout the atomic epilog expects) --
    acc_layout = fx.make_layout(4, 1)
    accm = [[fx.make_rmem_tensor(acc_layout, fx.Float32) for _ in range(num_acc_n)] for _ in range(m_repeat)]
    zero4 = Vec.filled(4, 0.0, fx.Float32)
    for mi in range_constexpr(m_repeat):
        for ni in range_constexpr(num_acc_n):
            accm[mi][ni].store(zero4)

    # Arch/dtype gate: gfx950 bf16 K=32 (one MFMA/K-step); gfx942 bf16 (use_k16) has no
    # 16x16x32 bf16, so each v8bf16 splits into two v4 halves -> TWO 16x16x16 into the
    # same acc. fp8: gfx942 *does* have 16x16x32 fp8, so one MFMA per K-step again --
    # half the MFMA instructions of the gfx942 bf16 path for identical work.
    _split_k16 = use_k16 and not _is_a_fp8
    if const_expr(_is_a_fp8):
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, _a_elem))
    elif const_expr(use_k16):
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
    else:
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))

    def _bf16_frag(v8):
        t = fx.make_rmem_tensor(fx.make_layout(8, 1), _a_elem)
        t.store(v8)
        return t

    def _bf16_frag4(v8, half):
        t = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.BFloat16)
        t.store(fx.Vector.from_elements([_raw(v8[half * 4 + j]) for j in range_constexpr(4)], fx.BFloat16))
        return t

    def _mma(acc, a8, b8):
        if const_expr(_split_k16):
            for h in range_constexpr(2):
                fx.gemm(mma_atom, acc, _bf16_frag4(a8, h), _bf16_frag4(b8, h), acc)
        else:
            fx.gemm(mma_atom, acc, _bf16_frag(a8), _bf16_frag(b8), acc)

    def _mma_scaled_add(acc, a8, b8_unscaled, scale_f32):
        # acc += (scale * _acc_fold) * (A @ b_unscaled). See gemm1's copy; the whole
        # lane shares one N, so one scalar covers all four accumulator elements.
        tmp = fx.make_rmem_tensor(acc_layout, fx.Float32)
        tmp.store(zero4)
        _mma(tmp, a8, b8_unscaled)
        eff = scale_f32 if const_expr(_acc_fold == 1.0) else scale_f32 * fx.Float32(_acc_fold)
        va = Vec(fx.memref_load_vec(acc))
        vt = Vec(fx.memref_load_vec(tmp))
        acc.store(Vec.from_elements([va[j] + eff * vt[j] for j in range_constexpr(4)], fx.Float32))

    def upconvert_b_raw(raw, ku):
        """Magnitude-only dequant for the accumulator-scaled path."""
        i32_val = fx.Int32(_raw(raw[ku // 4][ku % 4]))
        if const_expr(_is_int4):
            return _int4_nibble_to_bf16x8_raw(i32_val, use_k16=use_k16)
        return _fp4_nibble_to_bf16x8_raw(i32_val)

    for kt in range_constexpr(K_TILES_TOTAL):
        base_k = fx.Int32(kt * TILE_K)
        dma_a_tile_to_lds(base_k)
        if const_expr(_is_bf16):
            b_raw = [load_b_raw_bf16(base_k, n_blk_list[ni], n_intra_list[ni]) for ni in range_constexpr(num_acc_n)]
            b_sc = None
        else:
            b_raw = [load_b_raw(base_k, n_blk_list[ni], n_intra_list[ni]) for ni in range_constexpr(num_acc_n)]
            if const_expr(_is_a_fp8):
                # The "scale" carried forward is a magnitude-table pool per k0-block, not
                # an f32: the residual is folded into the FP4->fp8 lookup itself.
                b_sc = [
                    fp8_tables_for(
                        load_b_scale_byte(base_k, scale_mni_list[ni], scale_np_list[ni]),
                        sref_col[ni],
                    )
                    for ni in range_constexpr(num_acc_n)
                ]
            elif const_expr(_is_int4):
                b_sc = [load_b_scale_int4(base_k, scale_n_list[ni]) for ni in range_constexpr(num_acc_n)]
            else:
                b_sc = [
                    load_b_scale(base_k, scale_mni_list[ni], scale_np_list[ni]) for ni in range_constexpr(num_acc_n)
                ]
        gpu.barrier()
        for ni in range_constexpr(num_acc_n):
            for ku in range_constexpr(k_unroll):
                if const_expr(_is_a_fp8):
                    bb = upconvert_b_fp8(b_raw[ni], ku, b_sc[ni][ku // 4])
                    for mi in range_constexpr(m_repeat):
                        a8 = lds_load_a(mi, ku)
                        _mma(accm[mi][ni], a8, bb)
                    continue
                if const_expr(_acc_scale):
                    bb = upconvert_b_raw(b_raw[ni], ku)
                    for mi in range_constexpr(m_repeat):
                        a8 = lds_load_a(mi, ku)
                        _mma_scaled_add(accm[mi][ni], a8, bb, b_sc[ni][ku])
                    continue
                _bsc = None if const_expr(_is_bf16) else b_sc[ni][ku]
                bb = upconvert_b(b_raw[ni], ku, _bsc)
                for mi in range_constexpr(m_repeat):
                    a8 = lds_load_a(mi, ku)
                    _mma(accm[mi][ni], a8, bb)
        gpu.barrier()

    # ---- fp8 rescale: undo the two factors the MFMA could not carry ------------
    # 2^s_ref(n) is per output column and A's scale is per sorted row, so both are
    # constant over K and belong on the finished accumulator. Row/col mapping here must
    # match _atomic_bf16_epilog's: element v of accm[mi][ni] is row mi*16 +
    # lane_div_16*4 + v, column col_g_list[ni]. Padding rows need no mask -- a_scale is
    # sized to max_sorted like the intermediate it came from, so the read is in bounds,
    # and the epilogue drops those rows anyway on token_id >= i32_M.
    if const_expr(_is_a_fp8):
        sref_f32 = [
            _e8m0_byte_to_f32(sref_col[ni], fx.Int32(0)) for ni in range_constexpr(num_acc_n)
        ]
        for mi in range_constexpr(m_repeat):
            row_base = m_row + fx.Int32(mi * 16) + lane_div_16 * fx.Int32(4)
            a_sc = [
                fx.Float32(_global_f32_at(arg_a_scale, row_base + fx.Int32(v)))
                for v in range_constexpr(4)
            ]
            for ni in range_constexpr(num_acc_n):
                va = Vec(fx.memref_load_vec(accm[mi][ni]))
                accm[mi][ni].store(
                    Vec.from_elements(
                        [va[v] * a_sc[v] * sref_f32[ni] for v in range_constexpr(4)],
                        fx.Float32,
                    )
                )

    # ---- epilogue: atomic bf16 scatter (routing-weighted). K-loop done, so the A-LDS
    # region (offset 0) is reused for the epilog's f32 acc staging.
    gpu.barrier()
    lds_acc_base_i32 = fx.Int32(fx.ptrtoint(lds_raw_ptr))
    accm_v = [[accm[i][J].load().ir_value() for J in range(num_acc_n)] for i in range(m_repeat)]
    _atomic_bf16_epilog(
        lds_acc_base_i32,
        accm_v,
        arg_out,
        arg_stids,
        arg_sweights,
        m_row,
        n_block_idx,
        wave,
        lane,
        i32_M,
        BM,
        N_OUT,
        TILE_N,
    )


def gemm2_a16w4_grid(BM, *, N_OUT, TILE_N, max_m_blocks, persist=False):
    """Flattened launch grid for a16w4 gemm2.

    Non-persistent: one CTA per (m-block x n-block) tile over padded ``max_m_blocks``.
    Persistent: cap to ``min(total_work, cap)`` CTAs (only when padded work
    > ``cap*4``); each CTA loops over its real work-tiles.
    """
    total_work = int(max_m_blocks) * (N_OUT // TILE_N)
    cap = _persistent_cta_cap()
    if persist and total_work > cap * 4:
        return min(total_work, cap)
    return total_work


def compile_gemm2_a16w4_port(
    BM=32,
    *,
    NE,
    N_OUT,
    D_INTER,
    TILE_N=256,
    TILE_K=256,
    xcd_swizzle=1,
    b_cache_mod=2,
    waves_per_eu=None,
    w_dtype="mxfp4",
    a_dtype="bf16",
    persist=False,
):
    """a16w4/a16wi4/a16w16 (bf16 or fp8 intermediate A x mxfp4/int4/bf16 W2) stage2.

    N_OUT = model_dim (down-proj output). D_INTER = inter_dim (contraction). Output
    bf16 [tokens, model_dim] via atomic (routing-weighted) scatter.

    ``a_dtype='fp8'`` puts the stage1 intermediate on mfma_f32_16x16x32_fp8_fp8 and
    needs two extra pointers: the per-sorted-row A scale and the per-output-column
    reference exponent. See _gemm2_body_a16w4.

    ``xcd_swizzle`` (>0) bijectively round-robins the launch index across the 8 XCDs to
    balance per-XCD/HBM traffic (gemm2 is HBM-bound), + optional M-group swizzle for
    per-XCD L2 locality (group = xcd_swizzle m-blocks).
    """
    assert w_dtype in ("mxfp4", "int4", "bf16"), f"w_dtype must be 'mxfp4', 'int4' or 'bf16', got {w_dtype!r}"
    assert a_dtype in ("bf16", "fp8"), f"a_dtype must be 'bf16' or 'fp8', got {a_dtype!r}"
    # fp8 A decodes the weight through the mxfp4 byte-lookup plus the residual table, so
    # it is mxfp4-only (int4 has a bf16-pair scale, bf16 W has no scale at all).
    assert not (a_dtype == "fp8" and w_dtype != "mxfp4"), (
        f"a_dtype='fp8' is mxfp4-only, got w_dtype={w_dtype!r}"
    )
    # Arch-gate K=16 (gfx942) vs K=32 (gfx950); see a16wmix_use_k16.
    _use_k16 = a16wmix_use_k16()
    _K = D_INTER
    assert _K % TILE_K == 0, f"D_INTER (K) must be a multiple of {TILE_K}, got {_K}"
    assert N_OUT % TILE_N == 0, f"model_dim (N_OUT) must be a multiple of {TILE_N}, got {N_OUT}"
    assert BM % 16 == 0, f"BM must be a multiple of 16, got {BM}"
    _num_n_blocks = N_OUT // TILE_N
    _a_elem_bytes = 1 if a_dtype == "fp8" else 2
    KH_TILE_BYTES = TILE_K * _a_elem_bytes

    # LDS: A tile (BM x TILE_K, fp8 or bf16) then f32 accumulator region (BM x TILE_N).
    # The accumulator region dominates, so fp8 saves LDS but rarely changes occupancy.
    _a_bytes = BM * KH_TILE_BYTES
    _acc_bytes = lds_acc_bytes_for(BM, TILE_N)
    _lds_bytes = _a_bytes + _acc_bytes
    # Refuse rather than emit a kernel that cannot launch. Without this the overflow
    # surfaces at launch as hipErrorIllegalState ("the operation cannot be performed in
    # the present state"), which says nothing about the cause. BM=64 with TILE_N=256
    # needs 82 KB on fp8 and 98 KB on bf16, so this is not an fp8 limit -- the host
    # steps TILE_N down for it (see flydsl_a16w4_gemm2).
    if _lds_bytes > LDS_LIMIT_BYTES:
        raise NotImplementedError(
            f"a16w4 gemm2 needs {_lds_bytes} B of LDS for BM={BM} TILE_N={TILE_N} "
            f"TILE_K={TILE_K} a_dtype={a_dtype} ({_a_bytes} B A tile + {_acc_bytes} B "
            f"accumulator), over the {LDS_LIMIT_BYTES} B limit. Reduce TILE_N or BM."
        )

    _wd_tag = "" if w_dtype == "mxfp4" else f"_{w_dtype}"
    _ad_tag = "" if a_dtype == "bf16" else f"_a{a_dtype}"
    _name = f"gemm2_a16w4{_wd_tag}{_ad_tag}_port_ne{NE}_h{N_OUT}_i{_K}_bm{BM}_tn{TILE_N}"
    if b_cache_mod != 2:
        _name += f"_bcm{b_cache_mod}"
    if xcd_swizzle > 0:
        _name += f"_xcd{xcd_swizzle}"
    if waves_per_eu:
        _name += f"_w{waves_per_eu}"
    if persist:
        _name += "_persist"

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, _lds_bytes, 16]

    @flyc.kernel(name=_name, known_block_size=[256, 1, 1])
    def gemm2_kernel(
        arg_a: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        arg_out: fx.Int64,
        arg_a_scale: fx.Int64,
        arg_sref: fx.Int64,
    ):
        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        tx_i32 = fx.Int32(gpu.thread_id("x"))
        bx_i32 = fx.Int32(gpu.block_id("x"))
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
        cumsum0 = _global_i32_at(arg_cumsum, fx.Int32(0))
        total_m_blocks = cumsum0 // fx.Int32(BM)
        # Clamp to the host-side buffer size. The kernel previously trusted
        # cumsum0 unconditionally; a recycled/corrupted value walks off
        # sorted_ids / A / out and raises a GPU Memory Fault.
        total_m_blocks = fx.Int32(arith.minsi(_raw(total_m_blocks), _raw(i32_max_m_blocks)))
        bound = total_m_blocks * fx.Int32(_num_n_blocks)

        # Bijective XCD round-robin over valid tiles [0, bound) to balance per-XCD/HBM
        # traffic; xcd_swizzle>0 also M-group-swizzles for per-XCD L2 locality.
        _NXCD = 8
        _xq = _udiv(bound, _NXCD)
        _xr = _umod(bound, _NXCD)
        _SW = xcd_swizzle

        def _xcd_np(pid):
            xc = _umod(pid, _NXCD)
            wgid = xc * _xq + fx.Int32(arith.minsi(_raw(xc), _raw(_xr))) + _udiv(pid, _NXCD)
            if const_expr(_SW <= 0):
                return wgid
            _ng = fx.Int32(_SW * _num_n_blocks)
            group_id = wgid // _ng
            first_pid_m = group_id * fx.Int32(_SW)
            remaining_m = total_m_blocks - first_pid_m
            group_size_m = fx.Int32(arith.minsi(_raw(remaining_m), _raw(fx.Int32(_SW))))
            wig = wgid % _ng
            m_block = first_pid_m + (wig % group_size_m)
            n_block = wig // group_size_m
            return m_block * fx.Int32(_num_n_blocks) + n_block

        def _run_tile(tile):
            _gemm2_body_a16w4(
                lds_raw_ptr,
                arg_a,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_stids,
                arg_sweights,
                arg_out,
                tile,
                lane,
                wave,
                i32_M,
                arg_a_scale,
                arg_sref,
                BM=BM,
                TILE_N=TILE_N,
                TILE_K=TILE_K,
                N_OUT=N_OUT,
                INTER=_K,
                NE=NE,
                b_cache_mod=b_cache_mod,
                w_dtype=w_dtype,
                a_dtype=a_dtype,
                use_k16=_use_k16,
            )

        if const_expr(persist):
            # Persistent CU-limited grid (~NUM_CU CTAs): each CTA does tile bx_i32 then
            # strides by grid size over [0, bound); _xcd_np maps every visited index, so
            # each tile runs once (same mapping as non-persistent). Loop-top barrier
            # separates the prev tile's epilog LDS from the next tile's A-DMA.
            grid_nb = fx.Int32(gpu.grid_dim.x)
            if bx_i32 < bound:
                _run_tile(_xcd_np(bx_i32))
            for iv in range(bx_i32 + grid_nb, bound, gpu.grid_dim.x):
                gpu.barrier()
                _run_tile(_xcd_np(fx.Int32(iv)))
        else:
            if bx_i32 < bound:
                _run_tile(_xcd_np(bx_i32))

    @flyc.jit
    def launch_gemm2(
        arg_a: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        i32_grid: fx.Int32,
        arg_out: fx.Int64,
        arg_a_scale: fx.Int64,
        arg_sref: fx.Int64,
        stream: fx.Stream,
    ):
        grid_x = fx.Int64(i32_grid)
        gemm2_kernel(
            arg_a,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_stids,
            arg_sweights,
            i32_M,
            i32_max_m_blocks,
            arg_out,
            arg_a_scale,
            arg_sref,
            value_attrs={"rocdl.waves_per_eu": waves_per_eu} if waves_per_eu else None,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm2
