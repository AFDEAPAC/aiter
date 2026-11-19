# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""A8W4/A4W4 GEMM utilities for MegaMoE v2 Stage1."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..act import situ_mul, tanh_via_sigmoid_f32
from ..moe_2stage_a16wmix.utils import (
    _FP8_SWAR_HI_BASE,
    _FP8_SWAR_LO_BASE,
    _e8m0_byte_raw,
    _e8m0_byte_to_f32,
    _fp4_nibble_to_bf16x8_lut,
    _fp4_nibble_to_fp8x8_vec,
    _fp8_table_select,
    a16wmix_use_k16,
    fp8_elem_type,
)
from ..tensor_shim import _to_raw as _raw
from ..tensor_shim import buf_copy_load, buf_copy_store, ptr_buf_tensor
from . import envs

_PACK = 2  # fp4 micro-scale pack (per-32 E8M0): pack_M = pack_N = pack_K = 2


def fp8_e8m0_bias():
    """Exponent offset of the 1x32 FP8 quant scale: the scaled |max| stays below
    1.5 * 2**bias.  gfx942 packs FP8 as e4m3fnuz (max 240), gfx95x as OCP (max 448)."""
    return 7 if a16wmix_use_k16() else 8


def fp8x8_to_bf16x8(lo_i32, hi_i32, scale_f32):
    """Eight FP8 bytes (hardware FP8 format) times a power-of-two scale -> v8bf16.

    The product is exact in bf16, so the f32 -> bf16 step is a truncation."""
    out = []
    for word in (lo_i32, hi_i32):
        for sel in (False, True):
            pair = fx.Vector(rocdl.cvt_pk_f32_fp8(T.vec(2, T.f32), word, sel))
            lo_b = (fx.Float32(pair[0]) * scale_f32).bitcast(fx.Int32).shrui(fx.Int32(16))
            hi_b = (fx.Float32(pair[1]) * scale_f32).bitcast(fx.Int32) & fx.Int32(0xFFFF0000)
            out.append(lo_b | hi_b)
    return fx.Vector.from_elements(out, fx.Int32).bitcast(fx.BFloat16)


def dequant_b_fp4_k16(b_i32x4, sb_word, sb_byte):
    """B operand of one 128-K step (4 dwords of 8 FP4) -> 4 x v8bf16."""
    sb = _e8m0_byte_to_f32(fx.Int32(sb_word), fx.Int32(sb_byte))
    b_vec = Vec(b_i32x4)
    return [_fp4_nibble_to_bf16x8_lut(fx.Int32(b_vec[j]), sb) for j in range_constexpr(4)]


def k16_mma(atom, c_frag, a_bf16x8s, b_bf16x8s):
    """gfx942 replacement of one 16x16x128 FP8 x FP4 scaled MFMA.

    In the a16wmix K order lane group g owns K [32g, 32g+32) of the 128-K step,
    i.e. exactly one E8M0 block for both operands, so A and B are dequantized
    with their own scales and the step becomes 8 bf16 16x16x16 MFMAs."""
    for j in range_constexpr(4):
        a8 = a_bf16x8s[j]
        b8 = b_bf16x8s[j]
        for h in range_constexpr(2):
            a4 = fx.make_rmem_tensor(4, fx.BFloat16)
            b4 = fx.make_rmem_tensor(4, fx.BFloat16)
            a4.store(fx.Vector.from_elements([a8[h * 4 + i] for i in range_constexpr(4)], fx.BFloat16))
            b4.store(fx.Vector.from_elements([b8[h * 4 + i] for i in range_constexpr(4)], fx.BFloat16))
            fx.gemm(atom, c_frag, a4, b4, c_frag)


def a_per_token_enabled():
    """gfx942 GEMM1 with per-token FP8 A: FP8 MFMA, FP4 B decoded through the
    a16wmix residual tables, token and column scales applied in the epilogue."""
    return a16wmix_use_k16() and envs.AITER_MEGA_GFX942_A_PER_TOKEN


def a2_fp8_enabled():
    """gfx942 GEMM2 on FP8 MFMA with a shared exponent per 128-K step of A."""
    return a16wmix_use_k16() and envs.AITER_MEGA_GFX942_A2_FP8


def row_step_scales(sa_word, byte_sel):
    """2^e of the A rows this lane's accumulator holds (rows 4 * (lane // 16) + ii).

    ``sa_word`` holds, in lane (g, r), the scale bytes of A-operand row r; the rows
    of the accumulator are fetched from lanes 0..15 with ds_bpermute."""
    lane = fx.thread_idx.x % 64
    base = (lane // fx.Int32(16)) * fx.Int32(4)
    out = []
    for ii in range_constexpr(4):
        w = fx.Int32(fx.rocdl.ds_bpermute(T.i32, (base + fx.Int32(ii)) * fx.Int32(4), fx.Int32(sa_word)))
        out.append(_e8m0_byte_to_f32(w, fx.Int32(byte_sel)))
    return out


# Under EP the W2 column maximum spans the whole inter dim, and the V4.1 Flash
# checkpoint has one group at r = -7 (layer 39), one past the a16wmix tables.
# W1 stays within -6, and GEMM1 keeps the cheaper a16wmix decode.
MAX_NEG_RESIDUAL_W1 = 6
MAX_NEG_RESIDUAL_W2 = 7


def _fp8_table_select_r7(neg_r):
    """a16wmix ``_fp8_table_select`` extended to r = -7 on e4m3fnuz: there the 0.5
    magnitude is the subnormal 0x04 (the SWAR subtract leaves 0x00); every other
    byte still follows the subtract."""
    t = fx.Int32(arith.minsi(_raw(fx.Int32(neg_r)), _raw(fx.Int32(MAX_NEG_RESIDUAL_W2))))
    t = fx.Int32(arith.maxsi(_raw(t), _raw(fx.Int32(0))))
    sub = t * fx.Int32(0x08080808)
    lo = ((fx.Int32(_FP8_SWAR_LO_BASE) - sub) & fx.Int32(0xFFFFFF00)) | fx.Int32(0x01)
    lo = lo | ((t & (t >> fx.Int32(1)) & (t >> fx.Int32(2)) & fx.Int32(1)) << fx.Int32(10))
    hi = fx.Int32(_FP8_SWAR_HI_BASE) - sub
    return lo, hi


def decode_b_fp4_fp8(b_i32x4, sb_word, sb_byte, sref_i32, r7=False):
    """B operand of one 128-K step -> 4 x v8 e4m3fnuz with 2^(s - s_ref) folded in.

    s_ref is the column's largest E8M0 exponent, so the residual is <= 0; the
    epilogue multiplies the accumulator by 2^s_ref (a16wmix FP8 stage1 scheme).
    ``r7`` decodes residuals down to -7 instead of -6."""
    neg_r = fx.Int32(sref_i32) - _e8m0_byte_raw(fx.Int32(sb_word), fx.Int32(sb_byte))
    tab_lo, tab_hi = _fp8_table_select_r7(neg_r) if r7 else _fp8_table_select(neg_r)
    b_vec = Vec(b_i32x4)
    return [_fp4_nibble_to_fp8x8_vec(fx.Int32(b_vec[j]), tab_lo, tab_hi) for j in range_constexpr(4)]


def fp8_mma_k32(atom, c_frag, a_i32x8, b_fp8x8s):
    """One 128-K step as 4 FP8 16x16x32 MFMAs: in the a16wmix K order lane group g
    supplies A/B elements [32g + 8j, +8) to MFMA j."""
    a_vec = Vec(a_i32x8)
    elem = fp8_elem_type()
    for j in range_constexpr(4):
        a_frag = fx.make_rmem_tensor(8, elem)
        b_frag = fx.make_rmem_tensor(8, elem)
        a_frag.store(fx.Vector.from_elements([a_vec[2 * j], a_vec[2 * j + 1]], fx.Int32).bitcast(elem))
        b_frag.store(b_fp8x8s[j])
        fx.gemm(atom, c_frag, a_frag, b_frag, c_frag)


def scale_acc_per_token(acc, m_repeat, num_acc_n, token_scale_rsrc, row_base_i32, scale_row_i32, sref):
    """acc[mi, ni] *= token_scale[row] * 2^(s_ref[ni] - 127), before the activation.

    The token scale is the f32 in the first 4 bytes of each row of the dispatched
    scale payload (``scale_row_i32`` dwords per row)."""
    lane = fx.thread_idx.x % 64
    col = [(fx.Int32(sref[ni]) << fx.Int32(23)).bitcast(fx.Float32) for ni in range_constexpr(num_acc_n)]
    out = list(acc)
    for mi in range_constexpr(m_repeat):
        rows = []
        for ii in range_constexpr(4):
            row = row_base_i32 + fx.Int32(mi * 16) + (lane // fx.Int32(16)) * fx.Int32(4) + fx.Int32(ii)
            rows.append(fx.Int32(token_scale_rsrc[row * fx.Int32(scale_row_i32)]).bitcast(fx.Float32))
        for ni in range_constexpr(num_acc_n):
            idx = mi * num_acc_n + ni
            v = Vec(out[idx])
            out[idx] = Vec.from_elements([v[ii] * rows[ii] * col[ni] for ii in range_constexpr(4)], fx.Float32)
    return out


def dequant_a_fp8_k16(a_i32x8, sa_word, sa_byte):
    """A operand of one 128-K step (8 dwords, a16wmix K order) -> 4 x v8bf16."""
    sa = _e8m0_byte_to_f32(fx.Int32(sa_word), fx.Int32(sa_byte))
    a_vec = Vec(a_i32x8)
    return [
        fp8x8_to_bf16x8(fx.Int32(a_vec[2 * j]), fx.Int32(a_vec[2 * j + 1]), sa)
        for j in range_constexpr(4)
    ]


@flyc.jit
def _load_row_map_subgroup(row_map_rsrc, row_index, lane, subgroup_width):
    """Load and broadcast one source row per K-step copy subgroup."""
    subgroup_lane = lane % fx.Int32(subgroup_width)
    packed = fx.Int32(0)
    if subgroup_lane == fx.Int32(0):
        packed = row_map_rsrc[row_index]
    source_lane = lane - subgroup_lane
    return fx.Int32(
        fx.rocdl.ds_bpermute(
            T.i32,
            source_lane * fx.Int32(4),
            packed,
        )
    )


def wait_lds_barrier(vmcnt=63):
    """Drain LDS writes and optionally older VMEM while preserving newer loads."""
    fx.rocdl.s_waitcnt(vmcnt=vmcnt, lgkmcnt=0)
    fx.barrier()


class TileScheduler:
    """Per-tile expert / gate-base-row resolver (flat persistent loop + XCD swizzle live in kernel body)."""

    def __init__(self, *, expert_rsrc, inter_dim, expert_offset=0):
        self._expert_rsrc = expert_rsrc
        self._inter_dim = inter_dim
        # Dispatch emits global expert IDs while W1 is rank-local.
        self._expert_offset = int(expert_offset)

    def expert_of(self, m_tile_i32):
        g = self._expert_rsrc[m_tile_i32]
        if const_expr(self._expert_offset != 0):
            return g - fx.Int32(self._expert_offset)
        return g

    def gate_base_row(self, expert_i32):
        return expert_i32 * fx.Int32(2 * self._inter_dim)


class ATileLoader:
    """Load expert-major A rows from contiguous or source-key storage."""

    def __init__(
        self,
        *,
        row_bytes,
        sort_block_m,
        k_step_bytes,
        total_threads,
        swizzle=False,
        x_tensor=None,
        async_copy=False,
        indexed_input=False,
        row_map_rsrc=None,
        source_rows=0,
        a_dtype="fp8",
    ):
        self._sort_block_m = sort_block_m
        self._k_step_bytes = k_step_bytes
        self._total_threads = total_threads
        self._swizzle = swizzle
        self._row_bytes = row_bytes
        self._tx = fx.thread_idx.x
        self._wave = self._tx // 64
        self._x_tensor = x_tensor
        self._async_copy = bool(async_copy)
        self._indexed_input = bool(indexed_input)
        self._row_map_rsrc = row_map_rsrc
        self._source_rows = int(source_rows)
        assert x_tensor is not None
        if self._indexed_input:
            assert row_map_rsrc is not None and self._source_rows > 0
        self._is_fp4 = a_dtype == "fp4"
        assert self._is_fp4 or a_dtype == "fp8"
        if const_expr(self._async_copy):
            assert not a16wmix_use_k16(), "gfx942 has no 128-bit global-to-LDS copy"
            assert total_threads % 64 == 0
            total_chunks = sort_block_m * (k_step_bytes // 16)
            assert total_chunks % 64 == 0
            assert total_chunks < total_threads or total_chunks % total_threads == 0
            assert row_bytes % 16 == 0 and k_step_bytes % 16 == 0
            self._dma_atom = fx.make_copy_atom(
                fx.rocdl.BufferCopyLDS128b(),
                128,
            )

    def for_tile(self, tile_row_base_i32):
        """Precompute LDS and tile-local global offsets for one M tile."""
        if const_expr(self._indexed_input):
            input_iter = fx.get_iter(self._x_tensor)
            input_bytes = self._source_rows * self._row_bytes
            input_view = fx.Tensor(
                fx.make_view(input_iter, fx.make_layout(input_bytes, 1))
            )
        else:
            tile_iter = fx.add_offset(
                fx.get_iter(self._x_tensor),
                fx.Int64(tile_row_base_i32) * fx.Int64(self._row_bytes),
            )
            input_bytes = self._sort_block_m * self._row_bytes
            input_view = fx.Tensor(
                fx.make_view(tile_iter, fx.make_layout(input_bytes, 1))
            )
        self._tile_rsrc = ptr_buf_tensor(
            fx.get_iter(input_view),
            fx.Int32,
            unit_elems=4,
            num_records_bytes=input_bytes,
        )
        if const_expr(self._async_copy):
            tile_buffer = fx.rocdl.make_buffer_tensor(
                input_view,
                max_size=False,
                num_records_bytes=input_bytes,
            )
            self._tile_dma = fx.logical_divide(
                tile_buffer,
                fx.make_layout(1, 1),
            )
        chunks_per_row = self._k_step_bytes // 16
        row_stride_i32 = self._k_step_bytes // 4
        total_chunks = self._sort_block_m * chunks_per_row
        self._chunks = []
        for c in range_constexpr(0, total_chunks, self._total_threads):
            lin = fx.Int32(c) + fx.Int32(self._tx)
            row = lin // fx.Int32(chunks_per_row)
            chunk = lin % fx.Int32(chunks_per_row)
            source_row = row
            if const_expr(self._indexed_input):
                # One 16-lane subgroup copies one 256-byte K slice of a row.
                # Load its source key once and broadcast it within the wave.
                lane = self._tx & fx.Int32(63)
                packed = _load_row_map_subgroup(
                    self._row_map_rsrc,
                    tile_row_base_i32 + row,
                    lane,
                    chunks_per_row,
                )
                source_row = packed & fx.Int32(0xFFFFFF)
            row_byte = source_row * fx.Int32(self._row_bytes)
            if const_expr(self._swizzle):
                col_i32 = chunk * fx.Int32(4)
                swizzle_mask = (
                    (row & fx.Int32(14)) << fx.Int32(1)
                    if const_expr(self._is_fp4)
                    else (row & fx.Int32(15)) << fx.Int32(2)
                )
                swz = row * fx.Int32(row_stride_i32) + (col_i32 ^ swizzle_mask)
                lds_byte = swz * fx.Int32(4)
            else:
                lds_byte = lin * fx.Int32(16)
            self._chunks.append((lds_byte, row_byte + chunk * fx.Int32(16)))

    def load_regs(self, k_step_byte_off):
        """Read this K-step's 16-B chunks gmem->reg (VMEM); only the K-step offset varies. Returns (lds_off, vec4)."""
        koff = fx.Int32(k_step_byte_off)
        regs = []
        for lds_byte, chunk_base in self._chunks:
            group = (chunk_base + koff) // fx.Int32(16)
            regs.append(
                (
                    lds_byte,
                    buf_copy_load(self._tile_rsrc, group, fx.Int32, unit_elems=4),
                )
            )
        return regs

    def store(self, lds_dst, regs, base_i32=0):
        """Scatter loaded chunks into LDS via ds_write (precomputed lds_byte incl. swizzle). base_i32 = ping/pong."""
        base_bytes = fx.Int32(base_i32) * fx.Int32(4)
        for lds_byte, v in regs:
            dst = fx.make_view(
                fx.add_offset(
                    fx.recast_iter(fx.Int32, lds_dst.ptr),
                    (base_bytes + lds_byte) // fx.Int32(4),
                ),
                fx.make_layout(4, 1),
            )
            fragment = fx.make_rmem_tensor(4, fx.Int32)
            fragment.store(Vec(v))
            fx.copy(fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32), fragment, dst)

    def prefetch_to_lds(self, k_step_byte_off, lds_dst, base_i32=0):
        """Issue swizzled direct global-to-LDS copies with a wave-uniform LDS base."""
        koff = fx.Int32(k_step_byte_off)
        base_bytes = fx.Int32(base_i32) * fx.Int32(4)
        lds_elem = fx.recast_iter(
            fx.Uint8 if const_expr(self._is_fp4) else fx.Float8E4M3FN,
            lds_dst.ptr,
        )
        chunks_per_row = self._k_step_bytes // 16
        total_chunks = self._sort_block_m * chunks_per_row
        copy_waves = min(total_chunks, self._total_threads) // 64
        for round_base in range_constexpr(
            0,
            total_chunks,
            self._total_threads,
        ):
            copy_active = (
                self._wave < fx.Int32(copy_waves)
                if const_expr(total_chunks < self._total_threads)
                else fx.Boolean(True)
            )

            # issue_copy is invoked below in this same iteration and never
            # escapes it, so the late-binding B023 warns about cannot happen.
            @flyc.jit
            def issue_copy(active):
                if active:
                    physical = fx.Int32(round_base) + fx.Int32(self._tx)  # noqa: B023
                    row = physical // fx.Int32(chunks_per_row)
                    physical_chunk = physical % fx.Int32(chunks_per_row)
                    if const_expr(self._swizzle):
                        logical_chunk = (
                            physical_chunk ^ ((row & fx.Int32(14)) >> fx.Int32(1))
                            if const_expr(self._is_fp4)
                            else physical_chunk ^ (row & fx.Int32(15))
                        )
                    else:
                        logical_chunk = physical_chunk
                    source_row = row
                    if const_expr(self._indexed_input):
                        source_index = round_base // self._total_threads  # noqa: B023
                        source_byte = self._chunks[source_index][1]
                        source_row = source_byte // fx.Int32(self._row_bytes)
                    src_byte = (
                        source_row * fx.Int32(self._row_bytes)
                        + koff
                        + logical_chunk * fx.Int32(16)
                    )
                    src = fx.slice(
                        self._tile_dma,
                        (None, src_byte),
                    )
                    wave_base = base_bytes + fx.Int32(
                        (round_base + self._wave * 64) * 16  # noqa: B023
                    )
                    dst = fx.make_view(
                        fx.add_offset(lds_elem, wave_base),
                        fx.make_layout(1, 1),
                    )
                    fx.copy(self._dma_atom, src, dst)

            issue_copy(copy_active)


class AS2RLoader:
    """Load one FP8/FP4 scaled-MFMA A operand from row-major LDS."""

    def __init__(self, *, k_step_bytes, swizzle=False, a_dtype="fp8"):
        self._k_step_bytes = k_step_bytes
        self._swizzle = swizzle
        self._lane = fx.thread_idx.x % 64
        self._is_fp4 = a_dtype == "fp4"
        assert self._is_fp4 or a_dtype == "fp8"
        self._k16 = a16wmix_use_k16()
        assert not (self._k16 and self._is_fp4), "gfx942 MegaMoE supports a8w4 only"

    def _load_16b(self, lds_src, i32_off):
        # add_offset+recast preserves the LDS address space.
        byte_off = i32_off * fx.Int32(4)
        ptr_off = fx.add_offset(lds_src.ptr, fx.make_int_tuple(byte_off))
        i8_iter = fx.recast_iter(fx.Uint8, ptr_off)
        v = fx.make_view(i8_iter, fx.make_layout(16, 1)).load()
        return Vec(v).bitcast(fx.Int32)

    def load_operand(self, lds_src, mi, ksub, base_i32=0):
        """Load one K=128 A operand; FP4 uses one 16B chunk and FP8 uses two."""
        row = fx.Int32(mi * 16) + fx.Int32(self._lane % 16)
        row_i32 = row * fx.Int32(self._k_step_bytes // 4) + fx.Int32(base_i32)
        klane4 = fx.Int32(self._lane // 16) * fx.Int32(4)

        def _c(col):  # per-row XOR bank swizzle (must match ATileLoader.store)
            if const_expr(self._swizzle):
                mask = (
                    (row & fx.Int32(14)) << fx.Int32(1)
                    if const_expr(self._is_fp4)
                    else (row & fx.Int32(15)) << fx.Int32(2)
                )
                return col ^ mask
            return col

        col_lo = klane4 + fx.Int32(ksub * (16 if self._is_fp4 else 32))
        hi_step = 16
        if const_expr(self._k16):
            # a16wmix K order: lane group g reads K [32g, 32g+32) of this 128-K step.
            col_lo = klane4 * fx.Int32(2) + fx.Int32(ksub * 32)
            hi_step = 4
        lo = self._load_16b(lds_src, row_i32 + _c(col_lo)).bitcast(fx.Int64)
        if const_expr(self._is_fp4):
            return Vec(lo).bitcast(fx.Int32)
        hi = self._load_16b(lds_src, row_i32 + _c(col_lo + fx.Int32(hi_step))).bitcast(
            fx.Int64
        )
        return Vec.from_elements([lo[0], lo[1], hi[0], hi[1]], fx.Int64).bitcast(
            fx.Int32
        )


class BWeightLoader:
    """Per-K-step fp4 gate&up weights VMEM->reg (i32x8, 128b widened). shuffle_weight_w4 N-major layout."""

    def __init__(self, *, w_rsrc, num_acc_n, model_dim, cache_modifier=0):
        self._w_rsrc = w_rsrc
        self._num_acc_n = num_acc_n
        self._cache_modifier = int(cache_modifier)
        self._lane = fx.thread_idx.x % 64
        self._stride_nlane = 16
        self._stride_klane = 256
        self._stride_k0 = 1024
        self._stride_n0 = model_dim * 8

    def _load_pack(self, row_base_i32, ni, kstep_i32, ksub):
        lane_row = fx.Int32(self._lane % 16)
        lane_k = fx.Int32(self._lane // 16)
        n_blk = (row_base_i32 // fx.Int32(16)) + fx.Int32(ni)
        k0 = kstep_i32 * fx.Int32(2) + fx.Int32(ksub)
        byte = (
            n_blk * fx.Int32(self._stride_n0)
            + k0 * fx.Int32(self._stride_k0)
            + lane_k * fx.Int32(self._stride_klane)
            + lane_row * fx.Int32(self._stride_nlane)
        )
        return buf_copy_load(
            self._w_rsrc,
            byte // fx.Int32(16),
            fx.Int32,
            unit_elems=4,
            cache_modifier=self._cache_modifier,
        )

    def load_step(self, row_base_i32, kstep_i32):
        """list[num_acc_n] of [ksub0_i32x4, ksub1_i32x4] for this K-step."""
        return [
            self.load_ni(row_base_i32, ni, kstep_i32)
            for ni in range_constexpr(self._num_acc_n)
        ]

    def load_ni(self, row_base_i32, ni, kstep_i32):
        """One N-group's [ksub0, ksub1] fp4 packs (per-ni so the pipe can interleave one next-tile B load)."""
        return [
            self._load_pack(row_base_i32, ni, kstep_i32, ks)
            for ks in range_constexpr(_PACK)
        ]


class BScaleLoader:
    """Preshuffled per-1x32 E8M0 B scales; one packed i32 (4 e8m0) per K-step per 32-row pack-group."""

    def __init__(self, *, scale_rsrc, num_acc_n, model_dim):
        assert num_acc_n % _PACK == 0
        self._rsrc = scale_rsrc
        self._n_groups = num_acc_n // _PACK
        self._row_stride = (model_dim // 256) * 64
        lane = fx.thread_idx.x % 64
        self._lane_off = lane

    def load_step(self, base_row_i32, kstep_i32):
        base_group = base_row_i32 // fx.Int32(32)
        kterm = kstep_i32 * fx.Int32(64)
        lane = fx.Int32(self._lane_off)
        out = []
        for g in range_constexpr(self._n_groups):
            off = (base_group + fx.Int32(g)) * fx.Int32(self._row_stride) + kterm + lane
            out.append(self._rsrc[off])
        return out


class AScaleLoader:
    """Stage per-1x32 E8M0 A scales to LDS once per tile."""

    def __init__(
        self,
        *,
        scale_rsrc,
        m_repeat,
        model_dim,
        sort_block_m,
        total_threads,
        indexed_input=False,
        row_map_rsrc=None,
    ):
        self._rsrc = scale_rsrc
        self._n_scale = model_dim // 32
        self._lane = fx.thread_idx.x % 64
        self._n_groups = m_repeat // _PACK
        self._sort_block_m = sort_block_m
        self._total_threads = total_threads
        self._tx = fx.thread_idx.x
        self._indexed_input = bool(indexed_input)
        self._row_map_rsrc = row_map_rsrc
        if self._indexed_input:
            assert row_map_rsrc is not None

    def stage(self, lds_ascale, tile_row_base_i32):
        """Copy this tile's E8M0 scale block from global memory to LDS."""
        total = self._sort_block_m * self._n_scale
        assert total % 16 == 0, "A-scale tile must contain whole 16-byte copy chunks"
        base = tile_row_base_i32 * fx.Int32(self._n_scale)
        n16 = total // 16

        @flyc.jit
        def copy_group(lin: fx.Int32, source_group: fx.Int32):
            v = buf_copy_load(
                self._rsrc,
                source_group,
                fx.Int32,
                unit_elems=4,
            )
            dst = fx.make_view(
                fx.add_offset(
                    fx.recast_iter(fx.Int32, lds_ascale.ptr), lin * fx.Int32(4)
                ),
                fx.make_layout(4, 1),
            )
            fragment = fx.make_rmem_tensor(4, fx.Int32)
            fragment.store(v)
            fx.copy(fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32), fragment, dst)

        if const_expr(self._indexed_input and self._sort_block_m >= 128):
            groups_per_row = self._n_scale // 16
            # Preserve the tuned 16-slot geometry for current models while
            # covering every 16-byte scale group for model_dim > 8192.
            padded_groups_per_row = max(16, groups_per_row)
            indexed_slots = self._sort_block_m * padded_groups_per_row

            @flyc.jit
            def copy_indexed_slot(slot: fx.Int32):
                if slot < fx.Int32(indexed_slots):
                    row = slot // fx.Int32(padded_groups_per_row)
                    group = slot % fx.Int32(padded_groups_per_row)
                    if group < fx.Int32(groups_per_row):
                        lane = self._tx & fx.Int32(63)
                        packed = _load_row_map_subgroup(
                            self._row_map_rsrc,
                            tile_row_base_i32 + row,
                            lane,
                            padded_groups_per_row,
                        )
                        source_row = packed & fx.Int32(0xFFFFFF)
                        lin = row * fx.Int32(groups_per_row) + group
                        copy_group(
                            lin,
                            source_row * fx.Int32(groups_per_row) + group,
                        )

            for c in range_constexpr(0, indexed_slots, self._total_threads):
                copy_indexed_slot(fx.Int32(c) + fx.Int32(self._tx))
        else:

            @flyc.jit
            def copy_chunk(lin: fx.Int32):
                if lin < fx.Int32(n16):
                    source_group = (base + lin * fx.Int32(16)) // fx.Int32(16)
                    copy_group(lin, source_group)

            for c in range_constexpr(0, n16, self._total_threads):
                lin = fx.Int32(c) + fx.Int32(self._tx)
                # M32 has 448 chunks, which is not divisible by a
                # 256/512-thread CTA.
                copy_chunk(lin)

    def load_step(self, lds_ascale, kstep_i32):
        """Read one packed i32 per scale group from LDS."""
        lane_row = self._lane % 16
        col0 = kstep_i32 * 8 + self._lane // 16
        out = []
        for g in range_constexpr(self._n_groups):
            r0 = fx.Int32(g * 32) + lane_row
            r1 = r0 + fx.Int32(16)
            b = []
            for ksub in range_constexpr(_PACK):
                for rr in (r0, r1):
                    b.append(self._read_scale_lds(lds_ascale, rr, col0 + ksub * 4))
            packed = (
                b[0]
                | (b[1] << fx.Int32(8))
                | (b[2] << fx.Int32(16))
                | (b[3] << fx.Int32(24))
            )
            out.append(packed)
        return out

    def _read_scale_lds(self, lds_ascale, row_i32, col_i32):
        off = row_i32 * fx.Int32(self._n_scale) + col_i32
        ptr = fx.recast_iter(
            fx.Uint8, fx.add_offset(lds_ascale.ptr, fx.make_int_tuple(off))
        )
        v = fx.make_view(ptr, fx.make_layout(1, 1)).load()
        return Vec(v, dtype=fx.Uint8)[0].to(fx.Int32)


class MfmaScaleGU:
    """Gate/up scaled-MFMA atoms for FP8/FP4 A x FP4 B."""

    def __init__(self, *, m_repeat, num_acc_n, a_dtype="fp8", a_per_token=False):
        self._m_repeat = m_repeat
        self._num_acc_n = num_acc_n
        self._is_fp4 = a_dtype == "fp4"
        assert self._is_fp4 or a_dtype == "fp8"
        self._k16 = a16wmix_use_k16()
        self._a_per_token = bool(a_per_token)
        self.zero_value = Vec.filled(4, 0.0, fx.Float32)
        if const_expr(self._a_per_token):
            assert self._k16 and not self._is_fp4, "per-token FP8 A is the gfx942 a8w4 path"
            self._fp8_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fp8_elem_type()))
            self._sref = None
            self._deq = {}
            return
        if const_expr(self._k16):
            assert not self._is_fp4, "gfx942 MegaMoE supports a8w4 only"
            self._k16_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
            self._deq = {}
            return
        elem_a = fx.Float4E2M1FN if self._is_fp4 else fx.Float8E4M3FN
        self._atoms = {
            (osa, osb): fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(
                    16, 16, 128, elem_a, fx.Float4E2M1FN, opsel_a=osa, opsel_b=osb
                )
            )
            for osa in range(4)
            for osb in range(4)
        }

    def idx(self, mi, ni):
        return mi * self._num_acc_n + ni

    def _dequant(self, kind, op, word, byte):
        # Trace-time memo: one dequant per operand object, reused by every MFMA
        # that consumes it.  Holding ``op`` keeps its id() from being recycled.
        key = (kind, id(op), id(word), byte)
        hit = self._deq.get(key)
        if hit is None:
            fn = dequant_a_fp8_k16 if kind == "a" else dequant_b_fp4_k16
            hit = (op, word, fn(op, word, byte))
            self._deq[key] = hit
        return hit[2]

    def _sa(self, sa, mi):
        return None if const_expr(self._a_per_token) else sa[mi // _PACK]

    def set_tile_sref(self, sref):
        """Per-lane s_ref byte of each 16-column block of the current tile."""
        self._sref = sref

    def _decode_b_pt(self, b_op, sb_v, sb_byte, ni):
        key = ("b", id(b_op), id(sb_v), sb_byte)
        hit = self._deq.get(key)
        if hit is None:
            hit = (b_op, sb_v, decode_b_fp4_fp8(b_op, sb_v, sb_byte, self._sref[ni]))
            self._deq[key] = hit
        return hit[2]

    def _mfma(self, a_op, b_op, acc, sa_v, sb_v, ksub, ia, jb, ni=None):
        opsel_a = ksub * _PACK + ia
        opsel_b = ksub * _PACK + jb
        if const_expr(self._a_per_token):
            c_frag = fx.make_rmem_tensor(4, fx.Float32)
            c_frag.store(Vec(acc))
            fp8_mma_k32(self._fp8_atom, c_frag, a_op, self._decode_b_pt(b_op, sb_v, opsel_b, ni))
            return Vec(c_frag.load())
        if const_expr(self._k16):
            c_frag = fx.make_rmem_tensor(4, fx.Float32)
            c_frag.store(Vec(acc))
            k16_mma(
                self._k16_atom,
                c_frag,
                self._dequant("a", a_op, sa_v, opsel_a),
                self._dequant("b", b_op, sb_v, opsel_b),
            )
            return Vec(c_frag.load())
        a_frag = fx.make_rmem_tensor(4 if self._is_fp4 else 8, fx.Int32)
        b_frag = fx.make_rmem_tensor(4, fx.Int32)
        c_frag = fx.make_rmem_tensor(4, fx.Float32)
        a_frag.store(Vec(a_op))
        b_frag.store(Vec(b_op))
        c_frag.store(Vec(acc))
        fx.gemm(
            self._atoms[(opsel_a, opsel_b)],
            c_frag,
            a_frag,
            b_frag,
            c_frag,
            scale_a=sa_v,
            scale_b=sb_v,
        )
        return Vec(c_frag.load())

    def call(self, a_load, b, acc, sa, sb):
        """Non-pipelined lazy-A K-step: a_load(mi,ksub) on demand (one A operand live -> low VGPR, wins small bs)."""
        self._deq = {}
        for ksub in range_constexpr(_PACK):
            for mi in range_constexpr(self._m_repeat):
                a_op = a_load(mi, ksub)
                sa_v = self._sa(sa, mi)
                ia = mi % _PACK
                for ni in range_constexpr(self._num_acc_n):
                    aidx = self.idx(mi, ni)
                    acc[aidx] = self._mfma(
                        a_op,
                        b[ni][ksub],
                        acc[aidx],
                        sa_v,
                        sb[ni // _PACK],
                        ksub,
                        ia,
                        ni % _PACK,
                        ni=ni,
                    )
        return acc

    def call_pipe(self, a_load, b_prev, acc, sa, sb, load_next):
        """B-major pipe: MFMA b_prev while load_next(ni) issues next-tile B[ni] before its group."""
        self._deq = {}
        a_ops = []
        for mi in range_constexpr(self._m_repeat):
            a_ops.append([a_load(mi, ks) for ks in range_constexpr(_PACK)])
        b_next = []
        for ni in range_constexpr(self._num_acc_n):
            b_next.append(load_next(ni))
            jb = ni % _PACK
            sb_v = sb[ni // _PACK]
            for ksub in range_constexpr(_PACK):
                for mi in range_constexpr(self._m_repeat):
                    aidx = self.idx(mi, ni)
                    acc[aidx] = self._mfma(
                        a_ops[mi][ksub],
                        b_prev[ni][ksub],
                        acc[aidx],
                        self._sa(sa, mi),
                        sb_v,
                        ksub,
                        mi % _PACK,
                        jb,
                        ni=ni,
                    )
        return acc, b_next

    def call_pipe_am(self, a_load, b_prev, acc, sa, sb, load_next):
        """A-major pipe: lazy-A per (mi,ksub) overlaps next A ds_read with MFMA; next-B loads spread over groups."""
        self._deq = {}
        nn = self._num_acc_n
        b_next = [None] * nn
        ngrp = self._m_repeat * _PACK
        g = 0
        nb = 0
        for mi in range_constexpr(self._m_repeat):
            ia = mi % _PACK
            sa_v = self._sa(sa, mi)
            for ksub in range_constexpr(_PACK):
                a_op = a_load(mi, ksub)
                tgt = ((g + 1) * nn) // ngrp
                while nb < tgt:
                    b_next[nb] = load_next(nb)
                    nb += 1
                for ni in range_constexpr(nn):
                    aidx = self.idx(mi, ni)
                    acc[aidx] = self._mfma(
                        a_op,
                        b_prev[ni][ksub],
                        acc[aidx],
                        sa_v,
                        sb[ni // _PACK],
                        ksub,
                        ia,
                        ni % _PACK,
                        ni=ni,
                    )
                g += 1
        while nb < nn:
            b_next[nb] = load_next(nb)
            nb += 1
        return acc, b_next

    def call_pipe_am_final(self, a_load, b_prev, acc, sa, sb):
        """Consume the final prefetched B step without redundant loads."""
        self._deq = {}
        for mi in range_constexpr(self._m_repeat):
            ia = mi % _PACK
            sa_v = self._sa(sa, mi)
            for ksub in range_constexpr(_PACK):
                a_op = a_load(mi, ksub)
                for ni in range_constexpr(self._num_acc_n):
                    aidx = self.idx(mi, ni)
                    acc[aidx] = self._mfma(
                        a_op,
                        b_prev[ni][ksub],
                        acc[aidx],
                        sa_v,
                        sb[ni // _PACK],
                        ksub,
                        ia,
                        ni % _PACK,
                        ni=ni,
                    )
        return acc


def _group_absmax(vals, shuffle_offs, c64):
    """|max| over ``vals`` and over the lanes ``lane ^ off`` for each offset."""
    m = vals[0].maximumf(fx.Float32(0.0) - vals[0])
    for v in vals[1:]:
        m = m.maximumf(v.maximumf(fx.Float32(0.0) - v))
    for off in shuffle_offs:
        m = m.maximumf(m.shuffle_xor(fx.Int32(off), c64))
    return m


def _e8m0_scale(amax):
    """E8M0 exponent of a 1x32 group's |max| and the matching FP8 quant multiplier."""
    max_rounded = (amax.bitcast(fx.Int32) + fx.Int32(0x400000)) & fx.Int32(0xFF800000)
    e = (max_rounded >> fx.Int32(23)) - fx.Int32(fp8_e8m0_bias())
    e8m0 = (e > fx.Int32(0)).select(e, fx.Int32(0))
    quant_scale = ((fx.Int32(254) - e8m0) << fx.Int32(23)).bitcast(fx.Float32)
    return e8m0, quant_scale


class SiluQuantEpilogue:
    """silu(gate)*up -> fp8/fp4 + per-32 E8M0 out-scale."""

    # fmt: off
    def __init__(self, *, out_rsrc, out_scale_rsrc, sorted_rsrc, tokens, inter_dim, m_repeat, num_acc_n,
        sort_block_m, tile_n, num_waves, lds_out, evec, swiglu_limit=0.0, always_valid=False,
        out_tensor=None, out_dtype="fp8", act="silu", situ_beta=1.0, situ_linear_beta=1.0,
        shared_exp128=False):
    # fmt: on
        self._out_rsrc = out_rsrc
        self._out_scale_rsrc = out_scale_rsrc
        self._sorted_rsrc = sorted_rsrc
        self._tokens = tokens
        self._inter_dim = inter_dim
        self._m_repeat = m_repeat
        self._num_acc_n = num_acc_n
        self._sort_block_m = sort_block_m
        self._tile_n = tile_n
        self._num_waves = num_waves
        self._lds_out = lds_out
        self._swiglu_limit = float(swiglu_limit)
        self._act = act
        self._situ_beta = float(situ_beta)
        self._situ_linear_beta = float(situ_linear_beta)
        assert act in ("silu", "situv2"), f"unsupported mega_moe act {act!r}"
        if act == "situv2":
            assert self._situ_beta > 0.0 and self._situ_linear_beta > 0.0, \
                "situv2 requires positive beta / linear_beta"
        self._always_valid = always_valid
        self._out_tensor = out_tensor
        self._is_fp4 = out_dtype == "fp4"
        assert self._is_fp4 or out_dtype == "fp8"
        self._out_row_bytes = inter_dim // 2 if self._is_fp4 else inter_dim
        # Columns quantized per lane in the store pass. 2: one 16-bit store and a
        # 16-lane max per pair; 8: one 8-byte store and a 4-lane max per 8 values.
        # Without out_tensor, out_rsrc is the whole output in that store's width:
        # Int16 elements for 2, Int32 x2 units for 8.
        assert evec in (2, 8)
        # FP4 output is only implemented in the evec==2 store branch; the evec==8
        # vectorized store (_store_quant_vec8) is FP8-only.  Force evec=2 for FP4
        # so the default AITER_MEGA_S1_EPI_EVEC=8 never routes FP4 to the FP8
        # store path (which would write FP8 words into the Int8 FP4 buffer).
        self._evec = 2 if self._is_fp4 else evec
        self._lane = fx.thread_idx.x % 64
        self._sorted_scale_cols_i32 = (inter_dim // 32 + 7) // 8 * 8
        # GEMM2 FP8 needs one exponent per 128 columns (16 lanes of 8 in the vec8 store).
        # A compile parameter, not an env read here: FlyDSL keys its kernel cache on
        # source plus captured scalars, so a trace-time env read would hit stale code.
        self._shared_exp128 = bool(shared_exp128) and not self._is_fp4
        assert not (self._shared_exp128 and self._evec != 8), "shared 128-column exponent needs evec=8"

    def _silu(self, g):
        emu = (g * fx.Float32(-1.4426950408889634)).exp2()
        return g * (fx.Float32(1.0) / (fx.Float32(1.0) + emu))

    def _combine(self, acc):
        gui_n = self._num_acc_n // _PACK
        out = []
        for mi in range_constexpr(self._m_repeat):
            for ni in range_constexpr(gui_n):
                g_idx = mi * self._num_acc_n + ni * _PACK
                out.append(self._act_mul(acc[g_idx], acc[g_idx + 1]))
        return out

    def _act_mul(self, gate_v4, up_v4):
        gv = Vec(gate_v4)
        uv = Vec(up_v4)
        if const_expr(self._act == "situv2"):
            # SiTUv2 (Kimi-K3 hidden_act="situ"):
            #   situ(g)    = beta * tanh(g / beta) * sigmoid(g)
            #   situ_up(u) = linear_beta * tanh(u / linear_beta)
            # No hard clamp -- tanh self-saturates, matching SGLang SituAndMul and
            # the split-K silu_and_mul_fq situv2 path.  beta/linear_beta are baked
            # per network (one K3 value), so they stay compile-time here.
            b = fx.Float32(self._situ_beta)
            br = fx.Float32(1.0 / self._situ_beta)
            lb = fx.Float32(self._situ_linear_beta)
            lbr = fx.Float32(1.0 / self._situ_linear_beta)
            elems = [
                situ_mul(gv[i], uv[i], b, br, lb, lbr, tanh=tanh_via_sigmoid_f32)
                for i in range_constexpr(4)
            ]
            return Vec.from_elements(elems, fx.Float32)
        if self._swiglu_limit <= 0:
            elems = [self._silu(gv[i]) * uv[i] for i in range_constexpr(4)]
            return Vec.from_elements(elems, fx.Float32)
        limit = fx.Float32(self._swiglu_limit)
        elems = [
            self._silu(-(-gv[i]).maximumf(-limit)) * fx.clampf(uv[i], -limit, limit)
            for i in range_constexpr(4)
        ]
        return Vec.from_elements(elems, fx.Float32)

    def _tile_out(self, tile_i32, elem, unit_elems=1):
        """Buffer view of this tile's rows of ``out_tensor``."""
        tile_iter = fx.add_offset(
            fx.get_iter(self._out_tensor),
            fx.Int64(tile_i32) * fx.Int64(self._sort_block_m * self._inter_dim),
        )
        tile_view = fx.Tensor(fx.make_view(tile_iter, fx.make_layout(1, 1)))
        return ptr_buf_tensor(
            fx.get_iter(tile_view),
            elem,
            unit_elems=unit_elems,
            num_records_bytes=self._sort_block_m * self._inter_dim,
        )

    def _row_target(self, tile_i32, tile_row_base_i32, row):
        """Global row, validity and output byte base of tile row ``row``."""
        slot = tile_row_base_i32 + row
        row_g = tile_i32 * fx.Int32(self._sort_block_m) + row
        if const_expr(self._always_valid):
            valid = fx.Boolean(True)
            out_row_base = (
                row * fx.Int32(self._inter_dim)
                if self._out_tensor is not None
                else row_g * fx.Int32(self._inter_dim)
            )
        else:
            tok = self._sorted_rsrc[slot]
            valid = tok < fx.Int32(self._tokens)
            out_row_base = slot * fx.Int32(self._inter_dim)
        return row_g, valid, out_row_base

    def store(self, acc, tile_i32, tile_row_base_i32, n_tile_base_i32):
        combined = self._combine(acc)
        n_per = len(combined) // self._m_repeat
        if self._out_tensor is not None:
            tile_iter = fx.add_offset(
                fx.get_iter(self._out_tensor),
                fx.Int64(tile_i32) * fx.Int64(self._sort_block_m * self._out_row_bytes),
            )
            tile_view = fx.Tensor(fx.make_view(tile_iter, fx.make_layout(1, 1)))
            out_elem = fx.Int8 if const_expr(self._is_fp4) else fx.Int16
            out_rsrc = ptr_buf_tensor(
                fx.get_iter(tile_view),
                out_elem,
                num_records_bytes=self._sort_block_m * self._out_row_bytes,
            )
        else:
            out_rsrc = self._out_rsrc

        tx = fx.thread_idx.x
        wave = tx // fx.Int32(64)
        lane = tx % fx.Int32(64)
        l16 = lane % fx.Int32(16)
        ld4 = (lane // fx.Int32(16)) * fx.Int32(4)
        cs_tile_n = self._tile_n // 2
        nwo = cs_tile_n // self._num_waves
        cbase = wave * fx.Int32(nwo)
        cptr = self._lds_out.ptr

        for mi in range_constexpr(self._m_repeat):
            for nj in range_constexpr(n_per):
                v4 = Vec(combined[mi * n_per + nj])
                col = cbase + fx.Int32(nj * 16) + l16
                for ii in range_constexpr(4):
                    row = fx.Int32(mi * 16) + ld4 + fx.Int32(ii)
                    idx = row * fx.Int32(cs_tile_n) + col
                    ptr = fx.add_offset(cptr, fx.make_int_tuple(idx))
                    fx.ptr_store(Vec.from_elements([v4[ii]], fx.Float32), ptr)
        gpu.barrier()

        if const_expr(self._evec == 8):
            self._store_quant_vec8(tile_i32, tile_row_base_i32, n_tile_base_i32, cbase, cptr, cs_tile_n)
            return

        c64 = fx.Int32(64)
        n32 = fx.Int32(self._sorted_scale_cols_i32 * 32)
        NLANE = 32
        EVEC = 2
        rows_per_iter = self._num_waves * 64 // NLANE
        mlane = tx // fx.Int32(NLANE)
        nlane = tx % fx.Int32(NLANE)
        m_reps = self._sort_block_m // rows_per_iter
        n_reps = cs_tile_n // (NLANE * EVEC)
        out_tile_base = n_tile_base_i32 // fx.Int32(2) - cbase

        for mr in range_constexpr(m_reps):
            row = fx.Int32(mr * rows_per_iter) + mlane
            slot = tile_row_base_i32 + row
            row_g = tile_i32 * fx.Int32(self._sort_block_m) + row
            if const_expr(self._always_valid):
                valid = fx.Boolean(True)
                out_row_base = (
                    row * fx.Int32(self._out_row_bytes)
                    if self._out_tensor is not None
                    else row_g * fx.Int32(self._out_row_bytes)
                )
            else:
                tok = self._sorted_rsrc[slot]
                valid = tok < fx.Int32(self._tokens)
                out_row_base = slot * fx.Int32(self._out_row_bytes)
            for nr in range_constexpr(n_reps):
                col0 = fx.Int32(nr * NLANE * EVEC) + nlane * fx.Int32(EVEC)
                idx = row * fx.Int32(cs_tile_n) + col0
                f32_iter = fx.recast_iter(fx.Float32, fx.add_offset(cptr, fx.make_int_tuple(idx)))
                frag = fx.make_view(f32_iter, fx.make_layout(EVEC, 1)).load()
                v0 = frag[0]
                v1 = frag[1]
                a0 = v0.maximumf(fx.Float32(0.0) - v0)
                a1 = v1.maximumf(fx.Float32(0.0) - v1)
                m = a0.maximumf(a1)
                for off in (1, 2, 4, 8):
                    m = m.maximumf(m.shuffle_xor(fx.Int32(off), c64))
                max_rounded = (m.bitcast(fx.Int32) + fx.Int32(0x400000)) & fx.Int32(0xFF800000)
                _e = (max_rounded >> fx.Int32(23)) - fx.Int32(2 if self._is_fp4 else fp8_e8m0_bias())
                e8m0_v = (_e > fx.Int32(0)).select(_e, fx.Int32(0))
                quant_scale = ((fx.Int32(254) - e8m0_v) << fx.Int32(23)).bitcast(fx.Float32)
                gcol = out_tile_base + col0

                if const_expr(self._is_fp4):
                    dequant_scale = (e8m0_v << fx.Int32(23)).bitcast(fx.Float32)
                    packed = rocdl.cvt_scalef32_pk_fp4_f32(
                        T.i32,
                        fx.Int32(0),
                        v0,
                        v1,
                        dequant_scale,
                        0,
                    )
                    payload = fx.Int32(packed).to(fx.Int8)
                    out_byte = out_row_base + gcol // fx.Int32(2)
                else:
                    scaled0 = v0 * quant_scale
                    scaled1 = v1 * quant_scale
                    packed = rocdl.cvt_pk_fp8_f32(T.i32, scaled0, scaled1, fx.Int32(0), 0)
                    payload = fx.Int32(packed).to(fx.Int16)
                    out_byte = out_row_base + gcol
                out_byte = valid.select(out_byte, fx.Int32(0x40000000))
                if const_expr(self._is_fp4):
                    out_rsrc[out_byte] = payload
                else:
                    out_rsrc[out_byte // fx.Int32(2)] = payload

                col_s = gcol >> fx.Int32(5)
                is_writer = (gcol & fx.Int32(31)) == fx.Int32(0)
                d0 = row_g >> fx.Int32(5)
                d1 = (row_g >> fx.Int32(4)) & fx.Int32(1)
                d2 = row_g & fx.Int32(15)
                d3 = col_s >> fx.Int32(3)
                d4 = (col_s >> fx.Int32(2)) & fx.Int32(1)
                d5 = col_s & fx.Int32(3)
                byte_off = d0 * n32 + d3 * fx.Int32(256) + d5 * fx.Int32(64) + d2 * fx.Int32(4) + d4 * fx.Int32(2) + d1
                byte_off = is_writer.select(byte_off, fx.Int32(0x40000000))
                e8m0_i8 = e8m0_v.to(fx.Int8)
                self._out_scale_rsrc[byte_off] = e8m0_i8
        wait_lds_barrier()

    def _store_quant_vec8(self, tile_i32, tile_row_base_i32, n_tile_base_i32, cbase, cptr, cs_tile_n):
        """Store pass with 8 columns per lane: a 1x32 group spans 4 lanes (two
        xor shuffles for its max) and each lane writes its 8 FP8 values at once."""
        EVEC = 8
        NLANE = cs_tile_n // EVEC
        assert 64 % NLANE == 0 and NLANE * EVEC == cs_tile_n
        assert not self._shared_exp128 or NLANE % 16 == 0
        tx = fx.thread_idx.x
        c64 = fx.Int32(64)
        n32 = fx.Int32(self._sorted_scale_cols_i32 * 32)
        rows_per_iter = self._num_waves * 64 // NLANE
        mlane = tx // fx.Int32(NLANE)
        nlane = tx % fx.Int32(NLANE)
        m_reps = self._sort_block_m // rows_per_iter
        assert m_reps * rows_per_iter == self._sort_block_m
        out_tile_base = n_tile_base_i32 // fx.Int32(2) - cbase
        if self._out_tensor is not None:
            out_vec = self._tile_out(tile_i32, fx.Int32, unit_elems=2)
        else:
            out_vec = self._out_rsrc
        col0 = nlane * fx.Int32(EVEC)
        gcol = out_tile_base + col0
        is_writer = (nlane & fx.Int32(3)) == fx.Int32(0)
        col_s = gcol >> fx.Int32(5)
        d3 = col_s >> fx.Int32(3)
        d4 = (col_s >> fx.Int32(2)) & fx.Int32(1)
        d5 = col_s & fx.Int32(3)
        for mr in range_constexpr(m_reps):
            row = fx.Int32(mr * rows_per_iter) + mlane
            row_g, valid, out_row_base = self._row_target(tile_i32, tile_row_base_i32, row)
            idx = row * fx.Int32(cs_tile_n) + col0
            f32_iter = fx.recast_iter(fx.Float32, fx.add_offset(cptr, fx.make_int_tuple(idx)))
            frag = fx.make_view(f32_iter, fx.make_layout(EVEC, 1)).load()
            vals = [frag[i] for i in range_constexpr(EVEC)]
            m = _group_absmax(vals, (1, 2, 4, 8) if self._shared_exp128 else (1, 2), c64)
            e8m0_v, quant_scale = _e8m0_scale(m)
            words = []
            for w in range_constexpr(2):
                sc = [vals[w * 4 + i] * quant_scale for i in range_constexpr(4)]
                packed = rocdl.cvt_pk_fp8_f32(T.i32, sc[0], sc[1], fx.Int32(0), 0)
                packed = rocdl.cvt_pk_fp8_f32(T.i32, sc[2], sc[3], packed, 1)
                words.append(fx.Int32(packed))
            out_byte = out_row_base + gcol
            out_byte = valid.select(out_byte, fx.Int32(0x40000000))
            buf_copy_store(
                out_vec,
                out_byte // fx.Int32(8),
                Vec.from_elements(words, fx.Int32),
                fx.Int32,
                unit_elems=2,
            )
            d0 = row_g >> fx.Int32(5)
            d1 = (row_g >> fx.Int32(4)) & fx.Int32(1)
            d2 = row_g & fx.Int32(15)
            byte_off = d0 * n32 + d3 * fx.Int32(256) + d5 * fx.Int32(64) + d2 * fx.Int32(4) + d4 * fx.Int32(2) + d1
            byte_off = is_writer.select(byte_off, fx.Int32(0x40000000))
            self._out_scale_rsrc[byte_off] = e8m0_v.to(fx.Int8)
        wait_lds_barrier()
