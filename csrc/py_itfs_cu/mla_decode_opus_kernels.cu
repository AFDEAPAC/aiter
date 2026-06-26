// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// OPUS-based MLA absorbed decode (gfx950): host launcher + dtype dispatch.

#define MLA_DECODE_OPUS_IMPL
#include "mla_decode_opus.h"

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"

void mla_decode_opus_fwd(aiter_tensor_t& q,
                         aiter_tensor_t& unified_kv,
                         aiter_tensor_t& kv_indices,
                         aiter_tensor_t& kv_indptr,
                         aiter_tensor_t& attn_sink,
                         aiter_tensor_t& out,
                         float softmax_scale,
                         int qlen)
{
    AITER_CHECK(q.dim() == 4, "q must be 4-D [B, QLEN, H, D], got ndim=", q.dim());
    AITER_CHECK(unified_kv.dim() == 2, "unified_kv must be 2-D [total_pages, D], got ndim=", unified_kv.dim());
    AITER_CHECK(out.dim() == 4, "out must be 4-D [B, QLEN, H, D], got ndim=", out.dim());
    AITER_CHECK(kv_indptr.dim() == 1, "kv_indptr must be 1-D [B+1]");
    AITER_CHECK(kv_indices.dim() == 1, "kv_indices must be 1-D [nnz]");

    AITER_CHECK(q.dtype() == unified_kv.dtype() && q.dtype() == out.dtype(),
                "q/unified_kv/out must share dtype");
    AITER_CHECK(q.dtype() == AITER_DTYPE_bf16 || q.dtype() == AITER_DTYPE_fp16,
                "Only bf16/fp16 are supported for q/unified_kv/out");

    const int B = static_cast<int>(q.size(0));
    AITER_CHECK(static_cast<int>(q.size(1)) == qlen, "q.size(1) must equal qlen argument");
    const int H = static_cast<int>(q.size(2));
    const int D = static_cast<int>(q.size(3));

    AITER_CHECK(D == 512, "Only D=512 is compiled for mla_decode_opus_fwd, got D=", D);
    AITER_CHECK(unified_kv.size(1) == D, "unified_kv last dim must equal D");
    AITER_CHECK(out.size(0) == B && out.size(1) == qlen && out.size(2) == H && out.size(3) == D,
                "out shape must match q [B, QLEN, H, D]");

    AITER_CHECK(kv_indptr.dtype() == AITER_DTYPE_i32, "kv_indptr must be int32");
    AITER_CHECK(kv_indices.dtype() == AITER_DTYPE_i32, "kv_indices must be int32");
    AITER_CHECK(kv_indptr.size(0) == B + 1, "kv_indptr length must be B+1");

    AITER_CHECK(q.stride(3) == 1 && unified_kv.stride(1) == 1 && out.stride(3) == 1,
                "Q/UnifiedKV/out must be contiguous along D");
    AITER_CHECK(kv_indices.is_contiguous() && kv_indptr.is_contiguous(),
                "kv_indices and kv_indptr must be contiguous");

    const int use_sink = (attn_sink.numel() > 0) ? 1 : 0;
    if(use_sink) {
        AITER_CHECK(attn_sink.dim() == 1, "attn_sink must be 1-D [H]");
        AITER_CHECK(attn_sink.dtype() == AITER_DTYPE_fp32, "attn_sink must be fp32");
        AITER_CHECK(attn_sink.size(0) == H, "attn_sink length must equal H");
        AITER_CHECK(attn_sink.is_contiguous(), "attn_sink must be contiguous");
    }

    AITER_CHECK(qlen >= 1 && qlen <= 17, "qlen must be in [1,17], got qlen=", qlen);

    const int total_pages = static_cast<int>(unified_kv.size(0));

    if(B == 0)
        return;

    mla_decode_opus_kargs kargs{};
    kargs.q_ptr             = q.data_ptr();
    kargs.unified_kv_ptr    = unified_kv.data_ptr();
    kargs.attn_sink_ptr     = use_sink ? attn_sink.data_ptr() : nullptr;
    kargs.out_ptr           = out.data_ptr();
    kargs.kv_indptr         = reinterpret_cast<const int*>(kv_indptr.data_ptr());
    kargs.kv_indices        = reinterpret_cast<const int*>(kv_indices.data_ptr());
    kargs.B                 = B;
    kargs.QLEN              = qlen;
    kargs.H                 = H;
    kargs.D                 = D;
    kargs.total_pages       = total_pages;
    kargs.stride_q_b        = static_cast<int>(q.stride(0));
    kargs.stride_q_qlen     = static_cast<int>(q.stride(1));
    kargs.stride_qo_h       = static_cast<int>(q.stride(2));
    kargs.stride_o_h        = static_cast<int>(out.stride(2));
    kargs.stride_out_b      = static_cast<int>(out.stride(0));
    kargs.stride_out_qlen   = static_cast<int>(out.stride(1));
    kargs.stride_kv_page    = static_cast<int>(unified_kv.stride(0));
    kargs.softmax_scale     = softmax_scale;
    kargs.use_sink          = use_sink;

    HipDeviceGuard guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

#define LAUNCH_MLA_DECODE_16MX8(QL)                                                                                  \
    do {                                                                                                             \
        auto launch = [&](auto dtype_tag) {                                                                          \
            using Traits = mla_decode_16mx8_traits<QL, 16, 32, 512, 8, decltype(dtype_tag)>;                           \
            const int num_h_blocks = ceil_div(H, Traits::Q_TILE_SIZE * Traits::T_M);                               \
            dim3 grid(B, num_h_blocks, QL);                                                                          \
            dim3 block(Traits::BLOCK_SIZE);                                                                        \
            mla_decode_16mx8_32nx1_kernel<Traits><<<grid, block, 0, stream>>>(kargs);                                \
            HIP_CALL_LAUNCH(hipGetLastError());                                                                     \
        };                                                                                                           \
        if(q.dtype() == AITER_DTYPE_bf16)                                                                            \
            launch(bf16_t{});                                                                                        \
        else                                                                                                         \
            launch(fp16_t{});                                                                                        \
    } while(0)

#define LAUNCH_MLA_DECODE_16MX1(QL)                                                                                  \
    do {                                                                                                             \
        auto launch = [&](auto dtype_tag) {                                                                          \
            using Traits = mla_decode_16mx1_traits<QL, 16, 64, 512, 4, decltype(dtype_tag)>;                          \
            const int num_h_blocks = ceil_div(H, Traits::T_M * Traits::Q_TILE_SIZE);                                 \
            dim3 grid(B, num_h_blocks, QL);                                                                          \
            dim3 block(Traits::BLOCK_SIZE);                                                                        \
            mla_decode_16mx1_16nx4_kernel<Traits><<<grid, block, 0, stream>>>(kargs);                                \
            HIP_CALL_LAUNCH(hipGetLastError());                                                                     \
        };                                                                                                           \
        if(q.dtype() == AITER_DTYPE_bf16)                                                                            \
            launch(bf16_t{});                                                                                        \
        else                                                                                                         \
            launch(fp16_t{});                                                                                        \
    } while(0)

#define DISPATCH_QLEN_16MX8()             \
    switch(qlen)                         \
    {                                     \
    case 1:                               \
        LAUNCH_MLA_DECODE_16MX8(1);       \
        break;                            \
    case 2:                               \
        LAUNCH_MLA_DECODE_16MX8(2);       \
        break;                            \
    case 3:                               \
        LAUNCH_MLA_DECODE_16MX8(3);       \
        break;                            \
    case 4:                               \
        LAUNCH_MLA_DECODE_16MX8(4);       \
        break;                            \
    case 5:                               \
        LAUNCH_MLA_DECODE_16MX8(5);       \
        break;                            \
    case 6:                               \
        LAUNCH_MLA_DECODE_16MX8(6);       \
        break;                            \
    case 7:                               \
        LAUNCH_MLA_DECODE_16MX8(7);       \
        break;                            \
    case 8:                               \
        LAUNCH_MLA_DECODE_16MX8(8);       \
        break;                            \
    case 9:                               \
        LAUNCH_MLA_DECODE_16MX8(9);       \
        break;                            \
    case 10:                              \
        LAUNCH_MLA_DECODE_16MX8(10);      \
        break;                            \
    case 11:                              \
        LAUNCH_MLA_DECODE_16MX8(11);      \
        break;                            \
    case 12:                              \
        LAUNCH_MLA_DECODE_16MX8(12);      \
        break;                            \
    case 13:                              \
        LAUNCH_MLA_DECODE_16MX8(13);      \
        break;                            \
    case 14:                              \
        LAUNCH_MLA_DECODE_16MX8(14);      \
        break;                            \
    case 15:                              \
        LAUNCH_MLA_DECODE_16MX8(15);      \
        break;                            \
    case 16:                              \
        LAUNCH_MLA_DECODE_16MX8(16);      \
        break;                            \
    case 17:                              \
        LAUNCH_MLA_DECODE_16MX8(17);      \
        break;                            \
    default:                              \
        AITER_CHECK(false, "internal: qlen out of range"); \
    }

#define DISPATCH_QLEN_16MX1()             \
    switch(qlen)                         \
    {                                     \
    case 1:                               \
        LAUNCH_MLA_DECODE_16MX1(1);       \
        break;                            \
    case 2:                               \
        LAUNCH_MLA_DECODE_16MX1(2);       \
        break;                            \
    case 3:                               \
        LAUNCH_MLA_DECODE_16MX1(3);       \
        break;                            \
    case 4:                               \
        LAUNCH_MLA_DECODE_16MX1(4);       \
        break;                            \
    case 5:                               \
        LAUNCH_MLA_DECODE_16MX1(5);       \
        break;                            \
    case 6:                               \
        LAUNCH_MLA_DECODE_16MX1(6);       \
        break;                            \
    case 7:                               \
        LAUNCH_MLA_DECODE_16MX1(7);       \
        break;                            \
    case 8:                               \
        LAUNCH_MLA_DECODE_16MX1(8);       \
        break;                            \
    case 9:                               \
        LAUNCH_MLA_DECODE_16MX1(9);       \
        break;                            \
    case 10:                              \
        LAUNCH_MLA_DECODE_16MX1(10);      \
        break;                            \
    case 11:                              \
        LAUNCH_MLA_DECODE_16MX1(11);      \
        break;                            \
    case 12:                              \
        LAUNCH_MLA_DECODE_16MX1(12);      \
        break;                            \
    case 13:                              \
        LAUNCH_MLA_DECODE_16MX1(13);      \
        break;                            \
    case 14:                              \
        LAUNCH_MLA_DECODE_16MX1(14);      \
        break;                            \
    case 15:                              \
        LAUNCH_MLA_DECODE_16MX1(15);      \
        break;                            \
    case 16:                              \
        LAUNCH_MLA_DECODE_16MX1(16);      \
        break;                            \
    case 17:                              \
        LAUNCH_MLA_DECODE_16MX1(17);      \
        break;                            \
    default:                              \
        AITER_CHECK(false, "internal: qlen out of range"); \
    }

    if(H <= 32) {
        DISPATCH_QLEN_16MX1();
    } else {
        DISPATCH_QLEN_16MX8();
    }

#undef DISPATCH_QLEN_16MX1
#undef DISPATCH_QLEN_16MX8
#undef LAUNCH_MLA_DECODE_16MX1
#undef LAUNCH_MLA_DECODE_16MX8
}

// stage-2 launch shared by split-KV and qpack-split paths.
template <class Traits>
static void launch_splitkv_stage2(const mla_decode_opus_kargs& kargs, int B, int H, int qlen,
                                  hipStream_t stream)
{
    dim3 s2_block(128);  // 128 threads cover D=512 (4 elems each)
    // Tile D across grid.z when (B*qlen*H) blocks under-fill the GPU, so small-H /
    // small-batch reductions stay latency-hidden (else the reduce is ~31us launch-bound
    // for nhead=16 B=1: 16 blocks @ ~6% occupancy). Cap at D/blockDim so each thread
    // keeps >=1 element; no-op (d_tiles=1) once B*qlen*H already fills the device.
    const int D_v = Traits::D_V_TILE;
    const int base_blocks = B * qlen * H;
    // The reduce kernel is featherweight (low VGPR/LDS, ~8MB traffic) so it fits
    // many blocks/CU; target several waves/CU (4*num_cu) so mid-batch (e.g. B=16,
    // base=256=1/CU) still tiles D and hides the strided per-split load latency.
    const int target = 4 * 256;  // ~4 blocks/CU on gfx950
    int d_tiles = 1;
    if (base_blocks < target) {
        const int want = (target + base_blocks - 1) / base_blocks;
        const int max_dt = D_v / (int)s2_block.x;  // 512/128 = 4
        d_tiles = want < 1 ? 1 : (want < max_dt ? want : max_dt);
    }
    dim3 s2_grid(B * qlen, H, d_tiles);
    mla_decode_splitkv_s2_kernel<Traits><<<s2_grid, s2_block, 0, stream>>>(kargs);
    HIP_CALL_LAUNCH(hipGetLastError());
}

// ===========================================================================
// qlen-packed decode (16mx8, warp == position). H must be 16; qlen in {4,8,16}.
// ===========================================================================
void mla_decode_opus_qpack_fwd(aiter_tensor_t& q,
                               aiter_tensor_t& unified_kv,
                               aiter_tensor_t& kv_indices,
                               aiter_tensor_t& kv_indptr,
                               aiter_tensor_t& attn_sink,
                               aiter_tensor_t& out,
                               float softmax_scale,
                               int qlen)
{
    AITER_CHECK(q.dim() == 4, "q must be 4-D [B, QLEN, H, D]");
    const int B = static_cast<int>(q.size(0));
    AITER_CHECK(static_cast<int>(q.size(1)) == qlen, "q.size(1) must equal qlen");
    const int H = static_cast<int>(q.size(2));
    const int D = static_cast<int>(q.size(3));
    AITER_CHECK(D == 512, "Only D=512 is compiled");
    AITER_CHECK(H == 16, "qpack requires H == 16 (warp=position relies on H==W_M)");
    // NUM_WARPS(=qlen) must keep the pipelined stagger to <=2 groups (warp/4),
    // so qlen in {4,8}. (qlen=16 -> 4 stagger groups -> divergent barriers.)
    AITER_CHECK(qlen == 4 || qlen == 8, "qpack requires qlen in {4,8}");
    AITER_CHECK(q.dtype() == unified_kv.dtype() && q.dtype() == out.dtype(), "dtype mismatch");
    AITER_CHECK(q.dtype() == AITER_DTYPE_bf16 || q.dtype() == AITER_DTYPE_fp16, "bf16/fp16 only");
    AITER_CHECK(kv_indptr.size(0) == B + 1, "kv_indptr length must be B+1");
    AITER_CHECK(q.stride(3) == 1 && unified_kv.stride(1) == 1 && out.stride(3) == 1,
                "Q/UnifiedKV/out must be contiguous along D");

    const int use_sink = (attn_sink.numel() > 0) ? 1 : 0;
    if(B == 0) return;

    mla_decode_opus_kargs kargs{};
    kargs.q_ptr           = q.data_ptr();
    kargs.unified_kv_ptr  = unified_kv.data_ptr();
    kargs.attn_sink_ptr   = use_sink ? attn_sink.data_ptr() : nullptr;
    kargs.out_ptr         = out.data_ptr();
    kargs.kv_indptr       = reinterpret_cast<const int*>(kv_indptr.data_ptr());
    kargs.kv_indices      = reinterpret_cast<const int*>(kv_indices.data_ptr());
    kargs.B = B; kargs.QLEN = qlen; kargs.H = H; kargs.D = D;
    kargs.total_pages     = static_cast<int>(unified_kv.size(0));
    kargs.stride_q_b      = static_cast<int>(q.stride(0));
    kargs.stride_q_qlen   = static_cast<int>(q.stride(1));
    kargs.stride_qo_h     = static_cast<int>(q.stride(2));
    kargs.stride_o_h      = static_cast<int>(out.stride(2));
    kargs.stride_out_b    = static_cast<int>(out.stride(0));
    kargs.stride_out_qlen = static_cast<int>(out.stride(1));
    kargs.stride_kv_page  = static_cast<int>(unified_kv.stride(0));
    kargs.softmax_scale   = softmax_scale;
    kargs.use_sink        = use_sink;

    HipDeviceGuard guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

#define LAUNCH_QPACK(TRAITS_T, QL)                                                        \
    do {                                                                                   \
        auto launch = [&](auto tag) {                                                      \
            using Traits = TRAITS_T<QL, decltype(tag)>;                                    \
            dim3 grid(B, 1, 1);                                                            \
            dim3 block(Traits::BLOCK_SIZE);                                                \
            mla_decode_qpack_16mx8_kernel<Traits><<<grid, block, 0, stream>>>(kargs);      \
            HIP_CALL_LAUNCH(hipGetLastError());                                            \
        };                                                                                 \
        if(q.dtype() == AITER_DTYPE_bf16) launch(bf16_t{}); else launch(fp16_t{});         \
    } while(0)

    // Both use the pipelined 4x path. (Single-buffer was tried for qlen=4 to lift
    // occupancy 1->2 waves/SIMD but measured slower: losing load/compute overlap
    // over ~128 tiles outweighs the occupancy gain. See knowledge/mla_decode_opus.md.)
    switch(qlen) {
    case 4:  LAUNCH_QPACK(mla_decode_qpack_traits, 4); break;
    case 8:  LAUNCH_QPACK(mla_decode_qpack_traits, 8); break;
    default: AITER_CHECK(false, "qpack qlen unsupported");
    }
#undef LAUNCH_QPACK
}

void mla_decode_opus_qpack_h8_fwd(aiter_tensor_t& q,
                                  aiter_tensor_t& unified_kv,
                                  aiter_tensor_t& kv_indices,
                                  aiter_tensor_t& kv_indptr,
                                  aiter_tensor_t& attn_sink,
                                  aiter_tensor_t& out,
                                  float softmax_scale,
                                  int qlen)
{
    AITER_CHECK(q.dim() == 4, "q must be 4-D [B, QLEN, H, D]");
    const int B = static_cast<int>(q.size(0));
    AITER_CHECK(static_cast<int>(q.size(1)) == qlen, "q.size(1) must equal qlen");
    const int H = static_cast<int>(q.size(2));
    const int D = static_cast<int>(q.size(3));
    AITER_CHECK(D == 512 && H == 8, "qpack_h8 requires D=512, H=8");
    AITER_CHECK(qlen == 8, "qpack_h8 requires qlen==8 (NUM_WARPS=qlen/2=4)");
    AITER_CHECK(q.dtype() == unified_kv.dtype() && q.dtype() == out.dtype(), "dtype mismatch");
    AITER_CHECK(q.dtype() == AITER_DTYPE_bf16 || q.dtype() == AITER_DTYPE_fp16, "bf16/fp16 only");
    AITER_CHECK(kv_indptr.size(0) == B + 1, "kv_indptr length must be B+1");
    AITER_CHECK(q.stride(3) == 1 && unified_kv.stride(1) == 1 && out.stride(3) == 1,
                "Q/UnifiedKV/out must be contiguous along D");
    const int use_sink = (attn_sink.numel() > 0) ? 1 : 0;
    if(B == 0) return;

    mla_decode_opus_kargs kargs{};
    kargs.q_ptr = q.data_ptr(); kargs.unified_kv_ptr = unified_kv.data_ptr();
    kargs.attn_sink_ptr = use_sink ? attn_sink.data_ptr() : nullptr;
    kargs.out_ptr = out.data_ptr();
    kargs.kv_indptr = reinterpret_cast<const int*>(kv_indptr.data_ptr());
    kargs.kv_indices = reinterpret_cast<const int*>(kv_indices.data_ptr());
    kargs.B = B; kargs.QLEN = qlen; kargs.H = H; kargs.D = D;
    kargs.total_pages = static_cast<int>(unified_kv.size(0));
    kargs.stride_q_b = static_cast<int>(q.stride(0));
    kargs.stride_q_qlen = static_cast<int>(q.stride(1));
    kargs.stride_qo_h = static_cast<int>(q.stride(2));
    kargs.stride_o_h = static_cast<int>(out.stride(2));
    kargs.stride_out_b = static_cast<int>(out.stride(0));
    kargs.stride_out_qlen = static_cast<int>(out.stride(1));
    kargs.stride_kv_page = static_cast<int>(unified_kv.stride(0));
    kargs.softmax_scale = softmax_scale; kargs.use_sink = use_sink;

    HipDeviceGuard guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    auto launch = [&](auto tag) {
        using Traits = mla_decode_qpack_h8_traits<8, decltype(tag)>;
        dim3 grid(B, 1, 1);
        dim3 block(Traits::BLOCK_SIZE);
        mla_decode_qpack_h8_kernel<Traits><<<grid, block, 0, stream>>>(kargs);
        HIP_CALL_LAUNCH(hipGetLastError());
    };
    if(q.dtype() == AITER_DTYPE_bf16) launch(bf16_t{}); else launch(fp16_t{});
}

void mla_decode_opus_qpack_splitkv_fwd(aiter_tensor_t& q,
                                       aiter_tensor_t& unified_kv,
                                       aiter_tensor_t& kv_indices,
                                       aiter_tensor_t& kv_indptr,
                                       aiter_tensor_t& attn_sink,
                                       aiter_tensor_t& out,
                                       aiter_tensor_t& partial_o,
                                       aiter_tensor_t& partial_ml,
                                       float softmax_scale,
                                       int qlen,
                                       int num_splits)
{
    const int B = static_cast<int>(q.size(0));
    const int H = static_cast<int>(q.size(2));
    const int D = static_cast<int>(q.size(3));
    AITER_CHECK(D == 512 && H == 16, "qpack requires D=512, H=16");
    AITER_CHECK(qlen == 4 || qlen == 8, "qpack qlen in {4,8}");
    AITER_CHECK(num_splits >= 1 && num_splits <= 64, "num_splits in [1,64]");
    AITER_CHECK(partial_o.dtype() == AITER_DTYPE_fp32 && partial_ml.dtype() == AITER_DTYPE_fp32,
                "partials must be fp32");
    const int use_sink = (attn_sink.numel() > 0) ? 1 : 0;
    if(B == 0) return;

    mla_decode_opus_kargs kargs{};
    kargs.q_ptr=q.data_ptr(); kargs.unified_kv_ptr=unified_kv.data_ptr();
    kargs.attn_sink_ptr = use_sink ? attn_sink.data_ptr() : nullptr;
    kargs.out_ptr=out.data_ptr();
    kargs.kv_indptr=reinterpret_cast<const int*>(kv_indptr.data_ptr());
    kargs.kv_indices=reinterpret_cast<const int*>(kv_indices.data_ptr());
    kargs.B=B; kargs.QLEN=qlen; kargs.H=H; kargs.D=D;
    kargs.total_pages=static_cast<int>(unified_kv.size(0));
    kargs.stride_q_b=static_cast<int>(q.stride(0));
    kargs.stride_q_qlen=static_cast<int>(q.stride(1));
    kargs.stride_qo_h=static_cast<int>(q.stride(2));
    kargs.stride_o_h=static_cast<int>(out.stride(2));
    kargs.stride_out_b=static_cast<int>(out.stride(0));
    kargs.stride_out_qlen=static_cast<int>(out.stride(1));
    kargs.stride_kv_page=static_cast<int>(unified_kv.stride(0));
    kargs.softmax_scale=softmax_scale; kargs.use_sink=use_sink;
    kargs.num_splits=num_splits;
    kargs.partial_o_ptr=partial_o.data_ptr(); kargs.partial_ml_ptr=partial_ml.data_ptr();

    HipDeviceGuard guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

#define LAUNCH_QPACK_S1(TRAITS_T, QL)                                                     \
    do {                                                                                   \
        auto launch = [&](auto tag) {                                                      \
            using Traits = TRAITS_T<QL, decltype(tag)>;                                    \
            dim3 grid(B, 1, num_splits);                                                   \
            dim3 block(Traits::BLOCK_SIZE);                                                \
            mla_decode_qpack_s1_16mx8_kernel<Traits><<<grid, block, 0, stream>>>(kargs);   \
            HIP_CALL_LAUNCH(hipGetLastError());                                            \
            launch_splitkv_stage2<Traits>(kargs, B, H, qlen, stream);                      \
        };                                                                                 \
        if(q.dtype() == AITER_DTYPE_bf16) launch(bf16_t{}); else launch(fp16_t{});         \
    } while(0)
    switch(qlen) {
    case 4:  LAUNCH_QPACK_S1(mla_decode_qpack_traits, 4); break;
    case 8:  LAUNCH_QPACK_S1(mla_decode_qpack_traits, 8); break;
    default: AITER_CHECK(false, "qpack qlen unsupported");
    }
#undef LAUNCH_QPACK_S1
}

// ===========================================================================
// Split-KV (flash-decode): stage-1 partials + stage-2 reduce.
// ===========================================================================
template <class Traits>
static void launch_splitkv_8(const mla_decode_opus_kargs& kargs, int B, int H, int qlen,
                             int num_splits, hipStream_t stream)
{
    using T = Traits;
    const int num_h_blocks = ceil_div(H, T::Q_TILE_SIZE * T::T_M);
    dim3 s1_grid(B, num_h_blocks, qlen * num_splits);
    dim3 s1_block(T::BLOCK_SIZE);
    mla_decode_splitkv_s1_16mx8_kernel<T><<<s1_grid, s1_block, 0, stream>>>(kargs);
    HIP_CALL_LAUNCH(hipGetLastError());
    launch_splitkv_stage2<T>(kargs, B, H, qlen, stream);
}

template <class Traits>
static void launch_splitkv_1(const mla_decode_opus_kargs& kargs, int B, int H, int qlen,
                             int num_splits, hipStream_t stream)
{
    using T = Traits;
    const int num_h_blocks = ceil_div(H, T::T_M * T::Q_TILE_SIZE);
    dim3 s1_grid(B, num_h_blocks, qlen * num_splits);
    dim3 s1_block(T::BLOCK_SIZE);
    mla_decode_splitkv_s1_16mx1_kernel<T><<<s1_grid, s1_block, 0, stream>>>(kargs);
    HIP_CALL_LAUNCH(hipGetLastError());
    launch_splitkv_stage2<T>(kargs, B, H, qlen, stream);
}

void mla_decode_opus_splitkv_fwd(aiter_tensor_t& q,
                                 aiter_tensor_t& unified_kv,
                                 aiter_tensor_t& kv_indices,
                                 aiter_tensor_t& kv_indptr,
                                 aiter_tensor_t& attn_sink,
                                 aiter_tensor_t& out,
                                 aiter_tensor_t& partial_o,
                                 aiter_tensor_t& partial_ml,
                                 float softmax_scale,
                                 int qlen,
                                 int num_splits)
{
    AITER_CHECK(q.dim() == 4, "q must be 4-D [B, QLEN, H, D]");
    AITER_CHECK(out.dim() == 4, "out must be 4-D [B, QLEN, H, D]");
    AITER_CHECK(q.dtype() == unified_kv.dtype() && q.dtype() == out.dtype(),
                "q/unified_kv/out must share dtype");
    AITER_CHECK(q.dtype() == AITER_DTYPE_bf16 || q.dtype() == AITER_DTYPE_fp16,
                "Only bf16/fp16 supported");
    AITER_CHECK(partial_o.dtype() == AITER_DTYPE_fp32 && partial_ml.dtype() == AITER_DTYPE_fp32,
                "partial_o/partial_ml must be fp32");

    const int B = static_cast<int>(q.size(0));
    AITER_CHECK(static_cast<int>(q.size(1)) == qlen, "q.size(1) must equal qlen");
    const int H = static_cast<int>(q.size(2));
    const int D = static_cast<int>(q.size(3));
    AITER_CHECK(D == 512, "Only D=512 is compiled, got D=", D);
    AITER_CHECK(qlen >= 1 && qlen <= 17, "qlen must be in [1,17]");
    AITER_CHECK(num_splits >= 1 && num_splits <= 64, "num_splits must be in [1,64]");
    AITER_CHECK(kv_indptr.dtype() == AITER_DTYPE_i32 && kv_indices.dtype() == AITER_DTYPE_i32,
                "kv_indptr/kv_indices must be int32");
    AITER_CHECK(kv_indptr.size(0) == B + 1, "kv_indptr length must be B+1");
    AITER_CHECK(q.stride(3) == 1 && unified_kv.stride(1) == 1 && out.stride(3) == 1,
                "Q/UnifiedKV/out must be contiguous along D");

    const int use_sink = (attn_sink.numel() > 0) ? 1 : 0;
    if(use_sink) {
        AITER_CHECK(attn_sink.dtype() == AITER_DTYPE_fp32 && attn_sink.size(0) == H,
                    "attn_sink must be fp32 [H]");
    }

    // workspace size check: [B*QLEN*num_splits, H, D] and [..., H, 2]
    const long long rows = (long long)B * qlen * num_splits;
    AITER_CHECK(partial_o.numel() >= rows * H * D, "partial_o too small");
    AITER_CHECK(partial_ml.numel() >= rows * H * 2, "partial_ml too small");

    if(B == 0)
        return;

    mla_decode_opus_kargs kargs{};
    kargs.q_ptr           = q.data_ptr();
    kargs.unified_kv_ptr  = unified_kv.data_ptr();
    kargs.attn_sink_ptr   = use_sink ? attn_sink.data_ptr() : nullptr;
    kargs.out_ptr         = out.data_ptr();
    kargs.kv_indptr       = reinterpret_cast<const int*>(kv_indptr.data_ptr());
    kargs.kv_indices      = reinterpret_cast<const int*>(kv_indices.data_ptr());
    kargs.B               = B;
    kargs.QLEN            = qlen;
    kargs.H               = H;
    kargs.D               = D;
    kargs.total_pages     = static_cast<int>(unified_kv.size(0));
    kargs.stride_q_b      = static_cast<int>(q.stride(0));
    kargs.stride_q_qlen   = static_cast<int>(q.stride(1));
    kargs.stride_qo_h     = static_cast<int>(q.stride(2));
    kargs.stride_o_h      = static_cast<int>(out.stride(2));
    kargs.stride_out_b    = static_cast<int>(out.stride(0));
    kargs.stride_out_qlen = static_cast<int>(out.stride(1));
    kargs.stride_kv_page  = static_cast<int>(unified_kv.stride(0));
    kargs.softmax_scale   = softmax_scale;
    kargs.use_sink        = use_sink;
    kargs.num_splits      = num_splits;
    kargs.partial_o_ptr   = partial_o.data_ptr();
    kargs.partial_ml_ptr  = partial_ml.data_ptr();

    HipDeviceGuard guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

#define SPLITKV_8(QL)                                                                     \
    do {                                                                                   \
        if(q.dtype() == AITER_DTYPE_bf16)                                                  \
            launch_splitkv_8<mla_decode_16mx8_traits<QL, 16, 32, 512, 8, bf16_t>>(         \
                kargs, B, H, qlen, num_splits, stream);                                    \
        else                                                                               \
            launch_splitkv_8<mla_decode_16mx8_traits<QL, 16, 32, 512, 8, fp16_t>>(         \
                kargs, B, H, qlen, num_splits, stream);                                    \
    } while(0)
#define SPLITKV_1(QL)                                                                     \
    do {                                                                                   \
        if(q.dtype() == AITER_DTYPE_bf16)                                                  \
            launch_splitkv_1<mla_decode_16mx1_traits<QL, 16, 64, 512, 4, bf16_t>>(         \
                kargs, B, H, qlen, num_splits, stream);                                    \
        else                                                                               \
            launch_splitkv_1<mla_decode_16mx1_traits<QL, 16, 64, 512, 4, fp16_t>>(         \
                kargs, B, H, qlen, num_splits, stream);                                    \
    } while(0)
#define SPLITKV_DISPATCH(MACRO)            \
    switch(qlen) {                         \
    case 1: MACRO(1); break;               \
    case 2: MACRO(2); break;               \
    case 3: MACRO(3); break;               \
    case 4: MACRO(4); break;               \
    case 5: MACRO(5); break;               \
    case 6: MACRO(6); break;               \
    case 7: MACRO(7); break;               \
    case 8: MACRO(8); break;               \
    case 9: MACRO(9); break;               \
    case 10: MACRO(10); break;             \
    case 11: MACRO(11); break;             \
    case 12: MACRO(12); break;             \
    case 13: MACRO(13); break;             \
    case 14: MACRO(14); break;             \
    case 15: MACRO(15); break;             \
    case 16: MACRO(16); break;             \
    case 17: MACRO(17); break;             \
    default: AITER_CHECK(false, "qlen out of range"); \
    }

    if(H <= 32) {
        SPLITKV_DISPATCH(SPLITKV_1);
    } else {
        SPLITKV_DISPATCH(SPLITKV_8);
    }

#undef SPLITKV_DISPATCH
#undef SPLITKV_1
#undef SPLITKV_8
}

// ===========================================================================
// RoPE (D=576) MLA decode: QK over 576 (latent 512 + rope 64), V/output over
// the latent 512. H<=32 (16mx1) path only. q/unified_kv last dim = 576; out 512.
// The kernel does NOT rotate (applied upstream); it handles the asymmetric
// QK(576)/PV(512) contraction. See knowledge/mla_decode_opus.md.
// ===========================================================================
static void mla_decode_opus_rope_set_kargs(mla_decode_opus_kargs& kargs,
                                           aiter_tensor_t& q, aiter_tensor_t& unified_kv,
                                           aiter_tensor_t& kv_indices, aiter_tensor_t& kv_indptr,
                                           aiter_tensor_t& attn_sink, aiter_tensor_t& out,
                                           float softmax_scale, int B, int qlen, int H, int use_sink)
{
    kargs.q_ptr           = q.data_ptr();
    kargs.unified_kv_ptr  = unified_kv.data_ptr();
    kargs.attn_sink_ptr   = use_sink ? attn_sink.data_ptr() : nullptr;
    kargs.out_ptr         = out.data_ptr();
    kargs.kv_indptr       = reinterpret_cast<const int*>(kv_indptr.data_ptr());
    kargs.kv_indices      = reinterpret_cast<const int*>(kv_indices.data_ptr());
    kargs.B               = B;
    kargs.QLEN            = qlen;
    kargs.H               = H;
    kargs.D               = static_cast<int>(out.size(3));   // V/output dim (512)
    kargs.total_pages     = static_cast<int>(unified_kv.size(0));
    kargs.stride_q_b      = static_cast<int>(q.stride(0));
    kargs.stride_q_qlen   = static_cast<int>(q.stride(1));
    kargs.stride_qo_h     = static_cast<int>(q.stride(2));    // 576
    kargs.stride_o_h      = static_cast<int>(out.stride(2));  // 512
    kargs.stride_out_b    = static_cast<int>(out.stride(0));
    kargs.stride_out_qlen = static_cast<int>(out.stride(1));
    kargs.stride_kv_page  = static_cast<int>(unified_kv.stride(0));  // 576
    kargs.softmax_scale   = softmax_scale;
    kargs.use_sink        = use_sink;
}

static void mla_decode_opus_rope_checks(aiter_tensor_t& q, aiter_tensor_t& unified_kv,
                                        aiter_tensor_t& kv_indices, aiter_tensor_t& kv_indptr,
                                        aiter_tensor_t& attn_sink, aiter_tensor_t& out,
                                        int B, int qlen, int H, int use_sink)
{
    AITER_CHECK(q.dim() == 4, "q must be 4-D [B, QLEN, H, 576], got ndim=", q.dim());
    AITER_CHECK(unified_kv.dim() == 2, "unified_kv must be 2-D [total_pages, 576]");
    AITER_CHECK(out.dim() == 4, "out must be 4-D [B, QLEN, H, 512]");
    AITER_CHECK(q.dtype() == unified_kv.dtype() && q.dtype() == out.dtype(),
                "q/unified_kv/out must share dtype");
    AITER_CHECK(q.dtype() == AITER_DTYPE_bf16 || q.dtype() == AITER_DTYPE_fp16,
                "Only bf16/fp16 are supported");
    AITER_CHECK(static_cast<int>(q.size(3)) == 576, "RoPE: q last dim must be 576");
    AITER_CHECK(static_cast<int>(unified_kv.size(1)) == 576, "RoPE: unified_kv last dim must be 576");
    AITER_CHECK(static_cast<int>(out.size(3)) == 512, "RoPE: out last dim must be 512");
    AITER_CHECK(out.size(0) == B && out.size(1) == qlen && out.size(2) == H,
                "out [B,QLEN,H,512] must match q batch/qlen/head");
    // H<=32 -> 16mx1; H>32 -> 16mx8 with NUM_WARPS=4 (warps_d=1 so D_QK=576 tiles
    // cleanly; same config as qpack-rope, proven correct).
    AITER_CHECK(H >= 1, "H must be >= 1, got H=", H);
    AITER_CHECK(kv_indptr.dtype() == AITER_DTYPE_i32 && kv_indices.dtype() == AITER_DTYPE_i32,
                "kv_indptr/kv_indices must be int32");
    AITER_CHECK(kv_indptr.size(0) == B + 1, "kv_indptr length must be B+1");
    AITER_CHECK(q.stride(3) == 1 && unified_kv.stride(1) == 1 && out.stride(3) == 1,
                "Q/UnifiedKV/out must be contiguous along last dim");
    AITER_CHECK(qlen >= 1 && qlen <= 17, "qlen must be in [1,17]");
    if(use_sink) {
        AITER_CHECK(attn_sink.dim() == 1 && attn_sink.dtype() == AITER_DTYPE_fp32 &&
                    attn_sink.size(0) == H, "attn_sink must be fp32 [H]");
    }
}

void mla_decode_opus_rope_fwd(aiter_tensor_t& q,
                              aiter_tensor_t& unified_kv,
                              aiter_tensor_t& kv_indices,
                              aiter_tensor_t& kv_indptr,
                              aiter_tensor_t& attn_sink,
                              aiter_tensor_t& out,
                              float softmax_scale,
                              int qlen)
{
    const int B = static_cast<int>(q.size(0));
    AITER_CHECK(static_cast<int>(q.size(1)) == qlen, "q.size(1) must equal qlen");
    const int H = static_cast<int>(q.size(2));
    const int use_sink = (attn_sink.numel() > 0) ? 1 : 0;
    mla_decode_opus_rope_checks(q, unified_kv, kv_indices, kv_indptr, attn_sink, out, B, qlen, H, use_sink);
    if(B == 0) return;

    mla_decode_opus_kargs kargs{};
    mla_decode_opus_rope_set_kargs(kargs, q, unified_kv, kv_indices, kv_indptr, attn_sink, out,
                                   softmax_scale, B, qlen, H, use_sink);

    HipDeviceGuard guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

#define LAUNCH_MLA_ROPE(QL)                                                                        \
    do {                                                                                           \
        auto launch = [&](auto dtype_tag) {                                                        \
            using tag_t = decltype(dtype_tag);                                                     \
            if (H <= 32) {                                                                          \
                using Traits = mla_decode_rope_16mx1_traits<QL, tag_t>;                            \
                const int num_h_blocks = ceil_div(H, Traits::T_M * Traits::Q_TILE_SIZE);          \
                dim3 grid(B, num_h_blocks, QL); dim3 block(Traits::BLOCK_SIZE);                   \
                mla_decode_16mx1_16nx4_kernel<Traits><<<grid, block, 0, stream>>>(kargs);         \
            } else if (H % 128 == 0) {                                                            \
                using Traits = mla_decode_rope_16mx8_nw8_traits<QL, tag_t>;                        \
                const int num_h_blocks = ceil_div(H, Traits::NUM_WARPS * Traits::Q_TILE_SIZE);    \
                dim3 grid(B, num_h_blocks, QL); dim3 block(Traits::BLOCK_SIZE);                   \
                mla_decode_16mx8_32nx1_kernel<Traits><<<grid, block, 0, stream>>>(kargs);         \
            } else {                                                                               \
                using Traits = mla_decode_rope_16mx8_traits<QL, tag_t>;                            \
                const int num_h_blocks = ceil_div(H, Traits::NUM_WARPS * Traits::Q_TILE_SIZE);    \
                dim3 grid(B, num_h_blocks, QL); dim3 block(Traits::BLOCK_SIZE);                   \
                mla_decode_16mx8_32nx1_kernel<Traits><<<grid, block, 0, stream>>>(kargs);         \
            }                                                                                       \
            HIP_CALL_LAUNCH(hipGetLastError());                                                   \
        };                                                                                         \
        if(q.dtype() == AITER_DTYPE_bf16) launch(bf16_t{}); else launch(fp16_t{});                \
    } while(0)

    switch(qlen) {
    case 1:  LAUNCH_MLA_ROPE(1);  break;  case 2:  LAUNCH_MLA_ROPE(2);  break;
    case 3:  LAUNCH_MLA_ROPE(3);  break;  case 4:  LAUNCH_MLA_ROPE(4);  break;
    case 5:  LAUNCH_MLA_ROPE(5);  break;  case 6:  LAUNCH_MLA_ROPE(6);  break;
    case 7:  LAUNCH_MLA_ROPE(7);  break;  case 8:  LAUNCH_MLA_ROPE(8);  break;
    case 9:  LAUNCH_MLA_ROPE(9);  break;  case 10: LAUNCH_MLA_ROPE(10); break;
    case 11: LAUNCH_MLA_ROPE(11); break;  case 12: LAUNCH_MLA_ROPE(12); break;
    case 13: LAUNCH_MLA_ROPE(13); break;  case 14: LAUNCH_MLA_ROPE(14); break;
    case 15: LAUNCH_MLA_ROPE(15); break;  case 16: LAUNCH_MLA_ROPE(16); break;
    case 17: LAUNCH_MLA_ROPE(17); break;
    default: AITER_CHECK(false, "internal: qlen out of range");
    }
#undef LAUNCH_MLA_ROPE
}

void mla_decode_opus_rope_splitkv_fwd(aiter_tensor_t& q,
                                      aiter_tensor_t& unified_kv,
                                      aiter_tensor_t& kv_indices,
                                      aiter_tensor_t& kv_indptr,
                                      aiter_tensor_t& attn_sink,
                                      aiter_tensor_t& out,
                                      aiter_tensor_t& partial_o,
                                      aiter_tensor_t& partial_ml,
                                      float softmax_scale,
                                      int qlen,
                                      int num_splits)
{
    const int B = static_cast<int>(q.size(0));
    AITER_CHECK(static_cast<int>(q.size(1)) == qlen, "q.size(1) must equal qlen");
    const int H = static_cast<int>(q.size(2));
    const int use_sink = (attn_sink.numel() > 0) ? 1 : 0;
    mla_decode_opus_rope_checks(q, unified_kv, kv_indices, kv_indptr, attn_sink, out, B, qlen, H, use_sink);
    AITER_CHECK(num_splits >= 1 && num_splits <= 64, "num_splits must be in [1,64]");
    AITER_CHECK(partial_o.dtype() == AITER_DTYPE_fp32 && partial_ml.dtype() == AITER_DTYPE_fp32,
                "partial_o/partial_ml must be fp32");
    const int Dv = static_cast<int>(out.size(3));
    const long long rows = (long long)B * qlen * num_splits;
    AITER_CHECK(partial_o.numel() >= rows * H * Dv, "partial_o too small");
    AITER_CHECK(partial_ml.numel() >= rows * H * 2, "partial_ml too small");
    if(B == 0) return;

    mla_decode_opus_kargs kargs{};
    mla_decode_opus_rope_set_kargs(kargs, q, unified_kv, kv_indices, kv_indptr, attn_sink, out,
                                   softmax_scale, B, qlen, H, use_sink);
    kargs.num_splits     = num_splits;
    kargs.partial_o_ptr  = partial_o.data_ptr();
    kargs.partial_ml_ptr = partial_ml.data_ptr();

    HipDeviceGuard guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

#define LAUNCH_MLA_ROPE_S1(QL)                                                                     \
    do {                                                                                           \
        auto launch = [&](auto dtype_tag) {                                                        \
            using tag_t = decltype(dtype_tag);                                                     \
            if (H <= 32) {                                                                          \
                using Traits = mla_decode_rope_16mx1_traits<QL, tag_t>;                            \
                const int num_h_blocks = ceil_div(H, Traits::T_M * Traits::Q_TILE_SIZE);          \
                dim3 s1_grid(B, num_h_blocks, QL * num_splits); dim3 s1_block(Traits::BLOCK_SIZE);\
                mla_decode_splitkv_s1_16mx1_kernel<Traits><<<s1_grid, s1_block, 0, stream>>>(kargs); \
                HIP_CALL_LAUNCH(hipGetLastError());                                               \
                launch_splitkv_stage2<Traits>(kargs, B, H, qlen, stream);                         \
            } else if (H % 128 == 0) {                                                            \
                using Traits = mla_decode_rope_16mx8_nw8_traits<QL, tag_t>;                        \
                const int num_h_blocks = ceil_div(H, Traits::NUM_WARPS * Traits::Q_TILE_SIZE);    \
                dim3 s1_grid(B, num_h_blocks, QL * num_splits); dim3 s1_block(Traits::BLOCK_SIZE);\
                mla_decode_splitkv_s1_16mx8_kernel<Traits><<<s1_grid, s1_block, 0, stream>>>(kargs); \
                HIP_CALL_LAUNCH(hipGetLastError());                                               \
                launch_splitkv_stage2<Traits>(kargs, B, H, qlen, stream);                         \
            } else {                                                                               \
                using Traits = mla_decode_rope_16mx8_traits<QL, tag_t>;                            \
                const int num_h_blocks = ceil_div(H, Traits::NUM_WARPS * Traits::Q_TILE_SIZE);    \
                dim3 s1_grid(B, num_h_blocks, QL * num_splits); dim3 s1_block(Traits::BLOCK_SIZE);\
                mla_decode_splitkv_s1_16mx8_kernel<Traits><<<s1_grid, s1_block, 0, stream>>>(kargs); \
                HIP_CALL_LAUNCH(hipGetLastError());                                               \
                launch_splitkv_stage2<Traits>(kargs, B, H, qlen, stream);                         \
            }                                                                                       \
        };                                                                                         \
        if(q.dtype() == AITER_DTYPE_bf16) launch(bf16_t{}); else launch(fp16_t{});                \
    } while(0)

    switch(qlen) {
    case 1:  LAUNCH_MLA_ROPE_S1(1);  break;  case 2:  LAUNCH_MLA_ROPE_S1(2);  break;
    case 3:  LAUNCH_MLA_ROPE_S1(3);  break;  case 4:  LAUNCH_MLA_ROPE_S1(4);  break;
    case 5:  LAUNCH_MLA_ROPE_S1(5);  break;  case 6:  LAUNCH_MLA_ROPE_S1(6);  break;
    case 7:  LAUNCH_MLA_ROPE_S1(7);  break;  case 8:  LAUNCH_MLA_ROPE_S1(8);  break;
    case 9:  LAUNCH_MLA_ROPE_S1(9);  break;  case 10: LAUNCH_MLA_ROPE_S1(10); break;
    case 11: LAUNCH_MLA_ROPE_S1(11); break;  case 12: LAUNCH_MLA_ROPE_S1(12); break;
    case 13: LAUNCH_MLA_ROPE_S1(13); break;  case 14: LAUNCH_MLA_ROPE_S1(14); break;
    case 15: LAUNCH_MLA_ROPE_S1(15); break;  case 16: LAUNCH_MLA_ROPE_S1(16); break;
    case 17: LAUNCH_MLA_ROPE_S1(17); break;
    default: AITER_CHECK(false, "internal: qlen out of range");
    }
#undef LAUNCH_MLA_ROPE_S1
}

// ─── RoPE qpack (qlen-into-warps, H==16, qlen==4): closes the nhead=16 qlen=4
//     large-batch loss vs asm m16x4 by sharing one KV read across the 4 positions.
void mla_decode_opus_rope_qpack_fwd(aiter_tensor_t& q,
                                    aiter_tensor_t& unified_kv,
                                    aiter_tensor_t& kv_indices,
                                    aiter_tensor_t& kv_indptr,
                                    aiter_tensor_t& attn_sink,
                                    aiter_tensor_t& out,
                                    float softmax_scale,
                                    int qlen)
{
    const int B = static_cast<int>(q.size(0));
    AITER_CHECK(static_cast<int>(q.size(1)) == qlen, "q.size(1) must equal qlen");
    const int H = static_cast<int>(q.size(2));
    const int use_sink = (attn_sink.numel() > 0) ? 1 : 0;
    mla_decode_opus_rope_checks(q, unified_kv, kv_indices, kv_indptr, attn_sink, out, B, qlen, H, use_sink);
    AITER_CHECK((H == 16 && qlen == 4) || (H == 32 && (qlen == 2 || qlen == 4)),
                "rope qpack: H16/qlen4, or H32/qlen{2,4}");
    if(B == 0) return;

    mla_decode_opus_kargs kargs{};
    mla_decode_opus_rope_set_kargs(kargs, q, unified_kv, kv_indices, kv_indptr, attn_sink, out,
                                   softmax_scale, B, qlen, H, use_sink);

    HipDeviceGuard guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    auto launch = [&](auto tag) {
        dim3 grid(B, 1, 1);
        auto run = [&](auto Tr) {
            using Traits = decltype(Tr);
            mla_decode_qpack_16mx8_kernel<Traits><<<grid, dim3(Traits::BLOCK_SIZE), 0, stream>>>(kargs);
            HIP_CALL_LAUNCH(hipGetLastError());
        };
        using D = decltype(tag);
        if (H == 16)            run(mla_decode_qpack_rope_traits<4, D>{});
        else if (qlen == 2)     run(mla_decode_qpack_rope_h32_traits<2, D>{});
        else                    run(mla_decode_qpack_rope_h32_traits<4, D>{});
    };
    if(q.dtype() == AITER_DTYPE_bf16) launch(bf16_t{}); else launch(fp16_t{});
}

void mla_decode_opus_rope_qpack_splitkv_fwd(aiter_tensor_t& q,
                                            aiter_tensor_t& unified_kv,
                                            aiter_tensor_t& kv_indices,
                                            aiter_tensor_t& kv_indptr,
                                            aiter_tensor_t& attn_sink,
                                            aiter_tensor_t& out,
                                            aiter_tensor_t& partial_o,
                                            aiter_tensor_t& partial_ml,
                                            float softmax_scale,
                                            int qlen,
                                            int num_splits)
{
    const int B = static_cast<int>(q.size(0));
    AITER_CHECK(static_cast<int>(q.size(1)) == qlen, "q.size(1) must equal qlen");
    const int H = static_cast<int>(q.size(2));
    const int use_sink = (attn_sink.numel() > 0) ? 1 : 0;
    mla_decode_opus_rope_checks(q, unified_kv, kv_indices, kv_indptr, attn_sink, out, B, qlen, H, use_sink);
    AITER_CHECK((H == 16 && qlen == 4) || (H == 32 && (qlen == 2 || qlen == 4)),
                "rope qpack splitkv: H16/qlen4, or H32/qlen{2,4}");
    AITER_CHECK(num_splits >= 1 && num_splits <= 64, "num_splits must be in [1,64]");
    AITER_CHECK(partial_o.dtype() == AITER_DTYPE_fp32 && partial_ml.dtype() == AITER_DTYPE_fp32,
                "partial_o/partial_ml must be fp32");
    const int Dv = static_cast<int>(out.size(3));
    const long long rows = (long long)B * qlen * num_splits;
    AITER_CHECK(partial_o.numel() >= rows * H * Dv, "partial_o too small");
    AITER_CHECK(partial_ml.numel() >= rows * H * 2, "partial_ml too small");
    if(B == 0) return;

    mla_decode_opus_kargs kargs{};
    mla_decode_opus_rope_set_kargs(kargs, q, unified_kv, kv_indices, kv_indptr, attn_sink, out,
                                   softmax_scale, B, qlen, H, use_sink);
    kargs.num_splits     = num_splits;
    kargs.partial_o_ptr  = partial_o.data_ptr();
    kargs.partial_ml_ptr = partial_ml.data_ptr();

    HipDeviceGuard guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    auto launch = [&](auto tag) {
        dim3 grid(B, 1, num_splits);
        auto run = [&](auto Tr) {
            using Traits = decltype(Tr);
            mla_decode_qpack_s1_16mx8_kernel<Traits><<<grid, dim3(Traits::BLOCK_SIZE), 0, stream>>>(kargs);
            HIP_CALL_LAUNCH(hipGetLastError());
            launch_splitkv_stage2<Traits>(kargs, B, H, qlen, stream);
        };
        using D = decltype(tag);
        if (H == 16)        run(mla_decode_qpack_rope_traits<4, D>{});
        else if (qlen == 2) run(mla_decode_qpack_rope_h32_traits<2, D>{});
        else                run(mla_decode_qpack_rope_h32_traits<4, D>{});
    };
    if(q.dtype() == AITER_DTYPE_bf16) launch(bf16_t{}); else launch(fp16_t{});
}
