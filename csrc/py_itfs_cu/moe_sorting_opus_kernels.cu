// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Opus-based MOE sorting torch-free binding.
// Self-contained: no CK header dependency.

#define MOE_SORTING_OPUS_IMPL
#include "moe_sorting_opus.h"

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"
#include "mx_quant_utils.h"

#include <string>
#include <vector>

void moe_sorting_opus_fwd(aiter_tensor_t& topk_ids,
                          aiter_tensor_t& topk_weights,
                          aiter_tensor_t& sorted_token_ids,
                          aiter_tensor_t& sorted_weights,
                          aiter_tensor_t& sorted_expert_ids,
                          aiter_tensor_t& num_valid_ids,
                          aiter_tensor_t& moe_buf,
                          int num_experts,
                          int unit_size,
                          std::optional<aiter_tensor_t> local_expert_mask,
                          std::optional<aiter_tensor_t> num_local_tokens,
                          std::optional<aiter_tensor_t> workspace,
                          int dispatch_policy,
                          std::optional<aiter_tensor_t> local_topk_ids,
                          std::optional<aiter_tensor_t> m_indices,
                          std::optional<aiter_tensor_t> reverse_sorted)
{
    AITER_CHECK(topk_weights.dtype() == AITER_DTYPE_fp32,
                "topk_weights must be FP32 (float32)");

    auto dtype_str = AiterDtype_to_str(topk_ids.dtype());
    int num_tokens = topk_ids.size(0);
    int topk       = topk_ids.size(1);

    if(local_topk_ids.has_value())
    {
        auto& ids_out = local_topk_ids.value();
        AITER_CHECK(ids_out.dim() == 2 && ids_out.size(0) == topk_ids.size(0) &&
                        ids_out.size(1) == topk_ids.size(1),
                    "local_topk_ids must have the same [tokens, topk] shape as topk_ids");
        AITER_CHECK(ids_out.dtype() == topk_ids.dtype(),
                    "local_topk_ids dtype must match topk_ids");
        AITER_CHECK(ids_out.device_id == topk_ids.device_id,
                    "local_topk_ids must be on the same device as topk_ids");
        AITER_CHECK(ids_out.is_contiguous(), "local_topk_ids must be contiguous");
    }

    HipDeviceGuard device_guard(topk_ids.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    if(reverse_sorted.has_value())
    {
        auto& reverse = reverse_sorted.value();
        AITER_CHECK(reverse.dtype() == AITER_DTYPE_i32 && reverse.is_contiguous() &&
                        reverse.numel() >= static_cast<int64_t>(num_tokens) * topk,
                    "reverse_sorted must be a contiguous int32 buffer for tokens * topk routes");
    }

    void* ws_ptr = workspace.has_value() ? workspace.value().data_ptr() : nullptr;

    moe_sorting_opus(
        {
            dtype_str,
            "fp32",
            local_expert_mask.has_value(),
            true,
            dispatch_policy
        },
        {topk_ids.data_ptr(),
         topk_weights.data_ptr(),
         local_expert_mask.has_value() ? local_expert_mask.value().data_ptr() : nullptr,
         num_local_tokens.has_value() ? num_local_tokens.value().data_ptr() : nullptr,
         sorted_token_ids.data_ptr(),
         sorted_weights.data_ptr(),
         sorted_expert_ids.data_ptr(),
         num_valid_ids.data_ptr(),
         moe_buf.data_ptr(),
         ws_ptr,
         local_topk_ids.has_value() ? local_topk_ids.value().data_ptr() : nullptr,
         m_indices.has_value() ? m_indices.value().data_ptr() : nullptr,
         reverse_sorted.has_value() ? reverse_sorted.value().data_ptr() : nullptr,
         num_tokens,
         unit_size,
         num_experts,
         topk,
         static_cast<int>(moe_buf.size(-1)),
         static_cast<int>(moe_buf.element_size())},
        {stream});
}

namespace aiter::mxfp4_moe {

constexpr int kHiddenSize    = 4096;
constexpr int kBlockM        = 32;
constexpr int kThreads       = 1024;
constexpr int kSortQuantCtas = 16;
constexpr int kScaleCols     = kHiddenSize / 32;

// Routed-MoE shapes the sort kernel is instantiated for. NumExperts and TopK
// are template parameters (the expert histogram lives in LDS), so widening the
// support surface means adding a row here.
#define AITER_MXFP4_SORT_QUANT_ROUTES(_) \
    _(512, 10)                           \
    _(513, 11)

__device__ __forceinline__ int pack_token_topk_id(int token_id, int topk_id)
{
    return (token_id & 0x00ffffff) | ((topk_id & 0xff) << 24);
}

__device__ __forceinline__ int round_up_to_block(int value)
{
    return (value + kBlockM - 1) & ~(kBlockM - 1);
}

__device__ __forceinline__ int dpp_inclusive_scan_wave(int value)
{
    int tmp = __builtin_amdgcn_mov_dpp(value, 0x111, 0xF, 0xF, true);
    value += tmp;
    tmp = __builtin_amdgcn_mov_dpp(value, 0x112, 0xF, 0xF, true);
    value += tmp;
    tmp = __builtin_amdgcn_mov_dpp(value, 0x114, 0xF, 0xF, true);
    value += tmp;
    tmp = __builtin_amdgcn_mov_dpp(value, 0x118, 0xF, 0xF, true);
    value += tmp;
    tmp = __builtin_amdgcn_update_dpp(0, value, 0x142, 0xA, 0xF, true);
    value += tmp;
    tmp = __builtin_amdgcn_update_dpp(0, value, 0x143, 0xC, 0xF, true);
    return value + tmp;
}

template <int NumCtas>
__device__ __forceinline__ void zero_moe_output(void* output, int tokens)
{
    if(output == nullptr)
        return;

    constexpr long long row_bytes = static_cast<long long>(kHiddenSize) * 2;
    const long long total_vecs = (static_cast<long long>(tokens) * row_bytes) / sizeof(int4);
    const long long global_thread =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    constexpr long long stride = static_cast<long long>(NumCtas) * kThreads;
    auto* output_vec = reinterpret_cast<int4*>(output);
    const int4 zero = {0, 0, 0, 0};

    for(long long i = global_thread; i < total_vecs; i += stride)
        output_vec[i] = zero;
}

template <int NumExperts, int TopK>
__device__ __forceinline__ void build_sorted_metadata(
    const int32_t* topk_ids,
    const float* topk_weights,
    int32_t* sorted_token_ids,
    float* sorted_weights,
    int32_t* sorted_expert_ids,
    int32_t* num_valid_ids,
    int tokens)
{
    // Experts are scanned as complete 64-lane waves; a trailing partial wave
    // (E=385) is covered by the running prefix of the last full wave.
    constexpr int kFullExpertWaves = NumExperts / 64;
    __shared__ int expert_counts[NumExperts];
    __shared__ int expert_offsets[NumExperts + 1];
    __shared__ int expert_cursors[NumExperts];
    __shared__ int wave_totals[16];

    const int tid = threadIdx.x;
    for(int expert = tid; expert < NumExperts; expert += kThreads)
    {
        expert_counts[expert] = 0;
        expert_cursors[expert] = 0;
    }
    __syncthreads();

    const int total_pairs = tokens * TopK;
    for(int index = tid; index < total_pairs; index += kThreads)
    {
        const int expert = topk_ids[index];
        if(expert >= 0 && expert < NumExperts)
            atomicAdd(expert_counts + expert, 1);
    }
    __syncthreads();

    // At decode expert counts a small shared-memory prefix scan is cheaper
    // than the generic Opus one-shot bookkeeping.
    const int lane = tid & 63;
    const int wave = tid >> 6;
    int value = tid < NumExperts ? round_up_to_block(expert_counts[tid]) : 0;
    value = dpp_inclusive_scan_wave(value);
    if(lane == 63 && wave < kFullExpertWaves)
        wave_totals[wave] = value;
    __syncthreads();
    if(wave == 0)
    {
        int wave_value = lane < kFullExpertWaves ? wave_totals[lane] : 0;
        wave_value = dpp_inclusive_scan_wave(wave_value);
        if(lane < kFullExpertWaves)
            wave_totals[lane] = wave_value;
    }
    __syncthreads();

    const int wave_prefix = wave == 0 ? 0 : wave_totals[wave - 1];
    if(tid < NumExperts)
    {
        expert_offsets[tid + 1] = value + wave_prefix;
        expert_cursors[tid] = 0;
    }
    if(tid == 0)
        expert_offsets[0] = 0;
    __syncthreads();

    if(tid == 0)
    {
        num_valid_ids[0] = expert_offsets[NumExperts];
        num_valid_ids[1] = tokens;
    }
    __syncthreads();

    for(int index = tid; index < total_pairs; index += kThreads)
    {
        const int expert = topk_ids[index];
        if(expert < 0 || expert >= NumExperts)
            continue;

        const int token = index / TopK;
        const int topk = index - token * TopK;
        const int position = expert_offsets[expert] + atomicAdd(expert_cursors + expert, 1);
        sorted_token_ids[position] = pack_token_topk_id(token, topk);
        sorted_weights[position] = topk_weights[index];
    }
    __syncthreads();

    const int sentinel = pack_token_topk_id(tokens, 0);
    for(int expert = tid; expert < NumExperts; expert += kThreads)
    {
        const int count = expert_counts[expert];
        const int start = expert_offsets[expert];
        const int end = expert_offsets[expert + 1];

        for(int position = start + count; position < end; ++position)
        {
            sorted_token_ids[position] = sentinel;
            sorted_weights[position] = 0.0f;
        }
        for(int block = start / kBlockM; block < end / kBlockM; ++block)
            sorted_expert_ids[block] = expert;
    }
}

template <int QuantCtas>
__device__ __forceinline__ void quantize_activations_compact(
    const hip_bfloat16* hidden_states,
    uint8_t* activation_quant,
    uint8_t* activation_scale_token,
    int tokens)
{
    using bf16x2_t = __bf16 __attribute__((ext_vector_type(2)));
    constexpr int groups_per_token = kHiddenSize / 32;
    constexpr int groups_per_wave = 16;
    constexpr int groups_per_cta = (kThreads / 64) * groups_per_wave;

    const int tid = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int group_in_wave = lane >> 2;
    const int lane_in_group = lane & 3;
    const int total_groups = tokens * groups_per_token;
    const int total_batches = (total_groups + groups_per_cta - 1) / groups_per_cta;
    const int quant_cta = blockIdx.x - 1;
    const int batches_per_cta = (total_batches + QuantCtas - 1) / QuantCtas;
    const int batch_begin = quant_cta * batches_per_cta;
    const int batch_end = min(batch_begin + batches_per_cta, total_batches);

    for(int batch = batch_begin; batch < batch_end; ++batch)
    {
        const int group = batch * groups_per_cta + wave * groups_per_wave + group_in_wave;
        if(group >= total_groups)
            continue;

        const int element_offset = group * 32 + lane_in_group * 8;
        uint32_t raw[4];
        *reinterpret_cast<int4*>(raw) =
            *reinterpret_cast<const int4*>(hidden_states + element_offset);

        uint16_t local_amax = 0;
#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            const uint16_t lo = static_cast<uint16_t>(raw[i] & 0xffffu) & 0x7fffu;
            const uint16_t hi = static_cast<uint16_t>(raw[i] >> 16) & 0x7fffu;
            local_amax = max(local_amax, max(lo, hi));
        }

        uint32_t max_bits = local_amax;
        max_bits = max(
            max_bits,
            static_cast<uint32_t>(
                __builtin_amdgcn_mov_dpp(static_cast<int>(max_bits), 0xB1, 0xF, 0xF, true)));
        max_bits = max(
            max_bits,
            static_cast<uint32_t>(
                __builtin_amdgcn_mov_dpp(static_cast<int>(max_bits), 0x4E, 0xF, 0xF, true)));
        const float amax =
            __uint_as_float(static_cast<uint32_t>(max_bits & 0xffffu) << 16);
        const float dequant_scale = aiter::fp4_f32_to_e8m0_scale(amax);
        const uint8_t scale =
            (static_cast<uint32_t>(__builtin_bit_cast(uint32_t, dequant_scale)) >> 23) & 0xff;

        uint32_t packed = 0;
#if defined(__gfx950__)
        packed = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(
            packed, *reinterpret_cast<const bf16x2_t*>(&raw[0]), dequant_scale, 0);
        packed = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(
            packed, *reinterpret_cast<const bf16x2_t*>(&raw[1]), dequant_scale, 1);
        packed = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(
            packed, *reinterpret_cast<const bf16x2_t*>(&raw[2]), dequant_scale, 2);
        packed = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(
            packed, *reinterpret_cast<const bf16x2_t*>(&raw[3]), dequant_scale, 3);
#else
        __builtin_trap();
#endif

        *reinterpret_cast<uint32_t*>(
            activation_quant + static_cast<size_t>(group) * 16 + lane_in_group * 4) = packed;
        if(lane_in_group == 0)
            activation_scale_token[group] = scale;
    }
}

template <int NumExperts, int TopK>
__global__ void __launch_bounds__(kThreads) moe_sort_quant_kernel(
    const hip_bfloat16* hidden_states,
    const int32_t* topk_ids,
    const float* topk_weights,
    int32_t* sorted_token_ids,
    float* sorted_weights,
    int32_t* sorted_expert_ids,
    int32_t* num_valid_ids,
    void* moe_buf,
    uint8_t* activation_quant,
    uint8_t* activation_scale_token,
    int tokens)
{
    zero_moe_output<kSortQuantCtas>(moe_buf, tokens);
    if(blockIdx.x == 0)
    {
        build_sorted_metadata<NumExperts, TopK>(
            topk_ids,
            topk_weights,
            sorted_token_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            tokens);
    }
    else
    {
        quantize_activations_compact<kSortQuantCtas - 1>(
            hidden_states, activation_quant, activation_scale_token, tokens);
    }
}

// The activation quantization above lowers to v_cvt_scalef32_pk_fp4_bf16, which
// is part of the fp4-cvt-scale instruction set introduced with gfx950 (CDNA4).
// Earlier CDNA parts have no FP4 convert instruction at all, so there is no
// gfx942 equivalent to substitute -- and the MXFP4 MoE GEMM that consumes this
// kernel's output is gfx950-only for the same reason, so a software-emulated
// fallback would produce bytes nothing downstream could use. Reject other
// architectures on the host instead, so an unsupported call reports an error
// rather than executing the __builtin_trap() the device path compiles to.
//
// Cached per device: this sits on the decode path and hipGetDeviceProperties is
// not cheap enough to call per launch.
inline bool is_fp4_cvt_capable(int device_id)
{
    static const std::vector<char> capable = [] {
        int count = 0;
        HIP_CALL(hipGetDeviceCount(&count));
        std::vector<char> caps(static_cast<size_t>(count), 0);
        for(int i = 0; i < count; ++i)
        {
            hipDeviceProp_t prop;
            HIP_CALL(hipGetDeviceProperties(&prop, i));
            caps[static_cast<size_t>(i)] =
                std::string(prop.gcnArchName).find("gfx950") != std::string::npos;
        }
        return caps;
    }();
    return device_id >= 0 && device_id < static_cast<int>(capable.size()) &&
           capable[static_cast<size_t>(device_id)] != 0;
}

} // namespace aiter::mxfp4_moe

void mxfp4_moe_sort_quant_fwd(aiter_tensor_t& hidden_states,
                              aiter_tensor_t& topk_ids,
                              aiter_tensor_t& topk_weights,
                              aiter_tensor_t& sorted_token_ids,
                              aiter_tensor_t& sorted_weights,
                              aiter_tensor_t& sorted_expert_ids,
                              aiter_tensor_t& num_valid_ids,
                              aiter_tensor_t& moe_buf,
                              aiter_tensor_t& activation_quant,
                              aiter_tensor_t& activation_scale_token,
                              int num_experts)
{
    using namespace aiter::mxfp4_moe;

    const int device_id = hidden_states.device_id;
    AITER_CHECK(is_fp4_cvt_capable(device_id),
                __func__,
                ": requires gfx950; the fp4 activation quantization uses the "
                "scalef32 fp4-cvt instructions, which no earlier architecture has");
    // An off-device tensor would otherwise be dereferenced by the kernel under
    // this call's HipDeviceGuard and fault, or silently read another device's
    // memory. moe_buf is exempt when empty: the zeroing pass skips it entirely.
    AITER_CHECK(topk_ids.device_id == device_id && topk_weights.device_id == device_id &&
                    sorted_token_ids.device_id == device_id &&
                    sorted_weights.device_id == device_id &&
                    sorted_expert_ids.device_id == device_id &&
                    num_valid_ids.device_id == device_id &&
                    (moe_buf.numel() == 0 || moe_buf.device_id == device_id) &&
                    activation_quant.device_id == device_id &&
                    activation_scale_token.device_id == device_id,
                __func__,
                ": all tensors must be on the same device");

    AITER_CHECK(hidden_states.dtype() == AITER_DTYPE_bf16,
                __func__,
                ": hidden_states must be BF16");
    AITER_CHECK(topk_ids.dtype() == AITER_DTYPE_i32 &&
                    topk_weights.dtype() == AITER_DTYPE_fp32,
                __func__,
                ": expected int32 topk_ids and fp32 topk_weights");
    AITER_CHECK(sorted_token_ids.dtype() == AITER_DTYPE_i32 &&
                    sorted_weights.dtype() == AITER_DTYPE_fp32 &&
                    sorted_expert_ids.dtype() == AITER_DTYPE_i32 &&
                    num_valid_ids.dtype() == AITER_DTYPE_i32,
                __func__,
                ": expected int32 sorted_token_ids, fp32 sorted_weights, int32 "
                "sorted_expert_ids, and int32 num_valid_ids");
    const int topk = topk_ids.dim() == 2 ? static_cast<int>(topk_ids.size(1)) : 0;

#define AITER_MXFP4_SORT_QUANT_MATCH(NUM_EXPERTS, TOPK) \
    || (num_experts == (NUM_EXPERTS) && topk == (TOPK))
    const bool route_supported =
        false AITER_MXFP4_SORT_QUANT_ROUTES(AITER_MXFP4_SORT_QUANT_MATCH);
#undef AITER_MXFP4_SORT_QUANT_MATCH

    AITER_CHECK(hidden_states.dim() == 2 && hidden_states.size(1) == kHiddenSize &&
                    topk_ids.dim() == 2 && topk_ids.size(0) == hidden_states.size(0) &&
                    route_supported,
                __func__,
                ": only H=4096 with (num_experts, topk) in "
                "{(512,10), (513,11)} is supported");
    AITER_CHECK(hidden_states.is_contiguous() && topk_ids.is_contiguous() &&
                    topk_weights.is_contiguous() && sorted_token_ids.is_contiguous() &&
                    sorted_weights.is_contiguous() && sorted_expert_ids.is_contiguous() &&
                    num_valid_ids.is_contiguous() && moe_buf.is_contiguous() &&
                    activation_quant.is_contiguous() &&
                    activation_scale_token.is_contiguous(),
                __func__,
                ": all tensors must be contiguous");
    AITER_CHECK(activation_quant.dtype() == AITER_DTYPE_fp4x2 ||
                    activation_quant.dtype() == AITER_DTYPE_u8,
                __func__,
                ": activation_quant must be packed FP4");
    AITER_CHECK(activation_scale_token.dtype() == AITER_DTYPE_fp8_e8m0 ||
                    activation_scale_token.dtype() == AITER_DTYPE_u8,
                __func__,
                ": activation_scale_token must hold e8m0 bytes");

    const int tokens = static_cast<int>(hidden_states.size(0));
    AITER_CHECK(tokens > 0 && tokens <= 128,
                __func__,
                ": only decode M in [1, 128] is supported");
    const int64_t route_count = topk_ids.numel();
    const int64_t active_experts =
        route_count < static_cast<int64_t>(num_experts) ? route_count : num_experts;
    const int64_t required_sorted =
        ((route_count + active_experts * (kBlockM - 1) + kBlockM - 1) /
         kBlockM) *
        kBlockM;
    AITER_CHECK(topk_weights.numel() == topk_ids.numel() &&
                    activation_quant.numel() == hidden_states.numel() / 2 &&
                    activation_scale_token.numel() ==
                        hidden_states.size(0) * kScaleCols &&
                    num_valid_ids.numel() >= 2,
                __func__,
                ": incompatible fused sort/quant tensor sizes");
    AITER_CHECK(moe_buf.numel() == 0 ||
                    ((moe_buf.dtype() == AITER_DTYPE_bf16 ||
                      moe_buf.dtype() == AITER_DTYPE_fp16) &&
                     moe_buf.numel() >=
                         static_cast<int64_t>(tokens) * kHiddenSize),
                __func__,
                ": moe_buf must be empty or contain at least M*4096 BF16/FP16 elements");

    AITER_CHECK(sorted_token_ids.numel() >= required_sorted &&
                    sorted_weights.numel() >= required_sorted &&
                    sorted_expert_ids.numel() >= required_sorted / kBlockM,
                __func__,
                ": output routing buffers are too small");

    HipDeviceGuard device_guard(hidden_states.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    const hip_bfloat16* hidden_ptr =
        static_cast<const hip_bfloat16*>(hidden_states.data_ptr());
    const int32_t* topk_ids_ptr = static_cast<const int32_t*>(topk_ids.data_ptr());
    const float* topk_weights_ptr = static_cast<const float*>(topk_weights.data_ptr());
    int32_t* sorted_ids_ptr = static_cast<int32_t*>(sorted_token_ids.data_ptr());
    float* sorted_weights_ptr = static_cast<float*>(sorted_weights.data_ptr());
    int32_t* sorted_experts_ptr = static_cast<int32_t*>(sorted_expert_ids.data_ptr());
    int32_t* num_valid_ptr = static_cast<int32_t*>(num_valid_ids.data_ptr());
    void* moe_buf_ptr = moe_buf.numel() == 0 ? nullptr : moe_buf.data_ptr();
    uint8_t* activation_quant_ptr = static_cast<uint8_t*>(activation_quant.data_ptr());
    uint8_t* activation_scale_token_ptr =
        static_cast<uint8_t*>(activation_scale_token.data_ptr());
#define AITER_MXFP4_SORT_QUANT_LAUNCH(NUM_EXPERTS, TOPK)                            \
    if(num_experts == (NUM_EXPERTS) && topk == (TOPK))                              \
    {                                                                               \
        aiter::mxfp4_moe::moe_sort_quant_kernel<(NUM_EXPERTS), (TOPK)>              \
            <<<dim3(kSortQuantCtas), dim3(kThreads), 0, stream>>>(                  \
                hidden_ptr,                                                         \
                topk_ids_ptr,                                                       \
                topk_weights_ptr,                                                   \
                sorted_ids_ptr,                                                     \
                sorted_weights_ptr,                                                 \
                sorted_experts_ptr,                                                 \
                num_valid_ptr,                                                      \
                moe_buf_ptr,                                                        \
                activation_quant_ptr,                                               \
                activation_scale_token_ptr,                                         \
                tokens);                                                            \
    }
    AITER_MXFP4_SORT_QUANT_ROUTES(AITER_MXFP4_SORT_QUANT_LAUNCH)
#undef AITER_MXFP4_SORT_QUANT_LAUNCH

    const hipError_t launch_status = hipGetLastError();
    AITER_CHECK(
        launch_status == hipSuccess,
        __func__,
        ": fused sort+quant launch failed");
}

namespace aiter::a16w4_moe {

constexpr int kThreads       = 1024;
constexpr int kWaveSize      = 64; // gfx942 / gfx950
constexpr int kMaxTokens     = 128;
constexpr int kMaskWords     = kMaxTokens / 32;
// Quantized tokens per CTA (one wave each); the rest of the CTA only zeroes.
// A CU streams a few rows faster than one CTA working through 16.
constexpr int kTokensPerCta  = 4;

// (num_experts, topk, model_dim) the kernel is instantiated for. The expert
// table lives in LDS and the per-token quant is unrolled over model_dim, so
// each supported shape is a row here.
#define AITER_A16W4_SORT_QUANT_SHAPES(_) _(896, 16, 3584) _(384, 6, 5120)

// Single-CTA route sort producing the same metadata as the multi-phase
// moe_sorting_opus_fwd at these sizes: experts in ascending order, each
// expert's routes in ascending token order, every non-empty expert padded to
// BlockM with the (tokens, topk) sentinel and weight 0, empty experts skipped.
// A route's rank inside its expert is the number of earlier tokens in the
// expert's token bitmask, so the result does not depend on atomic ordering.
template <int NumExperts, int TopK, int BlockM>
__device__ __forceinline__ void sort_routes(const int32_t* topk_ids,
                                            const float* topk_weights,
                                            int32_t* sorted_token_ids,
                                            float* sorted_weights,
                                            int32_t* sorted_expert_ids,
                                            int32_t* num_valid_ids,
                                            int tokens)
{
    using aiter::mxfp4_moe::dpp_inclusive_scan_wave;
    using aiter::mxfp4_moe::pack_token_topk_id;
    static_assert((BlockM & (BlockM - 1)) == 0, "BlockM must be a power of two");
    constexpr int kFullExpertWaves = NumExperts / kWaveSize;
    static_assert(NumExperts <= kThreads && kFullExpertWaves <= kWaveSize,
                  "one expert per thread and wave totals in one wave scan");
    constexpr int kRoutesPerThread = (kMaxTokens * TopK + kThreads - 1) / kThreads;

    __shared__ uint32_t token_masks[NumExperts][kMaskWords];
    __shared__ int expert_offsets[NumExperts];
    __shared__ int expert_counts[NumExperts];
    __shared__ int wave_totals[kThreads / kWaveSize];

    const int tid = threadIdx.x;
    for(int i = tid; i < NumExperts * kMaskWords; i += kThreads)
        (&token_masks[0][0])[i] = 0;

    const int routes = tokens * TopK;
    int route_expert[kRoutesPerThread];
    float route_weight[kRoutesPerThread];
#pragma unroll
    for(int r = 0; r < kRoutesPerThread; ++r)
    {
        const int index = tid + r * kThreads;
        route_expert[r] = index < routes ? topk_ids[index] : -1;
        route_weight[r] = index < routes ? topk_weights[index] : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for(int r = 0; r < kRoutesPerThread; ++r)
    {
        const int expert = route_expert[r];
        if(expert >= 0 && expert < NumExperts)
        {
            const int token = (tid + r * kThreads) / TopK;
            atomicOr(&token_masks[expert][token >> 5], 1u << (token & 31));
        }
    }
    __syncthreads();

    int count = 0;
    if(tid < NumExperts)
    {
#pragma unroll
        for(int w = 0; w < kMaskWords; ++w)
            count += __popc(token_masks[tid][w]);
    }
    const int padded = (count + BlockM - 1) & ~(BlockM - 1);
    const int lane   = tid % kWaveSize;
    const int wave   = tid / kWaveSize;
    int value        = dpp_inclusive_scan_wave(padded);
    if(lane == kWaveSize - 1 && wave < kFullExpertWaves)
        wave_totals[wave] = value;
    __syncthreads();
    if(wave == 0)
    {
        int wave_value = lane < kFullExpertWaves ? wave_totals[lane] : 0;
        wave_value     = dpp_inclusive_scan_wave(wave_value);
        if(lane < kFullExpertWaves)
            wave_totals[lane] = wave_value;
    }
    __syncthreads();

    const int end = value + (wave == 0 ? 0 : wave_totals[wave - 1]);
    if(tid < NumExperts)
    {
        expert_offsets[tid] = end - padded;
        expert_counts[tid]  = count;
    }
    if(tid == NumExperts - 1)
    {
        num_valid_ids[0] = end;
        num_valid_ids[1] = tokens;
    }
    __syncthreads();

    // BlockM consecutive threads per expert write its padding rows and block
    // ids, so the stores coalesce and no thread loops over an expert's rows.
    // An expert has at most kMaxTokens / BlockM <= BlockM blocks.
    static_assert(kMaxTokens / BlockM <= BlockM, "one pass covers an expert's blocks");
    const int sentinel = pack_token_topk_id(tokens, TopK);
    for(int slot = tid; slot < NumExperts * BlockM; slot += kThreads)
    {
        const int expert = slot / BlockM;
        const int j      = slot % BlockM;
        const int e_cnt  = expert_counts[expert];
        if(e_cnt == 0)
            continue;
        const int e_start = expert_offsets[expert];
        const int e_pad   = ((e_cnt + BlockM - 1) & ~(BlockM - 1)) - e_cnt;
        if(j < e_pad)
        {
            sorted_token_ids[e_start + e_cnt + j] = sentinel;
            sorted_weights[e_start + e_cnt + j]   = 0.0f;
        }
        if(j < (e_cnt + BlockM - 1) / BlockM)
            sorted_expert_ids[e_start / BlockM + j] = expert;
    }

#pragma unroll
    for(int r = 0; r < kRoutesPerThread; ++r)
    {
        const int expert = route_expert[r];
        if(expert < 0 || expert >= NumExperts)
            continue;
        const int index   = tid + r * kThreads;
        const int token   = index / TopK;
        const int word    = token >> 5;
        const uint32_t* m = token_masks[expert];
        int rank          = __popc(m[word] & ((1u << (token & 31)) - 1u));
#pragma unroll
        for(int w = 0; w < kMaskWords; ++w)
            rank += w < word ? __popc(m[w]) : 0;
        const int position          = expert_offsets[expert] + rank;
        sorted_token_ids[position]  = pack_token_topk_id(token, index - token * TopK);
        sorted_weights[position]    = route_weight[r];
    }
}

// x * inverted_scale clamped to +-fp8 max and packed to two e4m3 bytes: the
// instruction sequence of aiter::fp32_to_fp8_scaled_x2 (aiter_opus_plus.h),
// which dynamic_per_token_scaled_quant uses. That header is not included
// here because its `using namespace opus` inside aiter clashes with this file.
__device__ __forceinline__ uint16_t fp32x2_to_fp8x2_scaled(float x0, float x1, float inverted_scale)
{
#if defined(__gfx942__) || defined(__gfx950__)
    using f32x2 = float __attribute__((ext_vector_type(2)));
    f32x2 v = {x0, x1};
    f32x2 s = {inverted_scale, inverted_scale};
    asm volatile("v_pk_mul_f32 %0, %1, %2" : "=v"(v) : "v"(v), "v"(s));
    const float hi = opus::finfo<opus::fp8_t>::max();
    const float lo = -hi;
    float a = v[0], b = v[1];
    int packed;
    asm volatile("v_med3_f32 %1, %1, %3, %4\n"
                 "v_med3_f32 %2, %2, %3, %4\n"
                 "v_cvt_pk_fp8_f32 %0, %1, %2"
                 : "=v"(packed), "+v"(a), "+v"(b)
                 : "v"(lo), "v"(hi));
    return static_cast<uint16_t>(packed);
#else
    // The host gate only selects the quant on gfx942.
    (void)x0;
    (void)x1;
    (void)inverted_scale;
    __builtin_trap();
    return 0;
#endif
}

// Per-token e4m3 quant of one row by one wave, with the arithmetic of
// dynamic_per_token_scaled_quant_kernel (row amax, scale = amax / fp8 max,
// rcp, then the same scaled fp8 pair conversion), so a_q / a_s are bitwise
// the same as that kernel's.
template <int HiddenSize>
__device__ __forceinline__ void quantize_token_fp8(const opus::bf16_t* hidden_states,
                                                   opus::fp8_t* activation_quant,
                                                   float* activation_scale,
                                                   int token)
{
    constexpr int kVec         = 8;
    constexpr int kVecsPerLane = HiddenSize / (kVec * kWaveSize);
    static_assert(HiddenSize % (kVec * kWaveSize) == 0, "model_dim must fill whole waves");
    using in_vec  = opus::vector_t<opus::bf16_t, kVec>;
    using out_vec = opus::vector_t<uint16_t, kVec / 2>;
    const float inverted_fp8_max = (1. / static_cast<float>(opus::finfo<opus::fp8_t>::max()));

    const int lane    = threadIdx.x % kWaveSize;
    const auto* src   = reinterpret_cast<const in_vec*>(hidden_states) +
                      static_cast<size_t>(token) * (HiddenSize / kVec);
    auto* dst         = reinterpret_cast<out_vec*>(activation_quant) +
                static_cast<size_t>(token) * (HiddenSize / kVec);
    in_vec v[kVecsPerLane];
#pragma unroll
    for(int i = 0; i < kVecsPerLane; ++i)
        v[i] = src[lane + i * kWaveSize];

    float amax = 0.f;
#pragma unroll
    for(int i = 0; i < kVecsPerLane; ++i)
    {
#pragma unroll
        for(int j = 0; j < kVec; ++j)
            amax = max(amax, abs(static_cast<float>(v[i][j])));
    }
#pragma unroll
    for(int offset = kWaveSize / 2; offset > 0; offset >>= 1)
        amax = max(amax, __shfl_xor(amax, offset));
    const float row_scale      = amax * inverted_fp8_max;
    const float inverted_scale = __builtin_amdgcn_rcpf(row_scale);
    if(lane == 0)
        activation_scale[token] = row_scale;

#pragma unroll
    for(int i = 0; i < kVecsPerLane; ++i)
    {
        out_vec q;
#pragma unroll
        for(int j = 0; j < kVec; j += 2)
            q[j / 2] = fp32x2_to_fp8x2_scaled(
                static_cast<float>(v[i][j]), static_cast<float>(v[i][j + 1]), inverted_scale);
        dst[lane + i * kWaveSize] = q;
    }
}

// CTA 0 sorts; CTAs 1.. zero moe_buf and, with kQuant, quantize one token per
// wave. Sorting and quantization share no data, so they run concurrently.
template <int NumExperts, int TopK, int HiddenSize, int BlockM, bool kQuant>
__global__ void __launch_bounds__(kThreads)
    a16w4_moe_sort_quant_kernel(const opus::bf16_t* hidden_states,
                                const int32_t* topk_ids,
                                const float* topk_weights,
                                int32_t* sorted_token_ids,
                                float* sorted_weights,
                                int32_t* sorted_expert_ids,
                                int32_t* num_valid_ids,
                                int4* moe_buf,
                                int64_t moe_buf_vecs,
                                opus::fp8_t* activation_quant,
                                float* activation_scale,
                                int tokens,
                                const int32_t* num_local_tokens)
{
    if(num_local_tokens != nullptr)
        tokens = max(0, min(tokens, *num_local_tokens));
    if(blockIdx.x == 0)
    {
        sort_routes<NumExperts, TopK, BlockM>(topk_ids,
                                              topk_weights,
                                              sorted_token_ids,
                                              sorted_weights,
                                              sorted_expert_ids,
                                              num_valid_ids,
                                              tokens);
        return;
    }

    const int worker = blockIdx.x - 1;
    if constexpr(kQuant)
    {
        const int wave  = static_cast<int>(threadIdx.x) / kWaveSize;
        const int token = worker * kTokensPerCta + wave;
        if(wave < kTokensPerCta && token < tokens)
            quantize_token_fp8<HiddenSize>(hidden_states, activation_quant, activation_scale, token);
    }

    const int64_t step = static_cast<int64_t>(gridDim.x - 1) * kThreads;
    const int4 zero    = {0, 0, 0, 0};
    for(int64_t i = static_cast<int64_t>(worker) * kThreads + threadIdx.x; i < moe_buf_vecs;
        i += step)
        moe_buf[i] = zero;
}

} // namespace aiter::a16w4_moe

void a16w4_moe_sort_quant_fwd(aiter_tensor_t& hidden_states,
                              aiter_tensor_t& topk_ids,
                              aiter_tensor_t& topk_weights,
                              aiter_tensor_t& sorted_token_ids,
                              aiter_tensor_t& sorted_weights,
                              aiter_tensor_t& sorted_expert_ids,
                              aiter_tensor_t& num_valid_ids,
                              aiter_tensor_t& moe_buf,
                              int num_experts,
                              int block_m,
                              std::optional<aiter_tensor_t> activation_quant,
                              std::optional<aiter_tensor_t> activation_scale,
                              std::optional<aiter_tensor_t> num_local_tokens)
{
    using namespace aiter::a16w4_moe;

    const bool quant = activation_quant.has_value();
    AITER_CHECK(quant == activation_scale.has_value(),
                __func__,
                ": activation_quant and activation_scale go together");
    const int device_id = hidden_states.device_id;
    AITER_CHECK(topk_ids.device_id == device_id && topk_weights.device_id == device_id &&
                    sorted_token_ids.device_id == device_id &&
                    sorted_weights.device_id == device_id &&
                    sorted_expert_ids.device_id == device_id &&
                    num_valid_ids.device_id == device_id &&
                    (moe_buf.numel() == 0 || moe_buf.device_id == device_id) &&
                    (!quant || (activation_quant->device_id == device_id &&
                                activation_scale->device_id == device_id)),
                __func__,
                ": all tensors must be on the same device");
    AITER_CHECK(hidden_states.dtype() == AITER_DTYPE_bf16 && hidden_states.dim() == 2,
                __func__,
                ": hidden_states must be 2-D BF16");
    AITER_CHECK(topk_ids.dtype() == AITER_DTYPE_i32 && topk_weights.dtype() == AITER_DTYPE_fp32 &&
                    sorted_token_ids.dtype() == AITER_DTYPE_i32 &&
                    sorted_weights.dtype() == AITER_DTYPE_fp32 &&
                    sorted_expert_ids.dtype() == AITER_DTYPE_i32 &&
                    num_valid_ids.dtype() == AITER_DTYPE_i32 && num_valid_ids.numel() >= 2,
                __func__,
                ": expected int32 ids and fp32 weights");
    AITER_CHECK(hidden_states.is_contiguous() && topk_ids.is_contiguous() &&
                    topk_weights.is_contiguous() && sorted_token_ids.is_contiguous() &&
                    sorted_weights.is_contiguous() && sorted_expert_ids.is_contiguous() &&
                    moe_buf.is_contiguous() &&
                    (!quant || (activation_quant->is_contiguous() &&
                                activation_scale->is_contiguous())),
                __func__,
                ": all tensors must be contiguous");

    const int tokens     = static_cast<int>(hidden_states.size(0));
    const int model_dim  = static_cast<int>(hidden_states.size(1));
    const int topk       = topk_ids.dim() == 2 ? static_cast<int>(topk_ids.size(1)) : 0;
    AITER_CHECK(tokens >= 1 && tokens <= kMaxTokens && topk_ids.size(0) == tokens &&
                    topk_weights.numel() == topk_ids.numel(),
                __func__,
                ": only decode M in [1, 128] is supported");
    AITER_CHECK(block_m == 16 || block_m == 32, __func__, ": block_m must be 16 or 32");

#define AITER_A16W4_SORT_QUANT_MATCH(E, K, D) \
    || (num_experts == (E) && topk == (K) && model_dim == (D))
    const bool shape_supported = false AITER_A16W4_SORT_QUANT_SHAPES(AITER_A16W4_SORT_QUANT_MATCH);
#undef AITER_A16W4_SORT_QUANT_MATCH
    AITER_CHECK(shape_supported,
                __func__,
                ": (num_experts, topk, model_dim) must be (896, 16, 3584) or (384, 6, 5120)");

    const int64_t routes         = topk_ids.numel();
    const int64_t active_experts = std::min<int64_t>(routes, num_experts);
    const int64_t required_sorted =
        (routes + active_experts * (block_m - 1) + block_m - 1) / block_m * block_m;
    AITER_CHECK(static_cast<int64_t>(sorted_token_ids.numel()) >= required_sorted &&
                    static_cast<int64_t>(sorted_weights.numel()) >= required_sorted &&
                    static_cast<int64_t>(sorted_expert_ids.numel()) >= required_sorted / block_m,
                __func__,
                ": output routing buffers are too small");

    int64_t moe_buf_vecs = 0;
    if(moe_buf.numel() != 0)
    {
        AITER_CHECK((moe_buf.dtype() == AITER_DTYPE_bf16 || moe_buf.dtype() == AITER_DTYPE_fp16) &&
                        static_cast<int64_t>(moe_buf.numel()) >=
                            static_cast<int64_t>(tokens) * model_dim,
                    __func__,
                    ": moe_buf must be empty or hold at least M*model_dim BF16/FP16 elements");
        moe_buf_vecs = static_cast<int64_t>(tokens) * model_dim * 2 / sizeof(int4);
    }
    if(quant)
    {
        AITER_CHECK(activation_quant->dtype() == AITER_DTYPE_fp8 &&
                        activation_quant->numel() == hidden_states.numel() &&
                        activation_scale->dtype() == AITER_DTYPE_fp32 &&
                        static_cast<int64_t>(activation_scale->numel()) >= tokens,
                    __func__,
                    ": expected fp8 [M, model_dim] activation_quant and fp32 [M, 1] activation_scale");
    }

    HipDeviceGuard device_guard(device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    const dim3 grid(1 + (tokens + kTokensPerCta - 1) / kTokensPerCta);

    const auto* hidden_ptr  = static_cast<const opus::bf16_t*>(hidden_states.data_ptr());
    const auto* ids_ptr     = static_cast<const int32_t*>(topk_ids.data_ptr());
    const auto* weights_ptr = static_cast<const float*>(topk_weights.data_ptr());
    auto* sorted_ids_ptr    = static_cast<int32_t*>(sorted_token_ids.data_ptr());
    auto* sorted_w_ptr      = static_cast<float*>(sorted_weights.data_ptr());
    auto* sorted_e_ptr      = static_cast<int32_t*>(sorted_expert_ids.data_ptr());
    auto* num_valid_ptr     = static_cast<int32_t*>(num_valid_ids.data_ptr());
    auto* moe_buf_ptr       = moe_buf_vecs == 0 ? nullptr : static_cast<int4*>(moe_buf.data_ptr());
    auto* a_q_ptr = quant ? static_cast<opus::fp8_t*>(activation_quant->data_ptr()) : nullptr;
    auto* a_s_ptr = quant ? static_cast<float*>(activation_scale->data_ptr()) : nullptr;
    const auto* local_ptr =
        num_local_tokens.has_value() ? static_cast<const int32_t*>(num_local_tokens->data_ptr())
                                     : nullptr;

    auto launch = [&](auto kernel) {
        kernel<<<grid, dim3(kThreads), 0, stream>>>(hidden_ptr,
                                                    ids_ptr,
                                                    weights_ptr,
                                                    sorted_ids_ptr,
                                                    sorted_w_ptr,
                                                    sorted_e_ptr,
                                                    num_valid_ptr,
                                                    moe_buf_ptr,
                                                    moe_buf_vecs,
                                                    a_q_ptr,
                                                    a_s_ptr,
                                                    tokens,
                                                    local_ptr);
    };
#define AITER_A16W4_SORT_QUANT_LAUNCH(E, K, D)                                              \
    if(num_experts == (E) && topk == (K) && model_dim == (D))                               \
    {                                                                                       \
        if(block_m == 16)                                                                   \
            quant ? launch(a16w4_moe_sort_quant_kernel<(E), (K), (D), 16, true>)            \
                  : launch(a16w4_moe_sort_quant_kernel<(E), (K), (D), 16, false>);          \
        else                                                                                \
            quant ? launch(a16w4_moe_sort_quant_kernel<(E), (K), (D), 32, true>)            \
                  : launch(a16w4_moe_sort_quant_kernel<(E), (K), (D), 32, false>);          \
    }
    AITER_A16W4_SORT_QUANT_SHAPES(AITER_A16W4_SORT_QUANT_LAUNCH)
#undef AITER_A16W4_SORT_QUANT_LAUNCH

    const hipError_t launch_status = hipGetLastError();
    AITER_CHECK(launch_status == hipSuccess, __func__, ": a16w4 sort+quant launch failed");
}
