// SPDX-License-Identifier: MIT
// Kimi-K3 decode MoE preamble in one launch: what moe_sorting (opus, accumulate) and
// per_token_quant_hip (fp8) produce, for T <= 32 tokens.
//   blocks [0, T): dynamic per-token fp8 quant of hidden row t + zero moe_buf row t
//   block  T     : counting sort of topk ids into the opus sorted-ids contract
#define AITER_NO_TORCH_TYPES
#include "aiter_hip_common.h"
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "aiter_opus_plus.h"
#include <hip/hip_runtime.h>
#include <tuple>

namespace aiter {
namespace k3rp {

constexpr int32_t BlockSize = 256;
constexpr int32_t kMaxExperts = 1024;
constexpr int32_t kMaxTokens = 32;

// Verbatim from quant_kernels.cu data_to_per_row_scale / scaled_quant_vgpr_impl (thread_data_size
// != 0 path), so the fp8 bytes and scales match dynamic_per_token_scaled_quant bit for bit.
template <typename DTYPE_I, typename DTYPE_O, int thread_data_size>
__device__ std::tuple<float, DTYPE_I*> row_scale(const DTYPE_I* __restrict__ input, const int32_t cols)
{
    static constexpr int32_t vec_size_i = thread_data_size;
    static constexpr int32_t load_chunk_bytes = sizeof(DTYPE_I) * vec_size_i % 16 == 0 ? 16 : (sizeof(DTYPE_I) * vec_size_i % 8 == 0 ? 8 : 4);
    using vec_i = opus::vector_t<DTYPE_I, vec_size_i>;
    const float inverted_DTYPE_MAX =
        (1. / static_cast<float>(opus::finfo<DTYPE_O>::max()));

    const int64_t row_offset        = blockIdx.x * cols;
    auto const* ptr_i               = reinterpret_cast<DTYPE_I const*>(input + row_offset);
    static constexpr int32_t ooba_i = 4 / sizeof(DTYPE_I);
    const int32_t oob_i             = (cols + ooba_i - 1) / ooba_i * ooba_i;
    auto buffer_i = opus::make_gmem<DTYPE_I>(ptr_i, oob_i * sizeof(DTYPE_I));

    const int32_t num_vecs = (cols + vec_size_i - 1) / vec_size_i;

    vec_i vec_cur;
    size_t vec_idx = threadIdx.x;
    if(vec_idx < num_vecs)
    {
        vec_cur = load_vector_nbytes<DTYPE_I, vec_size_i, load_chunk_bytes>(buffer_i, vec_idx * vec_size_i);
    }

    float absMax = 0.f;
    if(vec_idx < num_vecs)
    {
#pragma unroll
        for(size_t j = 0; j < vec_size_i; j++)
        {
            absMax = max(absMax, abs(static_cast<float>(vec_cur[j])));
        }
    }

    absMax = block_reduce<float, aiter::Max, BlockSize, true>(absMax, aiter::Max());

    float scale = absMax * inverted_DTYPE_MAX;
    return std::make_tuple(scale, reinterpret_cast<DTYPE_I*>(&vec_cur));
}

template <typename DTYPE_I, typename DTYPE_O, int thread_data_size>
__device__ void quant_vgpr(DTYPE_O* __restrict__ out,
                           DTYPE_I* __restrict__ input,
                           const float* __restrict__ scale,
                           const int cols,
                           int64_t out_offset)
{
    const float inverted_scale          = __builtin_amdgcn_rcpf(*scale);
    static constexpr int32_t vec_size_i = thread_data_size;
    static constexpr int32_t vec_size_o = vec_size_i;

    using DTYPE_STORE = DTYPE_O;
    using vec_i       = opus::vector_t<DTYPE_I, vec_size_i>;

    auto const* input_vecs          = reinterpret_cast<vec_i const*>(input);
    auto* ptr_o                     = reinterpret_cast<DTYPE_STORE*>(out + out_offset);
    static constexpr int32_t ooba_o = 4 / sizeof(DTYPE_O);
    const int32_t oob_o             = (cols + ooba_o - 1) / ooba_o * ooba_o;

    auto buffer_o = opus::make_gmem<DTYPE_STORE>(ptr_o, oob_o * sizeof(DTYPE_STORE));
    const int32_t num_vecs = (cols + vec_size_i - 1) / vec_size_i;

    if(threadIdx.x < num_vecs)
    {
        store_vector<DTYPE_STORE, DTYPE_I, thread_data_size, RT, false, WARP_SIZE, 1, DTYPE_O>(buffer_o, *input_vecs, threadIdx.x * vec_size_o, inverted_scale);
    }
}

template <typename DTYPE_I, typename DTYPE_O, int thread_data_size>
__global__ void __launch_bounds__(BlockSize)
route_prep_kernel(const int32_t* __restrict__ topk_ids,
                  const float* __restrict__ topk_weights,
                  DTYPE_I* __restrict__ hidden,
                  int32_t* __restrict__ sorted_ids,
                  float* __restrict__ sorted_weights,
                  int32_t* __restrict__ sorted_expert_ids,
                  int32_t* __restrict__ num_valid_ids,
                  uint32_t* __restrict__ moe_buf,
                  DTYPE_O* __restrict__ a_q,
                  float* __restrict__ a_scale,
                  const int32_t tokens,
                  const int32_t topk,
                  const int32_t num_experts,
                  const int32_t unit,
                  const int32_t cols,
                  const int32_t moe_buf_row_words)
{
    const int tid = threadIdx.x;
    if(blockIdx.x < tokens)
    {
        auto res        = row_scale<DTYPE_I, DTYPE_O, thread_data_size>(hidden, cols);
        float scale     = std::get<0>(res);
        DTYPE_I* vec_ptr = std::get<1>(res);
        if(tid == 0)
            a_scale[blockIdx.x] = scale;
        quant_vgpr<DTYPE_I, DTYPE_O, thread_data_size>(a_q, vec_ptr, &scale, cols, int64_t(blockIdx.x) * cols);
        uint32_t* row = moe_buf + int64_t(blockIdx.x) * moe_buf_row_words;
        for(int i = tid; i < moe_buf_row_words; i += BlockSize)
            row[i] = 0u;
        return;
    }

    // Opus contract: experts ascending, tokens ascending inside an expert, each expert padded to a
    // multiple of `unit` with id (topk << 24 | tokens) and weight 0, entries encoded token | slot << 24,
    // num_valid_ids = {padded total, tokens}, nothing written past the padded total.
    __shared__ uint32_t s_mask[kMaxExperts];
    __shared__ int32_t s_off[kMaxExperts];
    __shared__ int32_t s_wave[BlockSize / 64];

    for(int e = tid; e < kMaxExperts; e += BlockSize)
        s_mask[e] = 0u;
    __syncthreads();

    const int routes = tokens * topk;
    for(int r = tid; r < routes; r += BlockSize)
    {
        const int e = topk_ids[r];
        if(static_cast<uint32_t>(e) < static_cast<uint32_t>(num_experts))
            atomicOr(&s_mask[e], 1u << (r / topk));
    }
    __syncthreads();

    constexpr int kPer = kMaxExperts / BlockSize;
    int cnt[kPer], pad[kPer];
    int local = 0;
#pragma unroll
    for(int i = 0; i < kPer; ++i)
    {
        const int e = tid * kPer + i;
        cnt[i]      = e < num_experts ? __popc(s_mask[e]) : 0;
        pad[i]      = (cnt[i] + unit - 1) / unit * unit;
        local += pad[i];
    }

    const int lane = tid & 63, wid = tid >> 6;
    int incl = local;
#pragma unroll
    for(int o = 1; o < 64; o <<= 1)
    {
        const int y = __shfl_up(incl, o, 64);
        if(lane >= o)
            incl += y;
    }
    if(lane == 63)
        s_wave[wid] = incl;
    __syncthreads();
    int run = incl - local, total = 0;
#pragma unroll
    for(int w = 0; w < BlockSize / 64; ++w)
    {
        run += w < wid ? s_wave[w] : 0;
        total += s_wave[w];
    }

    const int pad_id = (topk << 24) | tokens;
#pragma unroll
    for(int i = 0; i < kPer; ++i)
    {
        const int e = tid * kPer + i;
        if(pad[i] > 0)
        {
            s_off[e] = run;
            for(int b = 0; b < pad[i]; b += unit)
                sorted_expert_ids[(run + b) / unit] = e;
            for(int j = cnt[i]; j < pad[i]; ++j)
            {
                sorted_ids[run + j]     = pad_id;
                sorted_weights[run + j] = 0.f;
            }
        }
        run += pad[i];
    }
    if(tid == 0)
    {
        num_valid_ids[0] = total;
        num_valid_ids[1] = tokens;
    }
    __syncthreads();

    for(int r = tid; r < routes; r += BlockSize)
    {
        const int e = topk_ids[r];
        if(static_cast<uint32_t>(e) >= static_cast<uint32_t>(num_experts))
            continue;
        const int t          = r / topk;
        const int pos        = s_off[e] + __popc(s_mask[e] & ((1u << t) - 1u));
        sorted_ids[pos]      = t | ((r - t * topk) << 24);
        sorted_weights[pos]  = topk_weights[r];
    }
}

} // namespace k3rp
} // namespace aiter

AITER_CTYPES_ERROR_DEF

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    k3_moe_route_prep,
    (aiter_tensor_t* topk_ids,          // [T, K] int32
     aiter_tensor_t* topk_weights,      // [T, K] fp32
     aiter_tensor_t* hidden,            // [T, D] bf16, contiguous
     aiter_tensor_t* sorted_ids,        // [T*K + E*unit - K] int32
     aiter_tensor_t* sorted_weights,    // same length, fp32
     aiter_tensor_t* sorted_expert_ids, // [ceil(len/unit)] int32
     aiter_tensor_t* num_valid_ids,     // [2] int32
     aiter_tensor_t* moe_buf,           // [T, D'] 16-bit, zeroed here
     aiter_tensor_t* a_q,               // [T, D] fp8
     aiter_tensor_t* a_scale,           // [T, 1] fp32
     int64_t num_experts,
     int64_t unit_size,
     hipStream_t stream),
    (topk_ids, topk_weights, hidden, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
     moe_buf, a_q, a_scale, num_experts, unit_size, stream))
{
    using namespace aiter::k3rp;
    const int tokens = static_cast<int>(topk_ids->size(0));
    const int topk   = static_cast<int>(topk_ids->size(1));
    const int cols   = static_cast<int>(hidden->size(-1));
    AITER_CHECK(tokens >= 1 && tokens <= kMaxTokens, "k3_moe_route_prep: tokens must be in [1, 32]");
    AITER_CHECK(num_experts >= 1 && num_experts <= kMaxExperts, "k3_moe_route_prep: num_experts must be <= 1024");
    AITER_CHECK(topk >= 1 && topk <= 255 && unit_size >= 1, "k3_moe_route_prep: bad topk/unit");
    AITER_CHECK(topk_ids->is_contiguous() && topk_weights->is_contiguous() && hidden->is_contiguous() &&
                    moe_buf->is_contiguous() && a_q->is_contiguous(),
                "k3_moe_route_prep: tensors must be contiguous");
    AITER_CHECK(topk_ids->dtype() == AITER_DTYPE_i32 && topk_weights->dtype() == AITER_DTYPE_fp32 &&
                    hidden->dtype() == AITER_DTYPE_bf16 && a_q->dtype() == AITER_DTYPE_fp8,
                "k3_moe_route_prep: dtype mismatch");
    AITER_CHECK(cols > 8 * BlockSize && cols <= 16 * BlockSize,
                "k3_moe_route_prep: hidden dim outside the thread_data_size=16 band");
    AITER_CHECK(moe_buf->size(0) >= tokens && (moe_buf->size(-1) * moe_buf->element_size()) % 4 == 0,
                "k3_moe_route_prep: moe_buf shape");
    const int row_words = static_cast<int>(moe_buf->size(-1) * moe_buf->element_size() / 4);

    using DI = opus::bf16_t;
    using DO = opus::fp8_t;
    aiter::k3rp::route_prep_kernel<DI, DO, 16><<<dim3(tokens + 1), dim3(BlockSize), 0, stream>>>(
        reinterpret_cast<const int32_t*>(topk_ids->data_ptr()),
        reinterpret_cast<const float*>(topk_weights->data_ptr()),
        reinterpret_cast<DI*>(hidden->data_ptr()),
        reinterpret_cast<int32_t*>(sorted_ids->data_ptr()),
        reinterpret_cast<float*>(sorted_weights->data_ptr()),
        reinterpret_cast<int32_t*>(sorted_expert_ids->data_ptr()),
        reinterpret_cast<int32_t*>(num_valid_ids->data_ptr()),
        reinterpret_cast<uint32_t*>(moe_buf->data_ptr()),
        reinterpret_cast<DO*>(a_q->data_ptr()),
        reinterpret_cast<float*>(a_scale->data_ptr()),
        tokens,
        topk,
        static_cast<int>(num_experts),
        static_cast<int>(unit_size),
        cols,
        row_words);
}
