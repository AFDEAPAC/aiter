// aiter op entry for the AVO fp32 per-row top-k kernels.
//
// scripts/export_aiter_op.py appends this file verbatim after the exported
// kernel region, so the entry and the kernels it calls live in one repo and
// move in one diff. Do not add anything here that the benchmark harness needs:
// this side is only reachable from aiter.
//
// The contract is mirrored from top_k_per_row_prefill
// (aiter csrc/kernels/topk_per_row_kernels.cu): host-side shape decisions key
// off `stride0`, the row pitch, because the per-row extents live on the device
// in rowEnds and the dispatcher has to pick a path, a block size and a grid
// without reading device memory back.

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"
#include <optional>

namespace avo {

// One caller-provided buffer, carved the way alloc_bufs() carves its separate
// hipMallocs. This is the single definition of that layout: workspace_bytes()
// publishes the total to Python and bind_bufs() reads the offsets, so the size
// Python allocates and the size the kernels address cannot disagree.
struct WsLayout {
  size_t threshold, threshold_f, cand_pack, cand_seg;
  size_t cand_count, cand_reserved, cand_bad, fb_rows, fb_count;
  size_t total;
};

static inline WsLayout ws_layout(int M, int cap) {
  // 256 B keeps every field on a dwordx4-friendly boundary regardless of M.
  auto align256 = [](size_t x) { return (x + 255u) & ~(size_t)255u; };
  const size_t row_slots = (size_t)std::max(CAND_SLOTS_PER_ROW, cap);
  size_t o = 0;
  auto take = [&](size_t bytes) {
    const size_t here = o;
    o = align256(o + bytes);
    return here;
  };
  WsLayout L{};
  L.threshold = take((size_t)M * sizeof(uint32_t));
  L.threshold_f = take((size_t)M * sizeof(float));
  L.cand_pack = take((size_t)M * row_slots * sizeof(uint64_t));
  L.cand_seg = take((size_t)M * MAX_WAVES_PER_BLOCK * sizeof(int));
  L.cand_count = take((size_t)M * sizeof(unsigned int));
  L.cand_reserved = take((size_t)M * sizeof(unsigned int));
  L.cand_bad = take((size_t)M * sizeof(unsigned int));
  L.fb_rows = take((size_t)M * sizeof(int));
  L.fb_count = take(sizeof(int));
  L.total = o;
  return L;
}

// Derived without consulting the harness globals, and in particular without
// topk_indices(), which assigns the derived S back into g_sample_s. In a
// library that write is a bug: the next call on a different shape would be
// handed the previous shape's S as an override and derive different parameters,
// making the result depend on call order.
// Every call here is ragged, so the geometry is sized by geometry_k_ragged()
// (topk_shape.hip.hpp) rather than the caller's k.
static inline ShapeParams params_for(int M, int N, int K) {
  return derive_shape_params(M, N, geometry_k_ragged(K, N), 0.0f, 0, 0, PATH_AUTO);
}

static inline Bufs bind_bufs(void* ws, const WsLayout& L, int cap) {
  char* base = static_cast<char*>(ws);
  Bufs b{};
  b.threshold = reinterpret_cast<uint32_t*>(base + L.threshold);
  b.threshold_f = reinterpret_cast<float*>(base + L.threshold_f);
  b.cand_pack = reinterpret_cast<uint64_t*>(base + L.cand_pack);
  b.cand_seg = reinterpret_cast<int*>(base + L.cand_seg);
  b.cand_count = reinterpret_cast<unsigned int*>(base + L.cand_count);
  b.cand_reserved = reinterpret_cast<unsigned int*>(base + L.cand_reserved);
  b.cand_bad = reinterpret_cast<unsigned int*>(base + L.cand_bad);
  b.fb_rows = reinterpret_cast<int*>(base + L.fb_rows);
  b.fb_count = reinterpret_cast<int*>(base + L.fb_count);
  b.C_alloc = cap;
  return b;
}

} // namespace avo

// Published to Python so the caller allocates the scratch (aiter's rule: host
// code here never allocates device memory). Plain scratch, not zeroed: Phase A
// clears the reservation counters and fb_count before any consumer reads them,
// Phase B assigns cand_count rather than accumulating into it, and the small_n
// path touches none of them.
int64_t topk_avo_workspace_size(int64_t numRows, int64_t stride0, int64_t k)
{
    if(numRows <= 0)
        return 1;
    const int M     = static_cast<int>(numRows);
    const int N     = static_cast<int>(stride0);
    const int K     = static_cast<int>(k);
    const auto sp   = avo::params_for(M, N, K);
    const int cap   = sp.path == PATH_SMALL_N ? CAND_SLOTS_PER_ROW : sp.cap;
    return static_cast<int64_t>(avo::ws_layout(M, cap).total);
}

// Reports whether this op can serve the shape at all, so Python can route
// around it instead of taking an AITER_CHECK abort. Kept in C++ because every
// term it tests (LDS residency, dwordx4 geometry, the Phase C LDS cap) is a
// property of these kernels, not of the caller.
bool topk_avo_supports(int64_t numRows, int64_t stride0, int64_t k)
{
    if(numRows <= 0 || stride0 <= 0 || k <= 0)
        return false;
    if(k > PHASE_C_CAP_MAX)
        return false;
    if(stride0 % FP32_EPT != 0)
        return false;
    return avo::params_for(
               static_cast<int>(numRows), static_cast<int>(stride0), static_cast<int>(k))
        .geom_ok;
}

// rowStarts[row] and rowEnds[row] bound [start, end); indices are absolute columns.
void top_k_per_row_prefill_avo(const aiter_tensor_t& logits,
                               const aiter_tensor_t& rowStarts,
                               const aiter_tensor_t& rowEnds,
                               aiter_tensor_t& indices,
                               std::optional<aiter_tensor_t> values,
                               int64_t numRows,
                               int64_t stride0,
                               int64_t stride1,
                               int64_t k                               = 2048,
                               std::optional<aiter_tensor_t> workspace = std::nullopt)
{
    if(numRows <= 0)
        return;

    AITER_CHECK(logits.dtype() == AITER_DTYPE_fp32,
                "top_k_per_row_prefill_avo: logits must be fp32");
    AITER_CHECK(stride1 == 1, "top_k_per_row_prefill_avo: logits inner stride must be 1");
    AITER_CHECK(indices.dtype() == AITER_DTYPE_i32,
                "top_k_per_row_prefill_avo: indices must be int32");
    AITER_CHECK(indices.numel() >= static_cast<size_t>(numRows) * static_cast<size_t>(k),
                "top_k_per_row_prefill_avo: indices must hold numRows*k entries");
    if(values.has_value())
    {
        AITER_CHECK(values.value().dtype() == AITER_DTYPE_fp32,
                    "top_k_per_row_prefill_avo: values must be fp32");
        AITER_CHECK(values.value().numel() >=
                        static_cast<size_t>(numRows) * static_cast<size_t>(k),
                    "top_k_per_row_prefill_avo: values must hold numRows*k entries");
    }
    AITER_CHECK(rowStarts.numel() >= static_cast<size_t>(numRows) &&
                    rowEnds.numel() >= static_cast<size_t>(numRows),
                "top_k_per_row_prefill_avo: rowStarts/rowEnds must have numRows entries");
    AITER_CHECK(topk_avo_supports(numRows, stride0, k),
                "top_k_per_row_prefill_avo: unsupported shape (numRows=%ld stride0=%ld k=%ld); "
                "ask topk_avo_supports() first",
                (long)numRows,
                (long)stride0,
                (long)k);
    AITER_CHECK(workspace.has_value(),
                "top_k_per_row_prefill_avo requires a caller-provided workspace "
                "(see top_k_per_row_prefill_avo in topk.py)");

    const int M = static_cast<int>(numRows);
    const int N = static_cast<int>(stride0);
    const int K = static_cast<int>(k);

    HipDeviceGuard device_guard(logits.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    const ShapeParams sp = avo::params_for(M, N, K);
    const float* in      = static_cast<const float*>(logits.data_ptr());
    const int* row_starts = static_cast<const int*>(rowStarts.data_ptr());
    const int* row_ends   = static_cast<const int*>(rowEnds.data_ptr());
    int* idx             = static_cast<int*>(indices.data_ptr());
    float* val = values.has_value() ? static_cast<float*>(values.value().data_ptr()) : nullptr;

    if(sp.path == PATH_SMALL_N)
    {
        if(val)
            topk_small_n<true, true>(in, M, N, row_starts, row_ends, K, idx, val, stream);
        else
            topk_small_n<true, false>(in, M, N, row_starts, row_ends, K, idx, nullptr, stream);
        return;
    }

    const auto L = avo::ws_layout(M, sp.cap);
    AITER_CHECK(workspace.value().numel() * workspace.value().element_size() >= L.total,
                "top_k_per_row_prefill_avo: workspace is %zu B, needs %zu B",
                workspace.value().numel() * workspace.value().element_size(),
                L.total);
    Bufs b = avo::bind_bufs(workspace.value().data_ptr(), L, sp.cap);
    if(val)
        topk_fused_impl<true, true>(in, M, N, row_starts, row_ends, K, idx, val, b, sp, stream);
    else
        topk_fused_impl<true, false>(in, M, N, row_starts, row_ends, K, idx, nullptr, b, sp, stream);
}
