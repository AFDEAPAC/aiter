// benchmark_topk.hip.cpp — fp32 per-row top-k indices for prefill.
// Contract: input fp32 [M,N], output int32 indices [M,K]. Target M=4096 N=131072 K=2048.
//
// Fused pipeline (default, --pipeline fused), 4 kernel launches per call:
//   A  phase_a_threshold : one block per row, samples SAMPLE_S contiguous-chunk elements,
//                          selects rank R entirely in LDS -> per-row threshold.
//   B  phase_b_filter    : the whole budget. Streams the row with dwordx4, ONE integer
//                          compare per element, wave64 ballot compression, one global
//                          atomic per wave, appends (key,index) to a per-row candidate area.
//   C  phase_c_select    : one block per row, loads the candidate set into LDS and does the
//                          exact 4x8-bit radix select + tie-correct gather in LDS.
//   D  phase_d_fallback  : rows whose candidate set is unusable (too few OR overflowed)
//                          are recomputed exactly by streaming the full row.
//
// --pipeline direct runs D over every row. That is the independent full-row oracle and is
// the same code the fallback uses, so both timed paths are separately verifiable.

#include "topk_common.hip.hpp"
#include "topk_shape.hip.hpp"

// ---- AVO variation surface -------------------------------------------------
static int g_sample_rank = 0;       // 0 => derive from margin
static int g_sample_s = 0;          // 0 => derive from shape (R_TARGET law)
static float g_margin = 0.0f;       // 0 => derive from the estimator's own noise

// The candidate count is the number of row elements above the rank-R value of S
// samples, so its spread is that estimator's noise: std ~ count/sqrt(R). A row
// undershoots K (and pays the exact fallback) when count < K, so the required
// over-collection factor is set by how noisy R is, NOT by a constant.
//
// Measured at S=8192, margin 1.4, N=131072 (min/mean, rows under K out of 4096):
//   K=2048 R=179  0.74  0 rows
//   K=1024 R= 89  0.67  4 rows
//   K= 512 R= 44  0.52 82 rows
// i.e. a fixed 1.4 is overfitted to K=2048. Requiring mean*(1 - 3/sqrt(R0)) > K
// with R0 = K*S/N (the margin-free rank) reproduces 1.36 at K=2048 and demands
// 2.13 at K=512, which is what the data shows.
// auto_margin() lives in topk_generalize.hip.hpp
static int g_cf_block = 512;        // Phase B block size
static int g_cf_gx = 16;            // Phase B blocks per row (grid.x)
static int g_use_nt_load = 0;       // non-temporal streaming loads in Phase B
// Phase B implementation: 3 = wave-private regions writing straight to global,
// 4 = same but passers staged in LDS and flushed in full contiguous bursts.
// Variants 0-2 (per-wave global atomic, block-aggregated atomic, LDS counter)
// were measured and superseded; see knowledge/known_bad.md and git history.
static int g_phase_b = 4;
static int g_phase_c_block = 0;     // 0 => derive from occupancy
static int g_phase_a_block = 0;     // 0 => derive from occupancy
// 3 passes give bit-identical candidate counts to 4 (the 4th byte never moves the
// bucket at fp32 precision) and 3 is safer than 2, whose spread ran to max=3919
// against C_alloc=4096.
static int g_phase_a_passes = 3;
static int g_pipeline_direct = 0;
static int g_inject_fault = 0;
static int g_dump_stats = 0;
static int g_ablate_store = 0;   // diagnostic only: produces WRONG results
// Phase C must use all 4 passes to be exact. Fewer is a TIMING ABLATION ONLY.
static int g_phase_c_passes = RADIX_PASSES;
static int g_path_override = PATH_AUTO;
static int g_coop_g = 0;
static int g_fuse_ab = 0;
static int g_use_hipgraph = 0;
static int g_small_n_block = 0;     // 0 => derive from the vec4 load count
static int g_small_n_passes = RADIX_PASSES;   // < 4 is a TIMING ABLATION (wrong results)
static int g_verify_sample_rows = 32;
static int g_verify_oracle_gpu = 1;

// ---------------------------------------------------------------------------
// Block-wide exact radix select over keys already resident in LDS.
// On return: pivot == the K-th largest sortable key, eq_needed == how many
// elements equal to pivot must be taken (so K - eq_needed are strictly greater).
// ---------------------------------------------------------------------------
// npasses < RADIX_PASSES yields a pivot truncated to the leading 8*npasses bits.
// Phase C and the fallback MUST use all 4 (their result is the answer). Phase A
// may use fewer: its pivot is only a filter threshold, and Phase C still does
// the exact select, so the only consequence is a shift in how many candidates
// Phase B collects.
__device__ __forceinline__ void block_select_lds(const uint32_t* __restrict__ s_keys, int c, int K,
                                                 uint32_t* __restrict__ s_hist,
                                                 uint32_t* __restrict__ s_red,
                                                 uint32_t* __restrict__ s_scan,
                                                 uint32_t* __restrict__ s_mm, uint32_t& pivot,
                                                 int& eq_needed, int npasses = RADIX_PASSES,
                                                 bool prefix_skip = false) {
  const int rep = threadIdx.x & (HIST_REP - 1);
  if (threadIdx.x == 0) {
    s_scan[0] = 0;
    s_scan[1] = 0;
  }

  // Phase C's candidates are all >= the Phase B threshold, so they share a high
  // prefix and the passes where min and max already agree can be skipped
  // outright. Phase A must NOT do this: its samples span the whole row, so no
  // pass is ever skipped and the min/max reduction is pure loss (measured
  // phase_c -6.0 us, phase_a +8.0 us).
  int start = 0;
  pivot = 0;
  if (prefix_skip) {
    uint32_t mn, mx;
    block_minmax_lds(s_keys, c, s_mm, mn, mx);
    start = common_prefix_passes(mn, mx);
    if (start >= RADIX_PASSES) {
      // Every key identical: the pivot is that value and all K come from ties.
      pivot = mn;
      eq_needed = K;
      return;
    }
    pivot = mn & ((start == 0) ? 0u : (0xFFFFFFFFu << (32 - 8 * start)));
  }

  int ek = K;
  for (int p = start; p < npasses; p++) {
    const int sh = radix_shift(p);
    const int hshift = sh + 8;
    const bool filter = (p > 0);
    for (int i = threadIdx.x; i < HIST_SLOTS; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();
    // Do NOT add an active-set min/max here to exit early once the pivot is
    // pinned. It was tried: accumulating amn/amx in this loop (the reads are
    // already happening) and breaking when they agree made small_n 21-39%
    // SLOWER and the anchor 615.5 -> 662.8 us. The two extra barriers per pass
    // in the reduction, plus the register pressure in this loop, cost far more
    // than the single pass the exit saves. See knowledge/known_bad.md.
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
      uint32_t k = s_keys[i];
      if (!filter || (k >> hshift) == (pivot >> hshift))
        atomicAdd(&s_hist[((k >> sh) & 0xFFu) * HIST_REP + rep], 1u);
    }
    __syncthreads();
    if (HIST_REP > 1) {
      for (int b = threadIdx.x; b < 256; b += blockDim.x) {
        uint32_t sum = 0;
#pragma unroll
        for (int r = 0; r < HIST_REP; r++) sum += s_hist[b * HIST_REP + r];
        s_red[b] = sum;
      }
      __syncthreads();
    }
    block_find_pivot_bucket(HIST_REP > 1 ? s_red : s_hist, s_scan, ek);
    pivot |= (s_scan[0] << sh);
    ek -= (int)s_scan[1];
  }
  eq_needed = ek;
}

// Same select but streaming the row from global memory (used by the fallback /
// direct oracle, where the row is far too large for LDS).
__device__ __forceinline__ void block_select_stream(const vfloat4* __restrict__ row4, int n4, int K,
                                                    uint32_t* __restrict__ s_hist,
                                                    uint32_t* __restrict__ s_red,
                                                    uint32_t* __restrict__ s_scan, uint32_t& pivot,
                                                    int& eq_needed) {
  pivot = 0;
  int ek = K;
  const int rep = threadIdx.x & (HIST_REP - 1);
  if (threadIdx.x == 0) {
    s_scan[0] = 0;
    s_scan[1] = 0;
  }
  for (int p = 0; p < RADIX_PASSES; p++) {
    const int sh = radix_shift(p);
    const int hshift = (p == 0) ? 0 : sh + 8;
    for (int i = threadIdx.x; i < HIST_SLOTS; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
      vfloat4 v = row4[i];
      uint32_t k[FP32_EPT] = {fp32_to_sortable(v[0]), fp32_to_sortable(v[1]),
                              fp32_to_sortable(v[2]), fp32_to_sortable(v[3])};
#pragma unroll
      for (int e = 0; e < FP32_EPT; e++) {
        if (p == 0 || (k[e] >> hshift) == (pivot >> hshift))
          atomicAdd(&s_hist[((k[e] >> sh) & 0xFFu) * HIST_REP + rep], 1u);
      }
    }
    __syncthreads();
    if (HIST_REP > 1) {
      for (int b = threadIdx.x; b < 256; b += blockDim.x) {
        uint32_t sum = 0;
#pragma unroll
        for (int r = 0; r < HIST_REP; r++) sum += s_hist[b * HIST_REP + r];
        s_red[b] = sum;
      }
      __syncthreads();
    }
    block_find_pivot_bucket(HIST_REP > 1 ? s_red : s_hist, s_scan, ek);
    pivot |= (s_scan[0] << sh);
    ek -= (int)s_scan[1];
  }
  eq_needed = ek;
}

// Exact full-row select for ONE row, streaming it from global memory. Shared by
// the fallback branch inside Phase C and by the standalone Phase D oracle, so
// the two can never drift apart. All threads of the block must call.
__device__ __forceinline__ void exact_row_select(const float* __restrict__ input, int N, int K,
                                                 int row, int* __restrict__ out,
                                                 uint32_t* __restrict__ s_hist,
                                                 uint32_t* __restrict__ s_red,
                                                 uint32_t* __restrict__ s_scan,
                                                 unsigned* __restrict__ s_wgt,
                                                 unsigned* __restrict__ s_weq) {
  const vfloat4* ri4 = reinterpret_cast<const vfloat4*>(input + (size_t)row * N);
  uint32_t pivot;
  int eq_needed;
  block_select_stream(ri4, N / FP32_EPT, K, s_hist, s_red, s_scan, pivot, eq_needed);
  if (threadIdx.x == 0) {
    *s_wgt = 0;
    *s_weq = 0;
  }
  __syncthreads();
  const float* rif = reinterpret_cast<const float*>(ri4);
  block_gather_topk(N, pivot, K - eq_needed, eq_needed, out, s_wgt, s_weq,
                    [&](int i) { return fp32_to_sortable(rif[i]); }, [](int i) { return i; });
}

#include "topk_generalize.hip.hpp"

// ---------------------------------------------------------------------------
// Phase A: per-row sampled threshold, fully in LDS, one kernel, one block/row.
// ---------------------------------------------------------------------------
// Also clears the counters the later phases accumulate into. Phase A already
// runs one block per row and completes before Phase B on the same stream, so
// this is free, where a hipMemsetAsync per counter was a full dispatch each
// (~2.6 us) -- 5 of the 9 dispatches on the decode path were memsets.
// cand_reserved / cand_bad may be null on the paths that do not reserve.
__global__ __launch_bounds__(1024) void phase_a_threshold(const float* __restrict__ input, int N,
                                                          int rank, int S, int npasses,
                                                          int chunk_stride,
                                                          uint32_t* __restrict__ threshold,
                                                          float* __restrict__ threshold_f,
                                                          unsigned int* __restrict__ cand_reserved,
                                                          unsigned int* __restrict__ cand_bad,
                                                          int* __restrict__ fb_count) {
  const int row = blockIdx.x;
  const float* ri = input + (size_t)row * N;

  if (threadIdx.x == 0) {
    if (cand_reserved) cand_reserved[row] = 0u;
    if (cand_bad) cand_bad[row] = 0u;
    if (row == 0) *fb_count = 0;
  }

  // Dynamic LDS so the footprint tracks the runtime S. A static [SAMPLE_S_MAX]
  // array costs 64 KB unconditionally and measurably crushes occupancy
  // (S=4096 regressed 0.774 -> 0.852 ms when it was sized statically).
  extern __shared__ uint32_t s_keys[];
  __shared__ uint32_t s_hist[HIST_SLOTS];
  __shared__ uint32_t s_red[256];
  __shared__ uint32_t s_scan[2];
  __shared__ uint32_t s_mm[2 * MAX_WAVES_PER_BLOCK];

  // dwordx4 per lane. A scalar `ri[chunk*stride+off]` loop moves only 4 B per
  // lane and left this kernel at 7x its own traffic floor.
  // chunk_stride arrives from the host (sample_chunk_stride), which is also what
  // decides servability, so the two cannot disagree. Passing it also takes an
  // integer division and a mask out of device code: computing it here cost 0.8%
  // on the decode geomean (30.91 -> 31.15 us, A/B'd on this machine) once the
  // mask was added, for a value the host already knew.
  const int v4_per_chunk = SAMPLE_CHUNK_ELEMS / FP32_EPT;
  const int total_v4 = S / FP32_EPT;
  for (int u = threadIdx.x; u < total_v4; u += blockDim.x) {
    const int chunk = u / v4_per_chunk;
    const int off4 = u % v4_per_chunk;
    vfloat4 v = *(reinterpret_cast<const vfloat4*>(ri + (size_t)chunk * chunk_stride) + off4);
    const int base = u * FP32_EPT;
    s_keys[base + 0] = fp32_to_sortable(v[0]);
    s_keys[base + 1] = fp32_to_sortable(v[1]);
    s_keys[base + 2] = fp32_to_sortable(v[2]);
    s_keys[base + 3] = fp32_to_sortable(v[3]);
  }
  __syncthreads();

  uint32_t pivot;
  int eq_needed;
  block_select_lds(s_keys, S, rank, s_hist, s_red, s_scan, s_mm, pivot, eq_needed, npasses);
  if (threadIdx.x == 0) {
    threshold[row] = pivot;
    // Exact round-trip: pivot is the sortable image of a real sampled value.
    threshold_f[row] = sortable_to_fp32(pivot);
  }
}

// Variant 3: wave-private output regions, so Phase B has NO atomic of any kind
// (variant 2 still paid ~512 LDS atomics per row on one address). Each wave
// keeps a wave-uniform register counter and writes into its own slice. Key and
// index go out as one packed 64-bit store instead of two 32-bit streams.
// Overflow of a slice is detected and sends the row to the exact fallback.
// ablate: 0 = normal, 1 = skip the candidate stores (loads/compares stay live
// via wcnt), 2 = skip the compaction entirely and only consume the loads.
// Used to attribute Phase B's gap to its own read floor.
template <int ABLATE>
__global__ void phase_b_filter_waveseg(const float* __restrict__ input, int N,
                                       const float* __restrict__ threshold_f,
                                       uint64_t* __restrict__ cand_pack,
                                       int* __restrict__ cand_seg,
                                       unsigned int* __restrict__ cand_count, int seg_stride) {
  const int row = blockIdx.x;
  const vfloat4* ri = reinterpret_cast<const vfloat4*>(input + (size_t)row * N);
  const float th = threshold_f[row];

  const int lane = threadIdx.x & (WAVE_SIZE - 1);
  const int wid = threadIdx.x / WAVE_SIZE;
  const int nwaves = blockDim.x / WAVE_SIZE;
  const uint64_t lt = (1ull << lane) - 1ull;

  uint64_t* seg = cand_pack + (size_t)row * CAND_SLOTS_PER_ROW + (size_t)wid * seg_stride;

  const int n4 = N / FP32_EPT;
  const int stride = blockDim.x;
  const int iters = (n4 + stride - 1) / stride;

  int wcnt = 0;       // wave-uniform: every lane holds the same running count
  bool overflow = false;

  for (int it = 0; it < iters; it++) {
    const int i = it * stride + threadIdx.x;
    vfloat4 v = {0.f, 0.f, 0.f, 0.f};
    const bool live = (i < n4);
    if (live) v = load_f4(ri + i);

    const uint64_t b0 = __ballot(live && !(v[0] < th));
    const uint64_t b1 = __ballot(live && !(v[1] < th));
    const uint64_t b2 = __ballot(live && !(v[2] < th));
    const uint64_t b3 = __ballot(live && !(v[3] < th));
    const int t0 = __popcll(b0);
    const int t1 = t0 + __popcll(b1);
    const int t2 = t1 + __popcll(b2);
    const int wtotal = t2 + __popcll(b3);

    if (ABLATE == 2) {
      wcnt += wtotal;
      continue;
    }

    if (wtotal > 0) {
      const int base_idx = i * FP32_EPT;
      if (b0 & (1ull << lane)) {
        int p = wcnt + __popcll(b0 & lt);
        if (ABLATE == 0 && p < seg_stride)
          seg[p] = ((uint64_t)__float_as_uint(v[0]) << 32) | (uint32_t)(base_idx + 0);
      }
      if (b1 & (1ull << lane)) {
        int p = wcnt + t0 + __popcll(b1 & lt);
        if (ABLATE == 0 && p < seg_stride)
          seg[p] = ((uint64_t)__float_as_uint(v[1]) << 32) | (uint32_t)(base_idx + 1);
      }
      if (b2 & (1ull << lane)) {
        int p = wcnt + t1 + __popcll(b2 & lt);
        if (ABLATE == 0 && p < seg_stride)
          seg[p] = ((uint64_t)__float_as_uint(v[2]) << 32) | (uint32_t)(base_idx + 2);
      }
      if (b3 & (1ull << lane)) {
        int p = wcnt + t2 + __popcll(b3 & lt);
        if (ABLATE == 0 && p < seg_stride)
          seg[p] = ((uint64_t)__float_as_uint(v[3]) << 32) | (uint32_t)(base_idx + 3);
      }
      wcnt += wtotal;
      if (wcnt > seg_stride) overflow = true;
    }
  }

  __shared__ int s_seg[MAX_WAVES_PER_BLOCK];
  if (lane == 0) s_seg[wid] = overflow ? -1 : wcnt;
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned total = 0;
    bool bad = false;
    for (int w = 0; w < nwaves; w++) {
      cand_seg[(size_t)row * MAX_WAVES_PER_BLOCK + w] = s_seg[w];
      if (s_seg[w] < 0) bad = true;
      else total += (unsigned)s_seg[w];
    }
    // 0xFFFFFFFF is unconditionally > PHASE_C_CAP, so Phase C routes it to the
    // exact fallback without needing a separate flag.
    cand_count[row] = bad ? 0xFFFFFFFFu : total;
  }
}

// Variant 4: same wave-private regions, but passers are staged in LDS and
// flushed only once the wave holds at least a full wave's worth, so the global
// writes go out as ~520 B contiguous bursts instead of ~45 B fragments.
//
// Ablation on variant 3 showed the candidate stores cost 139 us while the whole
// load+compare+compact path cost only 13 us over its 349 us read floor: a wave
// produces ~5.6 passers per iteration, so each 8 B-per-passer burst touched a
// 128 B line far below full width.
//
// The flush drains the WHOLE buffer, so no remainder has to be shifted down and
// wcnt simply stays unaligned; a 520 B contiguous burst spans 5 lines instead of
// 4, which is a boundary effect rather than per-element amplification.
// WSTAGE_* constants live in topk_generalize.hip.hpp
__global__ __launch_bounds__(512) void phase_b_filter_wavestage(
    const float* __restrict__ input, int N, const float* __restrict__ threshold_f,
    uint64_t* __restrict__ cand_pack, int* __restrict__ cand_seg,
    unsigned int* __restrict__ cand_count, int seg_stride) {
  const int row = blockIdx.x;
  const vfloat4* ri = reinterpret_cast<const vfloat4*>(input + (size_t)row * N);
  const float th = threshold_f[row];

  const int lane = threadIdx.x & (WAVE_SIZE - 1);
  const int wid = threadIdx.x / WAVE_SIZE;
  const int nwaves = blockDim.x / WAVE_SIZE;
  const uint64_t lt = (1ull << lane) - 1ull;

  __shared__ uint64_t wbuf[WSTAGE_WAVES * WSTAGE_CAP];
  uint64_t* buf = wbuf + (size_t)wid * WSTAGE_CAP;
  uint64_t* seg = cand_pack + (size_t)row * CAND_SLOTS_PER_ROW + (size_t)wid * seg_stride;

  const int n4 = N / FP32_EPT;
  const int stride = blockDim.x;
  const int iters = (n4 + stride - 1) / stride;

  int wcnt = 0;   // entries already flushed to global (wave-uniform)
  int bcnt = 0;   // entries currently staged in LDS (wave-uniform)
  bool overflow = false;

  for (int it = 0; it < iters; it++) {
    const int i = it * stride + threadIdx.x;
    vfloat4 v = {0.f, 0.f, 0.f, 0.f};
    const bool live = (i < n4);
    if (live) v = load_f4(ri + i);

    const uint64_t b0 = __ballot(live && !(v[0] < th));
    const uint64_t b1 = __ballot(live && !(v[1] < th));
    const uint64_t b2 = __ballot(live && !(v[2] < th));
    const uint64_t b3 = __ballot(live && !(v[3] < th));
    const int t0 = __popcll(b0);
    const int t1 = t0 + __popcll(b1);
    const int t2 = t1 + __popcll(b2);
    const int wtotal = t2 + __popcll(b3);

    if (wtotal > 0) {
      const int base_idx = i * FP32_EPT;
      if (b0 & (1ull << lane))
        buf[bcnt + __popcll(b0 & lt)] =
            ((uint64_t)__float_as_uint(v[0]) << 32) | (uint32_t)(base_idx + 0);
      if (b1 & (1ull << lane))
        buf[bcnt + t0 + __popcll(b1 & lt)] =
            ((uint64_t)__float_as_uint(v[1]) << 32) | (uint32_t)(base_idx + 1);
      if (b2 & (1ull << lane))
        buf[bcnt + t1 + __popcll(b2 & lt)] =
            ((uint64_t)__float_as_uint(v[2]) << 32) | (uint32_t)(base_idx + 2);
      if (b3 & (1ull << lane))
        buf[bcnt + t2 + __popcll(b3 & lt)] =
            ((uint64_t)__float_as_uint(v[3]) << 32) | (uint32_t)(base_idx + 3);
      bcnt += wtotal;
    }

    // Wave-uniform: every lane has the same bcnt.
    if (bcnt >= WAVE_SIZE) {
      __builtin_amdgcn_wave_barrier();
      if (wcnt + bcnt <= seg_stride) {
        for (int j = lane; j < bcnt; j += WAVE_SIZE) seg[wcnt + j] = buf[j];
      } else {
        overflow = true;
      }
      wcnt += bcnt;
      bcnt = 0;
    }
  }

  if (bcnt > 0) {
    __builtin_amdgcn_wave_barrier();
    if (wcnt + bcnt <= seg_stride) {
      for (int j = lane; j < bcnt; j += WAVE_SIZE) seg[wcnt + j] = buf[j];
    } else {
      overflow = true;
    }
    wcnt += bcnt;
  }

  __shared__ int s_seg[MAX_WAVES_PER_BLOCK];
  if (lane == 0) s_seg[wid] = overflow ? -1 : wcnt;
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned total = 0;
    bool bad = false;
    for (int w = 0; w < nwaves; w++) {
      cand_seg[(size_t)row * MAX_WAVES_PER_BLOCK + w] = s_seg[w];
      if (s_seg[w] < 0) bad = true;
      else total += (unsigned)s_seg[w];
    }
    cand_count[row] = bad ? 0xFFFFFFFFu : total;
  }
}

// Phase C fed by the wave-segmented layout: gathers the variable-length
// per-wave segments into one contiguous LDS array, then selects exactly.
//
// STATIC_CAP selects where the candidate staging area lives:
//   true  -> two __shared__ arrays at PHASE_C_CAP, so both base addresses are
//            compile-time constants. This is the shape the main config uses.
//   false -> one dynamic-LDS block split at the runtime `cap`, needed when cap
//            exceeds PHASE_C_CAP.
// Sizing the arrays statically at PHASE_C_CAP_MAX instead costs 32 KB of LDS
// unconditionally and halves occupancy (measured 0.6215 -> 0.6702 ms); making
// the main config pay the runtime split instead costs the +0.6% that the
// runtime base offset adds (measured 0.6254 vs 0.6215 ms).
template <bool STATIC_CAP>
__global__ void phase_c_select_waveseg(const float* __restrict__ input, int N,
                                       const uint64_t* __restrict__ cand_pack,
                                       const int* __restrict__ cand_seg,
                                       const unsigned int* __restrict__ cand_count, int seg_stride,
                                       int nwaves_b, int K, int cap, int* __restrict__ out_idx,
                                       int* __restrict__ fb_rows, int* __restrict__ fb_count,
                                       int npasses) {
  const int row = blockIdx.x;
  const unsigned int c_raw = cand_count[row];

  extern __shared__ uint32_t s_dyn[];
  __shared__ uint32_t s_keys_st[STATIC_CAP ? PHASE_C_CAP : 1];
  __shared__ int s_idx_st[STATIC_CAP ? PHASE_C_CAP : 1];
  uint32_t* s_keys = STATIC_CAP ? s_keys_st : s_dyn;
  int* s_idx = STATIC_CAP ? s_idx_st : reinterpret_cast<int*>(s_dyn + cap);
  __shared__ uint32_t s_hist[HIST_SLOTS];
  __shared__ uint32_t s_red[256];
  __shared__ uint32_t s_scan[2];
  __shared__ uint32_t s_mm[2 * MAX_WAVES_PER_BLOCK];
  __shared__ int s_cnt[MAX_WAVES_PER_BLOCK];
  __shared__ int s_off[MAX_WAVES_PER_BLOCK];
  __shared__ unsigned s_wgt, s_weq;

  int* out_row = out_idx + (size_t)row * K;
  if (c_raw < (unsigned)K || c_raw > (unsigned)cap) {
    // Unusable candidate set: do the exact full-row select HERE rather than in a
    // separate fallback kernel. This block already owns the row and already has
    // the histogram scratch, so folding it in removes the 4th dispatch from
    // every call (4.0 us of the 33 us at M=1 N=1M was an empty phase_d launch).
    // It is also more parallel in the worst case: when every row falls back, M
    // blocks share the work instead of FB_GRID=64.
    if (threadIdx.x == 0) fb_rows[atomicAdd(fb_count, 1)] = row;   // diagnostics only
    exact_row_select(input, N, K, row, out_row, s_hist, s_red, s_scan, &s_wgt, &s_weq);
    return;
  }
  const int c = (int)c_raw;

  if (threadIdx.x < nwaves_b) s_cnt[threadIdx.x] = cand_seg[(size_t)row * MAX_WAVES_PER_BLOCK + threadIdx.x];
  __syncthreads();
  if (threadIdx.x == 0) {
    int t = 0;
    for (int w = 0; w < nwaves_b; w++) {
      s_off[w] = t;
      t += s_cnt[w];
    }
    s_wgt = 0;
    s_weq = 0;
  }
  __syncthreads();

  const uint64_t* base = cand_pack + (size_t)row * CAND_SLOTS_PER_ROW;
  for (int w = 0; w < nwaves_b; w++) {
    const int cnt = s_cnt[w];
    const int off = s_off[w];
    for (int i = threadIdx.x; i < cnt; i += blockDim.x) {
      uint64_t p = base[(size_t)w * seg_stride + i];
      s_keys[off + i] = fp32_to_sortable_bits((uint32_t)(p >> 32));
      s_idx[off + i] = (int)(uint32_t)p;
    }
  }
  __syncthreads();

  uint32_t pivot;
  int eq_needed;
  block_select_lds(s_keys, c, K, s_hist, s_red, s_scan, s_mm, pivot, eq_needed, npasses,
                   /*prefix_skip=*/true);

  block_gather_topk(c, pivot, K - eq_needed, eq_needed, out_row, &s_wgt, &s_weq,
                    [&](int i) { return s_keys[i]; }, [&](int i) { return s_idx[i]; });
}

// ---------------------------------------------------------------------------
// Phase D: exact full-row select over a compacted row list. Now used only by
// --pipeline direct as the independent oracle; the fused path folds the same
// work into Phase C to save a dispatch.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(1024) void phase_d_fallback(const float* __restrict__ input, int N,
                                                         int K, const int* __restrict__ fb_rows,
                                                         const int* __restrict__ fb_count,
                                                         int* __restrict__ out_idx) {
  const int count = *fb_count;
  const int n4 = N / FP32_EPT;

  __shared__ uint32_t s_hist[HIST_SLOTS];
  __shared__ uint32_t s_red[256];
  __shared__ uint32_t s_scan[2];
  __shared__ unsigned s_wgt;
  __shared__ unsigned s_weq;

  // Grid-stride over the compacted row list: correct for any count, while the
  // dispatch stays at FB_GRID blocks instead of one per matrix row.
  (void)n4;
  for (int slot = blockIdx.y; slot < count; slot += gridDim.y) {
    const int row = fb_rows[slot];
    exact_row_select(input, N, K, row, out_idx + (size_t)row * K, s_hist, s_red, s_scan, &s_wgt,
                     &s_weq);
    __syncthreads();
  }
}

__global__ void fill_identity_rows(int* rows, int* count, int M) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < M) rows[i] = i;
  if (i == 0) *count = M;
}

__global__ void fill_random_fp32(float* data, size_t count, unsigned int seed, int mode, int N) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = idx; i < count; i += (size_t)gridDim.x * blockDim.x) {
    unsigned int h = (unsigned int)i ^ seed;
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    float v;
    if (mode == 1) {
      float u1 = (h & 0xFFFF) / 65535.f + 1e-6f;
      float u2 = ((h >> 16) & 0xFFFF) / 65535.f;
      v = sqrtf(-2.f * logf(u1)) * cosf(2.f * 3.14159265f * u2);
    } else if (mode == 2) {
      v = 1.0f;
    } else if (mode == 3) {
      v = ((i & 0xFF) == 0) ? __int_as_float(0x7f800000) : (float)(i & 0xFFFF) * 1e-6f;
    } else if (mode == 4) {
      int col = (int)(i % (size_t)N);
      v = (col >= N - 3000) ? 100.f + (float)(i & 0xF) : (float)(i & 0xFFFF) * 1e-6f;
    } else {
      v = ((int)(h & 0xFFFF) - 32768) / 32768.f;
    }
    data[i] = v;
  }
}

// ---------------------------------------------------------------------------
// Host orchestration
// ---------------------------------------------------------------------------
struct Bufs {
  uint32_t* threshold;
  float* threshold_f;
  uint64_t* cand_pack;
  int* cand_seg;
  unsigned int* cand_count;
  unsigned int* cand_reserved;
  unsigned int* cand_bad;
  int* fb_rows;
  int* fb_count;
  int C_alloc;
};

static void alloc_bufs(Bufs& b, int M, int K, int cap) {
  b.C_alloc = cap;
  const int row_slots = std::max(CAND_SLOTS_PER_ROW, cap);
  HIP_CHECK(hipMalloc(&b.threshold, (size_t)M * sizeof(uint32_t)));
  HIP_CHECK(hipMalloc(&b.threshold_f, (size_t)M * sizeof(float)));
  HIP_CHECK(hipMalloc(&b.cand_pack, (size_t)M * row_slots * sizeof(uint64_t)));
  HIP_CHECK(hipMalloc(&b.cand_seg, (size_t)M * MAX_WAVES_PER_BLOCK * sizeof(int)));
  HIP_CHECK(hipMalloc(&b.cand_count, (size_t)M * sizeof(unsigned int)));
  HIP_CHECK(hipMalloc(&b.cand_reserved, (size_t)M * sizeof(unsigned int)));
  HIP_CHECK(hipMalloc(&b.cand_bad, (size_t)M * sizeof(unsigned int)));
  HIP_CHECK(hipMalloc(&b.fb_rows, (size_t)M * sizeof(int)));
  HIP_CHECK(hipMalloc(&b.fb_count, sizeof(int)));
  (void)K;
}

static void free_bufs(Bufs& b) {
  (void)hipFree(b.threshold);
  (void)hipFree(b.threshold_f);
  (void)hipFree(b.cand_pack);
  (void)hipFree(b.cand_seg);
  (void)hipFree(b.cand_count);
  (void)hipFree(b.cand_reserved);
  (void)hipFree(b.cand_bad);
  (void)hipFree(b.fb_rows);
  (void)hipFree(b.fb_count);
}

// Block size for the small_n path. Sized from the dwordx4 LOAD count (n4 = N/4),
// not from N: sizing it from N gave a 16-wave block where only N/4 of the lanes
// ever issued a load (768 of 1024 idle at N=1024) and all 16 waves still paid
// the 4-pass x 4-barrier select over just 1024 elements.
//
// Lower bound 256 is a correctness constraint, not a tuning choice:
// block_find_pivot_bucket() indexes the 256 radix buckets by threadIdx.x, so a
// block under 256 threads silently drops the upper buckets and picks a wrong
// pivot.
// Sizing it from the load count (one dwordx4 per lane) is NOT optimal: measured
// M=4096 N=2048 wants 256 threads (39.7 us) where the load rule picks 512
// (50.2 us). What actually decides it is how many WAVES end up resident per CU,
// because a row's cost is dominated by the fixed 4-pass x 4-barrier select, not
// by its loads. Blocks per CU is capped by the dynamic LDS (the row itself), so
// at large N only a bigger block can supply enough waves, while at small N the
// cap is loose and the cheapest block wins.
//
// The row itself is the dynamic LDS here, and the load cap matters because a
// small row cannot occupy many waves: sizing purely from the load count picked a
// 512-thread block at M=4096 N=2048 (50.2 us) where 256 measures 39.7 us.
static int small_n_block(int M, int N) {
  if (g_small_n_block > 0) return std::max(256, std::min(1024, g_small_n_block));
  const int t = small_n_threads_from_table(M, N);
  if (t > 0) return t;
  return occupancy_block_threads(M, SMALL_N_STATIC_LDS + N * (int)sizeof(uint32_t),
                                 std::max(1, (N / FP32_EPT) / WAVE_SIZE));
}

static void topk_small_n(const float* d_in, int M, int N, int K, int* d_idx, hipStream_t s) {
  // g_small_n_passes < RADIX_PASSES is a TIMING ABLATION ONLY: the pivot is then
  // truncated and the result is WRONG. It exists to price the fixed per-pass
  // cost (zero 1024 hist slots, 256-bin reduce, scan, 4 barriers), because
  // passes 2..4 only histogram the elements matching the current prefix and so
  // carry almost no real work.
  phase_small_n_topk<<<M, small_n_block(M, N), (size_t)N * sizeof(uint32_t), s>>>(
      d_in, N, K, d_idx, g_small_n_passes);
}

static void topk_fused_impl(const float* d_in, int M, int N, int K, int* d_idx, Bufs& b,
                            const ShapeParams& sp, hipStream_t s) {
  const int S = sp.S;
  const float margin = sp.margin;
  const int rank = g_sample_rank > 0 ? g_sample_rank : sp.rank;
  const int cap = sp.cap;
  const int n4 = N / FP32_EPT;
  const int gx = std::max(1, std::min(g_cf_gx, n4 / g_cf_block));
  const int nwaves_b = std::max(1, g_cf_block / WAVE_SIZE);
  const int seg_stride = CAND_SLOTS_PER_ROW / nwaves_b;

  // Phase A and Phase C are one block per row and barrier-bound, so their block
  // size follows LDS-limited residency (S for A, cap for C), not a constant.
  const int a_block = g_phase_a_block > 0
                          ? g_phase_a_block
                          : occupancy_block_threads(M, PHASE_A_STATIC_LDS + S * 4, 0);
  const int c_block = g_phase_c_block > 0
                          ? g_phase_c_block
                          : occupancy_block_threads(M, PHASE_C_STATIC_LDS + cap * 8, 0);

  // No hipMemsetAsync here on purpose. cand_count is ASSIGNED (not accumulated)
  // by every Phase B variant, one block per row with no early return, so zeroing
  // it was always dead work; the reservation counters and fb_count are cleared
  // inside Phase A, which runs first on the same stream. This took the decode
  // path from 9 dispatches to 4 and the prefill path from 6 to 4.
  const bool coop = sp.coop_g > 1;

  const int chunk_stride = sample_chunk_stride(N, S / SAMPLE_CHUNK_ELEMS);

  if (g_fuse_ab) {
    phase_ab_fused<<<M, a_block, (size_t)S * sizeof(uint32_t), s>>>(
        d_in, N, rank, S, g_phase_a_passes, chunk_stride, seg_stride, b.cand_pack, b.cand_seg,
        b.cand_count, b.fb_count);
  } else {
    phase_a_threshold<<<M, a_block, (size_t)S * sizeof(uint32_t), s>>>(
        d_in, N, rank, S, g_phase_a_passes, chunk_stride, b.threshold, b.threshold_f,
        coop ? b.cand_reserved : nullptr, coop ? b.cand_bad : nullptr, b.fb_count);

    if (coop) {
      phase_b_filter_coop<<<dim3(sp.coop_g, M), g_cf_block, 0, s>>>(
          d_in, N, n4, b.threshold_f, b.cand_pack, b.cand_reserved, b.cand_bad, cap);
    } else if (g_phase_b == 4) {
      phase_b_filter_wavestage<<<M, g_cf_block, 0, s>>>(d_in, N, b.threshold_f, b.cand_pack,
                                                        b.cand_seg, b.cand_count, seg_stride);
    } else {
      phase_b_filter_waveseg<0><<<M, g_cf_block, 0, s>>>(d_in, N, b.threshold_f, b.cand_pack,
                                                          b.cand_seg, b.cand_count, seg_stride);
    }
  }

  // Phase C stages `cap` keys, and `cap` indices unless the keys-only variant
  // re-reads them from global. Sized here so the LDS footprint tracks the
  // runtime cap instead of PHASE_C_CAP_MAX.
  if (coop) {
    const size_t lds_c = (size_t)cap * (sp.keys_only_c ? sizeof(uint32_t)
                                                       : sizeof(uint32_t) + sizeof(int));
    phase_c_select_contig<<<M, c_block, lds_c, s>>>(
        d_in, N, b.cand_pack, b.cand_reserved, b.cand_bad, b.cand_count, cap, K, d_idx, b.fb_rows,
        b.fb_count, g_phase_c_passes, sp.keys_only_c);
  } else if (cap <= PHASE_C_CAP) {
    phase_c_select_waveseg<true><<<M, c_block, 0, s>>>(
        d_in, N, b.cand_pack, b.cand_seg, b.cand_count, seg_stride, nwaves_b, K, cap, d_idx,
        b.fb_rows, b.fb_count, g_phase_c_passes);
  } else {
    const size_t lds_c = (size_t)cap * (sizeof(uint32_t) + sizeof(int));
    phase_c_select_waveseg<false><<<M, c_block, lds_c, s>>>(
        d_in, N, b.cand_pack, b.cand_seg, b.cand_count, seg_stride, nwaves_b, K, cap, d_idx,
        b.fb_rows, b.fb_count, g_phase_c_passes);
  }

  // No Phase D launch: Phase C handles its own unusable rows inline. Phase D
  // still exists as the --pipeline direct oracle.
  (void)margin;
  (void)gx;
}

static void topk_indices(const float* d_in, int M, int N, int K, int* d_idx, Bufs& b, int smc,
                         hipStream_t s) {
  ShapeParams sp =
      derive_shape_params(M, N, K, g_margin, g_sample_s, g_coop_g, (TopkPath)g_path_override);
  g_sample_s = sp.S > 0 ? sp.S : g_sample_s;
  if (sp.path == PATH_SMALL_N) {
    topk_small_n(d_in, M, N, K, d_idx, s);
    return;
  }
  topk_fused_impl(d_in, M, N, K, d_idx, b, sp, s);
  (void)smc;
}

static void topk_fused(const float* d_in, int M, int N, int K, int* d_idx, Bufs& b, int smc,
                       hipStream_t s) {
  topk_indices(d_in, M, N, K, d_idx, b, smc, s);
}

static void topk_direct(const float* d_in, int M, int N, int K, int* d_idx, Bufs& b,
                        hipStream_t s) {
  fill_identity_rows<<<(M + 255) / 256, 256, 0, s>>>(b.fb_rows, b.fb_count, M);
  phase_d_fallback<<<dim3(1, FB_GRID), 1024, 0, s>>>(d_in, N, K, b.fb_rows, b.fb_count, d_idx);
}

static void run_topk(const float* d_in, int M, int N, int K, int* d_idx, Bufs& b, int smc,
                     hipStream_t s) {
  if (g_pipeline_direct)
    topk_direct(d_in, M, N, K, d_idx, b, s);
  else
    topk_fused(d_in, M, N, K, d_idx, b, smc, s);
}

// ---- AITER_EXPORT_END ----
// Everything above this line is the kernel plus its dispatch and is portable;
// scripts/export_aiter_op.py copies exactly that region into aiter and appends
// csrc/topk_aiter_entry.inc.hip. Everything below is harness only (verification
// oracles, timing, CLI). Moving code across this line changes what ships, so
// re-run the export and its diff check after doing so.

static bool verify_row_cpu(const float* row, int N, int K, const int* idx) {
  std::vector<uint32_t> sv(N);
  for (int i = 0; i < N; i++) sv[i] = fp32_to_sortable_host(row[i]);
  std::vector<uint32_t> got;
  got.reserve(K);
  for (int i = 0; i < K; i++) {
    if (idx[i] < 0 || idx[i] >= N) return false;
    got.push_back(sv[idx[i]]);
  }
  std::sort(got.begin(), got.end(), std::greater<uint32_t>());
  std::vector<uint32_t> ref(sv);
  std::partial_sort(ref.begin(), ref.begin() + K, ref.end(), std::greater<uint32_t>());
  ref.resize(K);
  return got == ref;
}

static bool verify_row_gpu_oracle(const float* d_in, int N, int K, int row, int* d_idx, Bufs& b) {
  int h_rows[1] = {row};
  int h_count = 1;
  HIP_CHECK(hipMemcpy(b.fb_rows, h_rows, sizeof(h_rows), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(b.fb_count, &h_count, sizeof(int), hipMemcpyHostToDevice));
  phase_d_fallback<<<dim3(1, 1), 1024>>>(d_in, N, K, b.fb_rows, b.fb_count, d_idx);
  HIP_CHECK(hipDeviceSynchronize());
  return true;
}

static bool row_idx_multiset_match(const float* row, int N, int K, const int* got,
                                   const int* ref) {
  std::vector<uint32_t> gv, rv;
  gv.reserve(K);
  rv.reserve(K);
  for (int i = 0; i < K; i++) {
    if (got[i] < 0 || got[i] >= N || ref[i] < 0 || ref[i] >= N) return false;
    gv.push_back(fp32_to_sortable_host(row[got[i]]));
    rv.push_back(fp32_to_sortable_host(row[ref[i]]));
  }
  std::sort(gv.begin(), gv.end(), std::greater<uint32_t>());
  std::sort(rv.begin(), rv.end(), std::greater<uint32_t>());
  return gv == rv;
}

static bool verify_rows_sampled(const float* d_in, int M, int N, int K, int* d_idx,
                                const int* h_idx, Bufs& b, int sample_n) {
  std::vector<int> rows;
  for (int i = 0; i < M; i += std::max(1, M / sample_n)) rows.push_back(i);
  if (rows.empty()) rows.push_back(0);

  std::vector<float> h_row((size_t)N);
  std::vector<int> oracle((size_t)K);
  for (int r : rows) {
    HIP_CHECK(hipMemcpy(h_row.data(), d_in + (size_t)r * N, (size_t)N * sizeof(float),
                        hipMemcpyDeviceToHost));
    if (g_verify_oracle_gpu) {
      verify_row_gpu_oracle(d_in, N, K, r, d_idx, b);
      HIP_CHECK(hipMemcpy(oracle.data(), d_idx + (size_t)r * K, (size_t)K * sizeof(int),
                          hipMemcpyDeviceToHost));
    } else {
      std::vector<uint32_t> sv(N);
      for (int i = 0; i < N; i++) sv[i] = fp32_to_sortable_host(h_row[i]);
      std::partial_sort(sv.begin(), sv.begin() + K, sv.end(), std::greater<uint32_t>());
      for (int i = 0; i < K; i++) {
        uint32_t want = sv[i];
        int found = -1;
        for (int j = 0; j < N; j++)
          if (fp32_to_sortable_host(h_row[j]) == want) {
            found = j;
            break;
          }
        if (found < 0) return false;
        oracle[i] = found;
      }
    }
    if (!row_idx_multiset_match(h_row.data(), N, K, h_idx + (size_t)r * K, oracle.data()))
      return false;
  }
  return true;
}

static double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

static double stddev_pct(const std::vector<double>& v, double med) {
  if (v.size() < 2 || med == 0.0) return 0.0;
  double sq = 0.0;
  for (double x : v) {
    double d = (x - med) / med;
    sq += d * d;
  }
  return 100.0 * std::sqrt(sq / (v.size() - 1));
}

static void usage(const char* prog) {
  fprintf(stderr,
          "Usage: %s --mode verify|time|verify_and_time [options]\n"
          "  --m M --n N --topk K --iters N --warmup N --repeats N\n"
          "  --dist uniform|gaussian|equal|inf|adversarial --seed S\n"
          "  --pipeline fused|direct\n"
          "  --margin F --sample-rank R --cf-block B --cf-gx G\n"
          "  --nt-load 0|1 --phase-b 0|1|2 --phase-c-block B --phase-a-block B\n"
          "  --input-bin PATH --dump-indices PATH --inject-fault 0|1\n"
          "  --path auto|small_n|prefill|decode --coop-g G --fuse-ab 0|1 --hipgraph 0|1\n"
          "  --small-n-block B (256..1024, 0=auto)  --s-rule 0|1 (0=legacy R_TARGET)\n"
          "  --verify-sample-rows N --verify-oracle gpu|cpu\n",
          prog);
}

int main(int argc, char** argv) {
  int M = 4096, N = 131072, K = 2048;
  int iters = 100, warmup = 20, repeats = 5;
  std::string mode = "verify_and_time";
  std::string dist = "uniform";
  std::string dump_path, input_path;
  unsigned seed = 42;

  for (int i = 1; i < argc; i++) {
    auto need = [&]() -> std::string {
      if (i + 1 >= argc) {
        usage(argv[0]);
        exit(1);
      }
      return std::string(argv[++i]);
    };
    std::string a = argv[i];
    if (a == "--mode") mode = need();
    else if (a == "--m") M = std::stoi(need());
    else if (a == "--n") N = std::stoi(need());
    else if (a == "--topk") K = std::stoi(need());
    else if (a == "--iters") iters = std::stoi(need());
    else if (a == "--warmup") warmup = std::stoi(need());
    else if (a == "--repeats") repeats = std::stoi(need());
    else if (a == "--dist") dist = need();
    else if (a == "--seed") seed = (unsigned)std::stoul(need());
    else if (a == "--pipeline") g_pipeline_direct = (need() == "direct") ? 1 : 0;
    else if (a == "--margin") g_margin = std::stof(need());
    else if (a == "--sample-rank") g_sample_rank = std::stoi(need());
    else if (a == "--sample-s") g_sample_s = std::stoi(need());
    else if (a == "--cf-block") g_cf_block = std::stoi(need());
    else if (a == "--cf-gx") g_cf_gx = std::stoi(need());
    else if (a == "--nt-load") g_use_nt_load = std::stoi(need());
    else if (a == "--phase-b") g_phase_b = std::stoi(need());
    else if (a == "--phase-c-block") g_phase_c_block = std::stoi(need());
    else if (a == "--phase-a-block") g_phase_a_block = std::stoi(need());
    else if (a == "--phase-a-passes") g_phase_a_passes = std::stoi(need());
    else if (a == "--phase-c-passes") g_phase_c_passes = std::stoi(need());
    else if (a == "--input-bin") input_path = need();
    else if (a == "--dump-indices") dump_path = need();
    else if (a == "--inject-fault") g_inject_fault = std::stoi(need());
    else if (a == "--dump-stats") g_dump_stats = std::stoi(need());
    else if (a == "--ablate-store") g_ablate_store = std::stoi(need());
    else if (a == "--path") {
      std::string p = need();
      if (p == "small_n") g_path_override = PATH_SMALL_N;
      else if (p == "prefill") g_path_override = PATH_PREFILL;
      else if (p == "decode") g_path_override = PATH_DECODE;
      else g_path_override = PATH_AUTO;
    } else if (a == "--coop-g") g_coop_g = std::stoi(need());
    else if (a == "--fuse-ab") g_fuse_ab = std::stoi(need());
    else if (a == "--hipgraph") g_use_hipgraph = std::stoi(need());
    else if (a == "--small-n-block") g_small_n_block = std::stoi(need());
    else if (a == "--s-rule") g_s_rule = std::stoi(need());
    else if (a == "--small-n-passes") g_small_n_passes = std::stoi(need());
    else if (a == "--verify-sample-rows") g_verify_sample_rows = std::stoi(need());
    else if (a == "--verify-oracle") g_verify_oracle_gpu = (need() == "gpu") ? 1 : 0;
    else {
      usage(argv[0]);
      exit(1);
    }
  }

  if (K > N) {
    fprintf(stderr, "ERROR: topk=%d exceeds N=%d\n", K, N);
    return 2;
  }
  if (N % FP32_EPT != 0) {
    fprintf(stderr, "ERROR: N=%d must be multiple of %d\n", N, FP32_EPT);
    return 2;
  }

  ShapeParams shape =
      derive_shape_params(M, N, K, g_margin, g_sample_s, g_coop_g, (TopkPath)g_path_override);
  if (g_sample_s <= 0 || g_path_override != PATH_SMALL_N) g_sample_s = shape.S > 0 ? shape.S : g_sample_s;
  if (!shape.geom_ok) {
    fprintf(stderr, "ERROR: shape M=%d N=%d K=%d incompatible (path=%d S=%d cap=%d)\n", M, N, K,
            (int)shape.path, shape.S, shape.cap);
    return 2;
  }
  if (K > shape.cap && shape.path != PATH_SMALL_N) {
    fprintf(stderr, "ERROR: topk=%d exceeds derived cap=%d\n", K, shape.cap);
    return 2;
  }

  GPUInfo info = get_gpu_info();
  const char* path_name =
      shape.path == PATH_SMALL_N ? "small_n" : (shape.path == PATH_DECODE ? "decode" : "prefill");
  printf("GPU: %s CUs=%d path=%s S=%d margin=%.3f cap=%d coop_g=%d fuse_ab=%d\n", info.name,
         info.sm_count, path_name, shape.S, shape.margin, shape.cap, shape.coop_g, g_fuse_ab);
  HIP_CHECK(hipMemcpyToSymbol(d_use_nt_load, &g_use_nt_load, sizeof(int)));

  int dist_mode = 0;
  if (dist == "gaussian") dist_mode = 1;
  else if (dist == "equal") dist_mode = 2;
  else if (dist == "inf") dist_mode = 3;
  else if (dist == "adversarial") dist_mode = 4;

  const size_t in_elems = (size_t)M * N;
  const size_t out_elems = (size_t)M * K;
  float* d_in = nullptr;
  int* d_idx = nullptr;
  HIP_CHECK(hipMalloc(&d_in, in_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_idx, out_elems * sizeof(int)));

  if (!input_path.empty()) {
    std::vector<float> h_in(in_elems);
    FILE* f = fopen(input_path.c_str(), "rb");
    if (!f) {
      perror("input-bin");
      return 2;
    }
    if (fread(h_in.data(), sizeof(float), in_elems, f) != in_elems) {
      fprintf(stderr, "input-bin short read\n");
      fclose(f);
      return 2;
    }
    fclose(f);
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), in_elems * sizeof(float), hipMemcpyHostToDevice));
  } else {
    int blocks = std::min(65535, (int)((in_elems + 255) / 256));
    fill_random_fp32<<<blocks, 256>>>(d_in, in_elems, seed, dist_mode, N);
    HIP_CHECK(hipDeviceSynchronize());
  }

  Bufs bufs{};
  alloc_bufs(bufs, M, K, shape.cap);

  HIP_CHECK(hipMemset(d_idx, 0xFF, out_elems * sizeof(int)));
  run_topk(d_in, M, N, K, d_idx, bufs, info.sm_count, 0);
  // A launch that never started is not slow, it is instant and wrong: an
  // over-large dynamic LDS request reported 4.40 us with garbage output before
  // this check existed.
  {
    hipError_t le = hipGetLastError();
    if (le != hipSuccess) {
      fprintf(stderr, "ERROR: kernel launch failed (%d: %s) for M=%d N=%d K=%d path=%s\n",
              (int)le, hipGetErrorString(le), M, N, K, path_name);
      return 2;
    }
  }
  HIP_CHECK(hipDeviceSynchronize());

  int fb = 0;
  HIP_CHECK(hipMemcpy(&fb, bufs.fb_count, sizeof(int), hipMemcpyDeviceToHost));

  // The small_n path never runs a candidate stage, so cand_count is untouched
  // and reading it reports under_K=M from uninitialised memory -- a false alarm
  // that made every small_n shape look like it was falling back.
  if (g_dump_stats && shape.path == PATH_SMALL_N) {
    printf("CANDSTATS path=small_n n/a (no candidate stage)\n");
  } else if (g_dump_stats) {
    std::vector<unsigned> cc(M);
    HIP_CHECK(hipMemcpy(cc.data(), bufs.cand_count, (size_t)M * sizeof(unsigned),
                        hipMemcpyDeviceToHost));
    std::vector<unsigned> sorted(cc);
    std::sort(sorted.begin(), sorted.end());
    double mean = std::accumulate(cc.begin(), cc.end(), 0.0) / M;
    int under = 0, over = 0;
    for (unsigned v : cc) {
      if (v < (unsigned)K) under++;
      if (v > (unsigned)bufs.C_alloc) over++;
    }
    printf("CANDSTATS K=%d C_alloc=%d min=%u p1=%u mean=%.1f p99=%u max=%u under_K=%d over_Calloc=%d\n",
           K, bufs.C_alloc, sorted.front(), sorted[M / 100], mean, sorted[M - 1 - M / 100],
           sorted.back(), under, over);
  }

  std::vector<int> h_idx(out_elems);
  HIP_CHECK(hipMemcpy(h_idx.data(), d_idx, out_elems * sizeof(int), hipMemcpyDeviceToHost));

  const bool do_verify = (mode == "verify" || mode == "verify_and_time");
  const bool do_time = (mode == "time" || mode == "verify_and_time");

  if (do_verify) {
    if (g_inject_fault) h_idx[0] ^= 1;
    int bad_rows = 0;
    const size_t full_elems = (size_t)M * N;
    if (full_elems <= 64 * 1024 * 1024 && M <= 256) {
      std::vector<float> h_in(full_elems);
      HIP_CHECK(hipMemcpy(h_in.data(), d_in, full_elems * sizeof(float), hipMemcpyDeviceToHost));
      for (int r = 0; r < M; r++)
        if (!verify_row_cpu(h_in.data() + (size_t)r * N, N, K, h_idx.data() + (size_t)r * K))
          bad_rows++;
    } else if (!verify_rows_sampled(d_in, M, N, K, d_idx, h_idx.data(), bufs,
                                      g_verify_sample_rows)) {
      bad_rows = 1;
    }
    printf("VERIFY rows_fail=%d fallback_rows=%d pipeline=%s inject_fault=%d path=%s\n", bad_rows,
           fb, g_pipeline_direct ? "direct" : "fused", g_inject_fault, path_name);
    if (bad_rows != 0) return 2;
    if (!dump_path.empty()) {
      FILE* f = fopen(dump_path.c_str(), "wb");
      if (!f) {
        perror("dump-indices");
        return 2;
      }
      fwrite(h_idx.data(), sizeof(int), out_elems, f);
      fclose(f);
    }
    printf("VERDICT PASS\n");
  }

  if (do_time) {
    HipTimer timer;
    std::vector<double> run_medians;
    hipGraph_t graph = nullptr;
    hipGraphExec_t graph_exec = nullptr;
    hipStream_t gs = nullptr;
    if (g_use_hipgraph && !g_pipeline_direct) {
      // Capture must run on a real stream: hipStreamBeginCapture on the legacy
      // default stream returns 900 "operation not permitted when stream is
      // capturing" regardless of what is being captured.
      HIP_CHECK(hipStreamCreateWithFlags(&gs, hipStreamNonBlocking));
      for (int w = 0; w < warmup; w++) run_topk(d_in, M, N, K, d_idx, bufs, info.sm_count, gs);
      HIP_CHECK(hipStreamSynchronize(gs));
      hipError_t cap_err = hipStreamBeginCapture(gs, hipStreamCaptureModeGlobal);
      if (cap_err == hipSuccess) {
        run_topk(d_in, M, N, K, d_idx, bufs, info.sm_count, gs);
        cap_err = hipStreamEndCapture(gs, &graph);
      }
      if (cap_err == hipSuccess && graph != nullptr) {
        HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
      } else {
        if (graph) {
          (void)hipGraphDestroy(graph);
          graph = nullptr;
        }
        fprintf(stderr, "WARN: hipGraph capture failed (%d: %s); timing without graph\n",
                (int)cap_err, hipGetErrorString(cap_err));
      }
    }
    for (int rep = 0; rep < repeats; rep++) {
      if (!g_use_hipgraph || g_pipeline_direct) {
        for (int w = 0; w < warmup; w++) run_topk(d_in, M, N, K, d_idx, bufs, info.sm_count, 0);
        HIP_CHECK(hipDeviceSynchronize());
      }
      std::vector<double> samples;
      samples.reserve(iters);
      for (int it = 0; it < iters; it++) {
        if (g_use_hipgraph && graph_exec) {
          timer.begin(gs);
          HIP_CHECK(hipGraphLaunch(graph_exec, gs));
          samples.push_back(timer.end(gs));
        } else {
          timer.begin();
          run_topk(d_in, M, N, K, d_idx, bufs, info.sm_count, 0);
          samples.push_back(timer.end());
        }
      }
      run_medians.push_back(median(samples));
    }
    if (graph_exec) HIP_CHECK(hipGraphExecDestroy(graph_exec));
    if (graph) HIP_CHECK(hipGraphDestroy(graph));
    if (gs) HIP_CHECK(hipStreamDestroy(gs));
    const double med = median(run_medians);
    const double sd = stddev_pct(run_medians, med);
    printf("TIMING wall_ms_median=%.4f stddev_pct=%.3f repeats=%d iters=%d warmup=%d\n", med, sd,
           repeats, iters, warmup);
    printf("RESULT pipeline=%s wall_ms=%.4f fallback_rows=%d hipgraph=%d\n",
           g_pipeline_direct ? "direct" : "fused", med, fb, g_use_hipgraph);
  }

  free_bufs(bufs);
  (void)hipFree(d_in);
  (void)hipFree(d_idx);
  return 0;
}
