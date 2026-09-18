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
// FALSIFIED as a perf change, kept only so the measurement can be reproduced:
// compacting the active set is correct (same pivot, same candidate counts on all
// five distributions) but buys nothing -- phase_a 65.7 -> 65.4 us and wall time
// WORSE on three of four shapes. The discarded reads were never the cost; the
// barriers are. See knowledge/known_bad.md.
static int g_phase_a_compact = 0;

// Folds the HIST_REP reduction into the pivot scan, taking a radix pass from 5
// block barriers to 4. Shipped on; build with `-DSELECT_FUSED_REDUCE=0` to A/B
// against the separate-reduction form.
//
// Build-time and not a runtime knob because it changes the barrier structure of
// every caller of block_select_lds at once, and a runtime branch around a
// barrier would not be a fair comparison. Measured wall time, 3 runs each at
// warmup 20 / iters 100 / repeats 9, 0 -> 1:
//   M=4096 N=131072  0.6132 -> 0.6090 ms  (-0.68%, the anchor)
//   M=4096 N=1048576 3.1741 -> 3.1680 ms  (-0.19%)
//   M=1    N=1048576 0.0316 -> 0.0309 ms  (-2.2%)
//   M=1024 N=65536   0.0793 -> 0.0781 ms  (-1.5%)
//   M=4096 N=8192    0.0995 -> 0.0947 ms  (-4.8%, small_n)
//   M=2048 N=4096    0.0372 -> 0.0355 ms  (-4.6%, small_n)
// The gain tracks how much of the kernel is the select, which is what the
// barrier-bound reading of phase_a/phase_c predicts.
#ifndef SELECT_FUSED_REDUCE
#define SELECT_FUSED_REDUCE 1
#endif

// Second barrier removal on the same lever: the scan re-zeroes each histogram
// bucket as it reads it, so the per-pass clear loop and its barrier disappear
// and a radix pass goes from 4 barriers to 3. Costs no LDS, unlike
// double-buffering the histogram (+4 KB, which would cut phase_a from 4 to 3
// blocks/CU at S=8192). Requires SELECT_FUSED_REDUCE.
//
// Shipped on, but it is a REGIME TRADE, not a free win, so read this before
// moving it. Wall time, 3 runs each at warmup 20 / iters 100 / repeats 9:
//   M=4096 N=131072  0.6090 -> 0.6069 ms  (-0.35%, the anchor)
//   M=1    N=1048576 0.0308 -> 0.0306 ms  (-0.8%)
//   M=4096 N=1048576 3.1658 -> 3.1652 ms  (neutral)
//   M=1024 N=65536   0.0781 -> 0.0782 ms  (neutral)
//   M=2048 N=4096    0.0354 -> 0.0353 ms  (neutral)
//   M=4096 N=8192    0.0948 -> 0.0965 ms  (**+1.8%, small_n**)
// Inner geomean -0.40% with decode -0.8% and prefill -0.8% against small_n
// +0.40%, which stays inside PATH_NOISE_BAND_PCT. Taken because the large-N
// paths are the target and N <= 32768 is routed to aiter's own prefill by the
// stride0 >= 32768 dispatch in aiter/ops/topk.py.
//
// [unverified hypothesis] for the small_n point: the clear now runs on the 256
// threads that also carry the wave-scan, where the old loop spread it over all
// blockDim.x threads (512 at that shape), so the work moved onto the critical
// path instead of disappearing.
#ifndef SELECT_CLEAR_ON_READ
#define SELECT_CLEAR_ON_READ 1
#endif

// Diagnostic only, PRODUCES WRONG RESULTS: drops the atomicity of the histogram
// increment so the per-element LDS atomic can be priced. Never ship non-zero.
// Measured ceiling for any aggregation of that atomic: phase_a 61.7 -> 53.0 us
// at the anchor (-14.1%), and -9.3%/-13.0% of wall on small_n M=4096 N=8192 /
// M=2048 N=4096. It says nothing about phase_c: a wrong Phase A threshold blows
// up the candidate counts and sends rows to the exact fallback, which took
// phase_c 67.3 -> 2119 us. An ablation is only a price when it leaves the path
// alone.
#ifndef ABLATE_HIST_ATOMIC
#define ABLATE_HIST_ATOMIC 0
#endif

// Rounds of wave-level aggregation before falling back to per-element atomics
// (0 = off, the shipped form). See hist_add_aggregated in topk_common.hip.hpp.
#ifndef HIST_AGG_ROUNDS
#define HIST_AGG_ROUNDS 0
#endif

// Third barrier removal: wave 0 alone scans all 256 buckets, so the cross-wave
// partial sums and their barrier disappear and a pass runs 2 barriers, not 3.
// See block_find_pivot_bucket_wave0 in topk_common.hip.hpp for why the
// everyone-scans-redundantly variant cannot reach 2 without +4 KB of LDS.
//
// Shipped on. It is close to the MIRROR of SELECT_CLEAR_ON_READ's regime trade:
// that one bought the anchor and cost small_n, this one buys small_n and the
// latency-bound small-M shapes and costs the anchor slightly.
//   M=4096 N=8192    0.0964 -> 0.0943 ms  (-2.2%, small_n)
//   M=1    N=1048576 0.0305 -> 0.0300 ms  (-1.6%)
//   M=2048 N=4096    0.0352 -> 0.0347 ms  (-1.4%)
//   M=4096 N=1048576 3.1647 -> 3.1616 ms  (neutral)
//   M=1024 N=65536   0.0776 -> 0.0776 ms  (neutral)
//   M=4096 N=131072  0.6071 -> 0.6095 ms  (**+0.4%, the anchor**)
// Inner geomean 62.66 -> 61.96/61.98 us (-1.1%, two runs) with small_n -2.3%,
// decode -1.3% and prefill neutral, so the aggregate is a clear win and the one
// regressing point sits far inside POINT_REGRESS_PCT.
#ifndef SELECT_WAVE0_SCAN
#define SELECT_WAVE0_SCAN 1
#endif
#if SELECT_CLEAR_ON_READ && !SELECT_FUSED_REDUCE
#error "SELECT_CLEAR_ON_READ needs SELECT_FUSED_REDUCE: only the fused scan clears"
#endif
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
static int g_ragged = 0;
// Mirrors aiter's create_row_boundaries(num_rows, num_prefix): row r has extent
// num_prefix + r + 1. The prefix is what decides WHICH ragged path is exercised
// and the two are disjoint, so a gate that only runs prefix 0 tests half the
// code: at prefix 0 every extent is <= M, so with S=8192 every row is routed to
// the identity/exact path and the SAMPLER never sees a ragged row at all
// (M=512 N=131072 reported fallback_rows for all 512 rows). aiter's real
// prefill config uses prefix 131072, where every extent is long and ragged.
static int g_ragged_prefix = 0;
static int g_row_starts_stride = 0;
static int g_values = 0;            // also emit the selected scores

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
#if SELECT_CLEAR_ON_READ
  // Zeroed once here; from then on the scan re-zeroes each bucket as it reads
  // it, so the per-pass clear loop and its barrier are gone.
  for (int i = threadIdx.x; i < HIST_SLOTS; i += blockDim.x) s_hist[i] = 0;
  __syncthreads();
#endif
  for (int p = start; p < npasses; p++) {
    const int sh = radix_shift(p);
    const int hshift = sh + 8;
    const bool filter = (p > 0);
#if !SELECT_CLEAR_ON_READ
    for (int i = threadIdx.x; i < HIST_SLOTS; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();
#endif
    // Do NOT add an active-set min/max here to exit early once the pivot is
    // pinned. It was tried: accumulating amn/amx in this loop (the reads are
    // already happening) and breaking when they agree made small_n 21-39%
    // SLOWER and the anchor 615.5 -> 662.8 us. The two extra barriers per pass
    // in the reduction, plus the register pressure in this loop, cost far more
    // than the single pass the exit saves. See knowledge/known_bad.md.
#if HIST_AGG_ROUNDS
    // Uniform trip count, because the aggregation ballots need every lane of the
    // wave in the same iteration; the strided form below exits at different
    // iterations per lane. Same shape as block_gather_topk's loop.
    for (int i0 = 0; i0 < c; i0 += blockDim.x) {
      const int i = i0 + threadIdx.x;
      const bool live = (i < c);
      const uint32_t k = live ? s_keys[i] : 0u;
      const bool act = live && (!filter || (k >> hshift) == (pivot >> hshift));
      hist_add_aggregated(s_hist, (k >> sh) & 0xFFu, rep, act, HIST_AGG_ROUNDS);
    }
#else
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
      uint32_t k = s_keys[i];
      if (!filter || (k >> hshift) == (pivot >> hshift))
#if ABLATE_HIST_ATOMIC
        // TIMING ABLATION, WRONG RESULTS: same address pattern and LDS traffic,
        // but no atomicity, so the delta is exactly what the atomic plus its
        // bucket conflict costs. Prices the ceiling of any wave-aggregation.
        s_hist[((k >> sh) & 0xFFu) * HIST_REP + rep] = 1u;
#else
        atomicAdd(&s_hist[((k >> sh) & 0xFFu) * HIST_REP + rep], 1u);
#endif
    }
#endif
    __syncthreads();
#if SELECT_WAVE0_SCAN
    block_find_pivot_bucket_wave0<SELECT_CLEAR_ON_READ != 0>(s_hist, s_scan, ek);
#elif SELECT_FUSED_REDUCE
    block_find_pivot_bucket_rep<SELECT_CLEAR_ON_READ != 0>(s_hist, s_scan, ek);
#else
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
#endif
    pivot |= (s_scan[0] << sh);
    ek -= (int)s_scan[1];
  }
  eq_needed = ek;
}

// Active-set compaction variant of the select above, for Phase A only.
//
// The filter-rescan form reads all `c` keys on every pass and discards the
// ~255/256 that do not match the pivot prefix. Those later passes were measured
// at 30-34% of phase_small_n_topk while carrying almost no work (see
// knowledge/known_bad.md, "Early-exiting the radix select once the pivot is
// pinned"). This form compacts the survivors after each pass, so pass p+1 reads
// only what pass p kept: ~2c key reads over three passes instead of 3c.
//
// The MECHANISM is the point, not the saving. An earlier attempt removed the
// same passes with a block-wide active-set min/max early exit and came out
// 21-39% SLOWER, because that test costs two barriers per pass. Compaction here
// is WAVE-PRIVATE, so it adds no barrier and no LDS -- the same ownership trick
// that lets the shipped Phase B filter run with no atomic of any kind.
//
// Wave w owns [lo, lo + n_active) of s_keys and compacts its own survivors to
// the front of its own segment, carrying the count in a wave-uniform register.
// In place is safe because within an iteration every lane reads before any lane
// writes, and a survivor lands at or below the index it came from
// (wcnt <= j and popcount(ballot & lt) <= lane), so nothing is overwritten
// before it has been read. Waves own disjoint segments, so there is no
// cross-wave hazard either.
//
// No prefix_skip path: a compacted pass needs no prefix filter to begin with,
// and Phase A never asked for prefix_skip anyway (its samples span the whole
// row, so no pass is ever skippable -- measured +8.0 us when tried).
__device__ __forceinline__ void block_select_lds_compact(uint32_t* __restrict__ s_keys, int c,
                                                         int K, uint32_t* __restrict__ s_hist,
                                                         uint32_t* __restrict__ s_red,
                                                         uint32_t* __restrict__ s_scan,
                                                         uint32_t& pivot, int& eq_needed,
                                                         int npasses) {
  const int rep = threadIdx.x & (HIST_REP - 1);
  const int lane = threadIdx.x & (WAVE_SIZE - 1);
  const int wid = threadIdx.x / WAVE_SIZE;
  const int nwaves = blockDim.x / WAVE_SIZE;
  const uint64_t lt = (1ull << lane) - 1ull;

  if (threadIdx.x == 0) {
    s_scan[0] = 0;
    s_scan[1] = 0;
  }

  const int chunk = (c + nwaves - 1) / nwaves;
  const int lo = min(wid * chunk, c);
  int n_active = min(lo + chunk, c) - lo;

  pivot = 0;
  int ek = K;
  for (int p = 0; p < npasses; p++) {
    const int sh = radix_shift(p);
    for (int i = threadIdx.x; i < HIST_SLOTS; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();
    for (int i = lane; i < n_active; i += WAVE_SIZE) {
      const uint32_t k = s_keys[lo + i];
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
    // Ends in a barrier, so every read of s_hist / s_red for this pass is done
    // before the next iteration zeroes them.
    block_find_pivot_bucket(HIST_REP > 1 ? s_red : s_hist, s_scan, ek);
    pivot |= (s_scan[0] << sh);
    ek -= (int)s_scan[1];

    if (p + 1 == npasses) break;
    const uint32_t want = pivot >> sh;
    int wcnt = 0;
    for (int j = 0; j < n_active; j += WAVE_SIZE) {
      const int i = j + lane;
      const bool live = (i < n_active);
      const uint32_t k = live ? s_keys[lo + i] : 0u;
      const bool keep = live && ((k >> sh) == want);
      const uint64_t bal = __ballot(keep);
      if (keep) s_keys[lo + wcnt + __popcll(bal & lt)] = k;
      wcnt += __popcll(bal);
    }
    n_active = wcnt;
  }
  eq_needed = ek;
}

// Same select but streaming the row from global memory (used by the fallback /
// direct oracle, where the row is far too large for LDS).
template <bool RAGGED>
__device__ __forceinline__ void block_select_stream(const float* __restrict__ row, int n4,
                                                    int len, int K, uint32_t* __restrict__ s_hist,
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
      vfloat4 v = load_row_f4<RAGGED>(row, i, len);
      uint32_t k[FP32_EPT] = {fp32_to_sortable(v[0]), fp32_to_sortable(v[1]),
                              fp32_to_sortable(v[2]), fp32_to_sortable(v[3])};
#pragma unroll
      for (int e = 0; e < FP32_EPT; e++) {
        const int col = i * FP32_EPT + e;
        if ((!RAGGED || col < len) &&
            (p == 0 || (k[e] >> hshift) == (pivot >> hshift)))
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
template <bool RAGGED, bool WRITE_VALUES>
__device__ __forceinline__ void exact_row_select(const float* __restrict__ input, int pitch,
                                                 RowExtents<RAGGED> extents, int K, int row,
                                                 int* __restrict__ out, float* __restrict__ out_val,
                                                 uint32_t* __restrict__ s_hist,
                                                 uint32_t* __restrict__ s_red,
                                                 uint32_t* __restrict__ s_scan,
                                                 unsigned* __restrict__ s_wgt,
                                                 unsigned* __restrict__ s_weq) {
  const int row_start = RAGGED ? extents.row_start(row, pitch) : 0;
  const int len = row_len_of<RAGGED>(row, pitch, extents);
  const float* rif0 = input + (size_t)row * pitch + row_start;
  if (RAGGED && len <= K) {
    emit_identity_row<WRITE_VALUES>(out, out_val, rif0, row_start, len, K);
    return;
  }
  const int k_out = RAGGED ? k_take_dev(K, len) : K;
  const int n4 = RAGGED ? n4_cover(len) : (pitch / FP32_EPT);
  uint32_t pivot;
  int eq_needed;
  block_select_stream<RAGGED>(rif0, n4, len, k_out, s_hist, s_red, s_scan, pivot, eq_needed);
  if (threadIdx.x == 0) {
    *s_wgt = 0;
    *s_weq = 0;
  }
  __syncthreads();
  const float* rif = rif0;
  if constexpr (RAGGED) {
    block_gather_topk<WRITE_VALUES>(len, pivot, k_out - eq_needed, eq_needed, out, out_val, s_wgt,
                                    s_weq, [&](int i) { return fp32_to_sortable(rif[i]); },
                                    [&](int i) { return row_start + i; });
  } else {
    block_gather_topk<WRITE_VALUES>(len, pivot, k_out - eq_needed, eq_needed, out, out_val, s_wgt,
                                    s_weq, [&](int i) { return fp32_to_sortable(rif[i]); },
                                    [](int i) { return i; });
  }
  if (RAGGED && k_out < K) {
    __syncthreads();
    pad_topk_tail<WRITE_VALUES>(out, out_val, k_out, K);
  }
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
// COMPACT selects the active-set-compacting select instead of the filter-rescan
// one. It is a template parameter and not a kernarg on purpose: one unused
// kernarg on phase_small_n_topk alone cost +1.0% of its geomean
// (knowledge/known_bad.md), so an A/B knob must compile out entirely.
template <bool RAGGED, bool COMPACT = false>
__global__ __launch_bounds__(1024) void phase_a_threshold(const float* __restrict__ input, int pitch,
                                                          RowExtents<RAGGED> extents, int rank,
                                                          int S, int npasses, int chunk_stride_host,
                                                          uint32_t* __restrict__ threshold,
                                                          float* __restrict__ threshold_f,
                                                          unsigned int* __restrict__ cand_reserved,
                                                          unsigned int* __restrict__ cand_bad,
                                                          int* __restrict__ fb_count, int K) {
  const int row = blockIdx.x;
  const int len = row_len_of<RAGGED>(row, pitch, extents);
  const float* ri = input + (size_t)row * pitch + (RAGGED ? extents.row_start(row, pitch) : 0);

  if (threadIdx.x == 0) {
    if (cand_reserved) cand_reserved[row] = 0u;
    if (cand_bad) cand_bad[row] = 0u;
    if (row == 0) *fb_count = 0;
  }

  // Two separate reasons a row cannot go through the sampler, and they do not
  // coincide: len <= K means every element is selected so there is nothing to
  // rank (aiter's identity case), while len < S means the sampler would read
  // past the row. Either way Phase B must collect nothing for this row, so the
  // threshold is +inf and Phase C takes it through the exact/identity path.
  if (RAGGED && (len <= K || len < S)) {
    // +inf is the whole routing signal: Phase B keeps nothing below it, so
    // cand_count lands under k_out and Phase C takes the row through the
    // exact/identity path. Deliberately NOT appended to fb_rows here -- Phase C
    // appends every row it routes, and doing it in both places counted a short
    // row twice, overflowing the M-entry fb_rows (M=512 triangular reported
    // fallback_rows=1024 and wrote 512 ints past the end of the buffer).
    if (threadIdx.x == 0) {
      threshold[row] = 0u;
      threshold_f[row] = __builtin_inff();
    }
    return;
  }

  extern __shared__ uint32_t s_keys[];
  __shared__ uint32_t s_hist[HIST_SLOTS];
  __shared__ uint32_t s_red[256];
  __shared__ uint32_t s_scan[2];
  __shared__ uint32_t s_mm[2 * MAX_WAVES_PER_BLOCK];

  const int chunks = S / SAMPLE_CHUNK_ELEMS;
  const int chunk_stride =
      RAGGED ? sample_chunk_stride(len, chunks) : chunk_stride_host;
  const int rank_row =
      RAGGED && len != pitch ? max(1, (int)((double)rank * pitch / len)) : rank;

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
  if constexpr (COMPACT) {
    block_select_lds_compact(s_keys, S, rank_row, s_hist, s_red, s_scan, pivot, eq_needed, npasses);
  } else {
    block_select_lds(s_keys, S, rank_row, s_hist, s_red, s_scan, s_mm, pivot, eq_needed, npasses);
  }
  if (threadIdx.x == 0) {
    threshold[row] = pivot;
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
template <int ABLATE, bool RAGGED>
__global__ void phase_b_filter_waveseg(const float* __restrict__ input, int pitch,
                                       RowExtents<RAGGED> extents,
                                       const float* __restrict__ threshold_f,
                                       uint64_t* __restrict__ cand_pack,
                                       int* __restrict__ cand_seg,
                                       unsigned int* __restrict__ cand_count, int seg_stride) {
  const int row = blockIdx.x;
  const int len = row_len_of<RAGGED>(row, pitch, extents);
  const float* ri = input + (size_t)row * pitch + (RAGGED ? extents.row_start(row, pitch) : 0);
  const float th = threshold_f[row];

  const int lane = threadIdx.x & (WAVE_SIZE - 1);
  const int wid = threadIdx.x / WAVE_SIZE;
  const int nwaves = blockDim.x / WAVE_SIZE;
  const uint64_t lt = (1ull << lane) - 1ull;

  uint64_t* seg = cand_pack + (size_t)row * CAND_SLOTS_PER_ROW + (size_t)wid * seg_stride;

  const int n4 = RAGGED ? n4_cover(len) : (pitch / FP32_EPT);
  const int stride = blockDim.x;
  const int iters = (n4 + stride - 1) / stride;

  int wcnt = 0;
  bool overflow = false;

  for (int it = 0; it < iters; it++) {
    const int i = it * stride + threadIdx.x;
    vfloat4 v = {0.f, 0.f, 0.f, 0.f};
    const bool live = (i < n4);
    if (live) v = load_row_f4<RAGGED>(ri, i, len);
    const int base_idx = i * FP32_EPT;

    const uint64_t b0 = __ballot(live && !(v[0] < th) && (!RAGGED || base_idx + 0 < len));
    const uint64_t b1 = __ballot(live && !(v[1] < th) && (!RAGGED || base_idx + 1 < len));
    const uint64_t b2 = __ballot(live && !(v[2] < th) && (!RAGGED || base_idx + 2 < len));
    const uint64_t b3 = __ballot(live && !(v[3] < th) && (!RAGGED || base_idx + 3 < len));
    const int t0 = __popcll(b0);
    const int t1 = t0 + __popcll(b1);
    const int t2 = t1 + __popcll(b2);
    const int wtotal = t2 + __popcll(b3);

    if (ABLATE == 2) {
      wcnt += wtotal;
      continue;
    }

    if (wtotal > 0) {
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
template <bool RAGGED>
__global__ __launch_bounds__(512) void phase_b_filter_wavestage(
    const float* __restrict__ input, int pitch, RowExtents<RAGGED> extents,
    const float* __restrict__ threshold_f, uint64_t* __restrict__ cand_pack, int* __restrict__ cand_seg,
    unsigned int* __restrict__ cand_count, int seg_stride) {
  const int row = blockIdx.x;
  const int len = row_len_of<RAGGED>(row, pitch, extents);
  const float* ri = input + (size_t)row * pitch + (RAGGED ? extents.row_start(row, pitch) : 0);
  const float th = threshold_f[row];

  const int lane = threadIdx.x & (WAVE_SIZE - 1);
  const int wid = threadIdx.x / WAVE_SIZE;
  const int nwaves = blockDim.x / WAVE_SIZE;
  const uint64_t lt = (1ull << lane) - 1ull;

  __shared__ uint64_t wbuf[WSTAGE_WAVES * WSTAGE_CAP];
  uint64_t* buf = wbuf + (size_t)wid * WSTAGE_CAP;
  uint64_t* seg = cand_pack + (size_t)row * CAND_SLOTS_PER_ROW + (size_t)wid * seg_stride;

  const int n4 = RAGGED ? n4_cover(len) : (pitch / FP32_EPT);
  const int stride = blockDim.x;
  const int iters = (n4 + stride - 1) / stride;

  int wcnt = 0;
  int bcnt = 0;
  bool overflow = false;

  for (int it = 0; it < iters; it++) {
    const int i = it * stride + threadIdx.x;
    vfloat4 v = {0.f, 0.f, 0.f, 0.f};
    const bool live = (i < n4);
    if (live) v = load_row_f4<RAGGED>(ri, i, len);

    const int base_idx = i * FP32_EPT;
    const uint64_t b0 = __ballot(live && !(v[0] < th) && (!RAGGED || base_idx + 0 < len));
    const uint64_t b1 = __ballot(live && !(v[1] < th) && (!RAGGED || base_idx + 1 < len));
    const uint64_t b2 = __ballot(live && !(v[2] < th) && (!RAGGED || base_idx + 2 < len));
    const uint64_t b3 = __ballot(live && !(v[3] < th) && (!RAGGED || base_idx + 3 < len));
    const int t0 = __popcll(b0);
    const int t1 = t0 + __popcll(b1);
    const int t2 = t1 + __popcll(b2);
    const int wtotal = t2 + __popcll(b3);

    if (wtotal > 0) {
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
template <bool STATIC_CAP, bool RAGGED, bool WRITE_VALUES>
__global__ void phase_c_select_waveseg(const float* __restrict__ input, int pitch,
                                       RowExtents<RAGGED> extents,
                                       const uint64_t* __restrict__ cand_pack,
                                       const int* __restrict__ cand_seg,
                                       const unsigned int* __restrict__ cand_count, int seg_stride,
                                       int nwaves_b, int K, int cap, TopkOut<WRITE_VALUES> dst,
                                       int* __restrict__ fb_rows, int* __restrict__ fb_count,
                                       int npasses) {
  const int row = blockIdx.x;
  const int row_start = RAGGED ? extents.row_start(row, pitch) : 0;
  const int len = row_len_of<RAGGED>(row, pitch, extents);
  const unsigned int c_raw = cand_count[row];
  const int k_out = RAGGED ? k_take_dev(K, len) : K;

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

  int* out_row = dst.idx_row(row, K);
  float* val_row = dst.val_row(row, K);
  // `len <= K` is routed unconditionally rather than via cand_count, so the
  // identity emit does not depend on how many candidates Phase B happened to
  // keep: under the `inf` distribution a row of +inf values passes the +inf
  // threshold and can push cand_count above k_out, which would otherwise send
  // an identity row down the candidate path and order it differently to aiter.
  if ((RAGGED && len <= K) || c_raw < (unsigned)k_out || c_raw > (unsigned)cap) {
    if (threadIdx.x == 0) fb_rows[atomicAdd(fb_count, 1)] = row;
    exact_row_select<RAGGED, WRITE_VALUES>(input, pitch, extents, K, row, out_row, val_row, s_hist,
                                           s_red, s_scan, &s_wgt, &s_weq);
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
  block_select_lds(s_keys, c, k_out, s_hist, s_red, s_scan, s_mm, pivot, eq_needed, npasses,
                   /*prefix_skip=*/true);

  if constexpr (RAGGED) {
    block_gather_topk<WRITE_VALUES>(c, pivot, k_out - eq_needed, eq_needed, out_row, val_row, &s_wgt,
                                    &s_weq, [&](int i) { return s_keys[i]; },
                                    [&](int i) { return row_start + s_idx[i]; });
  } else {
    block_gather_topk<WRITE_VALUES>(c, pivot, k_out - eq_needed, eq_needed, out_row, val_row, &s_wgt,
                                    &s_weq, [&](int i) { return s_keys[i]; },
                                    [&](int i) { return s_idx[i]; });
  }
  if (RAGGED && k_out < K) {
    __syncthreads();
    pad_topk_tail<WRITE_VALUES>(out_row, val_row, k_out, K);
  }
}

// ---------------------------------------------------------------------------
// Phase D: exact full-row select over a compacted row list. Now used only by
// --pipeline direct as the independent oracle; the fused path folds the same
// work into Phase C to save a dispatch.
// ---------------------------------------------------------------------------
template <bool RAGGED, bool WRITE_VALUES>
__global__ __launch_bounds__(1024) void phase_d_fallback(const float* __restrict__ input, int pitch,
                                                         RowExtents<RAGGED> extents, int K,
                                                         const int* __restrict__ fb_rows,
                                                         const int* __restrict__ fb_count,
                                                         TopkOut<WRITE_VALUES> dst) {
  const int count = *fb_count;

  __shared__ uint32_t s_hist[HIST_SLOTS];
  __shared__ uint32_t s_red[256];
  __shared__ uint32_t s_scan[2];
  __shared__ unsigned s_wgt;
  __shared__ unsigned s_weq;

  for (int slot = blockIdx.y; slot < count; slot += gridDim.y) {
    const int row = fb_rows[slot];
    const int len = row_len_of<RAGGED>(row, pitch, extents);
    exact_row_select<RAGGED, WRITE_VALUES>(input, pitch, extents, K, row, dst.idx_row(row, K),
                                           dst.val_row(row, K), s_hist, s_red, s_scan, &s_wgt,
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

// Builds the output bundle for the requested instantiation. The WRITE_VALUES
// =false form ignores d_val, so a stray non-null pointer cannot quietly turn
// into stores the caller did not ask for.
template <bool WRITE_VALUES>
static TopkOut<WRITE_VALUES> make_topk_out(int* d_idx, float* d_val);

template <>
TopkOut<false> make_topk_out<false>(int* d_idx, float*) {
  return TopkOut<false>{d_idx};
}

template <>
TopkOut<true> make_topk_out<true>(int* d_idx, float* d_val) {
  return TopkOut<true>{d_idx, d_val};
}

template <bool RAGGED>
static RowExtents<RAGGED> make_row_extents(const int* d_starts, const int* d_ends);
template <>
RowExtents<false> make_row_extents<false>(const int*, const int*) {
  return RowExtents<false>{nullptr};
}
template <>
RowExtents<true> make_row_extents<true>(const int* d_starts, const int* d_ends) {
  return RowExtents<true>{d_starts, d_ends};
}

template <bool RAGGED, bool WRITE_VALUES>
static void topk_small_n(const float* d_in, int M, int pitch, const int* d_row_starts,
                         const int* d_row_ends, int K,
                         int* d_idx, float* d_val, hipStream_t s) {
  phase_small_n_topk<RAGGED, WRITE_VALUES>
      <<<M, small_n_block(M, pitch), (size_t)pitch * sizeof(uint32_t), s>>>(
          d_in, pitch, make_row_extents<RAGGED>(d_row_starts, d_row_ends), K, make_topk_out<WRITE_VALUES>(d_idx, d_val),
          g_small_n_passes);
}

template <bool RAGGED, bool WRITE_VALUES>
static void topk_fused_impl(const float* d_in, int M, int pitch, const int* d_row_starts,
                            const int* d_row_ends, int K,
                            int* d_idx, float* d_val, Bufs& b, const ShapeParams& sp,
                            hipStream_t s) {
  const RowExtents<RAGGED> ext = make_row_extents<RAGGED>(d_row_starts, d_row_ends);
  const auto dst = make_topk_out<WRITE_VALUES>(d_idx, d_val);
  const int S = sp.S;
  const float margin = sp.margin;
  const int rank = g_sample_rank > 0 ? g_sample_rank : sp.rank;
  const int cap = sp.cap;
  const int n4 = pitch / FP32_EPT;
  const int gx = std::max(1, std::min(g_cf_gx, n4 / g_cf_block));
  const int nwaves_b = std::max(1, g_cf_block / WAVE_SIZE);
  const int seg_stride = CAND_SLOTS_PER_ROW / nwaves_b;

  const int a_block = g_phase_a_block > 0
                          ? g_phase_a_block
                          : occupancy_block_threads(M, PHASE_A_STATIC_LDS + S * 4, 0);
  const int c_block = g_phase_c_block > 0
                          ? g_phase_c_block
                          : occupancy_block_threads(M, PHASE_C_STATIC_LDS + cap * 8, 0);

  const bool coop = sp.coop_g > 1;

  const int chunk_stride = sample_chunk_stride(pitch, S / SAMPLE_CHUNK_ELEMS);

  if (g_fuse_ab) {
    phase_ab_fused<RAGGED><<<M, a_block, (size_t)S * sizeof(uint32_t), s>>>(
        d_in, pitch, ext, rank, S, g_phase_a_passes, chunk_stride, seg_stride, b.cand_pack,
        b.cand_seg, b.cand_count, b.fb_count, K);
  } else {
    if (g_phase_a_compact) {
      phase_a_threshold<RAGGED, true><<<M, a_block, (size_t)S * sizeof(uint32_t), s>>>(
          d_in, pitch, ext, rank, S, g_phase_a_passes, chunk_stride, b.threshold, b.threshold_f,
          coop ? b.cand_reserved : nullptr, coop ? b.cand_bad : nullptr, b.fb_count, K);
    } else {
      phase_a_threshold<RAGGED, false><<<M, a_block, (size_t)S * sizeof(uint32_t), s>>>(
          d_in, pitch, ext, rank, S, g_phase_a_passes, chunk_stride, b.threshold, b.threshold_f,
          coop ? b.cand_reserved : nullptr, coop ? b.cand_bad : nullptr, b.fb_count, K);
    }

    if (coop) {
      phase_b_filter_coop<RAGGED><<<dim3(sp.coop_g, M), g_cf_block, 0, s>>>(
          d_in, pitch, ext, n4, b.threshold_f, b.cand_pack, b.cand_reserved, b.cand_bad,
          cap);
    } else if (g_phase_b == 4) {
      phase_b_filter_wavestage<RAGGED><<<M, g_cf_block, 0, s>>>(
          d_in, pitch, ext, b.threshold_f, b.cand_pack, b.cand_seg, b.cand_count, seg_stride);
    } else {
      phase_b_filter_waveseg<0, RAGGED><<<M, g_cf_block, 0, s>>>(
          d_in, pitch, ext, b.threshold_f, b.cand_pack, b.cand_seg, b.cand_count, seg_stride);
    }
  }

  if (coop) {
    const size_t lds_c = (size_t)cap * (sp.keys_only_c ? sizeof(uint32_t)
                                                       : sizeof(uint32_t) + sizeof(int));
    phase_c_select_contig<RAGGED, WRITE_VALUES><<<M, c_block, lds_c, s>>>(
        d_in, pitch, ext, b.cand_pack, b.cand_reserved, b.cand_bad, b.cand_count, cap, K,
        dst, b.fb_rows, b.fb_count, g_phase_c_passes, sp.keys_only_c);
  } else if (cap <= PHASE_C_CAP) {
    phase_c_select_waveseg<true, RAGGED, WRITE_VALUES><<<M, c_block, 0, s>>>(
        d_in, pitch, ext, b.cand_pack, b.cand_seg, b.cand_count, seg_stride, nwaves_b, K,
        cap, dst, b.fb_rows, b.fb_count, g_phase_c_passes);
  } else {
    const size_t lds_c = (size_t)cap * (sizeof(uint32_t) + sizeof(int));
    phase_c_select_waveseg<false, RAGGED, WRITE_VALUES><<<M, c_block, lds_c, s>>>(
        d_in, pitch, ext, b.cand_pack, b.cand_seg, b.cand_count, seg_stride, nwaves_b, K,
        cap, dst, b.fb_rows, b.fb_count, g_phase_c_passes);
  }

  (void)margin;
  (void)gx;
}

// RAGGED and WRITE_VALUES are both compile-time, so a runtime dispatcher has to
// pick one of four instantiations. It is expanded in ONE place, here, rather
// than at each launch site: the four phase kernels would otherwise each carry
// the same 4-way if-tree, and the launch-site version is where a missed branch
// silently runs the wrong instantiation.
template <bool RAGGED, bool WRITE_VALUES>
static void topk_indices_inst(const float* d_in, int M, int pitch, const int* d_row_starts,
                              const int* d_row_ends, int K,
                              int* d_idx, float* d_val, Bufs& b, const ShapeParams& sp,
                              hipStream_t s) {
  if (sp.path == PATH_SMALL_N) {
    topk_small_n<RAGGED, WRITE_VALUES>(d_in, M, pitch, d_row_starts, d_row_ends, K, d_idx, d_val, s);
    return;
  }
  topk_fused_impl<RAGGED, WRITE_VALUES>(d_in, M, pitch, d_row_starts, d_row_ends, K, d_idx, d_val, b, sp, s);
}

static void topk_indices(const float* d_in, int M, int pitch, const int* d_row_starts,
                       const int* d_row_ends, int K,
                         int* d_idx, float* d_val, Bufs& b, int smc, hipStream_t s) {
  const int k_geom = g_ragged ? geometry_k_ragged(K, pitch) : K;
  ShapeParams sp = derive_shape_params(M, pitch, k_geom, g_margin, g_sample_s, g_coop_g,
                                       (TopkPath)g_path_override);
  g_sample_s = sp.S > 0 ? sp.S : g_sample_s;
  if (g_ragged) {
    if (d_val) topk_indices_inst<true, true>(d_in, M, pitch, d_row_starts, d_row_ends, K, d_idx, d_val, b, sp, s);
    else topk_indices_inst<true, false>(d_in, M, pitch, d_row_starts, d_row_ends, K, d_idx, nullptr, b, sp, s);
  } else {
    if (d_val) topk_indices_inst<false, true>(d_in, M, pitch, nullptr, nullptr, K, d_idx, d_val, b, sp, s);
    else topk_indices_inst<false, false>(d_in, M, pitch, nullptr, nullptr, K, d_idx, nullptr, b, sp, s);
  }
  (void)smc;
}

static void topk_fused(const float* d_in, int M, int pitch, const int* d_row_starts,
                       const int* d_row_ends, int K,
                       int* d_idx, float* d_val, Bufs& b, int smc, hipStream_t s) {
  topk_indices(d_in, M, pitch, d_row_starts, d_row_ends, K, d_idx, d_val, b, smc, s);
}

static void topk_direct(const float* d_in, int M, int pitch, const int* d_row_starts,
                       const int* d_row_ends, int K,
                        int* d_idx, float* d_val, Bufs& b, hipStream_t s) {
  fill_identity_rows<<<(M + 255) / 256, 256, 0, s>>>(b.fb_rows, b.fb_count, M);
  const dim3 g(1, FB_GRID);
  if (g_ragged) {
    const RowExtents<true> ext = make_row_extents<true>(d_row_starts, d_row_ends);
    if (d_val)
      phase_d_fallback<true, true><<<g, 1024, 0, s>>>(d_in, pitch, ext, K, b.fb_rows,
                                                      b.fb_count,
                                                      make_topk_out<true>(d_idx, d_val));
    else
      phase_d_fallback<true, false><<<g, 1024, 0, s>>>(d_in, pitch, ext, K, b.fb_rows,
                                                       b.fb_count,
                                                       make_topk_out<false>(d_idx, nullptr));
  } else {
    const RowExtents<false> ext = make_row_extents<false>(nullptr, nullptr);
    if (d_val)
      phase_d_fallback<false, true><<<g, 1024, 0, s>>>(d_in, pitch, ext, K, b.fb_rows, b.fb_count,
                                                       make_topk_out<true>(d_idx, d_val));
    else
      phase_d_fallback<false, false><<<g, 1024, 0, s>>>(d_in, pitch, ext, K, b.fb_rows, b.fb_count,
                                                        make_topk_out<false>(d_idx, nullptr));
  }
}

static void run_topk(const float* d_in, int M, int pitch, const int* d_row_starts,
                     const int* d_row_ends, int K, int* d_idx,
                     float* d_val, Bufs& b, int smc, hipStream_t s) {
  if (g_pipeline_direct)
    topk_direct(d_in, M, pitch, d_row_starts, d_row_ends, K, d_idx, d_val, b, s);
  else
    topk_fused(d_in, M, pitch, d_row_starts, d_row_ends, K, d_idx, d_val, b, smc, s);
}

// ---- AITER_EXPORT_END ----
// Everything above this line is the kernel plus its dispatch and is portable;
// scripts/export_aiter_op.py copies exactly that region into aiter and appends
// csrc/topk_aiter_entry.inc.hip. Everything below is harness only (verification
// oracles, timing, CLI). Moving code across this line changes what ships, so
// re-run the export and its diff check after doing so.

static bool verify_row_cpu(const float* row, int row_start, int len, int K, const int* idx) {
  const int k_take = len < K ? len : K;
  std::vector<uint32_t> sv(len);
  const int row_end = row_start + len;
  for (int i = 0; i < len; i++) sv[i] = fp32_to_sortable_host(row[row_start + i]);
  std::vector<uint32_t> got;
  got.reserve(k_take);
  for (int i = 0; i < k_take; i++) {
    if (idx[i] < row_start || idx[i] >= row_end) return false;
    got.push_back(fp32_to_sortable_host(row[idx[i]]));
  }
  for (int i = k_take; i < K; i++)
    if (idx[i] != -1) return false;
  std::sort(got.begin(), got.end(), std::greater<uint32_t>());
  std::vector<uint32_t> ref(sv);
  std::partial_sort(ref.begin(), ref.begin() + k_take, ref.end(), std::greater<uint32_t>());
  ref.resize(k_take);
  return got == ref;
}

static bool verify_row_gpu_oracle(const float* d_in, int pitch, const int* d_row_starts,
                                  const int* d_row_ends, int K, int row, int* d_idx, Bufs& b) {
  int h_rows[1] = {row};
  int h_count = 1;
  HIP_CHECK(hipMemcpy(b.fb_rows, h_rows, sizeof(h_rows), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(b.fb_count, &h_count, sizeof(int), hipMemcpyHostToDevice));
  // WRITE_VALUES=false: the oracle exists to produce indices to compare
  // against, and the value check is a separate, index-anchored one.
  if (g_ragged) {
    const RowExtents<true> ext = make_row_extents<true>(d_row_starts, d_row_ends);
    phase_d_fallback<true, false><<<dim3(1, 1), 1024>>>(d_in, pitch, ext, K, b.fb_rows, b.fb_count,
                                                       make_topk_out<false>(d_idx, nullptr));
  } else {
    const RowExtents<false> ext = make_row_extents<false>(nullptr, nullptr);
    phase_d_fallback<false, false><<<dim3(1, 1), 1024>>>(d_in, pitch, ext, K, b.fb_rows, b.fb_count,
                                                         make_topk_out<false>(d_idx, nullptr));
  }
  HIP_CHECK(hipDeviceSynchronize());
  return true;
}

static bool row_idx_multiset_match(const float* row, int row_start, int len, int K, const int* got,
                                   const int* ref) {
  const int row_end = row_start + len;
  const int k_take = len < K ? len : K;
  std::vector<uint32_t> gv, rv;
  gv.reserve(k_take);
  rv.reserve(k_take);
  for (int i = 0; i < k_take; i++) {
    if (got[i] < row_start || got[i] >= row_end || ref[i] < row_start || ref[i] >= row_end)
      return false;
    gv.push_back(fp32_to_sortable_host(row[got[i]]));
    rv.push_back(fp32_to_sortable_host(row[ref[i]]));
  }
  for (int i = k_take; i < K; i++)
    if (got[i] != -1 || ref[i] != -1) return false;
  std::sort(gv.begin(), gv.end(), std::greater<uint32_t>());
  std::sort(rv.begin(), rv.end(), std::greater<uint32_t>());
  return gv == rv;
}

static bool verify_rows_sampled(const float* d_in, int M, int pitch, const int* d_row_starts,
                                const int* d_row_ends, const int* h_row_starts,
                                const int* h_row_ends, int K, int* d_idx, const int* h_idx,
                                Bufs& b, int sample_n) {
  std::vector<int> rows;
  for (int i = 0; i < M; i += std::max(1, M / sample_n)) rows.push_back(i);
  if (rows.empty()) rows.push_back(0);

  std::vector<float> h_row((size_t)pitch);
  std::vector<int> oracle((size_t)K);
  for (int r : rows) {
    const int row_start = h_row_starts ? h_row_starts[r] : 0;
    const int len = h_row_ends ? (h_row_ends[r] - row_start) : pitch;
    HIP_CHECK(hipMemcpy(h_row.data(), d_in + (size_t)r * pitch, (size_t)pitch * sizeof(float),
                        hipMemcpyDeviceToHost));
    if (g_verify_oracle_gpu) {
      verify_row_gpu_oracle(d_in, pitch, d_row_starts, d_row_ends, K, r, d_idx, b);
      HIP_CHECK(hipMemcpy(oracle.data(), d_idx + (size_t)r * K, (size_t)K * sizeof(int),
                          hipMemcpyDeviceToHost));
    } else {
      const int k_take = len < K ? len : K;
      std::vector<uint32_t> sv(len);
      for (int i = 0; i < len; i++) sv[i] = fp32_to_sortable_host(h_row[row_start + i]);
      std::partial_sort(sv.begin(), sv.begin() + k_take, sv.end(), std::greater<uint32_t>());
      for (int i = 0; i < k_take; i++) {
        uint32_t want = sv[i];
        int found = -1;
        for (int j = 0; j < len; j++)
          if (fp32_to_sortable_host(h_row[row_start + j]) == want) {
            found = j;
            break;
          }
        if (found < 0) return false;
        oracle[i] = row_start + found;
      }
      for (int i = k_take; i < K; i++) oracle[i] = -1;
    }
    if (!row_idx_multiset_match(h_row.data(), row_start, len, K, h_idx + (size_t)r * K,
                                oracle.data()))
      return false;
  }
  return true;
}

// Values are checked AGAINST THE INDICES the same call emitted, not against a
// separately sorted reference: that is what makes the check able to fail. The
// kernel recovers the score from the sortable key rather than re-reading the
// row, so a wrong inverse, a wrong slot or a stale lane shows up as
// out_val[i] != input[row][out_idx[i]] -- a reference top-k of the same row
// would agree with a consistently-wrong value and prove nothing.
//
// Exact equality, not a tolerance: no arithmetic is performed on the value, so
// anything other than a bit-identical round-trip is a defect. NaN cannot appear
// from the inverse and is compared by bits so it is not silently accepted.
// Copies only the sampled rows off the device: a whole-input copy is 2.2 GB of
// host memory at M=4096 N=135168, which bounds which shapes the gate could
// include for no reason -- the check only ever reads `pitch` floats at a time.
static bool verify_values_rows(const float* d_in, int M, int pitch, const int* h_row_starts,
                               const int* h_row_ends, int K, const int* h_idx, const float* h_val,
                               int sample_n, std::string& why) {
  auto bits = [](float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
  };
  const uint32_t neg_inf = bits(-std::numeric_limits<float>::infinity());
  const int step = std::max(1, M / std::max(1, sample_n));
  std::vector<float> row((size_t)pitch);
  for (int r = 0; r < M; r += step) {
    HIP_CHECK(hipMemcpy(row.data(), d_in + (size_t)r * pitch, (size_t)pitch * sizeof(float),
                        hipMemcpyDeviceToHost));
    const int row_start = h_row_starts ? h_row_starts[r] : 0;
    const int len = h_row_ends ? (h_row_ends[r] - row_start) : pitch;
    const int row_end = row_start + len;
    const int k_take = std::min(K, len);
    for (int i = 0; i < K; i++) {
      const int idx = h_idx[(size_t)r * K + i];
      const uint32_t got = bits(h_val[(size_t)r * K + i]);
      if (i >= k_take || idx < 0) {
        if (idx != -1 || got != neg_inf) {
          char buf[192];
          snprintf(buf, sizeof(buf), "row %d slot %d: pad expected idx=-1 val=-inf, got idx=%d",
                   r, i, idx);
          why = buf;
          return false;
        }
        continue;
      }
      if (idx < row_start || idx >= row_end) {
        why = "index past the row extent";
        return false;
      }
      if (got != bits(row[idx])) {
        char buf[192];
        snprintf(buf, sizeof(buf), "row %d slot %d: val does not match input[%d]", r, i, idx);
        why = buf;
        return false;
      }
    }
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
          "  --phase-a-compact 0|1 (compact Phase A's active set between passes)\n"
          "  --input-bin PATH --dump-indices PATH --inject-fault 0|1|2 (1=index, 2=value)\n"
          "  --path auto|small_n|prefill|decode --coop-g G --fuse-ab 0|1 --hipgraph 0|1\n"
          "  --small-n-block B (256..1024, 0=auto)  --s-rule 0|1 (0=legacy R_TARGET)\n"
          "  --s-repair-search 0|1 (search for the smallest exact-stride S)\n"
          "  --verify-sample-rows N --verify-oracle gpu|cpu\n"
          "  --ragged 0|1 --ragged-prefix P (row r extent = P+r+1, clamped to N)\n"
          "  --row-starts-stride S (row r start = min(r*S, N-1); 0 = all zero)\n"
          "  --values 0|1 (also emit the selected scores)\n",
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
    else if (a == "--phase-a-compact") g_phase_a_compact = std::stoi(need());
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
    else if (a == "--s-repair-search") g_s_repair_search = std::stoi(need());
    else if (a == "--small-n-passes") g_small_n_passes = std::stoi(need());
    else if (a == "--verify-sample-rows") g_verify_sample_rows = std::stoi(need());
    else if (a == "--verify-oracle") g_verify_oracle_gpu = (need() == "gpu") ? 1 : 0;
    else if (a == "--ragged") g_ragged = std::stoi(need());
    else if (a == "--ragged-prefix") g_ragged_prefix = std::stoi(need());
    else if (a == "--row-starts-stride") g_row_starts_stride = std::stoi(need());
    else if (a == "--values") g_values = std::stoi(need());
    else {
      usage(argv[0]);
      exit(1);
    }
  }

  if (!g_ragged && K > N) {
    fprintf(stderr, "ERROR: topk=%d exceeds N=%d\n", K, N);
    return 2;
  }
  // An odd pitch is served through the RAGGED instantiation with full-row
  // extents, which is what the aiter entry always uses anyway. RAGGED=false
  // truncates its vector count (`pitch / FP32_EPT`) and applies no per-lane
  // bound, so it would drop the row's last 1-3 elements; giving it a bound would
  // cost the scored pow2 grid a compare per element for a case it never sees.
  // Routing instead keeps that path byte-identical and reuses a tested one.
  if (N % FP32_EPT != 0 && !g_ragged) {
    g_ragged = 1;
    g_ragged_prefix = N;      // every extent clamps to N, i.e. the whole row
    g_row_starts_stride = 0;  // every row starts at 0
  }

  ShapeParams shape =
      derive_shape_params(M, N, g_ragged ? geometry_k_ragged(K, N) : K, g_margin, g_sample_s,
                          g_coop_g, (TopkPath)g_path_override);
  if (g_sample_s <= 0 || g_path_override != PATH_SMALL_N) g_sample_s = shape.S > 0 ? shape.S : g_sample_s;
  if (!shape.geom_ok) {
    fprintf(stderr, "ERROR: shape M=%d N=%d K=%d incompatible (path=%d S=%d cap=%d)\n", M, N, K,
            (int)shape.path, shape.S, shape.cap);
    return 2;
  }
  if ((g_ragged ? geometry_k_ragged(K, N) : K) > shape.cap && shape.path != PATH_SMALL_N) {
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
  float* d_val = nullptr;
  int* d_row_ends = nullptr;
  int* d_row_starts = nullptr;
  std::vector<int> h_row_ends;
  std::vector<int> h_row_starts;
  HIP_CHECK(hipMalloc(&d_in, in_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_idx, out_elems * sizeof(int)));
  if (g_values) HIP_CHECK(hipMalloc(&d_val, out_elems * sizeof(float)));
  if (g_ragged) {
    // Triangular, the shape aiter's create_row_boundaries produces. Clamped to
    // the pitch because a row extent above it is not a ragged matrix at all:
    // at M > N the raw r+1 runs off the end of the allocation (M=4096 N=512
    // faulted with HIP 700 on row 512 onward).
    h_row_starts.resize(M);
    h_row_ends.resize(M);
    for (int r = 0; r < M; r++) {
      const int start =
          g_row_starts_stride > 0 ? std::min(r * g_row_starts_stride, std::max(0, N - 1)) : 0;
      h_row_starts[r] = start;
      h_row_ends[r] = std::min(g_ragged_prefix + r + 1, N);
      if (h_row_ends[r] <= start) h_row_ends[r] = std::min(start + 1, N);
    }
    HIP_CHECK(hipMalloc(&d_row_starts, (size_t)M * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_row_ends, (size_t)M * sizeof(int)));
    HIP_CHECK(hipMemcpy(d_row_starts, h_row_starts.data(), (size_t)M * sizeof(int),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_row_ends, h_row_ends.data(), (size_t)M * sizeof(int),
                        hipMemcpyHostToDevice));
  }

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
  run_topk(d_in, M, N, d_row_starts, d_row_ends, K, d_idx, d_val, bufs, info.sm_count, 0);
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
    // 1 corrupts an index, 2 corrupts only a VALUE. They have to be separate:
    // with one flag the index check fails first and returns, so the value check
    // never runs and is never shown to be failable.
    if (g_inject_fault == 1) h_idx[0] ^= 1;
    int bad_rows = 0;
    const size_t full_elems = (size_t)M * N;
    if (full_elems <= 64 * 1024 * 1024 && M <= 256) {
      std::vector<float> h_in(full_elems);
      HIP_CHECK(hipMemcpy(h_in.data(), d_in, full_elems * sizeof(float), hipMemcpyDeviceToHost));
      for (int r = 0; r < M; r++) {
        const int row_start = g_ragged ? h_row_starts[r] : 0;
        const int len = g_ragged ? (h_row_ends[r] - row_start) : N;
        if (!verify_row_cpu(h_in.data() + (size_t)r * N, row_start, len, K,
                            h_idx.data() + (size_t)r * K))
          bad_rows++;
      }
    } else if (!verify_rows_sampled(d_in, M, N, d_row_starts, d_row_ends,
                                    g_ragged ? h_row_starts.data() : nullptr,
                                    g_ragged ? h_row_ends.data() : nullptr, K, d_idx,
                                    h_idx.data(), bufs, g_verify_sample_rows)) {
      bad_rows = 1;
    }
    printf("VERIFY rows_fail=%d fallback_rows=%d pipeline=%s inject_fault=%d path=%s\n", bad_rows,
           fb, g_pipeline_direct ? "direct" : "fused", g_inject_fault, path_name);
    if (bad_rows != 0) return 2;

    if (g_values) {
      std::vector<float> h_val(out_elems);
      HIP_CHECK(hipMemcpy(h_val.data(), d_val, out_elems * sizeof(float), hipMemcpyDeviceToHost));
      if (g_inject_fault == 2) h_val[0] = -0.5f;
      std::string why;
      const bool ok = verify_values_rows(d_in, M, N, g_ragged ? h_row_starts.data() : nullptr,
                                         g_ragged ? h_row_ends.data() : nullptr, K, h_idx.data(),
                                         h_val.data(), g_verify_sample_rows, why);
      printf("VERIFY_VALUES ok=%d %s\n", ok ? 1 : 0, ok ? "" : why.c_str());
      if (!ok) return 2;
    }
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
      for (int w = 0; w < warmup; w++) run_topk(d_in, M, N, d_row_starts, d_row_ends, K, d_idx, d_val, bufs, info.sm_count, gs);
      HIP_CHECK(hipStreamSynchronize(gs));
      hipError_t cap_err = hipStreamBeginCapture(gs, hipStreamCaptureModeGlobal);
      if (cap_err == hipSuccess) {
        run_topk(d_in, M, N, d_row_starts, d_row_ends, K, d_idx, d_val, bufs, info.sm_count, gs);
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
        for (int w = 0; w < warmup; w++) run_topk(d_in, M, N, d_row_starts, d_row_ends, K, d_idx, d_val, bufs, info.sm_count, 0);
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
          run_topk(d_in, M, N, d_row_starts, d_row_ends, K, d_idx, d_val, bufs, info.sm_count, 0);
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
  if (d_val) (void)hipFree(d_val);
  if (d_row_starts) (void)hipFree(d_row_starts);
  if (d_row_ends) (void)hipFree(d_row_ends);
  return 0;
}
