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

// ---- AVO variation surface -------------------------------------------------
static int g_sample_rank = 0;       // 0 => derive from margin
static int g_sample_s = 8192;       // Phase A sample count (S=4096 leaves rows short of K)
static float g_margin = 1.4f;       // Phase B over-collection factor
static int g_cf_block = 512;        // Phase B block size
static int g_cf_gx = 16;            // Phase B blocks per row (grid.x)
static int g_use_nt_load = 0;       // non-temporal streaming loads in Phase B
// Phase B implementation: 0 = per-wave global atomic, 1 = block-aggregated
// global atomic, 2 = one block per row with an LDS counter and no global atomic.
static int g_phase_b = 3;
static int g_phase_c_block = 512;
static int g_phase_a_block = 512;
static int g_pipeline_direct = 0;
static int g_inject_fault = 0;
static int g_dump_stats = 0;

// ---------------------------------------------------------------------------
// Block-wide exact radix select over keys already resident in LDS.
// On return: pivot == the K-th largest sortable key, eq_needed == how many
// elements equal to pivot must be taken (so K - eq_needed are strictly greater).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void block_select_lds(const uint32_t* __restrict__ s_keys, int c, int K,
                                                 uint32_t* __restrict__ s_hist,
                                                 uint32_t* __restrict__ s_scan, uint32_t& pivot,
                                                 int& eq_needed) {
  pivot = 0;
  int ek = K;
  for (int p = 0; p < RADIX_PASSES; p++) {
    const int sh = radix_shift(p);
    const int hshift = (p == 0) ? 0 : sh + 8;
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
      uint32_t k = s_keys[i];
      if (p == 0 || (k >> hshift) == (pivot >> hshift))
        atomicAdd(&s_hist[(k >> sh) & 0xFFu], 1u);
    }
    __syncthreads();
    block_find_pivot_bucket(s_hist, s_scan, ek);
    pivot |= (s_scan[0] << sh);
    ek -= (int)s_scan[1];
    __syncthreads();
  }
  eq_needed = ek;
}

// Same select but streaming the row from global memory (used by the fallback /
// direct oracle, where the row is far too large for LDS).
__device__ __forceinline__ void block_select_stream(const vfloat4* __restrict__ row4, int n4, int K,
                                                    uint32_t* __restrict__ s_hist,
                                                    uint32_t* __restrict__ s_scan, uint32_t& pivot,
                                                    int& eq_needed) {
  pivot = 0;
  int ek = K;
  for (int p = 0; p < RADIX_PASSES; p++) {
    const int sh = radix_shift(p);
    const int hshift = (p == 0) ? 0 : sh + 8;
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
      vfloat4 v = row4[i];
      uint32_t k[FP32_EPT] = {fp32_to_sortable(v[0]), fp32_to_sortable(v[1]),
                              fp32_to_sortable(v[2]), fp32_to_sortable(v[3])};
#pragma unroll
      for (int e = 0; e < FP32_EPT; e++) {
        if (p == 0 || (k[e] >> hshift) == (pivot >> hshift))
          atomicAdd(&s_hist[(k[e] >> sh) & 0xFFu], 1u);
      }
    }
    __syncthreads();
    block_find_pivot_bucket(s_hist, s_scan, ek);
    pivot |= (s_scan[0] << sh);
    ek -= (int)s_scan[1];
    __syncthreads();
  }
  eq_needed = ek;
}

// ---------------------------------------------------------------------------
// Phase A: per-row sampled threshold, fully in LDS, one kernel, one block/row.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(1024) void phase_a_threshold(const float* __restrict__ input, int N,
                                                          int rank, int S,
                                                          uint32_t* __restrict__ threshold,
                                                          float* __restrict__ threshold_f) {
  const int row = blockIdx.x;
  const float* ri = input + (size_t)row * N;

  // Dynamic LDS so the footprint tracks the runtime S. A static [SAMPLE_S_MAX]
  // array costs 64 KB unconditionally and measurably crushes occupancy
  // (S=4096 regressed 0.774 -> 0.852 ms when it was sized statically).
  extern __shared__ uint32_t s_keys[];
  __shared__ uint32_t s_hist[256];
  __shared__ uint32_t s_scan[2];

  // dwordx4 per lane. A scalar `ri[chunk*stride+off]` loop moves only 4 B per
  // lane and left this kernel at 7x its own traffic floor.
  const int chunks = S / SAMPLE_CHUNK_ELEMS;
  const int chunk_stride = N / chunks;
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
  block_select_lds(s_keys, S, rank, s_hist, s_scan, pivot, eq_needed);
  if (threadIdx.x == 0) {
    threshold[row] = pivot;
    // Exact round-trip: pivot is the sortable image of a real sampled value.
    threshold_f[row] = sortable_to_fp32(pivot);
  }
}

// ---------------------------------------------------------------------------
// Phase B: the streaming filter. This is where the time budget lives.
// ---------------------------------------------------------------------------
__global__ void phase_b_filter(const float* __restrict__ input, int N,
                               const uint32_t* __restrict__ threshold,
                               uint32_t* __restrict__ cand_keys, int* __restrict__ cand_idx,
                               unsigned int* __restrict__ cand_count, int C_alloc) {
  const int row = blockIdx.y;
  const vfloat4* ri = reinterpret_cast<const vfloat4*>(input + (size_t)row * N);
  uint32_t* rk = cand_keys + (size_t)row * C_alloc;
  int* rx = cand_idx + (size_t)row * C_alloc;
  const uint32_t th = threshold[row];

  const int lane = threadIdx.x & (WAVE_SIZE - 1);
  const int n4 = N / FP32_EPT;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  for (; i < n4; i += stride) {
    vfloat4 v = load_f4(ri + i);
    uint32_t k[FP32_EPT] = {fp32_to_sortable(v[0]), fp32_to_sortable(v[1]), fp32_to_sortable(v[2]),
                            fp32_to_sortable(v[3])};
    unsigned pmask = 0;
#pragma unroll
    for (int e = 0; e < FP32_EPT; e++) {
      if (k[e] >= th) pmask |= (1u << e);
    }

    int cnt = __popc(pmask);
    int incl = cnt;
#pragma unroll
    for (int d = 1; d < WAVE_SIZE; d <<= 1) {
      int nb = __shfl_up(incl, d);
      if (lane >= d) incl += nb;
    }
    const int my_off = incl - cnt;
    const int wtotal = __shfl(incl, WAVE_SIZE - 1);

    if (wtotal > 0) {
      unsigned wbase = 0;
      if (lane == 0) wbase = atomicAdd(&cand_count[row], (unsigned)wtotal);
      wbase = __shfl(wbase, 0);
      unsigned rem = pmask;
      int j = 0;
      while (rem) {
        int e = __builtin_ctz(rem);
        rem &= rem - 1;
        unsigned pos = wbase + (unsigned)my_off + (unsigned)j;
        if (pos < (unsigned)C_alloc) {
          rk[pos] = k[e];
          rx[pos] = i * FP32_EPT + e;
        }
        j++;
      }
    }
  }
}

// Variant: one global atomic per block-iteration instead of per wave. Uniform
// trip count per block so the in-loop barriers are safe.
__global__ void phase_b_filter_blockagg(const float* __restrict__ input, int N,
                                        const uint32_t* __restrict__ threshold,
                                        uint32_t* __restrict__ cand_keys,
                                        int* __restrict__ cand_idx,
                                        unsigned int* __restrict__ cand_count, int C_alloc) {
  const int row = blockIdx.y;
  const vfloat4* ri = reinterpret_cast<const vfloat4*>(input + (size_t)row * N);
  uint32_t* rk = cand_keys + (size_t)row * C_alloc;
  int* rx = cand_idx + (size_t)row * C_alloc;
  const uint32_t th = threshold[row];

  __shared__ unsigned s_wave_tot[16];
  __shared__ unsigned s_wave_base[16];
  __shared__ unsigned s_block_base;

  const int lane = threadIdx.x & (WAVE_SIZE - 1);
  const int wid = threadIdx.x / WAVE_SIZE;
  const int nwaves = blockDim.x / WAVE_SIZE;
  const int n4 = N / FP32_EPT;
  const int stride = gridDim.x * blockDim.x;
  const int start = blockIdx.x * blockDim.x + threadIdx.x;
  // Uniform iteration count across the whole block.
  const int iters = (n4 - blockIdx.x * blockDim.x + stride - 1) / stride;

  for (int it = 0; it < iters; it++) {
    const int i = start + it * stride;
    unsigned pmask = 0;
    uint32_t k[FP32_EPT] = {0, 0, 0, 0};
    if (i < n4) {
      vfloat4 v = load_f4(ri + i);
      k[0] = fp32_to_sortable(v[0]);
      k[1] = fp32_to_sortable(v[1]);
      k[2] = fp32_to_sortable(v[2]);
      k[3] = fp32_to_sortable(v[3]);
#pragma unroll
      for (int e = 0; e < FP32_EPT; e++) {
        if (k[e] >= th) pmask |= (1u << e);
      }
    }

    int cnt = __popc(pmask);
    int incl = cnt;
#pragma unroll
    for (int d = 1; d < WAVE_SIZE; d <<= 1) {
      int nb = __shfl_up(incl, d);
      if (lane >= d) incl += nb;
    }
    const int my_off = incl - cnt;
    if (lane == WAVE_SIZE - 1) s_wave_tot[wid] = (unsigned)incl;
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned tot = 0;
      for (int w = 0; w < nwaves; w++) {
        s_wave_base[w] = tot;
        tot += s_wave_tot[w];
      }
      s_block_base = tot ? atomicAdd(&cand_count[row], tot) : 0u;
    }
    __syncthreads();
    const unsigned base = s_block_base + s_wave_base[wid] + (unsigned)my_off;
    unsigned rem = pmask;
    int j = 0;
    while (rem) {
      int e = __builtin_ctz(rem);
      rem &= rem - 1;
      unsigned pos = base + (unsigned)j;
      if (pos < (unsigned)C_alloc) {
        rk[pos] = k[e];
        rx[pos] = i * FP32_EPT + e;
      }
      j++;
    }
    __syncthreads();
  }
}

// Variant 2: one block owns the whole row, so it also owns that row's candidate
// area outright -- no global atomic is needed at all, just an LDS counter.
// Measured: the per-wave global atomicAdd on cand_count[row] was the real Phase B
// bottleneck (raising grid.x monotonically slowed the kernel: gx=2 -> 0.925 ms,
// gx=64 -> 3.42 ms, while a pure read at gx=1 hits 6.08 TB/s).
//
// Also drops fp32_to_sortable from the hot loop: the filter compares floats
// directly against the row threshold using !(v < thr), which admits v >= thr
// AND NaN. That is a superset of the sortable-order test in every case, and
// over-collection is harmless because Phase C still does the exact select.
__global__ void phase_b_filter_rowblock(const float* __restrict__ input, int N,
                                        const float* __restrict__ threshold_f,
                                        uint32_t* __restrict__ cand_raw, int* __restrict__ cand_idx,
                                        unsigned int* __restrict__ cand_count, int C_alloc) {
  const int row = blockIdx.x;
  const vfloat4* ri = reinterpret_cast<const vfloat4*>(input + (size_t)row * N);
  uint32_t* rk = cand_raw + (size_t)row * C_alloc;
  int* rx = cand_idx + (size_t)row * C_alloc;
  const float th = threshold_f[row];

  __shared__ unsigned s_n;
  if (threadIdx.x == 0) s_n = 0;
  __syncthreads();

  const int lane = threadIdx.x & (WAVE_SIZE - 1);
  const uint64_t lt = (1ull << lane) - 1ull;
  const int n4 = N / FP32_EPT;
  const int stride = blockDim.x;
  const int iters = (n4 + stride - 1) / stride;

  for (int it = 0; it < iters; it++) {
    const int i = it * stride + threadIdx.x;
    vfloat4 v = {0.f, 0.f, 0.f, 0.f};
    bool live = (i < n4);
    if (live) v = load_f4(ri + i);

    // One v_cmp per element. __popcll of a wave-uniform mask is a scalar op.
    const uint64_t b0 = __ballot(live && !(v[0] < th));
    const uint64_t b1 = __ballot(live && !(v[1] < th));
    const uint64_t b2 = __ballot(live && !(v[2] < th));
    const uint64_t b3 = __ballot(live && !(v[3] < th));
    const int t0 = __popcll(b0);
    const int t1 = t0 + __popcll(b1);
    const int t2 = t1 + __popcll(b2);
    const int wtotal = t2 + __popcll(b3);

    if (wtotal > 0) {
      unsigned wbase = 0;
      if (lane == 0) wbase = atomicAdd(&s_n, (unsigned)wtotal);
      wbase = __shfl(wbase, 0);

      const int base_idx = i * FP32_EPT;
      if (b0 & (1ull << lane)) {
        unsigned p = wbase + (unsigned)__popcll(b0 & lt);
        if (p < (unsigned)C_alloc) { rk[p] = __float_as_uint(v[0]); rx[p] = base_idx + 0; }
      }
      if (b1 & (1ull << lane)) {
        unsigned p = wbase + (unsigned)(t0 + __popcll(b1 & lt));
        if (p < (unsigned)C_alloc) { rk[p] = __float_as_uint(v[1]); rx[p] = base_idx + 1; }
      }
      if (b2 & (1ull << lane)) {
        unsigned p = wbase + (unsigned)(t1 + __popcll(b2 & lt));
        if (p < (unsigned)C_alloc) { rk[p] = __float_as_uint(v[2]); rx[p] = base_idx + 2; }
      }
      if (b3 & (1ull << lane)) {
        unsigned p = wbase + (unsigned)(t2 + __popcll(b3 & lt));
        if (p < (unsigned)C_alloc) { rk[p] = __float_as_uint(v[3]); rx[p] = base_idx + 3; }
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) cand_count[row] = s_n;
}

// Variant 3: wave-private output regions, so Phase B has NO atomic of any kind
// (variant 2 still paid ~512 LDS atomics per row on one address). Each wave
// keeps a wave-uniform register counter and writes into its own slice. Key and
// index go out as one packed 64-bit store instead of two 32-bit streams.
// Overflow of a slice is detected and sends the row to the exact fallback.
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

    if (wtotal > 0) {
      const int base_idx = i * FP32_EPT;
      if (b0 & (1ull << lane)) {
        int p = wcnt + __popcll(b0 & lt);
        if (p < seg_stride)
          seg[p] = ((uint64_t)__float_as_uint(v[0]) << 32) | (uint32_t)(base_idx + 0);
      }
      if (b1 & (1ull << lane)) {
        int p = wcnt + t0 + __popcll(b1 & lt);
        if (p < seg_stride)
          seg[p] = ((uint64_t)__float_as_uint(v[1]) << 32) | (uint32_t)(base_idx + 1);
      }
      if (b2 & (1ull << lane)) {
        int p = wcnt + t1 + __popcll(b2 & lt);
        if (p < seg_stride)
          seg[p] = ((uint64_t)__float_as_uint(v[2]) << 32) | (uint32_t)(base_idx + 2);
      }
      if (b3 & (1ull << lane)) {
        int p = wcnt + t2 + __popcll(b3 & lt);
        if (p < seg_stride)
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

// Phase C fed by the wave-segmented layout: gathers the variable-length
// per-wave segments into one contiguous LDS array, then selects exactly.
__global__ void phase_c_select_waveseg(const uint64_t* __restrict__ cand_pack,
                                       const int* __restrict__ cand_seg,
                                       const unsigned int* __restrict__ cand_count, int seg_stride,
                                       int nwaves_b, int K, int* __restrict__ out_idx,
                                       int* __restrict__ fb_rows, int* __restrict__ fb_count) {
  const int row = blockIdx.x;
  const unsigned int c_raw = cand_count[row];
  if (c_raw < (unsigned)K || c_raw > (unsigned)PHASE_C_CAP) {
    if (threadIdx.x == 0) fb_rows[atomicAdd(fb_count, 1)] = row;
    return;
  }
  const int c = (int)c_raw;

  __shared__ uint32_t s_keys[PHASE_C_CAP];
  __shared__ int s_idx[PHASE_C_CAP];
  __shared__ uint32_t s_hist[256];
  __shared__ uint32_t s_scan[2];
  __shared__ int s_cnt[MAX_WAVES_PER_BLOCK];
  __shared__ int s_off[MAX_WAVES_PER_BLOCK];
  __shared__ unsigned s_wgt, s_weq;

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
  block_select_lds(s_keys, c, K, s_hist, s_scan, pivot, eq_needed);

  int* out = out_idx + (size_t)row * K;
  const int ngt = K - eq_needed;
  for (int i = threadIdx.x; i < c; i += blockDim.x) {
    uint32_t k = s_keys[i];
    if (k > pivot) {
      unsigned p = atomicAdd(&s_wgt, 1u);
      if (p < (unsigned)ngt) out[p] = s_idx[i];
    } else if (k == pivot) {
      unsigned p = atomicAdd(&s_weq, 1u);
      if (p < (unsigned)eq_needed) out[ngt + p] = s_idx[i];
    }
  }
}

// ---------------------------------------------------------------------------
// Phase C: exact select on the candidate set, entirely in LDS, one block/row.
// Also decides which rows need the full-row fallback and compacts them.
//
// A row is unusable if it has FEWER than K candidates (threshold too high) OR
// MORE than C_alloc (threshold too low => Phase B dropped passers past the end
// of the per-row area, so the set is silently incomplete).
// ---------------------------------------------------------------------------
__global__ void phase_c_select(const uint32_t* __restrict__ cand_keys,
                               const int* __restrict__ cand_idx,
                               const unsigned int* __restrict__ cand_count, int C_alloc, int K,
                               int* __restrict__ out_idx, int* __restrict__ fb_rows,
                               int* __restrict__ fb_count, int keys_are_raw_float) {
  const int row = blockIdx.x;
  const unsigned int c_raw = cand_count[row];

  if (c_raw < (unsigned)K || c_raw > (unsigned)C_alloc) {
    if (threadIdx.x == 0) {
      int slot = atomicAdd(fb_count, 1);
      fb_rows[slot] = row;
    }
    return;
  }

  const int c = (int)c_raw;
  const uint32_t* rk = cand_keys + (size_t)row * C_alloc;
  const int* rx = cand_idx + (size_t)row * C_alloc;
  int* out = out_idx + (size_t)row * K;

  __shared__ uint32_t s_keys[PHASE_C_CAP];
  __shared__ int s_idx[PHASE_C_CAP];
  __shared__ uint32_t s_hist[256];
  __shared__ uint32_t s_scan[2];
  __shared__ unsigned s_wgt;
  __shared__ unsigned s_weq;

  for (int i = threadIdx.x; i < c; i += blockDim.x) {
    uint32_t w = rk[i];
    s_keys[i] = keys_are_raw_float ? fp32_to_sortable_bits(w) : w;
    s_idx[i] = rx[i];
  }
  if (threadIdx.x == 0) {
    s_wgt = 0;
    s_weq = 0;
  }
  __syncthreads();

  uint32_t pivot;
  int eq_needed;
  block_select_lds(s_keys, c, K, s_hist, s_scan, pivot, eq_needed);

  const int ngt = K - eq_needed;
  for (int i = threadIdx.x; i < c; i += blockDim.x) {
    uint32_t k = s_keys[i];
    if (k > pivot) {
      unsigned p = atomicAdd(&s_wgt, 1u);
      if (p < (unsigned)ngt) out[p] = s_idx[i];
    } else if (k == pivot) {
      unsigned p = atomicAdd(&s_weq, 1u);
      if (p < (unsigned)eq_needed) out[ngt + p] = s_idx[i];
    }
  }
}

// ---------------------------------------------------------------------------
// Phase D: exact full-row select. Fallback for unusable rows, and the
// independent oracle when --pipeline direct.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(1024) void phase_d_fallback(const float* __restrict__ input, int N,
                                                         int K, const int* __restrict__ fb_rows,
                                                         const int* __restrict__ fb_count,
                                                         int* __restrict__ out_idx) {
  const int count = *fb_count;
  const int n4 = N / FP32_EPT;

  __shared__ uint32_t s_hist[256];
  __shared__ uint32_t s_scan[2];
  __shared__ unsigned s_wgt;
  __shared__ unsigned s_weq;

  // Grid-stride over the compacted row list: correct for any count, while the
  // dispatch stays at FB_GRID blocks instead of one per matrix row.
  for (int slot = blockIdx.y; slot < count; slot += gridDim.y) {
    const int row = fb_rows[slot];
    const vfloat4* ri4 = reinterpret_cast<const vfloat4*>(input + (size_t)row * N);
    int* out = out_idx + (size_t)row * K;

    uint32_t pivot;
    int eq_needed;
    block_select_stream(ri4, n4, K, s_hist, s_scan, pivot, eq_needed);

    if (threadIdx.x == 0) {
      s_wgt = 0;
      s_weq = 0;
    }
    __syncthreads();

    const int ngt = K - eq_needed;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
      vfloat4 v = ri4[i];
      uint32_t k[FP32_EPT] = {fp32_to_sortable(v[0]), fp32_to_sortable(v[1]),
                              fp32_to_sortable(v[2]), fp32_to_sortable(v[3])};
#pragma unroll
      for (int e = 0; e < FP32_EPT; e++) {
        if (k[e] > pivot) {
          unsigned p = atomicAdd(&s_wgt, 1u);
          if (p < (unsigned)ngt) out[p] = i * FP32_EPT + e;
        } else if (k[e] == pivot) {
          unsigned p = atomicAdd(&s_weq, 1u);
          if (p < (unsigned)eq_needed) out[ngt + p] = i * FP32_EPT + e;
        }
      }
    }
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
  uint32_t* cand_keys;
  int* cand_idx;
  uint64_t* cand_pack;
  int* cand_seg;
  unsigned int* cand_count;
  int* fb_rows;
  int* fb_count;
  int C_alloc;
};

static void alloc_bufs(Bufs& b, int M, int K) {
  b.C_alloc = PHASE_C_CAP;
  HIP_CHECK(hipMalloc(&b.threshold, (size_t)M * sizeof(uint32_t)));
  HIP_CHECK(hipMalloc(&b.threshold_f, (size_t)M * sizeof(float)));
  HIP_CHECK(hipMalloc(&b.cand_keys, (size_t)M * b.C_alloc * sizeof(uint32_t)));
  HIP_CHECK(hipMalloc(&b.cand_idx, (size_t)M * b.C_alloc * sizeof(int)));
  HIP_CHECK(hipMalloc(&b.cand_pack, (size_t)M * CAND_SLOTS_PER_ROW * sizeof(uint64_t)));
  HIP_CHECK(hipMalloc(&b.cand_seg, (size_t)M * MAX_WAVES_PER_BLOCK * sizeof(int)));
  HIP_CHECK(hipMalloc(&b.cand_count, (size_t)M * sizeof(unsigned int)));
  HIP_CHECK(hipMalloc(&b.fb_rows, (size_t)M * sizeof(int)));
  HIP_CHECK(hipMalloc(&b.fb_count, sizeof(int)));
  (void)K;
}

static void free_bufs(Bufs& b) {
  (void)hipFree(b.threshold);
  (void)hipFree(b.threshold_f);
  (void)hipFree(b.cand_keys);
  (void)hipFree(b.cand_idx);
  (void)hipFree(b.cand_pack);
  (void)hipFree(b.cand_seg);
  (void)hipFree(b.cand_count);
  (void)hipFree(b.fb_rows);
  (void)hipFree(b.fb_count);
}

static void topk_fused(const float* d_in, int M, int N, int K, int* d_idx, Bufs& b, int smc,
                       hipStream_t s) {
  const int S = g_sample_s;
  const int rank = g_sample_rank > 0
                       ? g_sample_rank
                       : std::max(1, (int)(g_margin * (double)K * S / (double)N));
  const int n4 = N / FP32_EPT;
  const int gx = std::max(1, std::min(g_cf_gx, n4 / g_cf_block));
  (void)smc;

  HIP_CHECK(hipMemsetAsync(b.cand_count, 0, (size_t)M * sizeof(unsigned int), s));
  HIP_CHECK(hipMemsetAsync(b.fb_count, 0, sizeof(int), s));

  phase_a_threshold<<<M, g_phase_a_block, S * sizeof(uint32_t), s>>>(d_in, N, rank, S, b.threshold,
                                                                     b.threshold_f);

  if (g_phase_b == 3) {
    const int nwaves_b = std::max(1, g_cf_block / WAVE_SIZE);
    const int seg_stride = CAND_SLOTS_PER_ROW / nwaves_b;
    phase_b_filter_waveseg<<<M, g_cf_block, 0, s>>>(d_in, N, b.threshold_f, b.cand_pack, b.cand_seg,
                                                    b.cand_count, seg_stride);
    phase_c_select_waveseg<<<M, g_phase_c_block, 0, s>>>(b.cand_pack, b.cand_seg, b.cand_count,
                                                         seg_stride, nwaves_b, K, d_idx, b.fb_rows,
                                                         b.fb_count);
    phase_d_fallback<<<dim3(1, FB_GRID), 1024, 0, s>>>(d_in, N, K, b.fb_rows, b.fb_count, d_idx);
    return;
  }

  int keys_are_raw = 0;
  if (g_phase_b == 2) {
    keys_are_raw = 1;
    phase_b_filter_rowblock<<<M, g_cf_block, 0, s>>>(d_in, N, b.threshold_f, b.cand_keys,
                                                     b.cand_idx, b.cand_count, b.C_alloc);
  } else if (g_phase_b == 1) {
    phase_b_filter_blockagg<<<dim3(gx, M), g_cf_block, 0, s>>>(d_in, N, b.threshold, b.cand_keys,
                                                               b.cand_idx, b.cand_count, b.C_alloc);
  } else {
    phase_b_filter<<<dim3(gx, M), g_cf_block, 0, s>>>(d_in, N, b.threshold, b.cand_keys, b.cand_idx,
                                                      b.cand_count, b.C_alloc);
  }

  phase_c_select<<<M, g_phase_c_block, 0, s>>>(b.cand_keys, b.cand_idx, b.cand_count, b.C_alloc, K,
                                               d_idx, b.fb_rows, b.fb_count, keys_are_raw);

  phase_d_fallback<<<dim3(1, FB_GRID), 1024, 0, s>>>(d_in, N, K, b.fb_rows, b.fb_count, d_idx);
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
          "  --input-bin PATH --dump-indices PATH --inject-fault 0|1\n",
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
    else if (a == "--input-bin") input_path = need();
    else if (a == "--dump-indices") dump_path = need();
    else if (a == "--inject-fault") g_inject_fault = std::stoi(need());
    else if (a == "--dump-stats") g_dump_stats = std::stoi(need());
    else {
      usage(argv[0]);
      exit(1);
    }
  }

  if (K > PHASE_C_CAP) {
    fprintf(stderr, "ERROR: topk=%d exceeds PHASE_C_CAP=%d\n", K, PHASE_C_CAP);
    return 2;
  }
  if (g_sample_s > SAMPLE_S_MAX || g_sample_s % SAMPLE_CHUNK_ELEMS != 0) {
    fprintf(stderr, "ERROR: sample-s=%d must be a multiple of %d and <= %d\n", g_sample_s,
            SAMPLE_CHUNK_ELEMS, SAMPLE_S_MAX);
    return 2;
  }
  {
    const int chunks = g_sample_s / SAMPLE_CHUNK_ELEMS;
    // chunk_stride must keep every chunk start 16 B aligned for the dwordx4 load.
    if (N / chunks < SAMPLE_CHUNK_ELEMS || N % FP32_EPT != 0 || (N / chunks) % FP32_EPT != 0) {
      fprintf(stderr, "ERROR: N=%d incompatible with sampling geometry (chunks=%d)\n", N, chunks);
      return 2;
    }
  }

  GPUInfo info = get_gpu_info();
  printf("GPU: %s CUs=%d\n", info.name, info.sm_count);
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
  alloc_bufs(bufs, M, K);

  HIP_CHECK(hipMemset(d_idx, 0xFF, out_elems * sizeof(int)));
  run_topk(d_in, M, N, K, d_idx, bufs, info.sm_count, 0);
  HIP_CHECK(hipDeviceSynchronize());

  int fb = 0;
  HIP_CHECK(hipMemcpy(&fb, bufs.fb_count, sizeof(int), hipMemcpyDeviceToHost));

  if (g_dump_stats) {
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
    std::vector<float> h_in(in_elems);
    HIP_CHECK(hipMemcpy(h_in.data(), d_in, in_elems * sizeof(float), hipMemcpyDeviceToHost));
    int bad_rows = 0;
    for (int r = 0; r < M; r++)
      if (!verify_row_cpu(h_in.data() + (size_t)r * N, N, K, h_idx.data() + (size_t)r * K))
        bad_rows++;
    printf("VERIFY rows_fail=%d fallback_rows=%d pipeline=%s inject_fault=%d\n", bad_rows, fb,
           g_pipeline_direct ? "direct" : "fused", g_inject_fault);
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
    for (int rep = 0; rep < repeats; rep++) {
      for (int w = 0; w < warmup; w++) run_topk(d_in, M, N, K, d_idx, bufs, info.sm_count, 0);
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<double> samples;
      samples.reserve(iters);
      for (int it = 0; it < iters; it++) {
        timer.begin();
        run_topk(d_in, M, N, K, d_idx, bufs, info.sm_count, 0);
        samples.push_back(timer.end());
      }
      run_medians.push_back(median(samples));
    }
    const double med = median(run_medians);
    const double sd = stddev_pct(run_medians, med);
    printf("TIMING wall_ms_median=%.4f stddev_pct=%.3f repeats=%d iters=%d warmup=%d\n", med, sd,
           repeats, iters, warmup);
    printf("RESULT pipeline=%s wall_ms=%.4f fallback_rows=%d\n",
           g_pipeline_direct ? "direct" : "fused", med, fb);
  }

  free_bufs(bufs);
  (void)hipFree(d_in);
  (void)hipFree(d_idx);
  return 0;
}
