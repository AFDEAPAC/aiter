#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>

#define HIP_CHECK(call)                                                         \
  do {                                                                          \
    hipError_t _err = (call);                                                   \
    if (_err != hipSuccess) {                                                   \
      fprintf(stderr, "HIP error %d (%s) at %s:%d\n", (int)_err,              \
              hipGetErrorString(_err), __FILE__, __LINE__);                     \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

constexpr int WAVE_SIZE = 64;
constexpr int FP32_EPT = 4;          // floats per dwordx4 load
constexpr int RADIX_PASSES = 4;      // 4 x 8-bit covers all 32 sortable bits

// Phase A sampling geometry: NUM_CHUNKS contiguous runs of CHUNK floats each.
// Contiguous runs (not stride-1-of-32) so the DRAM traffic equals the useful
// bytes; a strided sample would fetch a whole 128 B line per useful float.
constexpr int SAMPLE_CHUNKS = 64;
constexpr int SAMPLE_CHUNK_ELEMS = 64;
constexpr int SAMPLE_S = SAMPLE_CHUNKS * SAMPLE_CHUNK_ELEMS;  // 4096

// LDS capacity for the Phase C candidate set (keys + indices).
constexpr int PHASE_C_CAP = 4096;    // 4096 * (4+4) B = 32 KB LDS

struct GPUInfo {
  char name[256];
  int compute_major;
  int compute_minor;
  int clock_khz;
  int sm_count;
  int mem_clock_khz;
  int mem_bus_width;
};

static inline GPUInfo get_gpu_info() {
  int dev = 0;
  HIP_CHECK(hipGetDevice(&dev));
  hipDeviceProp_t p;
  HIP_CHECK(hipGetDeviceProperties(&p, dev));
  GPUInfo g{};
  std::strncpy(g.name, p.name, sizeof(g.name) - 1);
  g.compute_major = p.major;
  g.compute_minor = p.minor;
  g.clock_khz = p.clockRate;
  g.sm_count = p.multiProcessorCount;
  g.mem_clock_khz = p.memoryClockRate;
  g.mem_bus_width = p.memoryBusWidth;
  return g;
}

class HipTimer {
 public:
  HipTimer() {
    (void)hipEventCreate(&start_);
    (void)hipEventCreate(&stop_);
  }
  ~HipTimer() {
    (void)hipEventDestroy(start_);
    (void)hipEventDestroy(stop_);
  }
  void begin(hipStream_t s = 0) { (void)hipEventRecord(start_, s); }
  double end(hipStream_t s = 0) {
    (void)hipEventRecord(stop_, s);
    (void)hipEventSynchronize(stop_);
    float ms = 0.f;
    (void)hipEventElapsedTime(&ms, start_, stop_);
    return (double)ms;
  }

 private:
  hipEvent_t start_, stop_;
};

// IEEE-754 fp32 -> monotone uint32. NaN lands above +INF, matching the
// "distort" trick from DeepSelect (csrc/hip_kernels/bit_utils_hip.cuh).
__host__ __device__ __forceinline__ uint32_t fp32_to_sortable_bits(uint32_t u) {
  return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

__device__ __forceinline__ uint32_t fp32_to_sortable(float v) {
  return fp32_to_sortable_bits(__float_as_uint(v));
}

__device__ __forceinline__ float sortable_to_fp32(uint32_t s) {
  uint32_t u = (s & 0x80000000u) ? (s ^ 0x80000000u) : ~s;
  return __uint_as_float(u);
}

static inline uint32_t fp32_to_sortable_host(float v) {
  uint32_t u;
  std::memcpy(&u, &v, sizeof(u));
  return fp32_to_sortable_bits(u);
}

// Selects the load flavour for the streaming filter pass. Non-temporal is the
// right hint for Phase B: pure streaming, zero reuse, so keeping the lines in
// L2 only evicts useful data.
__constant__ int d_use_nt_load = 0;

// Native ext_vector_type: __builtin_nontemporal_load rejects HIP_vector_type.
typedef float vfloat4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ vfloat4 load_f4(const vfloat4* p) {
  if (d_use_nt_load) return __builtin_nontemporal_load(p);
  return *p;
}

// Byte offset of radix pass p (MSB first).
__device__ __host__ __forceinline__ int radix_shift(int pass) { return 24 - 8 * pass; }

// Grid.y for the fallback kernel. Blocks loop over the compacted row list, so
// any number of fallback rows is handled; this only bounds the dispatch cost.
// grid.y = M would dispatch 4096 blocks to do work for a handful of rows.
constexpr int FB_GRID = 64;

// Wave-private candidate regions (Phase B variant 3). Each wave in the row's
// block owns a fixed slice, so it needs no atomic at all -- just a wave-uniform
// register counter. The PHYSICAL stride is deliberately generous: only the
// occupied slots are ever written or read, so a wide stride costs address space
// and nothing else, while making per-wave overflow ~25 sigma away instead of
// ~0 sigma (expected passers/wave is ~178 +/- 13 at K=2048).
constexpr int CAND_SLOTS_PER_ROW = 8192;
constexpr int MAX_WAVES_PER_BLOCK = 16;

// Turns s_hist[256] (per-bucket counts) into an INCLUSIVE SUFFIX sum in place,
// then finds the bucket where the running count from the top first reaches ek.
// Writes s_scan[0] = bucket, s_scan[1] = count strictly above that bucket.
//
// Replaces a serial 256-step walk by thread 0: that walk is a chain of
// dependent LDS reads and measured as the dominant cost of the select kernels
// (phase_a 118 us / phase_c 220 us for a few MB of traffic).
// Every thread in the block must call this (it contains barriers).
__device__ __forceinline__ void block_find_pivot_bucket(uint32_t* __restrict__ s_hist,
                                                        uint32_t* __restrict__ s_scan, int ek) {
  const int t = threadIdx.x;
  // Hillis-Steele inclusive suffix scan over the 256 buckets.
  for (int off = 1; off < 256; off <<= 1) {
    uint32_t add = 0;
    if (t < 256 && t + off < 256) add = s_hist[t + off];
    __syncthreads();
    if (t < 256) s_hist[t] += add;
    __syncthreads();
  }
  // Default matches the serial walk when the total never reaches ek.
  if (t == 0) {
    s_scan[0] = 0;
    s_scan[1] = s_hist[0];
  }
  __syncthreads();
  if (ek > 0 && t < 256) {
    const uint32_t here = s_hist[t];
    const uint32_t above = (t == 255) ? 0u : s_hist[t + 1];
    if (here >= (uint32_t)ek && above < (uint32_t)ek) {
      s_scan[0] = (uint32_t)t;
      s_scan[1] = above;
    }
  }
  __syncthreads();
}
