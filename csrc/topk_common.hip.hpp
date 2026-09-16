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
