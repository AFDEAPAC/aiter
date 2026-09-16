// bw_kernel.hip — achievable HBM bandwidth anchors for the topk-prefill workload.
//
// Reports three numbers, all with correct byte accounting:
//   READ_STREAM : pure read of a [M,N] fp32 buffer via dwordx4, value consumed into a sink
//                 so the loads cannot be dead-code eliminated. This is THE floor for topk.
//   COPY        : read+write of the same buffer (traffic = 2x bytes).
//   READ_ROWMAJOR: same total bytes but with the exact grid/row mapping the filter kernel
//                 uses, so the floor is comparable to the real Phase B.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define HIP_CHECK(call)                                                          \
  do {                                                                           \
    hipError_t _err = (call);                                                     \
    if (_err != hipSuccess) {                                                    \
      fprintf(stderr, "HIP error %d (%s) at %s:%d\n", (int)_err,                 \
              hipGetErrorString(_err), __FILE__, __LINE__);                       \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

// Pure read: grid-stride over uint4, accumulate so loads are live.
__global__ void read_stream(const float4* __restrict__ src, size_t n4,
                            float* __restrict__ sink) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  float acc = 0.f;
  for (; i < n4; i += stride) {
    float4 v = src[i];
    acc += v.x + v.y + v.z + v.w;
  }
  // Consume: only lane-0-of-block writes, and only if acc is NaN (never true for
  // real data), so the store costs nothing but the loads cannot be removed.
  if (acc != acc) sink[blockIdx.x] = acc;
}

// Same total bytes, but one block-row mapping like the filter kernel (grid.y = M).
__global__ void read_rowmajor(const float4* __restrict__ src, int n4_per_row,
                              float* __restrict__ sink) {
  int row = blockIdx.y;
  const float4* ri = src + (size_t)row * n4_per_row;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  float acc = 0.f;
  for (; i < n4_per_row; i += stride) {
    float4 v = ri[i];
    acc += v.x + v.y + v.z + v.w;
  }
  if (acc != acc) sink[blockIdx.x] = acc;
}

__global__ void copy_kernel(const float4* __restrict__ src, float4* __restrict__ dst, size_t n4) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) dst[i] = src[i];
}

__global__ void fill_kernel(float* p, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = (float)(i & 0xFFFF) * 1e-4f;
}

static double median(std::vector<double> v) {
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

int main() {
  // Exactly the topk-prefill workload footprint.
  const int M = 4096, N = 131072;
  const size_t n = (size_t)M * N;           // 536,870,912 floats
  const size_t bytes = n * sizeof(float);   // 2,147,483,648 B = 2.147 GB
  const size_t n4 = n / 4;
  const int n4_per_row = N / 4;

  float *d_a = nullptr, *d_b = nullptr, *d_sink = nullptr;
  HIP_CHECK(hipMalloc(&d_a, bytes));
  HIP_CHECK(hipMalloc(&d_b, bytes));
  HIP_CHECK(hipMalloc(&d_sink, 65536 * sizeof(float)));
  fill_kernel<<<65535, 256>>>(d_a, n);
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  const int warmup = 5, iters = 20, repeats = 5;
  const int threads = 256;
  const int blocks = 8192;

  auto time_it = [&](auto launch) {
    for (int w = 0; w < warmup; w++) launch();
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<double> run_medians;
    for (int rep = 0; rep < repeats; rep++) {
      std::vector<double> s;
      for (int it = 0; it < iters; it++) {
        HIP_CHECK(hipEventRecord(start));
        launch();
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        s.push_back((double)ms);
      }
      run_medians.push_back(median(s));
    }
    double med = median(run_medians);
    return std::make_pair(med, stddev_pct(run_medians, med));
  };

  auto report = [&](const char* tag, std::pair<double, double> r, double traffic_bytes) {
    double tbps = traffic_bytes / (r.first * 1e-3) / 1e12;
    printf("%s traffic_GB=%.3f median_ms=%.4f stddev_pct=%.3f bandwidth_TB_s=%.3f\n", tag,
           traffic_bytes / 1e9, r.first, r.second, tbps);
  };

  auto r_read = time_it([&] {
    read_stream<<<blocks, threads>>>(reinterpret_cast<const float4*>(d_a), n4, d_sink);
  });
  report("READ_STREAM", r_read, (double)bytes);

  auto r_row = time_it([&] {
    read_rowmajor<<<dim3(16, M), threads>>>(reinterpret_cast<const float4*>(d_a), n4_per_row, d_sink);
  });
  report("READ_ROWMAJOR", r_row, (double)bytes);

  auto r_copy = time_it([&] {
    copy_kernel<<<blocks, threads>>>(reinterpret_cast<const float4*>(d_a),
                                     reinterpret_cast<float4*>(d_b), n4);
  });
  report("COPY", r_copy, 2.0 * (double)bytes);

  printf("FLOOR_READ_ONLY_MS %.4f\n", r_read.first);
  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_sink));
  return 0;
}
