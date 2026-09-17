// floor_bench.hip — working-set bandwidth curve + empty-kernel launch overhead.
// Used by scripts/floor_model.py to build floor(M,N) estimates.
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

__global__ void read_stream(const float4* __restrict__ src, size_t n4, float* __restrict__ sink) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  float acc = 0.f;
  for (; i < n4; i += stride) {
    float4 v = src[i];
    acc += v.x + v.y + v.z + v.w;
  }
  if (acc != acc) sink[blockIdx.x] = acc;
}

__global__ void read_rowmajor(const float4* __restrict__ src, int n4_per_row, int M,
                              float* __restrict__ sink) {
  const int row = blockIdx.y;
  if (row >= M) return;
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

__global__ void read_plus_sparse_write(const float4* __restrict__ src, int n4_per_row, int M,
                                       unsigned long long* __restrict__ out, int w_per_row,
                                       int out_stride, float* __restrict__ sink) {
  const int row = blockIdx.y;
  if (row >= M) return;
  const float4* ri = src + (size_t)row * n4_per_row;
  unsigned long long* ro = out + (size_t)row * out_stride;
  float acc = 0.f;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (; i < n4_per_row; i += stride) {
    float4 v = ri[i];
    acc += v.x + v.y + v.z + v.w;
  }
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < w_per_row; j += stride)
    ro[j] = (unsigned long long)j;
  if (acc != acc) sink[blockIdx.x] = acc;
}

__global__ void empty_kernel(int* __restrict__ sink) {
  if (threadIdx.x == 0 && blockIdx.x == 0) atomicAdd(sink, 1);
}

__global__ void fill_kernel(float* p, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = (float)(i & 0xFFFF) * 1e-4f;
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

template <typename LaunchFn>
static std::pair<double, double> time_launch(LaunchFn launch, int warmup = 5, int iters = 20,
                                             int repeats = 5) {
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
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
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  double med = median(run_medians);
  return {med, stddev_pct(run_medians, med)};
}

int main(int argc, char** argv) {
  const int warmup = 5, iters = 20, repeats = 5;
  const int threads = 256;

  float* d_sink = nullptr;
  HIP_CHECK(hipMalloc(&d_sink, 65536 * sizeof(float)));
  int* d_counter = nullptr;
  HIP_CHECK(hipMalloc(&d_counter, sizeof(int)));

  printf("FLOOR_BENCH_BEGIN\n");

  // Empty kernel launch overhead: 1 launch, 4 sequential launches, M=4096 launches.
  HIP_CHECK(hipMemset(d_counter, 0, sizeof(int)));
  auto r1 = time_launch([&] { empty_kernel<<<1, 1>>>(d_counter); }, warmup, iters, repeats);
  printf("EMPTY_LAUNCH n=1 median_us=%.3f stddev_pct=%.3f\n", r1.first * 1000.0, r1.second);

  auto r4 = time_launch(
      [&] {
        empty_kernel<<<1, 1>>>(d_counter);
        empty_kernel<<<1, 1>>>(d_counter);
        empty_kernel<<<1, 1>>>(d_counter);
        empty_kernel<<<1, 1>>>(d_counter);
      },
      warmup, iters, repeats);
  printf("EMPTY_LAUNCH n=4 median_us=%.3f stddev_pct=%.3f per_launch_us=%.3f\n", r4.first * 1000.0,
         r4.second, r4.first * 1000.0 / 4.0);

  for (int M : {1, 4, 16, 64, 256, 1024, 4096}) {
    HIP_CHECK(hipMemset(d_counter, 0, sizeof(int)));
    auto rM = time_launch([&] { empty_kernel<<<M, 1>>>(d_counter); }, warmup, iters, repeats);
    printf("EMPTY_LAUNCH grid=%d median_us=%.3f stddev_pct=%.3f per_block_us=%.3f\n", M,
           rM.first * 1000.0, rM.second, rM.first * 1000.0 / (double)M);
  }

  // Working-set sweep: bytes from 4 KB to 2 GB (log-ish steps).
  const size_t sweep_bytes[] = {
      4096,      8192,       16384,      32768,      65536,      131072,     262144,
      524288,    1048576,    2097152,    4194304,    8388608,    16777216,   33554432,
      67108864,  134217728,  268435456,  536870912,  1073741824, 2147483648,
  };

  for (size_t bytes : sweep_bytes) {
    const size_t n = bytes / sizeof(float);
    const size_t n4 = n / 4;
    float* d_a = nullptr;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    fill_kernel<<<65535, 256>>>(d_a, n);
    HIP_CHECK(hipDeviceSynchronize());

    const int blocks = std::min(8192, (int)((n4 + threads - 1) / threads));
    auto r = time_launch(
        [&] { read_stream<<<blocks, threads>>>(reinterpret_cast<const float4*>(d_a), n4, d_sink); },
        warmup, iters, repeats);
    const double tbps = bytes / (r.first * 1e-3) / 1e12;
    printf("BW_SWEEP bytes=%zu median_us=%.3f stddev_pct=%.3f bandwidth_TB_s=%.4f\n", bytes,
           r.first * 1000.0, r.second, tbps);
    HIP_CHECK(hipFree(d_a));
  }

  // Read+write floor for the real access pattern, at SEVERAL N.
  //
  // One anchor is not enough. The achievable fraction of peak depends on how
  // much contiguous data ONE block streams, which is N*4 bytes, not on the total
  // payload. Anchoring only at N=131072 (512 KB per block) over-stated the floor
  // at N=1048576 (4 MB per block) enough that M=4096 N=1048576 scored 0.94x --
  // below its own floor, which is impossible, so it proved the model wrong
  // rather than the kernel fast.
  {
    const int w_per_row = 2867;
    const int out_stride = 4096;
    const int Ns[] = {16384, 131072, 1048576};
    const int Ms[] = {4096, 4096, 1024};
    for (int i = 0; i < 3; i++) {
      const int N = Ns[i], M = Ms[i];
      const size_t bytes = (size_t)M * N * sizeof(float);
      const int n4_per_row = N / 4;
      float* d_a = nullptr;
      unsigned long long* d_out = nullptr;
      HIP_CHECK(hipMalloc(&d_a, bytes));
      HIP_CHECK(hipMalloc(&d_out, (size_t)M * out_stride * sizeof(unsigned long long)));
      fill_kernel<<<65535, 256>>>(d_a, (size_t)M * N);
      HIP_CHECK(hipDeviceSynchronize());
      auto r = time_launch(
          [&] {
            read_plus_sparse_write<<<dim3(1, M), 512>>>(reinterpret_cast<const float4*>(d_a),
                                                        n4_per_row, M, d_out, w_per_row,
                                                        out_stride, d_sink);
          },
          warmup, iters, repeats);
      const double traffic = (double)bytes + (double)M * w_per_row * 8.0;
      printf("PHASEB_FLOOR M=%d N=%d traffic_GB=%.4f median_ms=%.4f stddev_pct=%.3f "
             "bytes_per_block=%zu achieved_TB_s=%.4f\n",
             M, N, traffic / 1e9, r.first, r.second, (size_t)N * 4,
             traffic / (r.first * 1e-3) / 1e12);
      HIP_CHECK(hipFree(d_a));
      HIP_CHECK(hipFree(d_out));
    }
  }

  printf("FLOOR_BENCH_END\n");
  HIP_CHECK(hipFree(d_sink));
  HIP_CHECK(hipFree(d_counter));
  return 0;
}
