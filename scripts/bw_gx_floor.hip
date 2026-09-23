// bw_gx_floor.hip -- read+sparse-write bandwidth floor at the EXACT grid shape
// phase_b_filter_coop uses, so phase_b can be judged against a comparable floor.
//
// scripts/bw_kernel.hip only measures PHASEB_FLOOR at dim3(1, M) with a
// grid-stride walk. The shipped kernel runs dim3(G, M) with G=8 and a CONTIGUOUS
// per-block chunk. Those are different access patterns, so the gx=1 floor is not
// the right yardstick. This sweeps G and the block size with phase_b geometry.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t _e = (call);                                                    \
    if (_e != hipSuccess) {                                                    \
      fprintf(stderr, "HIP error %d (%s) at %s:%d\n", (int)_e,                 \
              hipGetErrorString(_e), __FILE__, __LINE__);                      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Mirrors phase_b_filter_coop: blockIdx.y = row, blockIdx.x = chunk of that row,
// contiguous [i0, i1) per block, then a contiguous burst of 8 B candidate writes.
__global__ void read_chunk_plus_write(const float4* __restrict__ src, int n4_per_row,
                                      unsigned long long* __restrict__ out, int w_per_row,
                                      int out_stride, float* __restrict__ sink) {
  const int row = blockIdx.y;
  const float4* ri = src + (size_t)row * n4_per_row;
  unsigned long long* ro = out + (size_t)row * out_stride;
  const int G = gridDim.x, bid = blockIdx.x;
  const int chunk = (n4_per_row + G - 1) / G;
  const int i0 = bid * chunk;
  const int i1 = min(i0 + chunk, n4_per_row);
  const int stride = blockDim.x;
  float acc = 0.f;
  for (int i = i0 + threadIdx.x; i < i1; i += stride) {
    float4 v = ri[i];
    acc += v.x + v.y + v.z + v.w;
  }
  // This block writes its share of the row candidates, contiguously.
  const int w_chunk = (w_per_row + G - 1) / G;
  const int j0 = bid * w_chunk, j1 = min(j0 + w_chunk, w_per_row);
  for (int j = j0 + threadIdx.x; j < j1; j += stride) ro[j] = (unsigned long long)j;
  if (acc != acc) sink[blockIdx.x] = acc;
}

__global__ void fill_kernel(float* p, size_t n) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    p[i] = (float)(i & 0xFFFF) * 1e-4f;
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
static double stddev_pct(const std::vector<double>& v, double med) {
  if (v.size() < 2 || med == 0.0) return 0.0;
  double sq = 0.0;
  for (double x : v) { double d = (x - med) / med; sq += d * d; }
  return 100.0 * std::sqrt(sq / (v.size() - 1));
}

int main(int argc, char** argv) {
  const int M = 4096, N = 131072;
  const int w_per_row = (argc > 1) ? atoi(argv[1]) : 2867;  // margin 1.4 * K=2048
  const int out_stride = 4096;
  const size_t n = (size_t)M * N;
  const size_t bytes = n * sizeof(float);
  const int n4_per_row = N / 4;

  float *d_a = nullptr, *d_sink = nullptr;
  unsigned long long* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_a, bytes));
  HIP_CHECK(hipMalloc(&d_sink, 65536 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, (size_t)M * out_stride * sizeof(unsigned long long)));
  fill_kernel<<<65535, 256>>>(d_a, n);
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  const int warmup = 20, iters = 50, repeats = 5;

  auto time_it = [&](auto launch) {
    for (int w = 0; w < warmup; w++) launch();
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<double> reps;
    for (int r = 0; r < repeats; r++) {
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
      reps.push_back(median(s));
    }
    double m = median(reps);
    return std::make_pair(m, stddev_pct(reps, m));
  };

  const double traffic = (double)bytes + (double)M * w_per_row * 8.0;
  printf("# w_per_row=%d traffic_GB=%.3f (read %.3f + write %.3f)\n", w_per_row,
         traffic / 1e9, bytes / 1e9, (double)M * w_per_row * 8.0 / 1e9);
  for (int gx : {1, 2, 4, 8, 16, 32}) {
    for (int bs : {256, 512, 1024}) {
      auto r = time_it([&] {
        read_chunk_plus_write<<<dim3(gx, M), bs>>>(reinterpret_cast<const float4*>(d_a),
                                                   n4_per_row, d_out, w_per_row, out_stride,
                                                   d_sink);
      });
      printf("GXFLOOR_gx%-3d_wg%-5d median_us=%8.2f stddev_pct=%.3f bandwidth_TB_s=%.3f\n", gx,
             bs, r.first * 1e3, r.second, traffic / (r.first * 1e-3) / 1e12);
      fflush(stdout);
    }
  }
  return 0;
}
