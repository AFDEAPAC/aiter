// coop_spike.hip — does gfx950 / ROCm 10 support a cooperative grid sync, and
// how many blocks can actually be co-resident?
//
// S3 wants to spread phase_a's 3-pass radix select over many blocks, which needs
// a grid-wide barrier between passes. The alternative is one kernel launch per
// pass (~2.6 us each against an ~11 us target), so the whole direction depends
// on this answer.
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;

// Three "passes" separated by grid syncs, mimicking the radix structure: each
// pass accumulates into a global histogram, then every block reads the reduced
// result.
__global__ void coop_passes(unsigned* hist, unsigned* pivot, int n, int passes) {
  cg::grid_group grid = cg::this_grid();
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int p = 0; p < passes; p++) {
    for (int i = tid; i < 256; i += stride) hist[i] = 0;
    grid.sync();
    for (int i = tid; i < n; i += stride) atomicAdd(&hist[i & 255], 1u);
    grid.sync();
    if (tid == 0) {
      unsigned s = 0;
      for (int b = 255; b >= 0; b--) {
        s += hist[b];
        if (s >= (unsigned)(n / 2)) {
          *pivot = (unsigned)b;
          break;
        }
      }
    }
    grid.sync();
  }
}

int main() {
  int dev = 0;
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, dev) != hipSuccess) {
    printf("SPIKE: hipGetDeviceProperties failed\n");
    return 1;
  }
  printf("device            : %s  CUs=%d\n", prop.name, prop.multiProcessorCount);
  printf("cooperativeLaunch : %d\n", prop.cooperativeLaunch);
  printf("coopMultiDevice   : %d\n", prop.cooperativeMultiDeviceLaunch);

  for (int block : {256, 512, 1024}) {
    int per_cu = 0;
    hipError_t e = hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_cu, reinterpret_cast<const void*>(coop_passes), block, 0);
    printf("block=%-5d maxActiveBlocksPerCU=%-3d  => max coop grid = %-6d (err %d)\n", block,
           per_cu, per_cu * prop.multiProcessorCount, (int)e);
  }

  unsigned *hist = nullptr, *pivot = nullptr;
  if (hipMalloc(&hist, 256 * sizeof(unsigned)) != hipSuccess) return 1;
  if (hipMalloc(&pivot, sizeof(unsigned)) != hipSuccess) return 1;

  const int block = 512;
  int per_cu = 0;
  (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &per_cu, reinterpret_cast<const void*>(coop_passes), block, 0);
  const int grid = per_cu * prop.multiProcessorCount;
  const int n = 1 << 20;
  int passes = 3;
  void* args[] = {&hist, &pivot, (void*)&n, (void*)&passes};

  hipError_t le = hipLaunchCooperativeKernel(reinterpret_cast<const void*>(coop_passes),
                                             dim3(grid), dim3(block), args, 0, 0);
  printf("hipLaunchCooperativeKernel(grid=%d, block=%d) -> %d (%s)\n", grid, block, (int)le,
         hipGetErrorString(le));
  if (le == hipSuccess) {
    hipError_t se = hipDeviceSynchronize();
    unsigned p = 0;
    (void)hipMemcpy(&p, pivot, sizeof(p), hipMemcpyDeviceToHost);
    printf("sync -> %d (%s), pivot=%u (expect 127 or 128)\n", (int)se, hipGetErrorString(se), p);
  }

  // Does it reject an oversized grid, or silently deadlock? Launch, do not sync
  // if it already failed.
  const int too_big = grid * 4;
  le = hipLaunchCooperativeKernel(reinterpret_cast<const void*>(coop_passes), dim3(too_big),
                                  dim3(block), args, 0, 0);
  printf("oversized grid=%d -> %d (%s)%s\n", too_big, (int)le, hipGetErrorString(le),
         le == hipSuccess ? "  *** ACCEPTED: would deadlock, must clamp by hand ***" : "  (rejected, good)");
  if (le == hipSuccess) {
    printf("  not syncing: a co-residency violation hangs the queue\n");
  }
  // Price the mechanism itself. S3 would need 2 grid syncs per radix pass x 3
  // passes = 6, against an ~11 us target, so a sync costing more than ~1 us
  // kills the direction before any kernel is written.
  printf("\n--- cost of a bare grid.sync() ---\n");
  hipEvent_t a, b;
  (void)hipEventCreate(&a);
  (void)hipEventCreate(&b);
  for (int blk : {256, 512}) {
    int pcu = 0;
    (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &pcu, reinterpret_cast<const void*>(coop_passes), blk, 0);
    const int gmax = pcu * prop.multiProcessorCount;
    for (int g : {64, 256, gmax}) {
      if (g > gmax) continue;
      int zero = 0, one = 12;
      int n0 = 1;                       // minimal work so we time the syncs
      void* a0[] = {&hist, &pivot, (void*)&n0, (void*)&zero};
      void* a1[] = {&hist, &pivot, (void*)&n0, (void*)&one};
      // warm
      for (int w = 0; w < 5; w++) {
        (void)hipLaunchCooperativeKernel(reinterpret_cast<const void*>(coop_passes), dim3(g),
                                         dim3(blk), a1, 0, 0);
      }
      (void)hipDeviceSynchronize();
      auto timeit = [&](void** ar) {
        double best = 1e9;
        for (int r = 0; r < 5; r++) {
          (void)hipEventRecord(a);
          for (int it = 0; it < 20; it++)
            (void)hipLaunchCooperativeKernel(reinterpret_cast<const void*>(coop_passes), dim3(g),
                                             dim3(blk), ar, 0, 0);
          (void)hipEventRecord(b);
          (void)hipEventSynchronize(b);
          float ms = 0.f;
          (void)hipEventElapsedTime(&ms, a, b);
          if (ms / 20.0 < best) best = ms / 20.0;
        }
        return best * 1000.0;   // us per launch
      };
      const double t0 = timeit(a0);    // 0 passes  -> 0 syncs
      const double t1 = timeit(a1);    // 12 passes -> 36 syncs
      printf("block=%-5d grid=%-5d  0 syncs %6.2f us   36 syncs %6.2f us   => %5.3f us/sync\n",
             blk, g, t0, t1, (t1 - t0) / 36.0);
    }
  }
  printf("SPIKE_DONE\n");
  return 0;
}
