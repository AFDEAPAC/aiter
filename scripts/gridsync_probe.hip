// What does cooperative_groups::this_grid().sync() cost on this box?
//
// Gate 0 for replacing the three-kernel top-k pipeline with one cooperative
// kernel. The pipeline's three launches cost about 7.9 us of wall time here
// (knowledge/known_bad.md: four empty launches measure 10.5 us, 2.6 us each),
// and a fused kernel would trade two of those launches for two grid syncs. If a
// sync costs more than a launch the trade is a loss and there is nothing to
// build, so price it before writing any of it.
//
// Method: a cooperative kernel whose entire body is N grid syncs, swept over N,
// slope taken by least squares. The intercept is the kernel's own fixed cost and
// is reported separately because it is the other half of the trade -- a fused
// kernel pays one intercept instead of three.
//
// The same sweep is run for ordinary launches (N back-to-back empty kernels on
// one stream) so both halves of the comparison come from the same box, the same
// timer and the same run rather than from a number recorded months ago.
//
//   hipcc --offload-arch=gfx950 -O3 -Wall -Wextra scripts/gridsync_probe.hip -o gridsync_probe
//   ./gridsync_probe

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

#define HIP_CHECK(x)                                                          \
    do {                                                                      \
        hipError_t e_ = (x);                                                  \
        if (e_ != hipSuccess) {                                               \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,                 \
                    hipGetErrorString(e_));                                   \
            return 1;                                                         \
        }                                                                     \
    } while (0)

namespace cg = cooperative_groups;

// sink keeps the loop from being optimised away without adding memory traffic
// that would pollute the slope: one store per block, after the last sync.
__global__ void sync_kern(int n, unsigned* __restrict__ sink) {
    cg::grid_group grid = cg::this_grid();
    unsigned acc = threadIdx.x;
    for (int i = 0; i < n; i++) {
        grid.sync();
        acc += i;
    }
    if (threadIdx.x == 0 && acc == 0xFFFFFFFFu) sink[blockIdx.x] = acc;
}

__global__ void empty_kern(unsigned* __restrict__ sink) {
    if (threadIdx.x == 0xFFFF) sink[blockIdx.x] = 1u;
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Least squares slope and intercept of t = a + b*n.
static void fit(const std::vector<int>& ns, const std::vector<double>& ts,
                double& a, double& b) {
    double sn = 0, st = 0, snn = 0, snt = 0;
    const double m = (double)ns.size();
    for (size_t i = 0; i < ns.size(); i++) {
        sn += ns[i]; st += ts[i];
        snn += (double)ns[i] * ns[i];
        snt += (double)ns[i] * ts[i];
    }
    b = (m * snt - sn * st) / (m * snn - sn * sn);
    a = (st - b * sn) / m;
}

int main() {
    int coop = 0;
    HIP_CHECK(hipDeviceGetAttribute(&coop, hipDeviceAttributeCooperativeLaunch, 0));
    if (!coop) { printf("cooperative launch unsupported\n"); return 1; }

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    unsigned* d_sink = nullptr;
    HIP_CHECK(hipMalloc(&d_sink, 4096 * sizeof(unsigned)));

    hipEvent_t beg, end;
    HIP_CHECK(hipEventCreate(&beg));
    HIP_CHECK(hipEventCreate(&end));

    const int TB = 1024;
    int max_blocks_per_cu = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_cu,
                                                           (const void*)sync_kern, TB, 0));
    const int fit_limit = max_blocks_per_cu * prop.multiProcessorCount;
    printf("gfx=%s CUs=%d  blocks/CU at %d threads=%d  co-resident limit=%d blocks\n",
           prop.gcnArchName, prop.multiProcessorCount, TB, max_blocks_per_cu, fit_limit);

    const int ns_arr[] = {0, 1, 2, 4, 8, 16, 32, 64};
    const int grids[] = {64, 128, 256, 512};

    printf("\ncooperative kernel, body is N grid syncs, median of 9 runs of 20 iters\n");
    printf("%8s | %s | %11s %11s\n", "blocks", "  N=0     N=1     N=2     N=4     N=8    N=16    N=32    N=64",
           "us/sync", "intercept");
    for (int gi = 0; gi < 4; gi++) {
        const int blocks = grids[gi];
        if (blocks > fit_limit) { printf("%8d | does not fit co-resident\n", blocks); continue; }
        std::vector<int> ns;
        std::vector<double> ts;
        printf("%8d |", blocks);
        for (int k = 0; k < 8; k++) {
            int n = ns_arr[k];
            void* args[] = {&n, &d_sink};
            std::vector<double> samples;
            for (int rep = 0; rep < 9; rep++) {
                // warm
                HIP_CHECK(hipLaunchCooperativeKernel((const void*)sync_kern, dim3(blocks),
                                                     dim3(TB), args, 0, 0));
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipEventRecord(beg));
                for (int it = 0; it < 20; it++)
                    HIP_CHECK(hipLaunchCooperativeKernel((const void*)sync_kern, dim3(blocks),
                                                         dim3(TB), args, 0, 0));
                HIP_CHECK(hipEventRecord(end));
                HIP_CHECK(hipEventSynchronize(end));
                float ms = 0;
                HIP_CHECK(hipEventElapsedTime(&ms, beg, end));
                samples.push_back(ms * 1000.0 / 20.0);
            }
            const double us = median(samples);
            printf(" %7.2f", us);
            ns.push_back(n);
            ts.push_back(us);
        }
        double a, b;
        fit(ns, ts, a, b);
        printf(" | %9.3f %9.3f\n", b, a);
    }

    printf("\nordinary launches for comparison, N back-to-back empty kernels on one stream\n");
    printf("%8s | %s | %11s %11s\n", "blocks", "  N=1     N=2     N=4     N=8    N=16    N=32    N=64",
           "us/launch", "intercept");
    for (int gi = 0; gi < 4; gi++) {
        const int blocks = grids[gi];
        std::vector<int> ns;
        std::vector<double> ts;
        printf("%8d |", blocks);
        for (int k = 1; k < 8; k++) {
            const int n = ns_arr[k];
            std::vector<double> samples;
            for (int rep = 0; rep < 9; rep++) {
                for (int it = 0; it < n; it++)
                    empty_kern<<<blocks, TB>>>(d_sink);
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipEventRecord(beg));
                for (int it = 0; it < 20 * n; it++) empty_kern<<<blocks, TB>>>(d_sink);
                HIP_CHECK(hipEventRecord(end));
                HIP_CHECK(hipEventSynchronize(end));
                float ms = 0;
                HIP_CHECK(hipEventElapsedTime(&ms, beg, end));
                samples.push_back(ms * 1000.0 / 20.0);
            }
            const double us = median(samples);
            printf(" %7.2f", us);
            ns.push_back(n);
            ts.push_back(us);
        }
        double a, b;
        fit(ns, ts, a, b);
        printf(" | %9.3f %9.3f\n", b, a);
    }

    printf("\nVERDICT: a fused kernel trades 2 launches for 2 syncs and saves 2 intercepts.\n");
    HIP_CHECK(hipFree(d_sink));
    return 0;
}
