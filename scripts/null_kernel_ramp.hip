// null_kernel_ramp.hip -- what a kernel costs before it does anything.
//
// phase_a and phase_c are one block per row. At m=8 that is 8 workgroups on 256
// CUs, and rocprof-compute measures 1.38% CU utilisation for phase_a there. The
// question this answers is whether the ~7us per-block floor those kernels show
// is work that more blocks could spread, or a dispatch ramp that no amount of
// blocks can remove -- because if it is the ramp, three kernels cost three ramps
// and the only lever is launching fewer of them.
//
// The LDS sweep is not decoration: phase_a asks for S*4 bytes of dynamic LDS
// (26 KB at S=6528) and phase_c for cap*8 (64 KB at cap=8192), and a workgroup
// that reserves 64 KB cannot be placed as freely as one that reserves none.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void nop_kernel(float* __restrict__ sink)
{
    // Touch the dynamic LDS so the allocation is not elided, but keep the work
    // to one store per block: this must measure the launch, not the body.
    extern __shared__ char s_dyn[];
    if(threadIdx.x == 0)
    {
        s_dyn[0] = (char)blockIdx.x;
        if(blockIdx.x == 0xFFFFFF) sink[0] = (float)s_dyn[0];
    }
}

int main(int argc, char** argv)
{
    const int iters = (argc > 1) ? atoi(argv[1]) : 40;
    float* d_sink = nullptr;
    (void)hipMalloc(&d_sink, 1024);

    const int grids[]  = {8, 16, 32, 64, 128, 256, 512, 1024, 4096};
    const int blocks[] = {256, 512, 1024};
    const size_t ldsb[] = {0, 8192, 32768, 65536};

    printf("%8s %8s %10s\n", "grid", "block", "lds_KB");
    for(size_t li = 0; li < sizeof(ldsb) / sizeof(ldsb[0]); li++)
        for(size_t bi = 0; bi < sizeof(blocks) / sizeof(blocks[0]); bi++)
            for(size_t gi = 0; gi < sizeof(grids) / sizeof(grids[0]); gi++)
            {
                const int g = grids[gi], b = blocks[bi];
                const size_t L = ldsb[li];
                for(int w = 0; w < 5; w++) nop_kernel<<<g, b, L>>>(d_sink);
                (void)hipDeviceSynchronize();
                hipError_t e = hipGetLastError();
                if(e != hipSuccess)
                {
                    printf("%8d %8d %10zu   LAUNCH-FAILED %s\n", g, b, L / 1024,
                           hipGetErrorString(e));
                    continue;
                }
                for(int it = 0; it < iters; it++) nop_kernel<<<g, b, L>>>(d_sink);
                (void)hipDeviceSynchronize();
                printf("%8d %8d %10zu\n", g, b, L / 1024);
            }
    return 0;
}
