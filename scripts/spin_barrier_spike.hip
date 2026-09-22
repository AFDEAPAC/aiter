// spin_barrier_spike.hip -- what does a HAND-ROLLED barrier over G blocks cost,
// against the 2.63 us of an extra kernel launch?
//
// `knowledge/known_bad.md` ("Cooperative grid.sync() costs 7-196 us") killed the
// multi-block radix select, but it measured `cg::this_grid().sync()`: a sync of
// the WHOLE grid, under cooperative-launch semantics. `phase_d_fallback` needs
// something smaller -- a barrier among the G blocks that share one row, with G
// in the low tens -- and that has not been measured. If it is under one kernel
// launch, spreading `block_select_stream`'s four full-row passes over G blocks
// is viable; if not, the direction dies here with a number.
//
// The barrier is the standard sense-reversing counter so it can be re-entered
// without a reset between passes.
#include <hip/hip_runtime.h>
#include <cstdio>

struct Barrier
{
    unsigned* count; // one counter per group
    unsigned* sense; // one sense flag per group
};

__device__ __forceinline__ void
group_barrier(Barrier b, int group, int nblocks, unsigned& local_sense)
{
    __syncthreads();
    if(threadIdx.x == 0)
    {
        local_sense ^= 1u;
        __threadfence();
        const unsigned arrived = atomicAdd(&b.count[group], 1u) + 1u;
        if(arrived == (unsigned)nblocks)
        {
            b.count[group] = 0;
            __threadfence();
            atomicExch(&b.sense[group], local_sense);
        }
        else
        {
            while(atomicAdd(&b.sense[group], 0u) != local_sense)
                __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();
}

// `passes` barriers and nothing else, so the difference between two pass counts
// is the barrier cost alone.
__global__ __launch_bounds__(1024) void
barrier_only(Barrier b, int nblocks, int groups, int passes, unsigned* sink)
{
    const int group = blockIdx.x / nblocks;
    if(group >= groups)
        return;
    unsigned local_sense = 0;
    for(int p = 0; p < passes; p++)
        group_barrier(b, group, nblocks, local_sense);
    if(threadIdx.x == 0 && blockIdx.x == 0)
        *sink = local_sense;
}

static float time_kernel(int nblocks, int groups, int passes, Barrier b, unsigned* sink)
{
    const dim3 grid(nblocks * groups), block(1024);
    for(int i = 0; i < 3; i++)
        hipLaunchKernelGGL(barrier_only, grid, block, 0, 0, b, nblocks, groups, passes, sink);
    hipDeviceSynchronize();
    hipEvent_t s, e;
    hipEventCreate(&s);
    hipEventCreate(&e);
    const int iters = 50;
    hipEventRecord(s);
    for(int i = 0; i < iters; i++)
        hipLaunchKernelGGL(barrier_only, grid, block, 0, 0, b, nblocks, groups, passes, sink);
    hipEventRecord(e);
    hipEventSynchronize(e);
    float ms = 0;
    hipEventElapsedTime(&ms, s, e);
    hipEventDestroy(s);
    hipEventDestroy(e);
    return ms * 1000.0f / iters; // us per launch
}

int main()
{
    unsigned *count, *sense, *sink;
    hipMalloc(&count, 4096 * sizeof(unsigned));
    hipMalloc(&sense, 4096 * sizeof(unsigned));
    hipMalloc(&sink, sizeof(unsigned));
    hipMemset(count, 0, 4096 * sizeof(unsigned));
    hipMemset(sense, 0, 4096 * sizeof(unsigned));
    Barrier b{count, sense};

    printf("hand-rolled sense-reversing barrier, 1024 threads/block\n");
    printf("%8s %8s %12s %12s %14s\n", "G", "groups", "0 barriers", "8 barriers", "us/barrier");
    for(int g : {2, 4, 8, 16, 32, 64})
    {
        for(int groups : {1, 8})
        {
            if(g * groups > 1024)
                continue;
            const float t0 = time_kernel(g, groups, 0, b, sink);
            const float t8 = time_kernel(g, groups, 8, b, sink);
            printf("%8d %8d %12.3f %12.3f %14.3f\n", g, groups, t0, t8, (t8 - t0) / 8.0f);
        }
    }
    printf("\nreference: one empty kernel launch on this box is ~2.63 us\n");
    hipFree(count);
    hipFree(sense);
    hipFree(sink);
    return 0;
}
