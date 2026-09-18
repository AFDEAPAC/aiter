// Vendored 2026-09-18 from https://amd-afde.top/rocm/select_grid.hip (HTTP 200,
// 21581 bytes, 501 lines). Colleague deliverable, unmodified below this header.
// Used by bench/floor_grid.py to produce the per-cell floor in
// reports/ceiling_report.html. Build:
//   hipcc --offload-arch=gfx950 -O3 -Wall -Wextra scripts/select_grid.hip -o select_grid

// Ideal-selector floor, swept over the whole (m, n, k) heat map.
//
// Generalisation of select_min.hip: M, N, K and G became runtime parameters
// so 555 cells cost one compile instead of 555.  TB and MLP stay in the
// template because they change the generated code; TAIL does too, so that a
// cell whose N is a multiple of 4 pays nothing for the bounds check that a
// cell like N=131071 needs.
//
// One workgroup per (row, chunk).  Every thread loads 128 bits at a time,
// compares each value against the row's threshold, and whichever pass write
// their index.  One pass over the data, no radix, no histogram.
//
// The data is a hash of the global index, so one buffer filled once serves
// every cell: a cell just reads the first m * n4full * 4 floats of it.
//
// =========================================================================
//  HOW TO USE
// =========================================================================
//
// What this gives you.  A lower bound for any correct per-row top-k on this
// hardware: the time to read the data once, compare every element against a
// threshold, and write out the indices that pass.  Nothing else.  Compare a
// real top-k against it to see how much of the gap is the algorithm and how
// much is the memory system.
//
// ---- Build -------------------------------------------------------------
//
//   hipcc --offload-arch=gfx950 -O3 -Wall -Wextra select_grid.hip -o select_grid
//
// If ROCm came from a Python wheel rather than /opt/rocm, libamdhip64.so is
// not on the default path and the binary will not start.  Point at it:
//
//   export LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib:$LD_LIBRARY_PATH
//
// ---- Run ---------------------------------------------------------------
//
//   ./select_grid spec.txt > out.txt
//
// One line per cell, "#" and blank lines ignored:
//
//   m  n  k  g  mlp  tb  iters
//
//   m      rows
//   n      values per row
//   k      indices to emit per row; k == n is legal and degenerates to a copy
//   g      workgroups per row (see below)
//   mlp    loads issued back to back before any is consumed: 1, 2, 4 or 8
//   tb     threads per workgroup: 64, 256 or 1024
//   iters  timed launches per pass; 3 passes are always run
//
// Example, the cell this was built for:
//
//   # m    n       k     g mlp tb   iters
//   4096   131072  2048  1 4   1024 50
//
// ---- Choosing g, mlp, tb -----------------------------------------------
//
// These were measured on MI355X (256 CUs), not guessed.  Start here:
//
//   tb  = 1024, except n <= 4096 where 256 is 15-17% faster
//   g   = clamp(512 / m, 1, 32), rounded UP to a power of two
//   mlp = 4 when g == 1, otherwise 1
//
// The reasoning: 512 workgroups of 1024 threads fill the machine (256 CUs x
// 2 blocks/CU), so g exists to reach that count when m alone cannot.  It is
// worth about two thirds of the time at m <= 64 and 8.8% at m = 4096.  mlp
// hides the LDS atomics behind load latency, but each extra stream in flight
// spreads the address footprint and costs read bandwidth, so once g has
// already split the row, mlp = 1 wins.
//
// Do not use mlp = 8.  The scalar tail loop it needs costs read_kern 12 VGPRs
// (42 -> 54), which makes the read control slower than the kernel it is
// supposed to be controlling, and the paired difference goes negative.
//
// Sweeping 2-3 candidates per cell and keeping the best is cheap and worth it.
//
// ---- Reading the output ------------------------------------------------
//
// A "#COLS" header then one "#R" row per cell.  The columns that matter:
//
//   sel1 sel2 sel3   select kernel, median us of each pass -> THE FLOOR
//   rd1 rd2 rd3      read-only control, same geometry, same loads
//   d1 d2 d3         paired difference, what selection costs on top of reading
//   hit_rate         measured, from a counter -- check it against k/n
//   drop_pct         hits beyond the slots a block owns, dropped
//
// Take the median of sel1..sel3 as the floor for that cell.
//
// ---- Four things worth knowing before you trust a number ---------------
//
// 1. The floor is end to end.  Read and select have OPPOSING optimal
//    geometries: on the 4096 x 131072 cell, g=32/mlp=1 reads fastest
//    (303.6 us) but selects slowest (372.6 us), while g=1/mlp=4 selects
//    fastest (342.4 us) with a slower read (335.8 us).  Never build a floor
//    by adding a best-case read to a selection cost -- no geometry delivers
//    both.  Take the smallest end-to-end select.
//
// 2. Check hit_rate against k/n every time.  It is read back from a counter
//    rather than assumed, so it catches a wrong threshold, a bad fill, or a
//    cell that silently did nothing.
//
// 3. drop_pct is expected, not a bug.  The threshold is a fixed quantile, so
//    per-row hit counts are binomial around k (sd ~ sqrt(n*p*(1-p)), about 45
//    at the reference cell).  Rows that run over the slots they own drop the
//    excess.  At the reference cell that is 0.89% of hits and 0.013% of the
//    time.  It grows as sqrt(g), so watch it when g is large.
//
// 4. Small cells measure dispatch, not selection.  An empty kernel costs
//    about 6 us here.  Below roughly m = 128 the floor IS that launch cost --
//    m=4 and m=16 both land at 6.2 us.  Those cells tell you nothing about a
//    selector.
//
// ---- Memory ------------------------------------------------------------
//
// One input buffer is allocated for the largest cell in the spec and filled
// once; every smaller cell reads a prefix of it, because fill_randn hashes
// the global index and any prefix is still valid randn.  4096 x 1048576 needs
// 17.2 GB.  Put the big cells and the small ones in the same spec file and
// you pay the fill once for all of them.
//
// ---- Reference numbers -------------------------------------------------
//
// MI355X, ROCm 10.0, m=4096 n=131072 k=2048 g=1 mlp=4 tb=1024:
//
//   read 335.8 us    select 342.4 us    difference 6.6 us
//   hit_rate 0.015628 against k/n 0.015625    drop 0.89%
//
// If your build lands far from that on the same hardware, something in the
// toolchain or the machine differs -- find out what before using the numbers.

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HIP_CHECK(x)                                                        \
    do {                                                                    \
        hipError_t e_ = (x);                                                \
        if (e_ != hipSuccess) {                                             \
            fprintf(stderr, "%s:%d  %s\n", __FILE__, __LINE__,              \
                    hipGetErrorString(e_));                                 \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static constexpr int PASSES = 3;

// =========================================================================

// Block (chunk, row) walks float4 groups [chunk*per4, min(chunk*per4+per4,
// n4full)) of its row and owns SLOTS of that row's output.  Rows are padded
// to a whole float4 so every row start is 16-byte aligned; TAIL drops the
// padding elements by index.
// EXACT means every chunk is a whole number of TB*MLP batches, so the scalar
// tail loop can go.  It is not just dead code when present: at MLP=8 it costs
// read_kern 12 VGPRs (42 -> 54), which is enough to slow the read control
// below the kernel it is supposed to be a control for.
template <unsigned TB, unsigned MLP, bool TAIL, bool EXACT>
__global__ __launch_bounds__(TB) void select_kern(const float4* __restrict in,
                                                  unsigned* __restrict out,
                                                  unsigned* __restrict counts,
                                                  unsigned n, unsigned n4full,
                                                  unsigned per4, unsigned slots,
                                                  float thr) {
    __shared__ unsigned cnt;
    if (threadIdx.x == 0) cnt = 0u;
    __syncthreads();

    const unsigned chunk = blockIdx.x;
    const unsigned row   = blockIdx.y;
    const unsigned bid   = row * gridDim.x + chunk;

    const unsigned beg = chunk * per4;
    const unsigned end = (beg + per4 < n4full) ? beg + per4 : n4full;

    const float4* a = in + (size_t)row * n4full;
    unsigned*     o = out + (size_t)bid * slots;

    auto emit = [&](const float4& v, unsigned g4) {
        const float s[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (unsigned j = 0; j < 4; j++) {
            const unsigned idx = g4 * 4u + j;
            if (TAIL && idx >= n) continue;          // padding, not data
            if (s[j] > thr) {                        // the one compare
                const unsigned slot = atomicAdd(&cnt, 1u);
                if (slot < slots) o[slot] = idx;
            }
        }
    };

    auto batch = [&](unsigned i) {
        float4 v[MLP];
#pragma unroll
        for (unsigned b = 0; b < MLP; b++)
            get_native_vector(v[b]) =
                __builtin_nontemporal_load(&get_native_vector(a[i + b * TB]));
#pragma unroll
        for (unsigned b = 0; b < MLP; b++) emit(v[b], i + b * TB);
    };

    unsigned i = beg + threadIdx.x;
    if constexpr (EXACT) {
        for (; i < end; i += TB * MLP) batch(i);
    } else {
        // Unrolled while all MLP loads are in range, then one group at a
        // time.  The two together cover [beg, end) exactly once, any per4.
        for (; i + (MLP - 1) * TB < end; i += TB * MLP) batch(i);
        for (; i < end; i += TB) {
            float4 v;
            get_native_vector(v) =
                __builtin_nontemporal_load(&get_native_vector(a[i]));
            emit(v, i);
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) counts[bid] = cnt;
}

// Read-only control: same loads, same chunking, no compare and no store.
template <unsigned TB, unsigned MLP, bool EXACT>
__global__ __launch_bounds__(TB) void read_kern(const float4* __restrict in,
                                                float* __restrict sink,
                                                unsigned n4full, unsigned per4) {
    const unsigned beg = blockIdx.x * per4;
    const unsigned end = (beg + per4 < n4full) ? beg + per4 : n4full;
    const float4* a = in + (size_t)blockIdx.y * n4full;

    float4 acc = {0.f, 0.f, 0.f, 0.f};
    auto batch = [&](unsigned i) {
        float4 v[MLP];
#pragma unroll
        for (unsigned b = 0; b < MLP; b++)
            get_native_vector(v[b]) =
                __builtin_nontemporal_load(&get_native_vector(a[i + b * TB]));
#pragma unroll
        for (unsigned b = 0; b < MLP; b++) {
            acc.x += v[b].x; acc.y += v[b].y;
            acc.z += v[b].z; acc.w += v[b].w;
        }
    };

    unsigned i = beg + threadIdx.x;
    if constexpr (EXACT) {
        for (; i < end; i += TB * MLP) batch(i);
    } else {
        for (; i + (MLP - 1) * TB < end; i += TB * MLP) batch(i);
        for (; i < end; i += TB) {
            float4 v;
            get_native_vector(v) =
                __builtin_nontemporal_load(&get_native_vector(a[i]));
            acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
        }
    }
    const float t = acc.x + acc.y + acc.z + acc.w;
    if (t == 1e30f)
        sink[(blockIdx.y * gridDim.x + blockIdx.x) * TB + threadIdx.x] = t;
}

// Box-Muller off a hashed index, so the tail area of a threshold is real.
__global__ void fill_randn(float* a, size_t n) {
    size_t       i      = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        unsigned h1 = (unsigned)i * 2654435761u + 12345u;
        h1 ^= h1 >> 15; h1 *= 2246822519u; h1 ^= h1 >> 13;
        h1 *= 3266489917u; h1 ^= h1 >> 16;
        unsigned h2 = h1 * 2246822519u + 0x9e3779b9u;
        h2 ^= h2 >> 15; h2 *= 2246822519u; h2 ^= h2 >> 13;
        h2 *= 3266489917u; h2 ^= h2 >> 16;
        const float u1 = ((h1 >> 8) + 1u) * (1.0f / 16777217.0f);   // (0,1)
        const float u2 = (h2 >> 8) * (1.0f / 16777216.0f);          // [0,1)
        a[i] = sqrtf(-2.0f * __logf(u1)) * __cosf(6.2831853f * u2);
    }
}

// =========================================================================

// Upper-tail quantile of the standard normal at area p, by bisection on erfc
// (which is exact to double precision and needs no table).  p == 1 means
// every element is wanted, so the threshold is -inf and the cell degenerates
// to a copy.
static double normal_upper_quantile(double p) {
    if (p >= 1.0) return -INFINITY;
    if (p <= 0.0) return INFINITY;
    double lo = -40.0, hi = 40.0;
    for (int it = 0; it < 200; it++) {
        const double mid = 0.5 * (lo + hi);
        if (std::erfc(mid / std::sqrt(2.0)) > 2.0 * p) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

struct Spec { unsigned m, n, k, g, mlp, tb; int iters; };

// ---- launch dispatch over the templated knobs ---------------------------

struct Args {
    const float4* in; unsigned* out; unsigned* counts; float* sink;
    dim3 grid; unsigned tb;
    unsigned n, n4full, per4, slots; float thr;
};

template <unsigned TB, unsigned MLP, bool EXACT>
static void launch_pair(const Args& a, bool tail, bool do_select) {
    if (do_select) {
        if (tail)
            select_kern<TB, MLP, true, EXACT><<<a.grid, TB>>>(
                a.in, a.out, a.counts, a.n, a.n4full, a.per4, a.slots, a.thr);
        else
            select_kern<TB, MLP, false, EXACT><<<a.grid, TB>>>(
                a.in, a.out, a.counts, a.n, a.n4full, a.per4, a.slots, a.thr);
    } else {
        read_kern<TB, MLP, EXACT><<<a.grid, TB>>>(a.in, a.sink, a.n4full, a.per4);
    }
}

template <unsigned TB, unsigned MLP>
static void dispatch_exact(const Args& a, bool tail, bool sel, bool exact) {
    if (exact) launch_pair<TB, MLP, true>(a, tail, sel);
    else       launch_pair<TB, MLP, false>(a, tail, sel);
}

template <unsigned TB>
static void dispatch_mlp(const Args& a, unsigned mlp, bool tail, bool sel,
                         bool exact) {
    switch (mlp) {
        case 1: dispatch_exact<TB, 1>(a, tail, sel, exact); break;
        case 2: dispatch_exact<TB, 2>(a, tail, sel, exact); break;
        case 4: dispatch_exact<TB, 4>(a, tail, sel, exact); break;
        case 8: dispatch_exact<TB, 8>(a, tail, sel, exact); break;
        default: fprintf(stderr, "bad mlp %u\n", mlp); exit(1);
    }
}

static void dispatch(const Args& a, unsigned mlp, bool tail, bool sel,
                     bool exact) {
    switch (a.tb) {
        case 64:   dispatch_mlp<64>(a, mlp, tail, sel, exact); break;
        case 256:  dispatch_mlp<256>(a, mlp, tail, sel, exact); break;
        case 1024: dispatch_mlp<1024>(a, mlp, tail, sel, exact); break;
        default: fprintf(stderr, "bad tb %u\n", a.tb); exit(1);
    }
}

// =========================================================================

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s spec.txt\n", argv[0]); return 1; }

    std::vector<Spec> specs;
    {
        FILE* f = fopen(argv[1], "r");
        if (!f) { perror("spec"); return 1; }
        char line[512];
        while (fgets(line, sizeof line, f)) {
            if (line[0] == '#' || line[0] == '\n') continue;
            Spec s{};
            if (sscanf(line, "%u %u %u %u %u %u %d", &s.m, &s.n, &s.k, &s.g,
                       &s.mlp, &s.tb, &s.iters) == 7)
                specs.push_back(s);
        }
        fclose(f);
    }
    fprintf(stderr, "specs: %zu\n", specs.size());

    // One buffer for every cell.  Rows are padded to a whole float4.
    size_t max_in4 = 0, max_out = 0, max_blocks = 0, max_sink = 0;
    for (const Spec& s : specs) {
        const size_t n4full = (s.n + 3) / 4;
        const size_t slots  = (s.k + s.g - 1) / s.g;
        max_in4    = std::max(max_in4, (size_t)s.m * n4full);
        max_blocks = std::max(max_blocks, (size_t)s.m * s.g);
        max_out    = std::max(max_out, (size_t)s.m * s.g * slots);
        max_sink   = std::max(max_sink, (size_t)s.m * s.g * s.tb);
    }
    const size_t in_floats = max_in4 * 4;
    fprintf(stderr, "input %.2f GB  out %.1f MB  sink %.1f MB  blocks %zu\n",
            in_floats * 4 / 1e9, max_out * 4 / 1e6, max_sink * 4 / 1e6,
            max_blocks);

    float*    d_in     = nullptr;
    unsigned* d_out    = nullptr;
    unsigned* d_counts = nullptr;
    float*    d_sink   = nullptr;
    HIP_CHECK(hipMalloc(&d_in, in_floats * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, max_out * sizeof(unsigned)));
    HIP_CHECK(hipMalloc(&d_counts, max_blocks * sizeof(unsigned)));
    HIP_CHECK(hipMalloc(&d_sink, max_sink * sizeof(float)));

    fill_randn<<<8192, 256>>>(d_in, in_floats);
    HIP_CHECK(hipDeviceSynchronize());
    fprintf(stderr, "filled\n");

    const float4* in4 = reinterpret_cast<const float4*>(d_in);

    hipEvent_t beg, end;
    HIP_CHECK(hipEventCreate(&beg));
    HIP_CHECK(hipEventCreate(&end));

    std::vector<unsigned> counts;
    printf("#COLS m n k g mlp tb tail exact blocks per4 slots thr hit_rate hits_total "
           "hits_mean hits_min hits_max written dropped drop_pct "
           "rd1 rd2 rd3 sel1 sel2 sel3 d1 d2 d3\n");

    for (size_t si = 0; si < specs.size(); si++) {
        const Spec& s = specs[si];
        const unsigned n4full = (s.n + 3) / 4;
        const unsigned per4   = (n4full + s.g - 1) / s.g;
        const unsigned slots  = (s.k + s.g - 1) / s.g;
        const bool     tail   = (s.n % 4u) != 0u;
        // Every chunk a whole number of TB*MLP batches: the last chunk is
        // full (n4full divides into per4) and per4 divides into batches.
        const bool     exact  = (n4full % per4 == 0u) &&
                                (per4 % (s.tb * s.mlp) == 0u);
        const double   p      = (double)s.k / (double)s.n;
        const double   thrd   = normal_upper_quantile(p);
        const float    thr    = (float)thrd;

        Args a{};
        a.in = in4; a.out = d_out; a.counts = d_counts; a.sink = d_sink;
        a.grid = dim3(s.g, s.m); a.tb = s.tb;
        a.n = s.n; a.n4full = n4full; a.per4 = per4; a.slots = slots;
        a.thr = thr;

        // Hit counts first, from one clean launch.
        HIP_CHECK(hipMemset(d_counts, 0, (size_t)s.m * s.g * sizeof(unsigned)));
        dispatch(a, s.mlp, tail, true, exact);
        HIP_CHECK(hipDeviceSynchronize());
        counts.assign((size_t)s.m * s.g, 0u);
        HIP_CHECK(hipMemcpy(counts.data(), d_counts,
                            (size_t)s.m * s.g * sizeof(unsigned),
                            hipMemcpyDeviceToHost));

        size_t total = 0, written = 0;
        std::vector<size_t> rowhits(s.m, 0);
        for (size_t b = 0; b < counts.size(); b++) {
            rowhits[b / s.g] += counts[b];
            written += std::min<size_t>(counts[b], slots);
        }
        size_t lo = rowhits[0], hi = rowhits[0];
        for (size_t c : rowhits) { total += c; lo = std::min(lo, c); hi = std::max(hi, c); }

        // Paired timing: read and select measured back to back, PASSES times.
        double rd[PASSES], se[PASSES];
        for (int pass = 0; pass < PASSES; pass++) {
            for (int w = 0; w < 5; w++) { dispatch(a, s.mlp, tail, false, exact);
                                          dispatch(a, s.mlp, tail, true, exact); }
            HIP_CHECK(hipDeviceSynchronize());
            std::vector<double> tr(s.iters), ts(s.iters);
            for (int i = 0; i < s.iters; i++) {
                float ms;
                HIP_CHECK(hipEventRecord(beg));
                dispatch(a, s.mlp, tail, false, exact);
                HIP_CHECK(hipEventRecord(end));
                HIP_CHECK(hipEventSynchronize(end));
                HIP_CHECK(hipEventElapsedTime(&ms, beg, end));
                tr[i] = ms * 1000.0;

                HIP_CHECK(hipEventRecord(beg));
                dispatch(a, s.mlp, tail, true, exact);
                HIP_CHECK(hipEventRecord(end));
                HIP_CHECK(hipEventSynchronize(end));
                HIP_CHECK(hipEventElapsedTime(&ms, beg, end));
                ts[i] = ms * 1000.0;
            }
            std::sort(tr.begin(), tr.end());
            std::sort(ts.begin(), ts.end());
            rd[pass] = tr[s.iters / 2];
            se[pass] = ts[s.iters / 2];
        }

        printf("#R %u %u %u %u %u %u %d %d %zu %u %u %.6g %.6f %zu %.3f %zu %zu "
               "%zu %zu %.4f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
               s.m, s.n, s.k, s.g, s.mlp, s.tb, tail ? 1 : 0, exact ? 1 : 0,
               (size_t)s.m * s.g, per4, slots, thrd,
               (double)total / ((double)s.m * s.n), total,
               (double)total / s.m, lo, hi, written, total - written,
               total ? 100.0 * (total - written) / total : 0.0,
               rd[0], rd[1], rd[2], se[0], se[1], se[2],
               se[0] - rd[0], se[1] - rd[1], se[2] - rd[2]);
        if ((si & 31u) == 0u) {
            fflush(stdout);
            fprintf(stderr, "  %zu/%zu\n", si, specs.size());
        }
    }
    fflush(stdout);

    HIP_CHECK(hipEventDestroy(beg));
    HIP_CHECK(hipEventDestroy(end));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipFree(d_counts));
    HIP_CHECK(hipFree(d_sink));
    return 0;
}
