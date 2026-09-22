// shape_probe.cpp -- print the sampled selector's shape plan, per (M, N).
//
// `fb_count` says N=524288 is the only width in the customer range where a row
// misses its sampled threshold and drops into the exact fallback, and that the
// rate is 1.2% under the small-S rule against 0.005% above it. This prints the
// numbers that rule produces, so the miss can be attributed to one of them
// rather than guessed at.
//
//   hipcc -O2 --offload-arch=gfx950 shape_probe.cpp -o shape_probe && ./shape_probe
#include "topk_sampled/topk_common.hip.hpp"
#include "topk_sampled/topk_shape.hip.hpp"
#include <cstdio>

int main()
{
    const int K       = 2048;
    const int Ms[]    = {1, 16, 32, 64, 128, 256, 1024, 4096};
    const int Ns[]    = {131072, 262144, 524288, 1048576};
    printf("k=%d.  R = margin*K*S/N is the estimator's rank; the candidate count\n", K);
    printf("has spread ~ count/sqrt(R), so a small R is a wide spread and a miss.\n\n");
    printf("%6s %9s %7s %8s %7s %7s %8s %10s %7s\n",
           "M", "N", "S", "margin", "rank", "cap", "R", "hi/cap", "s_rule");
    for(int M : Ms)
    {
        for(int N : Ns)
        {
            const ShapeParams p =
                derive_shape_params(M, N, K, 0.f, 0, 0, PATH_AUTO);
            const double R  = (double)p.margin * K * p.S / (double)N;
            const double hi = candidate_hi(K, p.S, N, p.margin);
            printf("%6d %9d %7d %8.3f %7d %7d %8.1f %10.3f %7d\n",
                   M, N, p.S, p.margin, p.rank, p.cap, R, hi / (double)p.cap,
                   effective_s_rule(M));
        }
        printf("\n");
    }
    return 0;
}
