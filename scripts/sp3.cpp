#include "topk_common.hip.hpp"
#include "topk_shape.hip.hpp"
#include <cstdio>
int main() {
  const int K = 2048, M = 4096;
  printf("%9s %7s %8s %7s %7s %9s %8s %8s %7s %s\n",
         "N","S","margin","rank","cap","cand_hi","hi/cap","sigma","exact","verdict");
  for (int N : {131072, 262144}) {
    for (int S : {4096, 6144, 8192, 10240, 12288, 14336, 16384}) {
      ShapeParams p = derive_shape_params(M, N, K, 0.f, S, 0, PATH_AUTO);
      const double hi = candidate_hi(K, p.S, N, p.margin);
      const double exp_c = (double)p.margin * K;
      const double sig = (hi - exp_c) / 3.0;
      printf("%9d %7d %8.3f %7d %7d %9.0f %8.3f %8.2f %7s %s\n",
             N, p.S, p.margin, p.rank, p.cap, hi, hi / p.cap,
             sig > 0 ? (p.cap - exp_c) / sig : 0.0,
             sample_stride_exact(N, p.S) ? "yes" : "no",
             hi <= CAP_SAFE_FILL * p.cap ? "" : "  <-- over CAP_SAFE_FILL");
    }
    printf("\n");
  }
  return 0;
}
