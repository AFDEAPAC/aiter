#include "topk_common.hip.hpp"
#include "topk_shape.hip.hpp"
#include <cstdio>
int main() {
  const int K = 2048;
  printf("%6s %9s %7s %8s %7s %7s %6s  %14s\n",
         "M", "N", "S", "margin", "rank", "cap", "coop", "exp cands/row");
  for (int M : {4096})
    for (int N : {131072, 262144, 524288, 1048576}) {
      ShapeParams p = derive_shape_params(M, N, K, 0.f, 0, 0, PATH_AUTO);
      printf("%6d %9d %7d %8.3f %7d %7d %6d  %14.0f\n",
             M, N, p.S, p.margin, p.rank, p.cap, p.coop_g, (double)p.margin * K);
    }
  return 0;
}
