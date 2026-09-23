"""Why does the plain path lose 72.5% of the elements at N=131077?

Dropping the last one to three columns can cost at most three of 2048, so the
truncation cannot be the whole story. This asks the output which story it tells:

  trunc   does the answer equal torch.topk over the first (N/4)*4 columns? Then
          it IS only the truncation, and the large mismatch count against the
          FULL row must come from somewhere else in the comparison.
  in_tail how many returned columns lie in the part the plain path never reads?
  dup     are there duplicates, i.e. is the gather emitting the same column
          more than once?
  spread  where do the returned columns sit -- uniformly, or clustered?
"""
import os

import torch

import aiter

FP32_EPT = 4
K = 2048


def call(x, k):
    r = aiter.topk_select(x, k)
    if isinstance(r, (tuple, list)):
        r = [t for t in r if t is not None][-1]
    return r


for N in (131072, 131073, 131075, 131077, 131079, 262147, 262151, 524291):
    M = 4
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    idx = call(x, K)
    r = 0
    gi = idx[r].to(torch.int64)
    n4 = (N // FP32_EPT) * FP32_EPT
    got = x[r][gi].sort().values
    ref_full = torch.topk(x[r].float(), K).values.sort().values
    ref_trunc = torch.topk(x[r][:n4].float(), K).values.sort().values
    bad_full = int((got != ref_full).sum())
    bad_trunc = int((got != ref_trunc).sum())
    in_tail = int((gi >= n4).sum())
    dup = K - int(torch.unique(gi).numel())
    # how far down the true ranking do the returned values reach
    allv = x[r].float()
    thresh = float(ref_full[0])
    rank_of_min = int((allv >= float(got[0])).sum())
    print("N=%-9d n4x4=%-9d bad_vs_full=%-6d bad_vs_trunc=%-6d in_tail=%-3d dup=%-5d "
          "worst_rank=%d"
          % (N, n4, bad_full, bad_trunc, in_tail, dup, rank_of_min), flush=True)
