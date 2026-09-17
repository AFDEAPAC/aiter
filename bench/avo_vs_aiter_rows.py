#!/usr/bin/env python3
"""Per-row AVO vs aiter comparison, to classify a value-sum divergence.

The contract audit found AVO and aiter reporting different value sums on the
baseline shape while both matched torch's multiset on row 0. Either the
difference is tie-breaking at the K-th boundary (both correct) or it is a real
disagreement. A sum cannot tell those apart, so this compares row by row:

  - value multiset equality per row (the only ordering-free correctness test)
  - for rows that differ, whether the differing slots hold EQUAL values, which
    is what a tie looks like
  - whether each side's multiset matches torch.topk for that row

Both backends run in one process; only the non-faulting shapes belong here.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aiter_contract_audit import stub_flydsl, use_mounted_aiter  # noqa: E402


def run_backend(aiter_mod, torch, logits, starts, ends, M, N, stride1, K, use_avo):
    os.environ["AITER_DISABLE_TOPK_AVO"] = "0" if use_avo else "1"
    idx = torch.full((M, K), -1, dtype=torch.int32, device="cuda")
    vals = torch.full((M, K), 0.0, dtype=torch.float32, device="cuda")
    aiter_mod.top_k_per_row_prefill(logits, starts, ends, idx, vals,
                                    M, N, stride1, K, False)
    torch.cuda.synchronize()
    return idx.clone(), vals.clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--n", type=int, default=65536)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--start-stride", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    use_mounted_aiter()
    stub_flydsl()
    import torch
    import aiter

    M, N, K = args.m, args.n, args.k
    torch.manual_seed(args.seed)
    logits = torch.randn((M, N), dtype=torch.float32, device="cuda")
    starts = (torch.arange(M, dtype=torch.int32, device="cuda") * args.start_stride)
    ends = torch.full((M,), N, dtype=torch.int32, device="cuda")

    a_idx, a_val = run_backend(aiter, torch, logits, starts, ends, M, N, 1, K, True)
    b_idx, b_val = run_backend(aiter, torch, logits, starts, ends, M, N, 1, K, False)

    lg = logits.cpu()
    ai, bi = a_idx.cpu().long(), b_idx.cpu().long()
    st = starts.cpu().tolist()

    n_idx_same = n_multiset_same = n_tie_only = n_real = 0
    n_avo_vs_torch = n_aiter_vs_torch = 0
    first_real = None

    for r in range(M):
        s = st[r]
        rl = N - s
        n = min(K, rl)
        av = torch.sort(lg[r][ai[r, :n]], descending=True).values
        bv = torch.sort(lg[r][bi[r, :n]], descending=True).values
        tv = torch.sort(torch.topk(lg[r, s:N], n).values, descending=True).values

        if torch.equal(ai[r, :n], bi[r, :n]):
            n_idx_same += 1
        if torch.equal(av, bv):
            n_multiset_same += 1
        else:
            # A tie at the boundary: the two selections hold the same values
            # everywhere except slots whose values are equal to the K-th value.
            kth = tv[-1]
            a_extra = av[av != bv] if av.shape == bv.shape else av
            if bool((a_extra == kth).all().item()):
                n_tie_only += 1
            else:
                n_real += 1
                if first_real is None:
                    first_real = r
        if torch.equal(av, tv):
            n_avo_vs_torch += 1
        if torch.equal(bv, tv):
            n_aiter_vs_torch += 1

    print("M=%d N=%d K=%d start_stride=%d" % (M, N, K, args.start_stride))
    print("  rows with identical index lists      %d / %d" % (n_idx_same, M))
    print("  rows with identical value multisets  %d / %d" % (n_multiset_same, M))
    print("  rows differing only at K-th ties     %d" % n_tie_only)
    print("  rows differing for another reason    %d%s"
          % (n_real, (" (first: row %d)" % first_real) if first_real is not None else ""))
    print("  AVO   multiset == torch.topk         %d / %d" % (n_avo_vs_torch, M))
    print("  aiter multiset == torch.topk         %d / %d" % (n_aiter_vs_torch, M))
    print("  value sums: avo %.4f  aiter %.4f"
          % (a_val.cpu().sum().item(), b_val.cpu().sum().item()))
    return 0 if n_real == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
