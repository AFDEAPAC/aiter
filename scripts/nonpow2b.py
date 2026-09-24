import os

import aiter
import torch

print("AITER:", aiter.__file__)
K = 2048


def call(x, k):
    r = aiter.topk_select(x, k)
    if isinstance(r, (tuple, list)):
        r = [t for t in r if t is not None][-1]
    return r


def check(M, N):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    idx = call(x, K)
    worst = 0.0
    for r in range(M):
        gi = idx[r].to(torch.int64)
        if int(gi.min()) < 0 or int(gi.max()) >= N:
            return 1.0
        got = x[r][gi].sort().values
        ref = torch.topk(x[r].float(), K).values.sort().values
        worst = max(worst, int((got != ref).sum()) / K)
    return worst


M = int(os.environ.get("M", 8))
print("%6s %9s %6s %11s" % ("M", "N", "N%4", "worst_bad"))
for base in (131072, 262144, 524288, 1048576):
    for off in (0, 1, 2, 3, 5, 7):
        N = base + off
        w = check(M, N)
        print(
            "%6d %9d %6d %11.4f %s" % (M, N, N % 4, w, "  <-- WRONG" if w > 0 else "")
        )
