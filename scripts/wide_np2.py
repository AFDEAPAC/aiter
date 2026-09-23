import torch, aiter, os

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


bad = 0
tot = 0
print("%6s %9s %6s %11s" % ("M", "N", "N%4", "worst_bad"))
for M in (4, 8, 64):
    for base in (131072, 262144, 524288, 1048576):
        for off in range(0, 9):
            N = base + off
            w = check(M, N)
            tot += 1
            if w > 0:
                bad += 1
                print("%6d %9d %6d %11.4f  <-- WRONG" % (M, N, N % 4, w))
print()
print("checked %d (M, N) combinations, every row of each: %d wrong" % (tot, bad))
