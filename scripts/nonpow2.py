import torch, aiter

print("AITER:", aiter.__file__)
K = 2048


def call(x, k, **kw):
    r = aiter.topk_select(x, k, **kw)
    if isinstance(r, (tuple, list)):
        r = [t for t in r if t is not None][-1]
    return r


def check(M, N, rows_to_check=3, **kw):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    idx = call(x, K, **kw)
    worst = 0.0
    for r in [0, M // 2, M - 1][:rows_to_check]:
        got_i = idx[r].to(torch.int64)
        if int(got_i.min()) < 0 or int(got_i.max()) >= N:
            return 1.0, "index out of range"
        got = x[r][got_i].sort().values
        ref = torch.topk(x[r].float(), K).values.sort().values
        bad = int((got != ref).sum())
        worst = max(worst, bad / K)
    return worst, ""


print()
print("%6s %9s %8s %10s  %s" % ("M", "N", "N%4", "worst_bad", "note"))
for M in (8, 64, 512):
    for base in (131072, 262144):
        for off in (0, 1, 2, 3):
            N = base + off
            w, note = check(M, N)
            flag = "" if w == 0 else ("  <-- %.1f%% of elements wrong" % (100 * w))
            print("%6d %9d %8d %10.4f  %s%s" % (M, N, N % 4, w, note, flag))
