import torch, aiter
print("AITER:", aiter.__file__)
K = 2048
def call(x, k, **kw):
    r = aiter.topk_select(x, k, **kw)
    if isinstance(r, (tuple, list)):
        r = [t for t in r if t is not None][-1]
    return r
bad = 0
for M, N in [(1,131072),(16,131072),(64,262144),(256,131072),(512,262144),(2048,131072),(4096,131072),(128,1048576)]:
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    # plain: no row bounds -> the new ragged=False path
    idx = call(x, K)
    for r in (0, M // 2, M - 1):
        got = x[r][idx[r].to(torch.int64)].sort().values
        ref = torch.topk(x[r].float(), K).values.sort().values
        if not torch.equal(got, ref):
            print("  MISMATCH plain m=%d n=%d row=%d" % (M, N, r)); bad += 1
    # ragged: real row bounds -> must still take the old path and stay correct
    ends = torch.full((M,), N, device="cuda", dtype=torch.int32)
    ends[: max(1, M // 4)] = N - 7
    idx2 = call(x, K, end=ends)
    for r in (0, M - 1):
        L = int(ends[r])
        got = x[r][idx2[r].to(torch.int64)].sort().values
        ref = torch.topk(x[r][:L].float(), K).values.sort().values
        if not torch.equal(got, ref):
            print("  MISMATCH ragged m=%d n=%d row=%d" % (M, N, r)); bad += 1
print("MISMATCHES:", bad)
