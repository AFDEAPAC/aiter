import torch, aiter, os
print("AITER:", aiter.__file__)
K = 2048
def call(x, k):
    r = aiter.topk_select(x, k)
    if isinstance(r, (tuple, list)):
        r = [t for t in r if t is not None][-1]
    return r
for M, N in [(2048, 131072), (256, 131072), (512, 262144)]:
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    idx = call(x, K)
    got = x[0][idx[0].to(torch.int64)].sort().values
    ref = torch.topk(x[0].float(), K).values.sort().values
    ok = bool(torch.equal(got, ref))
    for _ in range(8): call(x, K)
    torch.cuda.synchronize()
    for _ in range(20): call(x, K)
    torch.cuda.synchronize()
    print("  m=%-5d n=%-8d row0 matches torch.topk: %s" % (M, N, ok))
