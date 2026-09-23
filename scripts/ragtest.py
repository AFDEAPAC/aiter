import torch, aiter, os, time
print("AITER:", aiter.__file__)
K = 2048
for M, N in [(2048, 131072), (256, 131072), (512, 262144), (4096, 131072)]:
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    idx = aiter.topk_select(x, K)
    got = idx[0].to(torch.int64).sort().values
    ref = torch.topk(x[0].float(), K).indices.sort().values
    same = bool((x[0][got].sort().values == x[0][ref].sort().values).all())
    for _ in range(5): aiter.topk_select(x, K)
    torch.cuda.synchronize()
    for _ in range(20): aiter.topk_select(x, K)
    torch.cuda.synchronize()
    print("  m=%-5d n=%-8d row0 value-multiset matches torch.topk: %s" % (M, N, same))
