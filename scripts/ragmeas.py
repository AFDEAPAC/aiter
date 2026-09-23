import torch, aiter, os
K = 2048
FORCE = os.environ.get("FORCE_RAGGED", "") == "1"
def call(x, k):
    idx = torch.empty(x.shape[0], k, device=x.device, dtype=torch.int32)
    if FORCE:
        r = aiter.topk_select(x, k, end=torch.full((x.shape[0],), x.shape[1],
                                                   device=x.device, dtype=torch.int32))
    else:
        r = aiter.topk_select(x, k)
    return r
for M, N in [(2048,131072),(1024,131072),(512,131072),(256,131072),(512,262144),(256,262144),(4096,131072),(128,1048576)]:
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    for _ in range(8): call(x, K)
    torch.cuda.synchronize()
    for _ in range(20): call(x, K)
    torch.cuda.synchronize()
