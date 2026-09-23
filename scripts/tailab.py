import torch, aiter, os

K = 2048
M = int(os.environ.get("M", 512))
FORCE = os.environ.get("FORCE_RAGGED", "") == "1"
for N in (131072, 131073, 131075, 262144, 262147, 524291):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    end = torch.full((M,), N, device="cuda", dtype=torch.int32) if FORCE else None
    for _ in range(8):
        aiter.topk_select(x, K, end=end) if FORCE else aiter.topk_select(x, K)
    torch.cuda.synchronize()
    for _ in range(20):
        aiter.topk_select(x, K, end=end) if FORCE else aiter.topk_select(x, K)
    torch.cuda.synchronize()
