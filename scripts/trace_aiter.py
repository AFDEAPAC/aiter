import torch, aiter, os
print("AITER:", aiter.__file__)
M, N, K = int(os.environ.get("M", 2048)), int(os.environ.get("N", 131072)), 2048
torch.manual_seed(0); torch.cuda.manual_seed_all(0)
x = torch.randn(M, N, device="cuda", dtype=torch.float32)
for _ in range(5):
    aiter.topk_select(x, K)
torch.cuda.synchronize()
for _ in range(20):
    aiter.topk_select(x, K)
torch.cuda.synchronize()
print("done")
