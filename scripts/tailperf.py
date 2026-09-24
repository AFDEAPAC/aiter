import os

import aiter
import torch

K = 2048
M = int(os.environ.get("M", 512))
for N in (131072, 131073, 131074, 131075, 131076, 262144, 262145, 262147):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    for _ in range(8):
        aiter.topk_select(x, K)
    torch.cuda.synchronize()
    for _ in range(20):
        aiter.topk_select(x, K)
    torch.cuda.synchronize()
    print("done %d %d" % (M, N), flush=True)
