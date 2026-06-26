# SPDX-License-Identifier: MIT
# Clock-fair within-process A/B with min-of-many (peak-clock, contention-robust):
#   parallel = grid-z QLEN (1 launch, q[:, :N])
#   serial   = N launches of qlen=1 (emulates old in-block serial loop)
# Reports MIN over many repeat windows (noise/DVFS only slows things down).

import math, torch, aiter
from aiter.ops.mla_decode_opus import mla_decode_opus

def csr(B, L, dev):
    ip = torch.arange(0, (B + 1) * L, L, dtype=torch.int32, device=dev)
    ix = torch.arange(L, dtype=torch.int32, device=dev).repeat(B)
    return ip, ix

def timed(fn, iters):
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us

def best(fn, iters=30, repeats=20):
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    return min(timed(fn, iters) for _ in range(repeats))

def run(B=256, L=4096):
    dev = "cuda"
    ip, ix = csr(B, L, dev)
    ss = 1.0 / math.sqrt(512)
    print(f"B={B} L={L}  (min over repeats, us)")
    print("H\tQLEN\tparallel\tserial\tspeedup")
    for h in (16, 128):
        ukv = (torch.randn(L, 512, device=dev) * 0.5).bfloat16()
        sink = (torch.randn(h, device=dev) * 0.25)
        for N in (1, 2, 4, 8):
            q = (torch.randn(B, N, h, 512, device=dev) * 0.5).bfloat16()
            out = torch.empty_like(q)
            par = best(lambda: mla_decode_opus(q, ukv, ix, ip, sink, ss, out=out))
            q1 = [q[:, p:p+1].contiguous() for p in range(N)]
            o1 = [torch.empty_like(t) for t in q1]
            def serial():
                for p in range(N):
                    mla_decode_opus(q1[p], ukv, ix, ip, sink, ss, out=o1[p])
            ser = best(serial)
            print(f"{h}\t{N}\t{par:.1f}\t\t{ser:.1f}\t\t{ser/par:.2f}x")

if __name__ == "__main__":
    run()
