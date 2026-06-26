# SPDX-License-Identifier: MIT
# Single-pass vs split-KV (kernel-only: workspace pre-allocated, splits fixed).

import math, sys, torch, aiter
from aiter.ops.mla_decode_opus import (
    mla_decode_opus, mla_decode_opus_fwd, mla_decode_opus_splitkv_fwd, _pick_num_splits,
)

def csr(B, L, dev):
    ip = torch.arange(0, (B + 1) * L, L, dtype=torch.int32, device=dev)
    ix = torch.arange(L, dtype=torch.int32, device=dev).repeat(B)
    return ip, ix

def best(fn, iters=30, repeats=20):
    for _ in range(40): fn()
    torch.cuda.synchronize()
    def w():
        s=torch.cuda.Event(True);e=torch.cuda.Event(True);s.record()
        for _ in range(iters): fn()
        e.record();torch.cuda.synchronize();return s.elapsed_time(e)/iters*1e3
    return min(w() for _ in range(repeats))

def run(L=4096, qlen=1):
    dev="cuda"; d=512; ss=1.0/math.sqrt(d)
    print(f"L={L} qlen={qlen}  (min us, kernel-only)")
    print("H\tB\tsingle\tsplitKV\tsplits\tspeedup")
    for h in (16, 128):
        ukv=(torch.randn(L,d,device=dev)*0.5).bfloat16()
        sink=(torch.randn(h,device=dev)*0.25)
        nhb = max(1, h//128) if h>32 else 1
        for B in (1, 4, 16, 32, 64, 128, 256):
            q=(torch.randn(B,qlen,h,d,device=dev)*0.5).bfloat16()
            ip,ix=csr(B,L,dev); out=torch.empty_like(q)
            sp=best(lambda: mla_decode_opus_fwd(q,ukv,ix,ip,sink,out,ss,qlen))
            ns=_pick_num_splits(B,qlen,nhb,L)
            if ns<=1:
                print(f"{h}\t{B}\t{sp:.1f}\t{sp:.1f}\t1\t1.00x")
                continue
            rows=B*qlen*ns
            po=torch.empty((rows,h,d),dtype=torch.float32,device=dev)
            pml=torch.empty((rows,h,2),dtype=torch.float32,device=dev)
            out2=torch.empty_like(q)
            sk=best(lambda: mla_decode_opus_splitkv_fwd(q,ukv,ix,ip,sink,out2,po,pml,ss,qlen,ns))
            print(f"{h}\t{B}\t{sp:.1f}\t{sk:.1f}\t{ns}\t{sp/sk:.2f}x")

if __name__=="__main__":
    run(int(sys.argv[1]) if len(sys.argv)>1 else 4096, int(sys.argv[2]) if len(sys.argv)>2 else 1)
