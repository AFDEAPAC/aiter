# SPDX-License-Identifier: MIT
# Batch-size + KV-length sweep (min-of-many) to locate the bound regime.

import math, sys, torch, aiter
from aiter.ops.mla_decode_opus import mla_decode_opus

def csr(B, L, dev):
    ip = torch.arange(0, (B + 1) * L, L, dtype=torch.int32, device=dev)
    ix = torch.arange(L, dtype=torch.int32, device=dev).repeat(B)
    return ip, ix

def best(fn, iters=30, repeats=15):
    for _ in range(40):
        fn()
    torch.cuda.synchronize()
    def w():
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(iters): fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e)/iters*1e3
    return min(w() for _ in range(repeats))

def run(qlen, L):
    dev="cuda"; d=512; ss=1.0/math.sqrt(d)
    print(f"qlen={qlen} L={L}  (min us; CUs=256)")
    print("H\tB\tblocks\tlat_us\tTFLOPS\tus/token")
    for h in (16, 128):
        ukv=(torch.randn(L,d,device=dev)*0.5).bfloat16()
        sink=(torch.randn(h,device=dev)*0.25)
        hb = 1 if h<=32 else max(1, h//128)  # num_h_blocks
        for B in (1, 4, 16, 32, 64, 128, 256):
            q=(torch.randn(B,qlen,h,d,device=dev)*0.5).bfloat16()
            ip,ix=csr(B,L,dev); out=torch.empty_like(q)
            lat=best(lambda: mla_decode_opus(q,ukv,ix,ip,sink,ss,out=out))
            vis=sum(max(0,L-qlen+p) for p in range(qlen))
            tf=4.0*h*vis*d*B/(lat*1e-6)/1e12
            blocks=B*hb*qlen
            print(f"{h}\t{B}\t{blocks}\t{lat:.1f}\t{tf:.0f}\t{lat/(B*qlen):.2f}")

if __name__=="__main__":
    run(int(sys.argv[1]) if len(sys.argv)>1 else 1, int(sys.argv[2]) if len(sys.argv)>2 else 4096)
