# SPDX-License-Identifier: MIT
# Compare aiter's production asm MLA decode (bf16, RoPE qk=576) vs the OPUS
# absorbed-512 decode (mla_decode_opus / split-KV). Matched (B, nhead, ctx, qlen).
# Caveat: asm does 576-dim QK (512 nope + 64 rope); OPUS does 512 (absorbed, no rope)
# -> asm has ~12% more QK FLOPs. Treat as a ballpark, not bit-identical work.

import math, sys, torch, aiter
import aiter.mla as amla
from aiter.ops.mla_decode_opus import (
    mla_decode_opus_fwd, mla_decode_opus_splitkv_fwd, _pick_num_splits,
)

dev = "cuda"
KV_LORA = 512
ROPE = 64
QK = KV_LORA + ROPE  # 576
VHD = 512

def best(fn, iters=30, repeats=20):
    for _ in range(40): fn()
    torch.cuda.synchronize()
    def w():
        s=torch.cuda.Event(True);e=torch.cuda.Event(True);s.record()
        for _ in range(iters): fn()
        e.record();torch.cuda.synchronize();return s.elapsed_time(e)/iters*1e3
    return min(w() for _ in range(repeats))

def run_asm(B, nhead, ctx, qlen):
    total_q = B * qlen
    num_page = B * ctx + 16
    q = torch.randn((total_q, nhead, QK), dtype=torch.bfloat16, device=dev)
    kv_buffer = torch.randn((num_page, 1, 1, QK), dtype=torch.bfloat16, device=dev)
    qo_indptr = torch.arange(0, (B+1)*qlen, qlen, dtype=torch.int32, device=dev)
    kv_indptr = torch.arange(0, (B+1)*ctx, ctx, dtype=torch.int32, device=dev)
    kv_indices = torch.arange(B*ctx, dtype=torch.int32, device=dev)
    kv_last = torch.ones(B, dtype=torch.int32, device=dev)
    out = torch.empty((total_q, nhead, VHD), dtype=torch.bfloat16, device=dev)
    ss = 1.0/math.sqrt(QK)
    def call():
        amla.mla_decode_fwd(q, kv_buffer.view(num_page,1,1,QK), out, qo_indptr,
                            kv_indptr, kv_indices, kv_last, qlen, 1, 1, ss)
    return best(call)

def run_opus(B, nhead, ctx, qlen):
    q = torch.randn((B, qlen, nhead, KV_LORA), dtype=torch.bfloat16, device=dev)
    ukv = torch.randn((B*ctx, KV_LORA), dtype=torch.bfloat16, device=dev)
    kv_indptr = torch.arange(0, (B+1)*ctx, ctx, dtype=torch.int32, device=dev)
    kv_indices = torch.arange(B*ctx, dtype=torch.int32, device=dev)
    sink = torch.empty(0, dtype=torch.float32, device=dev)
    out = torch.empty_like(q)
    ss = 1.0/math.sqrt(KV_LORA)
    nhb = max(1, nhead//128) if nhead>32 else 1
    ns = _pick_num_splits(B, qlen, nhb, ctx, blocks_per_cu=(2 if nhead<=32 else 1))
    if ns <= 1:
        return best(lambda: mla_decode_opus_fwd(q,ukv,kv_indices,kv_indptr,sink,out,ss,qlen)), ns
    rows=B*qlen*ns
    po=torch.empty((rows,nhead,KV_LORA),dtype=torch.float32,device=dev)
    pml=torch.empty((rows,nhead,2),dtype=torch.float32,device=dev)
    return best(lambda: mla_decode_opus_splitkv_fwd(q,ukv,kv_indices,kv_indptr,sink,out,po,pml,ss,qlen,ns)), ns

if __name__ == "__main__":
    ctx = int(sys.argv[1]) if len(sys.argv)>1 else 4096
    print(f"ctx(kv_len)={ctx}  bf16  (min us)   [asm qk=576 incl RoPE; opus qk=512 absorbed]")
    print("nhead\tqlen\tB\tasm_us\topus_us\tsplits\topus/asm")
    for nhead in (16, 128):
        for qlen in (1, 2, 4):
            for B in (1, 16, 64, 128):
                try:
                    a = run_asm(B, nhead, ctx, qlen)
                except Exception as e:
                    a = float("nan"); print(f"  asm fail nhead={nhead} qlen={qlen} B={B}: {e}", file=sys.stderr)
                try:
                    o, ns = run_opus(B, nhead, ctx, qlen)
                except Exception as e:
                    o, ns = float("nan"), 0; print(f"  opus fail: {e}", file=sys.stderr)
                rel = (o/a) if (a==a and o==o and a>0) else float("nan")
                print(f"{nhead}\t{qlen}\t{B}\t{a:.1f}\t{o:.1f}\t{ns}\t{rel:.2f}")
