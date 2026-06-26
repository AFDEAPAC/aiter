# SPDX-License-Identifier: MIT
# Apples-to-apples: aiter production asm MLA decode (bf16, RoPE qk=576 / v=512) vs
# the OPUS RoPE decode (mla_decode_opus_rope, same qk=576 / v=512). Both consume the
# SAME q(576)/kv(576) data and produce out(512) -> identical FLOPs, true comparison
# (unlike the absorbed-512 cmp which gave asm ~12% more QK work).
#
# Usage: python op_tests/cmp_asm_vs_opus_mla_decode_rope.py [ctx]
# Our RoPE path is the 16mx1 (H<=32) variant, so nhead in {8,16} only.

import math, sys, torch, aiter
import aiter.mla as amla
from aiter.ops.mla_decode_opus import (
    mla_decode_opus_rope, mla_decode_opus_rope_fwd, mla_decode_opus_rope_splitkv_fwd,
    mla_decode_opus_workspace, _pick_num_splits, _qpack_min_b,
)

dev = "cuda"
KV_LORA = 512
ROPE = 64
QK = KV_LORA + ROPE  # 576
VHD = 512


def best(fn, iters=100, repeats=10):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()

    def w():
        s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
        for _ in range(iters):
            fn()
        e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / iters * 1e3
    return min(w() for _ in range(repeats))


def make_inputs(B, nhead, ctx, qlen, seed=0):
    torch.manual_seed(seed)
    q = (torch.randn((B, qlen, nhead, QK), dtype=torch.float32, device=dev) * 0.5).to(torch.bfloat16)
    # kv generated bf16-direct (no fp32 temp): the big-ctx kv would otherwise need a
    # 2x-larger fp32 scratch (e.g. 38GB at B256/ctx64K) -> OOM on a busy shared GPU.
    kv = torch.empty((B * ctx, QK), dtype=torch.bfloat16, device=dev).normal_(0.0, 0.5)
    return q, kv


def run_asm(q, kv, B, nhead, ctx, qlen):
    total_q = B * qlen
    num_page = B * ctx
    q_asm = q.reshape(total_q, nhead, QK).contiguous()
    kv_buffer = kv.reshape(num_page, 1, 1, QK).contiguous()
    qo_indptr = torch.arange(0, (B + 1) * qlen, qlen, dtype=torch.int32, device=dev)
    kv_indptr = torch.arange(0, (B + 1) * ctx, ctx, dtype=torch.int32, device=dev)
    kv_indices = torch.arange(B * ctx, dtype=torch.int32, device=dev)
    kv_last = torch.ones(B, dtype=torch.int32, device=dev)
    out = torch.empty((total_q, nhead, VHD), dtype=torch.bfloat16, device=dev)
    ss = 1.0 / math.sqrt(QK)

    def call():
        amla.mla_decode_fwd(q_asm, kv_buffer, out, qo_indptr,
                            kv_indptr, kv_indices, kv_last, qlen, 1, 1, ss)
    return best(call), out, call


def run_opus(q, kv, B, nhead, ctx, qlen):
    # Uses the auto-routing wrapper (split-KV at small batch, qpack for
    # H==16/qlen==4 large batch) so the bench reflects real dispatch.
    kv_indptr = torch.arange(0, (B + 1) * ctx, ctx, dtype=torch.int32, device=dev)
    kv_indices = torch.arange(B * ctx, dtype=torch.int32, device=dev)
    out = torch.empty((B, qlen, nhead, VHD), dtype=torch.bfloat16, device=dev)
    ss = 1.0 / math.sqrt(QK)
    qpack = (nhead == 16 and qlen == 4 and B >= _qpack_min_b(qlen, ctx)) \
            or (nhead == 32 and qlen in (2, 4))   # H32 qpack wins at every B for qlen>=2
    if qpack:
        ns = _pick_num_splits(B, 1, 1, ctx, blocks_per_cu=1)
    else:
        bpc = 2 if nhead <= 32 else 1          # 16mx1 (2/CU) vs 16mx8 (1/CU)
        hpb = 16 if nhead <= 32 else (128 if nhead % 128 == 0 else 64)
        nhb = max(1, -(-nhead // hpb))         # heads/block per variant
        ns = _pick_num_splits(B, qlen, nhb, ctx, blocks_per_cu=bpc)
    po, pml = mla_decode_opus_workspace(B, qlen, nhead, VHD, max(ns, 1), dev)
    call = lambda: mla_decode_opus_rope(
        q, kv, kv_indices, kv_indptr, None, ss, out=out,
        num_splits=ns, partial_o=po, partial_ml=pml, use_qpack=qpack,
    )
    tag = (f"qp{ns}" if qpack else f"s{ns}")
    return best(call), tag, out, call


if __name__ == "__main__":
    ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    check = "--check" in sys.argv
    print(f"ctx(kv_len)={ctx}  bf16  (min us)   [BOTH qk=576 incl RoPE, v=512 -> identical work]")
    print("nhead\tqlen\tB\tasm_us\topus_us\tsplits\topus/asm" + ("\tmax|d|" if check else ""))
    # Shapes mirror aiter's official decode optest + all other bf16 qh that asm ships
    # (a16w16 kernels: qh 8/16/32/64/128). qlen per asm kernel availability:
    #   qh8 -> qseqlen1; qh16/qh32 -> qseqlen{1,2,4}; qh64 -> qseqlen1; qh128 -> {1,2}.
    # B in {1,3,5,16,32,64,128,256}; ctx extended to 64K via the CLI arg.
    for nhead, qlen in ((128, 1), (128, 2), (64, 1),
                        (32, 1), (32, 2), (32, 4),
                        (16, 1), (16, 2), (16, 4), (8, 1)):
            for B in (1, 3, 5, 16, 32, 64, 128, 256):
                q, kv = make_inputs(B, nhead, ctx, qlen)
                try:
                    a, a_out, _ = run_asm(q, kv, B, nhead, ctx, qlen)
                except Exception as e:
                    a = float("nan"); a_out = None
                    print(f"  asm fail nhead={nhead} qlen={qlen} B={B}: {e}", file=sys.stderr)
                try:
                    o, ns, o_out, _ = run_opus(q, kv, B, nhead, ctx, qlen)
                except Exception as e:
                    o, ns, o_out = float("nan"), 0, None
                    print(f"  opus fail: {e}", file=sys.stderr)
                rel = (o / a) if (a == a and o == o and a > 0) else float("nan")
                extra = ""
                if check and a_out is not None and o_out is not None:
                    d = (o_out.float().reshape(B * qlen, nhead, VHD) - a_out.float()).abs().max().item()
                    extra = f"\t{d:.4g}"
                print(f"{nhead}\t{qlen}\t{B}\t{a:.1f}\t{o:.1f}\t{ns}\t{rel:.2f}{extra}")
