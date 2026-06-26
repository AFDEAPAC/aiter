# SPDX-License-Identifier: MIT
# Lean perf sweep of qh64/qlen1 bf16 MLA decode (HK when AITER_ENABLE_EXPERIMENTAL=1, asm
# when =0) over a ctx x B grid. Builds the SAME persistent metadata the test uses (so HK is
# actually dispatched) but NO fp32 reference / checkAllclose (avoids OOM at large ctx*B).
# Timing only via cuda events + min-of-repeats. Prints CSV "B,ctx,us". Run twice (exp 1/0).
import math, os, torch
import aiter
import aiter.mla as amla
from aiter import dtypes

dev = "cuda"
KV_LORA, ROPE, VHD = 512, 64, 512
QK = KV_LORA + ROPE  # 576
NHEAD, NHEAD_KV, QLEN = 64, 1, 1
PAGE = 1
MAX_SPLIT = 32

CTXS = [1200, 3200, 5200, 8000, 16384, 32768, 65536]
BS = [1, 16, 32, 64, 128, 256]


def best(fn, iters=50, repeats=5, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    def w():
        s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
        for _ in range(iters):
            fn()
        e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / iters * 1e3
    return min(w() for _ in range(repeats))


def run_cell(B, ctx):
    dtype = torch.bfloat16
    kvtype = torch.bfloat16
    num_page = B * ctx  # page_size==1, non-varlen
    qo_indptr = torch.arange(0, (B + 1) * QLEN, QLEN, dtype=torch.int32, device=dev)
    kv_indptr = torch.arange(0, (B + 1) * ctx, ctx, dtype=torch.int32, device=dev)
    kv_indices = torch.randperm(num_page, dtype=torch.int32, device=dev)
    kv_last_page_lens = torch.ones(B, dtype=torch.int32, device=dev)
    q = (torch.randn((B * QLEN, NHEAD, QK), dtype=torch.float32, device=dev) * 0.5).to(dtype)
    kv_buffer = torch.empty((num_page, PAGE, 1, QK), dtype=kvtype, device=dev).normal_(0.0, 0.5)
    out = torch.empty((B * QLEN, NHEAD, VHD), dtype=torch.bfloat16, device=dev)
    sm_scale = 1.0 / math.sqrt(QK)
    max_seqlen_qo = QLEN

    (
        (wmd_sz, wmd_t), (wi_sz, wi_t), (wis_sz, wis_t),
        (ri_sz, ri_t), (rf_sz, rf_t), (rp_sz, rp_t),
    ) = aiter.get_mla_metadata_info_v1(
        B, max_seqlen_qo, NHEAD, dtype, kvtype,
        is_sparse=False, fast_mode=True, num_kv_splits=MAX_SPLIT, intra_batch_mode=False,
    )
    work_meta_data = torch.empty(wmd_sz, dtype=wmd_t, device=dev)
    work_indptr = torch.empty(wi_sz, dtype=wi_t, device=dev)
    work_info_set = torch.empty(wis_sz, dtype=wis_t, device=dev)
    reduce_indptr = torch.empty(ri_sz, dtype=ri_t, device=dev)
    reduce_final_map = torch.empty(rf_sz, dtype=rf_t, device=dev)
    reduce_partial_map = torch.empty(rp_sz, dtype=rp_t, device=dev)
    aiter.get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_lens, NHEAD // NHEAD_KV, NHEAD_KV, False,
        work_meta_data, work_info_set, work_indptr, reduce_indptr, reduce_final_map,
        reduce_partial_map, page_size=PAGE, kv_granularity=max(PAGE, 16),
        max_seqlen_qo=int(max_seqlen_qo), uni_seqlen_qo=QLEN, fast_mode=True,
        max_split_per_batch=MAX_SPLIT, intra_batch_mode=False, dtype_q=dtype, dtype_kv=kvtype,
    )

    def call():
        amla.mla_decode_fwd(
            q, kv_buffer.view(num_page, PAGE, NHEAD_KV, QK), out, qo_indptr, kv_indptr,
            kv_indices, kv_last_page_lens, max_seqlen_qo, PAGE, NHEAD_KV, sm_scale,
            num_kv_splits=MAX_SPLIT, work_meta_data=work_meta_data, work_indptr=work_indptr,
            work_info_set=work_info_set, reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map, reduce_partial_map=reduce_partial_map,
            intra_batch_mode=False, return_lse=False,
        )

    t = best(call)
    del q, kv_buffer, out, kv_indices, kv_indptr, qo_indptr, kv_last_page_lens
    del work_meta_data, work_indptr, work_info_set, reduce_indptr, reduce_final_map, reduce_partial_map
    torch.cuda.empty_cache()
    return t


if __name__ == "__main__":
    tag = "HK" if os.environ.get("AITER_ENABLE_EXPERIMENTAL") == "1" else "asm"
    print(f"# mode={tag}")
    print("B,ctx,us")
    for ctx in CTXS:
        for B in BS:
            try:
                print(f"{B},{ctx},{run_cell(B, ctx):.2f}", flush=True)
            except Exception as ex:
                print(f"{B},{ctx},ERR:{type(ex).__name__}", flush=True)
                torch.cuda.empty_cache()
