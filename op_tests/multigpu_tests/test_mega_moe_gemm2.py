# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MegaMoE Stage2 (GEMM2 + P2P scatter + combine) checked in isolation from GEMM1.

The reference consumes the kernel's own Stage1 intermediate (FP8 rows plus their
preshuffled 1x32 E8M0 scales), so GEMM1 numerics drop out and the tolerance can
be tight. Sorted row m uses expert sorted_expert_ids[m // SBM] and the route
(source key | slot << 24, weight) at tile_row_base[m // SBM] + m % SBM, the same
mapping mega_moe_stage2 uses. Run with torchrun (2 ranks keeps V4.1's 384
experts within the 256 experts-per-rank limit).
"""

import argparse
import os
import sys

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "16G")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_mega_moe_v2 as t  # noqa: E402
from aiter import dtypes  # noqa: E402
from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2  # noqa: E402
from aiter.utility import fp4_utils  # noqa: E402


def fp8_dtype():
    arch = torch.cuda.get_device_properties(0).gcnArchName
    return torch.float8_e4m3fnuz if "gfx94" in arch else torch.float8_e4m3fn


def stage2_reference(moe, w2_q, w2_scale, bs):
    op = moe._s1_op
    sbm = int(moe._s1_active_tile_m)
    num_valid = int(op.num_valid[0].item())
    inter, d = moe.inter_dim, moe.model_dim
    m = torch.arange(num_valid, device=moe.dev)
    tile = m // sbm
    route_row = op.tile_row_base[tile].long() + m % sbm
    packed = op.srcmap_em.view(torch.int32)[route_row].long()
    weight = op.wts_em.view(torch.float32)[route_row]
    expert = op.sorted_expert_ids[tile].long() - moe.rank * moe.epr
    key = packed & 0xFFFFFF
    valid = (key < moe.world_size * moe.mtpr) & (expert >= 0) & (expert < moe.epr)
    scale_cols = (inter // 32 + 7) // 8 * 8
    r = m[:, None]
    c = torch.arange(inter // 32, device=moe.dev)[None, :]
    off = ((r >> 5) * scale_cols * 32 + (c >> 3) * 256 + (c & 3) * 64 + (r & 15) * 4
           + ((c >> 2) & 1) * 2 + ((r >> 4) & 1))
    a_scale = fp4_utils.e8m0_to_f32(moe._s1_osd[off].view(dtypes.fp8_e8m0))
    a = moe._s1_out.view(torch.uint8)[:num_valid].view(fp8_dtype()).float()
    a = a * a_scale.repeat_interleave(32, dim=-1)
    out = torch.zeros(moe.world_size * moe.mtpr, d, dtype=torch.float32, device=moe.dev)
    for e in torch.unique(expert[valid]).tolist():
        sel = valid & (expert == e)
        w2 = t._dequant_expert(w2_q[e], w2_scale[e], d, inter)
        out.index_add_(0, key[sel], (a[sel] @ w2.T) * weight[sel, None])
    dist.all_reduce(out)
    return out[moe.rank * moe.mtpr: moe.rank * moe.mtpr + bs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--network", default="v41_flash")
    p.add_argument("--bs-list", default="128,512")
    p.add_argument("--max-tok-per-rank", type=int, default=8192)
    p.add_argument("--rtol", type=float, default=0.01)
    args = p.parse_args()
    rank, world, device = t._setup_dist()
    failed = 0
    try:
        net = t.NETWORKS[args.network]
        epr = net["experts"] // world
        w1, w1s, w2, w2s, _, _, w2_q, w2_ref_scale = t._quantize_weights(
            net["model_dim"], net["inter_dim"], epr, rank, 123, device)
        bss = [int(v) for v in args.bs_list.split(",")]
        x_all, wts_all, ids_all = t._make_inputs(
            max(bss), net["model_dim"], net["experts"], net["topk"], rank, 123, device,
            force_fanout_boundary=False, inject_invalid_route=False, force_padding_boundary=False)
        moe = MegaMoEV2(rank=rank, world_size=world, quant="a8w4", w1=w1, w1_scale=w1s, w2=w2,
                        w2_scale=w2s, max_tok_per_rank=args.max_tok_per_rank, **net)
        for bs in bss:
            x, wts, ids = (v[:bs].contiguous() for v in (x_all, wts_all, ids_all))
            out = moe(x, wts, ids)[:bs].float()
            torch.cuda.synchronize()
            t._barrier()
            ref = stage2_reference(moe, w2_q, w2_ref_scale, bs)
            rel = t._rel_l2(out, ref)
            ok = 0 <= rel < args.rtol
            failed += not ok
            if rank == 0:
                print(f"[GEMM2] {'PASS' if ok else 'FAIL'} bs={bs} relL2={rel:.6f} "
                      f"SBM={moe._s1_active_tile_m} G2_BM={moe._g2_active_block_m}", flush=True)
        # Negative control: Stage2 alone on the same intermediate with W2 scales
        # rolled by one K group must miss the reference.
        saved = moe.w2_scale
        moe.w2_scale = torch.roll(saved.view(torch.uint8), 1, dims=-1).contiguous().view(saved.dtype)
        out = moe._run_stage2(bs, None, True, moe._active_config).float()
        torch.cuda.synchronize()
        t._barrier()
        moe.w2_scale = saved
        rel = t._rel_l2(out, ref)
        neg_ok = rel > 10 * args.rtol
        failed += not neg_ok
        if rank == 0:
            print(f"[GEMM2] negative control (W2 scales rolled): relL2={rel:.6f} "
                  f"{'detected (PASS)' if neg_ok else 'NOT detected (FAIL)'}", flush=True)
    finally:
        t._cleanup()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
