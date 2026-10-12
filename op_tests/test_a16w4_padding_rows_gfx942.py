#!/usr/bin/env python3
"""Rows past ``num_valid`` in ``inter_sorted`` must not be able to reach the output.

``fused_moe_a16wmix`` allocates the intermediate with ``torch.empty`` and bounds the
stage2 quantization with ``num_rows=num_valid_ids``, so the tail of that buffer is never
written and never read. This asserts that, at production dimensions, where the tail is
most of the buffer: at E=896 with one token, 14320 of 14336 sorted rows are padding.

Run directly (needs a gfx942/gfx950 GPU and the FlyDSL kernels):

    AITER_A16WMIX_FP8=1 AITER_A16WMIX_FP8_S2=1 AITER_A16W4_ALLOW_EP=1 \
        python3 op_tests/test_a16w4_padding_rows_gfx942.py

Three things about the method, each of which produced a wrong verdict first:

1. ``bughunt_fp8_moe.py`` and ``sweep_fp8_shapes.py`` cannot substitute for this. They
   call ``flydsl_a16w4_gemm1/gemm2`` directly with their own intermediate buffer and
   never enter the wrapper, so they cannot see this behaviour at all.
2. Bit-identity is not a usable gate. gemm2's epilogue accumulates with atomic-fadd, so
   two identical clean calls already differ by ~9.8e-4 here, one bf16 ULP. Every
   tolerance below is measured from the kernel itself rather than assumed.
3. Poisoning the caching allocator and hoping ``torch.empty`` picks the block up is
   vacuous: measured, a same-shape allocation straight after poisoning was 100% NaN
   while the wrapper's own buffer came back 0% NaN, because moe_sorting and the quant
   buffers are allocated in between. The poison is therefore written into the exact
   allocation, and ``injected`` asserts it happened.

The companion two-sided control lives in ``padding_rows_two_sided_control`` below: the
same NaN over the LIVE rows must move the output by orders more, otherwise this test is
merely insensitive and proves nothing.
"""
from __future__ import annotations

import torch

from aiter import dtypes
from aiter.ops.quant import per_1x32_f4_quant
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.utility import fp4_utils
import aiter.ops.flydsl.a16wmix_fused_moe as a16wmix
import aiter.ops.quant as quant_mod

DEV = "cuda"
E, MODEL_DIM, INTER_DIM, TOPK = 896, 3584, 384, 16

_real_empty = torch.empty


def _quant_chunked(rows, cols, scale, device, chunk=32768, seed=0):
    """FP4-quantize a [rows, cols] weight without ever holding it in f32.

    At these dims w1 is 896x768x3584, ~9.85 GB in f32 and the quantizer copies it.
    """
    qs, ss = [], []
    g = torch.Generator(device=device).manual_seed(seed)
    for base in range(0, rows, chunk):
        n = min(chunk, rows - base)
        blk = torch.randn((n, cols), device=device, dtype=torch.float32, generator=g) * scale
        q, s = per_1x32_f4_quant(blk)
        qs.append(q)
        ss.append(s)
        del blk, q, s
    return torch.cat(qs, 0), torch.cat(ss, 0)


def _build(tokens, seed=0):
    """Weights in the layout vLLM hands the kernel (preshuffled at load time)."""
    torch.manual_seed(seed)
    n_out, scale = 2 * INTER_DIM, 0.2
    x = torch.randn((tokens, MODEL_DIM), device=DEV, dtype=torch.float32) * scale
    w1_q, w1_s = _quant_chunked(E * n_out, MODEL_DIM, scale, DEV, seed=seed + 1)
    w2_q, w2_s = _quant_chunked(E * MODEL_DIM, INTER_DIM, scale / INTER_DIM**0.5, DEV, seed=seed + 2)
    score = torch.rand((tokens, E), device=DEV, dtype=torch.float32)
    tv, tid = torch.topk(score, k=TOPK, dim=1)
    fp4 = dtypes.fp4x2
    return dict(
        x=x.to(torch.bfloat16),
        topk_ids=tid.to(torch.int32),
        topk_weights=torch.softmax(tv, dim=1).float(),
        w1=shuffle_weight_a16w4(w1_q.view(fp4).reshape(E, n_out, -1), 16, False).view(torch.uint8),
        w1s=shuffle_scale_a16w4(w1_s.view(E * n_out, MODEL_DIM // 32), E, False).view(torch.uint8),
        w2=shuffle_weight_a16w4(w2_q.view(fp4).reshape(E, MODEL_DIM, -1), 16, False).view(torch.uint8),
        w2s=fp4_utils.e8m0_shuffle(w2_s.view(E * MODEL_DIM, INTER_DIM // 32)).view(torch.uint8),
    )


def _call(d, tile_m, poison_value=None):
    """Run the wrapper, optionally filling inter_sorted with poison at allocation."""
    hits = {"n": 0}

    def spy(*a, **k):
        t = _real_empty(*a, **k)
        if len(a) == 2 and a[1] == INTER_DIM and k.get("dtype") == torch.bfloat16:
            t.fill_(poison_value)
            hits["n"] += 1
        return t

    if poison_value is not None:
        torch.empty = spy
    try:
        out = a16wmix.fused_moe_a16wmix(
            d["x"], d["w1"], d["w2"], d["topk_weights"], d["topk_ids"],
            w1_scale=d["w1s"], w2_scale=d["w2s"],
            act="situv2", block_m=tile_m, tile_m=tile_m, w1_layout="standard",
        ).clone()
    finally:
        torch.empty = _real_empty
    return out, hits["n"]


def test_padding_rows_cannot_reach_the_output():
    bad = []
    for tokens in (1, 2, 16, 129, 1024):
        d = _build(tokens)
        for tile_m in (16, 32):
            torch.cuda.empty_cache()
            clean = [_call(d, tile_m)[0] for _ in range(3)]
            a = clean[0]
            floor = max((a.float() - c.float()).abs().max().item() for c in clean[1:])
            cmags = [c.abs().max().item() for c in clean]
            mag_lo, mag_hi = min(cmags), max(cmags)

            diffs, mags, injected = {}, {}, {}
            for name, val in (("NaN", float("nan")), ("1e30", 1e30)):
                p, n = _call(d, tile_m, poison_value=val)
                injected[name] = n
                diffs[name] = (a.float() - p.float()).abs().max().item()
                mags[name] = p.abs().max().item()

            worst = max(diffs.values())
            # An order-of-magnitude band, not a tight one: max|out| takes a handful of
            # discrete bf16 values, so a band drawn from three samples is flaky. What this
            # must catch is poison REACHING the output -- NaN or 1e30 against a clean ~0.09.
            ok = (
                all(n >= 1 for n in injected.values())
                and worst <= max(2.0 * floor, 2e-3)
                and torch.isfinite(a).all().item()
                and all(0.5 * mag_lo <= m <= 2.0 * mag_hi for m in mags.values())
            )
            print(
                f"  tokens={tokens:5d} tile_m={tile_m:2d} injected={injected['NaN']} "
                f"floor={floor:.3e} poison_worst={worst:.3e} {'PASS' if ok else 'FAIL'}"
            )
            if not ok:
                bad.append((tokens, tile_m))
    assert not bad, f"padding rows reached the output for {bad}"


def padding_rows_two_sided_control():
    """The above passing only means something if corrupting LIVE rows does move the output.

    Hook the quantization call rather than an allocation: at that point inter_sorted
    already holds gemm1's output and num_rows carries num_valid, so both ranges are
    addressable. Patch it on aiter.ops.quant -- the wrapper imports the name inside the
    function body, so the wrapper module has no such attribute.

    The wrapper drives ``dynamic_per_token_scaled_quant(out, input, scales, ...)``
    directly, so that is what is hooked; ``input`` is already the live-prefix view, and
    the padding range inside it is [num_valid, view_rows). Rows past the view are never
    quantized at all and are covered by the allocation-poison test above instead.
    """
    real_q = quant_mod.dynamic_per_token_scaled_quant

    def run(d, corrupt):
        info = {"nv": None, "rows": 0}

        def patched(out, x, scales, *a, **k):
            nr = k.get("num_rows")
            if corrupt and x.dim() == 2 and x.shape[1] == INTER_DIM and nr is not None:
                nv = int(nr.flatten()[0].item())
                info["nv"] = nv
                sl = slice(nv, x.shape[0]) if corrupt == "pad" else slice(0, nv)
                info["rows"] = max(sl.stop - sl.start, 0)
                if info["rows"]:
                    x[sl].fill_(float("nan"))
            return real_q(out, x, scales, *a, **k)

        quant_mod.dynamic_per_token_scaled_quant = patched
        try:
            out = a16wmix.fused_moe_a16wmix(
                d["x"], d["w1"], d["w2"], d["topk_weights"], d["topk_ids"],
                w1_scale=d["w1s"], w2_scale=d["w2s"],
                act="situv2", block_m=16, tile_m=16, w1_layout="standard",
            ).clone()
        finally:
            quant_mod.dynamic_per_token_scaled_quant = real_q
        return out, info

    for tokens in (1, 16, 1024):
        d = _build(tokens)
        a, _ = run(d, None)
        b, _ = run(d, None)
        floor = (a.float() - b.float()).abs().max().item()
        pa, ia = run(d, "pad")
        pl, il = run(d, "live")
        if ia["nv"] is None:
            # The hook is on the stage2 quantization, which only runs with
            # AITER_A16WMIX_FP8_S2=1. On the bf16 and stage1-only arms there is nothing to
            # hook, so skip rather than crash on a None -- the allocation-poison test below
            # still covers those arms, and it is the one that exercises torch.empty.
            print("  stage2 quant did not run (arm has FP8_S2 off) -- control not applicable")
            return
        da = (a.float() - pa.float()).abs().max().item()
        dl = (a.float() - pl.float()).abs().max().item()
        print(
            f"  tokens={tokens:5d} num_valid={ia['nv']:6d}  "
            f"pad {ia['rows']:6d} rows -> {da / floor:5.1f}x floor   "
            f"live {il['rows']:6d} rows -> {dl / floor:5.1f}x floor"
        )
        assert dl > 10 * floor, "corrupting live rows did not move the output; test is insensitive"
        assert da <= max(2 * floor, 2e-3), "corrupting padding rows moved the output"


if __name__ == "__main__":
    print(f"real dims: E={E} model_dim={MODEL_DIM} inter_dim={INTER_DIM} topk={TOPK}")
    print("two-sided control (live rows must move it, padding rows must not):")
    padding_rows_two_sided_control()
    print("\npoison injected into inter_sorted at allocation:")
    test_padding_rows_cannot_reach_the_output()
    print("\nALL PASS")
