# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Fused a16w4/a16wi4/a16w16 (bf16 A x mxfp4/int4/bf16 W) 2-stage MoE kernels.

CDNA MFMA pipeline. bf16 A (no A-scale), W1/W2 upconverted to bf16 in-kernel,
non-scaled ``MFMA(16,16,32,bf16)``:

  - stage1 (:mod:`gemm1`): fused gate+up GEMM + SiLU/SiTUv2 -> bf16 intermediate
    ``[sorted_size, inter_dim]`` by sorted position (no requant/scale).
  - stage2 (:mod:`gemm2`): down-proj GEMM + routing-weighted atomic bf16 scatter
    to ``[tokens, model_dim]``.

Reuses the standard sorting/cumsum/m_indices contract and the
shuffle_weight+e8m0_shuffle W layout. Shared low-level helpers live in
:mod:`gemm1` (imported by :mod:`gemm2`); host-side launch glue is defined below.

Launch args are raw device pointers (``fx.Int64``); tensors passed as
``.data_ptr()``.
"""

import csv
import functools
import os
import re

import torch

from .kernels.tensor_shim import _run_compiled
from .kernels.moe_2stage_a16wmix.gemm1 import compile_gemm1_a16w4_port, gemm1_a16w4_grid
from .kernels.moe_2stage_a16wmix.gemm2 import (
    LDS_LIMIT_BYTES as _LDS_LIMIT,
    compile_gemm2_a16w4_port,
    gemm2_a16w4_grid,
)
from .kernels.moe_2stage_a16wmix.utils import lds_acc_bytes_for

__all__ = [
    "compile_gemm1_a16w4_port",
    "gemm1_a16w4_grid",
    "compile_gemm2_a16w4_port",
    "gemm2_a16w4_grid",
    "flydsl_a16w4_gemm1",
    "flydsl_a16w4_gemm2",
    "a16wi4_scale_to_kernel_layout",
    "a16wi4_recommend_block_m",
    "pick_a16w4_config",
    "resolve_a16w4_gemm1_config",
    "resolve_a16w4_gemm2_config",
    "resolve_a16wmix_gemm1_config",
    "resolve_a16wmix_gemm2_config",
]


@functools.cache
def _get_compiled_gemm1_a16w4(
    BM,
    D_HIDDEN,
    D_INTER,
    NE,
    topk,
    TILE_N,
    TILE_K,
    act,
    b_cache_mod,
    xcd_swizzle,
    waves_per_eu,
    w_dtype="mxfp4",
    w_layout="standard",
    a_dtype="bf16",
    k_wave=1,
):
    return compile_gemm1_a16w4_port(
        BM=BM,
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        NE=NE,
        TOPK=topk,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        act=act,
        b_cache_mod=b_cache_mod,
        xcd_swizzle=xcd_swizzle,
        waves_per_eu=waves_per_eu,
        w_dtype=w_dtype,
        w_layout=w_layout,
        a_dtype=a_dtype,
        k_wave=k_wave,
    )


@functools.cache
def _get_compiled_gemm2_a16w4(
    BM,
    NE,
    N_OUT,
    D_INTER,
    TILE_N,
    TILE_K,
    b_cache_mod=2,
    xcd_swizzle=1,
    waves_per_eu=None,
    w_dtype="mxfp4",
    a_dtype="bf16",
    persist=False,
):
    return compile_gemm2_a16w4_port(
        BM=BM,
        NE=NE,
        N_OUT=N_OUT,
        D_INTER=D_INTER,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        b_cache_mod=b_cache_mod,
        xcd_swizzle=xcd_swizzle,
        waves_per_eu=waves_per_eu,
        w_dtype=w_dtype,
        a_dtype=a_dtype,
        persist=persist,
    )


def _default_tile_n(N, *, w_dtype="mxfp4"):
    """Adaptive N-tile default. mxfp4/bf16: 256 when N % 256 == 0 else 128 (aiter fat
    tile). int4 (bandwidth/grid-fill-bound): prefers 128, falling to 64 when N % 128 != 0.

    Part of the built-in basic/core configs (used unless an override CSV is set).
    """
    if w_dtype == "int4":
        if N % 128 == 0:
            return 128
        return 64 if N % 64 == 0 else 128
    return 256 if N % 256 == 0 else 128


# =============================================================================
# Per-(shape, token) tile config. By default the built-in basic/core configs
# (``_default_tiles_fallback``) are used -- a small, hand-maintained heuristic
# that runs any shape. An OPTIONAL tuned-tile CSV may override it per cell:
# set ``FLYDSL_A16WMIX_TUNED_CSV=/path/to.csv`` (columns: w_dtype, model_dim,
# inter_dim, experts, topk, token, stage, tile_m, tile_n, tile_k, k_wave,
# xcd_swizzle, b_nt). When the env is unset or the file is missing, the resolver
# returns None for every cell and the basic configs are used.
# =============================================================================

_A16WMIX_CSV_ENV = "FLYDSL_A16WMIX_TUNED_CSV"


def _ours_tuned_csv_path():
    """Optional tuned-tile CSV path from ``FLYDSL_A16WMIX_TUNED_CSV``, or None when
    unset / not a file (-> basic-config fallback)."""
    p = os.environ.get(_A16WMIX_CSV_ENV)
    return p if (p and os.path.isfile(p)) else None


@functools.cache
def _load_ours_tuned_csv(csv_path):
    """Parse a tuned-tile CSV into
    {(w_dtype, model_dim, inter_dim, experts, topk, token, stage): cfg}."""
    table = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = (
                    row["w_dtype"],
                    int(row["model_dim"]),
                    int(row["inter_dim"]),
                    int(row["experts"]),
                    int(row["topk"]),
                    int(row["token"]),
                    int(row["stage"]),
                )
                cfg = {
                    "tile_m": int(row["tile_m"]),
                    "tile_n": int(row["tile_n"]),
                    "tile_k": int(row["tile_k"]),
                    "k_wave": int(row["k_wave"]),
                    "xcd_swizzle": int(row["xcd_swizzle"]),
                    "b_nt": int(row["b_nt"]),
                }
            except (KeyError, ValueError):
                continue
            table[key] = cfg
    return table


def _resolve_ours_tuned(*, w_dtype, model_dim, inter_dim, experts, topk, tokens, stage, csv_path=None):
    """Return the tuned tile-config dict for one (shape, token, stage) from the optional
    override CSV, or None when no CSV is configured / no exact row exists. None -> the
    caller uses the built-in basic configs (_default_tiles_fallback)."""
    path = csv_path or _ours_tuned_csv_path()
    if not path or not os.path.isfile(path):
        return None
    table = _load_ours_tuned_csv(path)
    tok = int(tokens)
    hit = table.get((w_dtype, model_dim, inter_dim, experts, topk, tok, stage))
    if hit is not None:
        return hit
    # Serving sees far more token counts than anyone tunes, so fall back to the nearest
    # tuned row rather than dropping to the built-in default: the largest tuned token at
    # or below the request, else the smallest tuned token.
    cands = [
        (t, cfg)
        for (wd, md, inter, e, tk, t, st), cfg in table.items()
        if (wd, md, inter, e, tk, st) == (w_dtype, model_dim, inter_dim, experts, topk, stage)
    ]
    if not cands:
        return None
    cands.sort()
    below = [c for c in cands if c[0] <= tok]
    return (below[-1] if below else cands[0])[1]


def _default_tiles_fallback(*, D_HIDDEN, D_INTER, tokens, w_dtype, tile_m, stage):
    """Built-in basic/core tile config -- the maintained default used whenever no
    override CSV (FLYDSL_A16WMIX_TUNED_CSV) supplies a row for this (shape, token).

    A small per-token heuristic that runs any shape, consolidated in one place (not
    scattered inline in the launchers). Returns the same dict shape as the CSV resolver.
    Assumes the caller left all tile args at their defaults (explicit caller overrides
    are applied afterwards, upstream).
    """
    BM = int(tile_m)
    _m = int(tokens)
    if stage == 1:
        TILE_K = 256
        k_wave = 1
        xcd_swizzle = 0
        b_nt = 2 if (16 <= _m <= 1024) else 0
        # mxfp4 high-token: shorter K-tiles + XCD remap from tok>=16 (K % 128 == 0).
        if w_dtype == "mxfp4" and _m >= 16 and D_HIDDEN % 128 == 0:
            TILE_K = 128
            xcd_swizzle = 1
        # TILE_N resolution (mirrors the old tile_n=None branch order).
        if w_dtype == "mxfp4" and _m <= 2 and D_INTER % 64 == 0:
            TILE_N = 64
            # tok<=2 4-way slice-K (needs K % (4*128) == 0).
            if D_HIDDEN % 512 == 0:
                TILE_K = 128
                k_wave = 4
        elif w_dtype == "int4" and BM == 64 and D_INTER % 64 == 0:
            TILE_N = 64
        elif w_dtype == "mxfp4" and D_INTER % 128 == 0:
            TILE_N = 128
        else:
            TILE_N = _default_tile_n(D_INTER, w_dtype=w_dtype)
        return {
            "tile_m": BM,
            "tile_n": TILE_N,
            "tile_k": TILE_K,
            "k_wave": k_wave,
            "xcd_swizzle": xcd_swizzle,
            "b_nt": b_nt,
        }
    # stage 2 (down-proj): fixed 4-wave N-split; tile_n=256/tile_k=256/xcd=1 defaults.
    b_nt = 0 if (_m <= 16 or _m >= 2048) else 2
    return {
        "tile_m": BM,
        "tile_n": 256,
        "tile_k": 256,
        "k_wave": 1,
        "xcd_swizzle": 1,
        "b_nt": b_nt,
    }



def _fp8_tile_override(cfg, *, stage, tile_m):
    """Tiles chosen while A was bf16 are not the fp8 optimum.

    a16wmix_tuned.csv predates the fp8 path and has no a_dtype column, so the fp8 arm has
    always inherited bf16's tiles. Swept against the fp8 kernels at production dims
    (3584/384/896, topk 16), the two that move are:

      stage 2  tile_n 256 -> 128, xcd 1 -> 0.  tile_n=128 won at every token count tried
               (1, 2, 8, 16, 32, 8192) and the gap to 256 is a cliff, not a slope: at
               prefill the four best configs are all tile_n=128 (3523-3792 us) and the
               best 256 is 5660 us. Gains where the CSV picks 256 -- which is the decode
               band -- are 1.13x / 1.17x / 1.11x / 1.06x at token 1 / 2 / 8 / 16. Token
               counts whose CSV row already says 128 are unaffected.
      stage 1  tile_k 128 -> 256 at BM<=16, worth 1.06-1.08x at the decode shape. Prefill
               (BM=32) measured no gain, so it keeps 128.

    bf16 is untouched: this runs only when a_dtype == "fp8".
    """
    cfg = dict(cfg)
    if stage == 2:
        cfg["tile_n"] = 128
        cfg["xcd_swizzle"] = 0
    elif int(tile_m) <= 16:
        cfg["tile_k"] = 256
        # fp8 bumps K-tile 128->256; k_wave=4 slice-K needs K % (4*128)==0 and
        # breaks once TILE_K becomes 256 (3584 % 1024 != 0).
        if int(cfg.get("k_wave", 1)) > 1:
            cfg["k_wave"] = 1
    return cfg


def resolve_a16wmix_gemm1_config(*, w_dtype, model_dim, inter_dim, experts, topk, tokens, tile_m, csv_path=None, a_dtype="bf16"):
    """Resolve the ours-tuned gemm1 tile-config: exact CSV row if present, else the
    slim documented fallback. Never returns None (arbitrary shapes still run)."""
    cfg = _resolve_ours_tuned(
        w_dtype=w_dtype,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tokens=tokens,
        stage=1,
        csv_path=csv_path,
    )
    if cfg is None:
        cfg = _default_tiles_fallback(
            D_HIDDEN=model_dim, D_INTER=inter_dim, tokens=tokens, w_dtype=w_dtype, tile_m=tile_m, stage=1
        )
        return _fp8_tile_override(cfg, stage=1, tile_m=tile_m) if a_dtype == "fp8" else cfg
    # int4 gemm1 tile_n is tile_m-dependent (the BM==64 W1-reuse occupancy gate), while
    # the CSV row assumes the recommended tile_m. Re-derive tile_n from the ACTUAL BM so
    # an explicit caller tile_m stays correct; the CSV row still owns the other fields.
    cfg = dict(cfg)
    if w_dtype == "int4" and inter_dim % 64 == 0:
        cfg["tile_n"] = 64 if int(tile_m) == 64 else _default_tile_n(inter_dim, w_dtype=w_dtype)
    cfg["tile_m"] = int(tile_m)
    return _fp8_tile_override(cfg, stage=1, tile_m=tile_m) if a_dtype == "fp8" else cfg


def resolve_a16wmix_gemm2_config(*, w_dtype, model_dim, inter_dim, experts, topk, tokens, tile_m, csv_path=None, a_dtype="bf16"):
    """Resolve the ours-tuned gemm2 tile-config: exact CSV row if present, else the
    slim documented fallback. Never returns None."""
    cfg = _resolve_ours_tuned(
        w_dtype=w_dtype,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tokens=tokens,
        stage=2,
        csv_path=csv_path,
    )
    if cfg is None:
        cfg = _default_tiles_fallback(
            D_HIDDEN=model_dim, D_INTER=inter_dim, tokens=tokens, w_dtype=w_dtype, tile_m=tile_m, stage=2
        )
    return _fp8_tile_override(cfg, stage=2, tile_m=tile_m) if a_dtype == "fp8" else cfg


def a16wi4_recommend_block_m(tokens, experts, topk, *, base_block_m=32):
    """Recommend the routing/gemm1 block_m (tile_m) for the a16wi4 (int4-W) stage1.

    int4 gemm1 is W-load bound (W1 re-fetched per padded m-block). At the fill point
    where each expert has exactly two half-full 32-row m-blocks, block_m=64 collapses
    them into one full 64-row block, halving W1 HBM re-reads (~2.27x fewer misses, ~7%
    stage1). Outside that band 64 wastes padding or halves grid parallelism.

    Decides on ceil(tokens*topk/experts / base_block_m) (avg padded m-blocks/expert ==
    aiter's estimated_m_per_expert): return 2*base_block_m iff that count == 2.
    base_block_m==32 only; other block_m pass through.

    IMPORTANT: block_m sizes the moe_sorting padding, so the caller MUST build the
    routing buffers with the SAME block_m passed to gemm1 (this is a dispatcher-side
    recommendation, not a gemm1-internal override).
    """
    if int(base_block_m) != 32 or int(experts) <= 0:
        return int(base_block_m)
    routes = int(tokens) * int(topk)
    # ceil(avg routes-per-expert / 32) == avg padded m-blocks/expert at block_m=32.
    m_blocks_32 = -(-routes // (int(experts) * 32))
    return 64 if m_blocks_32 == 2 else int(base_block_m)


def a16wi4_scale_to_kernel_layout(scale_ng):
    """Re-layout a logical int4 scale ``[E, N, G]`` into the ``(E, N, G//2, 2)``
    bf16-pair layout the kernel expects (dword = n*(G//2) + group//2, even/odd group ->
    lo/hi bf16). ``G`` must be even; input is already N-major.
    """
    E, N, G = scale_ng.shape
    assert G % 2 == 0, f"num_groups must be even for bf16-pair packing, got {G}"
    s = scale_ng.to(torch.bfloat16).contiguous().view(E, N, G // 2, 2).contiguous()
    return s


def flydsl_a16w4_gemm1(
    *,
    a_bf16,
    w1_u8,
    w1_scale_u8,
    sorted_expert_ids,
    cumsum_tensor,
    m_indices,
    inter_sorted_bf16,
    n_tokens,
    NE,
    D_HIDDEN,
    D_INTER,
    topk,
    tile_m=32,
    tile_n=None,
    tile_k=256,
    waves_per_eu=None,
    k_batch=1,
    k_wave=1,
    b_nt=None,
    xcd_swizzle=0,
    gate_mode="separated",
    act="silu",
    situ_beta=1.0,
    situ_linear_beta=1.0,
    swiglu_limit=float("inf"),
    w_dtype="mxfp4",
    w_layout="standard",
    a_dtype="bf16",
    a_scale=None,
    w1_sref=None,
    use_csv_config=False,  # opt-in: default uses our tuned tile_n; CSV params for aiter-compare / when requested
    csv_path=None,
    stream=None,
):
    """a16w4/a16wi4/a16w16 fused stage1: gate+up GEMM + SiLU -> bf16 intermediate.

    ``a_dtype="fp8"`` switches the MFMA to ``mfma_f32_16x16x32_fp8_fp8``: ``a_bf16``
    then carries e4m3fnuz bytes, ``a_scale`` is the f32 per-token quantization scale,
    and ``w1_sref`` is the per-(expert, output column) E8M0 reference exponent from
    ``a16wmix_fp8_prep``. The weight and its scale tensor are unchanged -- the kernel
    forms the residual itself as ``sref - scale_byte``.

    ``w_dtype="mxfp4"`` (default): W1 mxfp4, ``w1_scale_u8`` = shuffled e8m0. ``"int4"``:
    W1 packed signed int4 (same preshuffle as mxfp4), ``w1_scale_u8`` groupwise bf16 in
    the ``(E, N_OUT, G//2, 2)`` layout (see :func:`a16wi4_scale_to_kernel_layout`).
    ``"bf16"``: RAW bf16 W1 preshuffled ``shuffle_weight (16,16)``; ``w1_scale_u8`` unused.

    ``w_layout="standard"`` (default) consumes the N-major GGUU preshuffle. ``"guinterleave"``
    (mxfp4 only) consumes aiter's native GUGU stage1 W1+scale layout
    (``shuffle_weight_a16w4``/``shuffle_scale_a16w4``) directly, with no host relayout.

    ``a_bf16`` is bf16 ``[n_tokens, D_HIDDEN]``. Writes the bf16 intermediate
    ``[sorted_size, D_INTER]`` (by sorted position) into ``inter_sorted_bf16``.

    Tile config: ``tile_m/n/k`` -> BM/TILE_N/TILE_K, ``waves_per_eu`` ->
    rocdl.waves_per_eu, ``b_nt`` -> W-load cache modifier (0=cached, 2=nt),
    ``xcd_swizzle`` -> XCD/HBM grid remap, ``k_wave`` -> intra-block slice-K ({1,2,4}).
    ``k_batch``/``gate_mode`` accepted for parity (only k_batch=1/separated supported).
    ``tile_n=None`` picks the largest N tile dividing D_INTER. ``b_nt=None`` uses the
    per-M U-shape (nt mid-band, cached at ends).
    """
    if k_batch != 1:
        raise NotImplementedError(f"a16w4 gemm1 only supports k_batch=1, got {k_batch}")
    if gate_mode != "separated":
        raise NotImplementedError(f"a16w4 gemm1 only supports gate_mode='separated', got {gate_mode!r}")

    # CSV-driven per-token config (mxfp4 only, opt-in): aiter's tuned tile geometry.
    # Falls back to adaptive default on no match; explicit caller overrides win.
    if use_csv_config and w_dtype == "mxfp4":
        cfg = resolve_a16w4_gemm1_config(
            model_dim=D_HIDDEN, inter_dim=D_INTER, experts=NE, topk=topk, tokens=int(n_tokens), csv_path=csv_path
        )
        if cfg is not None:
            if tile_n is None:
                tile_n = cfg["tile_n"]
            if tile_k == 256:
                tile_k = cfg["tile_k"]
            if k_wave == 1:
                k_wave = cfg["k_wave"]
            if waves_per_eu is None:
                waves_per_eu = cfg["waves_per_eu"]
            if xcd_swizzle == 0:
                xcd_swizzle = cfg["xcd_swizzle"]
            if b_nt is None:
                b_nt = cfg["b_nt"]

    BM = tile_m
    TILE_K = tile_k
    _m = int(n_tokens)
    TILE_N = tile_n
    # Per-token tile config (basic/core configs, or an override CSV via
    # FLYDSL_A16WMIX_TUNED_CSV). Fills only the tile args the caller left at default;
    # explicit caller overrides always win. The aiter-compare CSV path (use_csv_config,
    # resolved above) takes precedence when enabled.
    # A16WMIX_FORCE_TILES=1 makes the caller's explicit tiles final: the tuner
    # sweeps geometries the resolver has never heard of, and silently rewriting
    # them from a tuned row would make it measure the wrong kernel.
    if not use_csv_config and os.environ.get("A16WMIX_FORCE_TILES", "0") != "1":
        _o = resolve_a16wmix_gemm1_config(
            a_dtype=a_dtype,
            w_dtype=w_dtype,
            model_dim=D_HIDDEN,
            inter_dim=D_INTER,
            experts=NE,
            topk=topk,
            tokens=_m,
            tile_m=BM,
        )
        if b_nt is None:
            b_nt = _o["b_nt"]
        if xcd_swizzle == 0:
            xcd_swizzle = _o["xcd_swizzle"]
        # The slice-K config (k_wave > 1: the tok<=2 4-way split) is coupled to the
        # tile_n=64 branch and to tile_k/k_wave being at defaults -- apply it as one
        # unit only when the caller left tile_n unset. The k_wave==1 tile_k bump
        # (tok>=16 shorter K-tiles) is independent of tile_n.
        if _o["k_wave"] > 1:
            if TILE_N is None and tile_k == 256 and k_wave == 1:
                TILE_K = _o["tile_k"]
                k_wave = _o["k_wave"]
        elif tile_k == 256:
            TILE_K = _o["tile_k"]
        if TILE_N is None:
            TILE_N = _o["tile_n"]
    b_cache_mod = (2 if (16 <= _m <= 1024) else 0) if b_nt is None else b_nt
    if TILE_N is None:
        TILE_N = _default_tile_n(D_INTER, w_dtype=w_dtype)
    if D_HIDDEN % TILE_K != 0:
        raise NotImplementedError(f"a16w4 gemm1 requires D_HIDDEN (K) % {TILE_K} == 0, got H={D_HIDDEN}")
    if (2 * D_INTER) % 256 != 0:
        raise NotImplementedError(f"a16w4 gemm1 requires 2*D_INTER % 256 == 0, got D_INTER={D_INTER}")
    if D_INTER % TILE_N != 0:
        raise NotImplementedError(f"a16w4 gemm1 requires D_INTER % TILE_N({TILE_N}) == 0, got D_INTER={D_INTER}")

    launch = _get_compiled_gemm1_a16w4(
        BM,
        D_HIDDEN,
        D_INTER,
        NE,
        topk,
        TILE_N,
        TILE_K,
        act,
        b_cache_mod,
        xcd_swizzle,
        waves_per_eu,
        w_dtype,
        w_layout,
        a_dtype,
        k_wave,
    )
    if a_dtype == "fp8":
        if a_scale is None or w1_sref is None:
            raise ValueError("a_dtype='fp8' needs both a_scale and w1_sref")
        _a_scale_ptr = a_scale.data_ptr()
        _sref_ptr = w1_sref.data_ptr()
    else:
        # Unused by the bf16 kernel (const_expr'd out), but the launcher signature is
        # shared, so pass something addressable rather than a null.
        _a_scale_ptr = a_bf16.data_ptr()
        _sref_ptr = a_bf16.data_ptr()
    max_m_blocks = min(int(sorted_expert_ids.numel()), int(m_indices.numel()) // BM)
    grid = gemm1_a16w4_grid(BM, INTER=D_INTER, TILE_N=TILE_N, max_m_blocks=max_m_blocks)
    # SiTUv2 beta/linear_beta + swiglu_limit -> runtime f32 scalars (host precomputes
    # reciprocals; no device rcp). swiglu_limit is the SiTUv2 clamp bound (+inf = no
    # clamp), matching the a8w4/mixed_moe situv2 clamp. Unused by the silu kernel.
    _beta = float(situ_beta)
    _lbeta = float(situ_linear_beta)
    if _beta <= 0.0 or _lbeta <= 0.0:
        raise ValueError(f"situ_beta/situ_linear_beta must be > 0, got {_beta!r}/{_lbeta!r}")
    _run_compiled(
        launch,
        a_bf16.data_ptr(),
        w1_u8.data_ptr(),
        w1_scale_u8.data_ptr(),
        sorted_expert_ids.data_ptr(),
        cumsum_tensor.data_ptr(),
        m_indices.data_ptr(),
        int(n_tokens),
        int(grid),
        _beta,
        1.0 / _beta,
        _lbeta,
        1.0 / _lbeta,
        float(swiglu_limit),
        _a_scale_ptr,
        _sref_ptr,
        inter_sorted_bf16.data_ptr(),
        torch.cuda.current_stream() if stream is None else stream,
    )
    return inter_sorted_bf16


def flydsl_a16w4_gemm2(
    *,
    inter_sorted_bf16,
    w2_u8,
    w2_scale_u8,
    sorted_expert_ids,
    cumsum_tensor,
    sorted_token_ids,
    sorted_weights,
    flat_out,
    M_logical,
    max_sorted,
    NE,
    D_HIDDEN,
    D_INTER,
    topk,
    tile_m=32,
    tile_n=256,
    tile_k=256,
    waves_per_eu=None,
    k_batch=1,
    b_nt=None,
    xcd_swizzle=1,
    w_dtype="mxfp4",
    a_dtype="bf16",
    a_scale=None,
    w2_sref=None,
    use_csv_config=False,  # opt-in: default uses our tuned tile_n; CSV params for aiter-compare / when requested
    csv_path=None,
    persist=None,
    stream=None,
):
    """a16w4/a16wi4/a16w16 fused stage2 (down-proj). Consumes the bf16 [sorted_size,
    D_INTER] intermediate; scatters routing-weighted bf16 into ``flat_out``.

    Tile config: ``tile_m/n/k`` -> BM/TILE_N/TILE_K, ``waves_per_eu`` ->
    rocdl.waves_per_eu, ``b_nt`` -> W-load cache modifier, ``xcd_swizzle`` -> XCD/HBM
    grid remap. ``k_batch`` for parity (must be 1). ``b_nt=None`` keeps the per-M
    U-shape (cached at ends, nt mid-band).
    """
    if k_batch != 1:
        raise NotImplementedError(f"a16w4 gemm2 only supports k_batch=1, got {k_batch}")
    if a_dtype == "fp8":
        # inter_sorted is fp8 here and its scale is per SORTED ROW, so it must be as long
        # as the padded sorted dimension -- a per-token array would be indexed with a
        # sorted position and read the wrong scale for every route past the first.
        if a_scale is None or w2_sref is None:
            raise ValueError("a_dtype='fp8' needs both a_scale and w2_sref")
        _rows = int(inter_sorted_bf16.shape[0])
        if int(a_scale.numel()) < _rows:
            raise ValueError(
                f"fp8 gemm2 a_scale has {int(a_scale.numel())} entries but the sorted "
                f"intermediate has {_rows} rows"
            )

    # CSV-driven per-token config (mxfp4 only, opt-in). Falls back to adaptive default
    # on no match / divisibility violation; explicit caller overrides win.
    if use_csv_config and w_dtype == "mxfp4":
        cfg = resolve_a16w4_gemm2_config(
            model_dim=D_HIDDEN, inter_dim=D_INTER, experts=NE, topk=topk, tokens=int(M_logical), csv_path=csv_path
        )
        if cfg is not None:
            if tile_n == 256 and D_HIDDEN % cfg["tile_n"] == 0:
                tile_n = cfg["tile_n"]
            if tile_k == 256:
                tile_k = cfg["tile_k"]
            if b_nt is None:
                b_nt = cfg["b_nt"]
            if xcd_swizzle == 1:
                xcd_swizzle = cfg["xcd_swizzle"]

    BM = tile_m
    TILE_N = tile_n
    TILE_K = tile_k
    _m = int(M_logical)
    # Per-token tile config (basic/core configs, or an override CSV via
    # FLYDSL_A16WMIX_TUNED_CSV). Fills only the tile args the caller left at default;
    # explicit caller overrides always win. gemm2 defaults are tile_n=256/tile_k=256/
    # xcd_swizzle=1 (fixed 4-wave N-split, no k_wave). The aiter-compare CSV path
    # (use_csv_config, resolved above) takes precedence when enabled.
    # A16WMIX_FORCE_TILES=1 makes the caller's explicit tiles final: the tuner
    # sweeps geometries the resolver has never heard of, and silently rewriting
    # them from a tuned row would make it measure the wrong kernel.
    if not use_csv_config and os.environ.get("A16WMIX_FORCE_TILES", "0") != "1":
        _o = resolve_a16wmix_gemm2_config(
            a_dtype=a_dtype,
            w_dtype=w_dtype,
            model_dim=D_HIDDEN,
            inter_dim=D_INTER,
            experts=NE,
            topk=topk,
            tokens=_m,
            tile_m=BM,
        )
        # gemm2's tile_n IS token-dependent, so take it from the resolver like the
        # other fields. The adaptive default below stays as the fallback for shapes the
        # resolver has no row for.
        if tile_k == 256:
            TILE_K = _o["tile_k"]
        if b_nt is None:
            b_nt = _o["b_nt"]
        if xcd_swizzle == 1:
            xcd_swizzle = _o["xcd_swizzle"]
        if TILE_N is None:
            TILE_N = _o["tile_n"]
    if TILE_N is None:
        # Adaptive default: largest N tile dividing model_dim (int4 prefers 128).
        TILE_N = _default_tile_n(D_HIDDEN, w_dtype=w_dtype)
    if D_INTER % TILE_K != 0:
        # gemm2 contracts over D_INTER, which at Kimi-K3 TP=8 is 384 -- not a multiple
        # of the 256 default. Step down rather than refuse: the caller's tile_k is a
        # preference, and a K tile that does not divide K is simply unusable.
        for cand in (256, 128, 64):
            if D_INTER % cand == 0:
                TILE_K = cand
                break
        else:
            raise NotImplementedError(f"a16w4 gemm2 requires D_INTER (K) % {TILE_K} == 0, got D_INTER={D_INTER}")
    if D_HIDDEN % TILE_N != 0:
        raise NotImplementedError(f"a16w4 gemm2 requires D_HIDDEN (model_dim) % {TILE_N} == 0, got H={D_HIDDEN}")
    # LDS = A tile (BM x TILE_K x elem) + f32 accumulator (BM x TILE_N x 4), and the
    # accumulator dominates. BM=64 with the usual TILE_N=256 needs 82 KB on fp8 / 98 KB
    # on bf16 against a 64 KB limit, so it would build a kernel that cannot launch.
    # Step TILE_N down instead: BM=64 is perfectly usable at 128, and fp8 makes it more
    # attractive than before (it drops VGPR back under the 256 ceiling).
    _elem = 1 if a_dtype == "fp8" else 2
    def _lds_need(tn):
        return BM * TILE_K * _elem + lds_acc_bytes_for(BM, tn)
    if _lds_need(TILE_N) > _LDS_LIMIT:
        for _cand in (256, 128, 64):
            if _cand <= TILE_N and D_HIDDEN % _cand == 0 and _lds_need(_cand) <= _LDS_LIMIT:
                TILE_N = _cand
                break
        else:
            raise NotImplementedError(
                f"a16w4 gemm2 cannot fit LDS for BM={BM} TILE_K={TILE_K} "
                f"a_dtype={a_dtype}: needs {_lds_need(TILE_N)} B > {_LDS_LIMIT} B and no "
                f"smaller TILE_N divides model_dim={D_HIDDEN}. Reduce tile_m."
            )

    # B cache modifier per-token U-shape: cached (0) at both ends (small M reuse / large
    # M L2 residency), nt (2) mid-band (32..1024). Caller may override via b_nt.
    _b_cache_mod = (0 if (_m <= 16 or _m >= 2048) else 2) if b_nt is None else b_nt
    max_m_blocks = min(int(sorted_expert_ids.numel()), int(sorted_token_ids.numel()) // BM)
    # Persistent CU-limited grid, on by default. The earlier reading -- that this does
    # not help because a padded launch's empty CTAs early-return for free -- holds for
    # the empty CTAs but not for the conclusion: trimming sorted_expert_ids so the grid
    # covers only the valid blocks is worth 1.00x, while making the CTAs persistent over
    # the same grid is worth 2.0-3.4x. What pays is CTA reuse, not launch count.
    # It reorders the atomic scatter, which is already order-dependent run to run.
    _persist = True if persist is None else bool(persist)
    launch = _get_compiled_gemm2_a16w4(
        BM, NE, D_HIDDEN, D_INTER, TILE_N, TILE_K, _b_cache_mod, xcd_swizzle, waves_per_eu,
        w_dtype, a_dtype, _persist,
    )
    grid = gemm2_a16w4_grid(BM, N_OUT=D_HIDDEN, TILE_N=TILE_N, max_m_blocks=max_m_blocks, persist=_persist)
    _run_compiled(
        launch,
        inter_sorted_bf16.data_ptr(),
        w2_u8.data_ptr(),
        w2_scale_u8.data_ptr(),
        sorted_expert_ids.data_ptr(),
        cumsum_tensor.data_ptr(),
        sorted_token_ids.data_ptr(),
        sorted_weights.data_ptr(),
        int(M_logical),
        int(max_m_blocks),
        int(grid),
        flat_out.data_ptr(),
        a_scale.data_ptr() if a_dtype == "fp8" else 0,
        w2_sref.data_ptr() if a_dtype == "fp8" else 0,
        torch.cuda.current_stream() if stream is None else stream,
    )
    return flat_out


# =============================================================================
# aiter tuned-CSV config loader for bf16-A MoE. Decodes each
# ``flydsl_moe{1,2}_abf16_w{fp4,int4,bf16}`` kernelName into a tile-config dict.
# Only the tile GEOMETRY is used (aiter's gemm bodies differ).
# =============================================================================

# kernelName tokens:  flydsl_moe{stage}_abf16_w{fmt}_bf16_t{m}x{n}x{k}
#   [_w{N}]=waves_per_eu [_xcd{N}]=xcd_swizzle [_bnt{N}]=b_nt [_kw{N}]=k_wave
#   [_kb{N}]=k_batch (aiter grid split-K, mapped onto k_wave; see _kwave_from_kbatch).
_A16W4_TILE_RE = re.compile(r"_t(\d+)x(\d+)x(\d+)")
_A16W4_W_RE = re.compile(r"_w(\d+)")
_A16W4_XCD_RE = re.compile(r"_xcd(\d+)")
_A16W4_BNT_RE = re.compile(r"_bnt(\d+)")
_A16W4_KW_RE = re.compile(r"_kw(\d+)")
_A16W4_KB_RE = re.compile(r"_kb(\d+)")


def _kwave_from_kbatch(k_batch):
    """Map aiter's grid split-K (``_kb{N}``) onto intra-block slice-K: kb<=1 -> 1,
    kb==2 -> 2, kb>2 -> 4 (k_wave only supports {1,2,4})."""
    if k_batch <= 1:
        return 1
    return 2 if k_batch == 2 else 4


def _decode_a16w4_kname(kname):
    """Decode an ``abf16_w{fp4,int4,bf16}`` kernelName into a tile-config dict, or None."""
    m = _A16W4_TILE_RE.search(kname)
    if m is None:
        return None
    tile_m, tile_n, tile_k = int(m.group(1)), int(m.group(2)), int(m.group(3))
    w = _A16W4_W_RE.search(kname)
    xcd = _A16W4_XCD_RE.search(kname)
    bnt = _A16W4_BNT_RE.search(kname)
    kw = _A16W4_KW_RE.search(kname)
    kb = _A16W4_KB_RE.search(kname)
    k_batch = int(kb.group(1)) if kb else 1
    # Explicit _kw wins; else derive k_wave from aiter's split-K (_kb).
    k_wave = int(kw.group(1)) if kw else _kwave_from_kbatch(k_batch)
    return {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "b_nt": int(bnt.group(1)) if bnt else 2,  # aiter default 2 when token absent
        "waves_per_eu": int(w.group(1)) if w else None,
        "xcd_swizzle": int(xcd.group(1)) if xcd else 0,
        "k_wave": k_wave,
        "k_batch": k_batch,
    }


@functools.cache
def _load_a16w4_csv(csv_path):
    """Parse the tuned CSV into {(model_dim,inter,E,topk,stage,tokens): cfg}."""
    table = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                key_shape = (
                    int(row["model_dim"]),
                    int(row["inter_dim"]),
                    int(row["expert"]),
                    int(row["topk"]),
                    int(row["token"]),
                )
            except (KeyError, ValueError):
                continue
            for stage, col in ((1, "kernelName1"), (2, "kernelName2")):
                kname = row.get(col, "")
                # bf16-A rows across all weight formats (fp4/int4/bf16); tile geometry only.
                if not any(w in kname for w in ("abf16_wfp4", "abf16_wint4", "abf16_wbf16")):
                    continue
                cfg = _decode_a16w4_kname(kname)
                if cfg is not None:
                    table[key_shape + (stage,)] = cfg
    return table


def pick_a16w4_config(csv_path, *, model_dim, inter_dim, experts, topk, tokens, stage):
    """Return aiter's tuned tile-config for one (shape, tokens, stage), or None.

    Exact ``tokens`` row if present, else nearest tuned token (largest <= requested,
    or smallest). ``stage`` is 1 (gemm1) or 2 (gemm2).
    """
    table = _load_a16w4_csv(csv_path)
    exact = table.get((model_dim, inter_dim, experts, topk, tokens, stage))
    if exact is not None:
        return exact
    cand = sorted(
        t for (md, i, e, k, t, s) in table if (md, i, e, k, s) == (model_dim, inter_dim, experts, topk, stage)
    )
    if not cand:
        return None
    le = [t for t in cand if t <= tokens]
    pick = le[-1] if le else cand[0]
    return table[(model_dim, inter_dim, experts, topk, pick, stage)]


# Candidate locations for aiter's tuned fp4 fmoe CSV (env override wins).
_A16W4_CSV_ENV = "FLYDSL_A16W4_TUNED_CSV"
_A16W4_CSV_CANDIDATES = ("/root/aiter/aiter/configs/model_configs/kimik3_fp4_tuned_fmoe.csv",)


@functools.cache
def _default_a16w4_csv_path():
    """Locate aiter's tuned fp4 fmoe CSV (``FLYDSL_A16W4_TUNED_CSV`` overrides), or None."""
    env = os.environ.get(_A16W4_CSV_ENV)
    if env:
        return env if os.path.isfile(env) else None
    for p in _A16W4_CSV_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


def _kw_tile_k_for(cfg, *, K):
    """Return (k_wave, tile_k, note) from ``cfg`` correct for contraction ``K``.

    gemm1 requires ``K % (k_wave * tile_k) == 0``. Keep the CSV's k_wave and pick the
    largest tile_k in {256,128,64} that divides; if none works, drop k_wave to 1.
    ``note`` is set when a fallback was applied.
    """
    kw = int(cfg.get("k_wave", 1))
    tk = int(cfg["tile_k"])
    if kw == 1:
        return 1, tk, ""
    if K % (kw * tk) == 0:
        return kw, tk, ""
    for cand_tk in (256, 128, 64):
        if cand_tk <= tk and K % (kw * cand_tk) == 0:
            return kw, cand_tk, f"kw{kw}:tile_k {tk}->{cand_tk}"
    # No tile_k divides for this k_wave; fall back to no slice-K.
    return 1, tk, f"kw{kw}->1 (no divisible tile_k at K={K})"


def resolve_a16w4_gemm1_config(*, model_dim, inter_dim, experts, topk, tokens, csv_path=None):
    """Resolve the per-token gemm1 tile-config from the tuned CSV.

    Returns a kwargs dict (+ ``_note``), or None when no CSV/row matches (caller uses
    adaptive default). K=model_dim; the kw/tile_k pair is corrected for it.
    """
    path = csv_path or _default_a16w4_csv_path()
    if path is None:
        return None
    cfg = pick_a16w4_config(
        path, model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk, tokens=tokens, stage=1
    )
    if cfg is None:
        return None
    # gemm1 requires inter_dim % tile_n == 0; skip CSV tile_n if it does not divide.
    tile_n = int(cfg["tile_n"])
    if inter_dim % tile_n != 0:
        return None
    kw, tile_k, note = _kw_tile_k_for(cfg, K=model_dim)
    return {
        "tile_m": int(cfg["tile_m"]),
        "tile_n": tile_n,
        "tile_k": tile_k,
        "waves_per_eu": cfg.get("waves_per_eu"),
        "xcd_swizzle": int(cfg.get("xcd_swizzle", 0)),
        "b_nt": int(cfg["b_nt"]),
        "k_wave": kw,
        "_note": note,
    }


def resolve_a16w4_gemm2_config(*, model_dim, inter_dim, experts, topk, tokens, csv_path=None):
    """Resolve the per-token gemm2 tile-config from the tuned CSV.

    gemm2 has no k_wave (fixed 4-wave N-split). Requires D_INTER % tile_k == 0 and
    model_dim % tile_n == 0; a row violating either is skipped (None -> adaptive default).
    """
    path = csv_path or _default_a16w4_csv_path()
    if path is None:
        return None
    cfg = pick_a16w4_config(
        path, model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk, tokens=tokens, stage=2
    )
    if cfg is None:
        return None
    tile_n = int(cfg["tile_n"])
    tile_k = int(cfg["tile_k"])
    if model_dim % tile_n != 0 or inter_dim % tile_k != 0:
        return None
    return {
        "tile_m": int(cfg["tile_m"]),
        "tile_n": tile_n,
        "tile_k": tile_k,
        "waves_per_eu": cfg.get("waves_per_eu"),
        "xcd_swizzle": int(cfg.get("xcd_swizzle", 1)),
        "b_nt": int(cfg["b_nt"]),
        "_note": "",
    }
