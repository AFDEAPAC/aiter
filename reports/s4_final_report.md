# S4 Final Report — topk-prefill fp32 AVO (MI355X)

## Verdict

Seed kernel **x_0** is **correct** (five oracle gates + inject-fault red) and **1.65× faster than `torch.topk`** on the frozen shape, but **does not meet the 760 µs gluon target**. Best measured wall time: **2.940 ms** (5-run median, stddev 0.087%).

| Anchor | ms | vs gluon 760 µs |
|--------|-----|-----------------|
| HBM read floor (2.147 GB @ 8898 GiB/s) | 0.253 | 0.33× |
| Customer gluon (external) | 0.760 | 1.00× |
| **Shipped x_0 (g_cf_gmult=16)** | **2.940** | **3.87× slower** |
| torch.topk (local) | 4.841 | 6.37× slower |

## What shipped

- **Repo:** `/home/mh/topk-prefill-avo`
- **Binary:** `./benchmark_topk` — three-stage sampled fp32 top-k (Phase A sample + Phase B `coarse_filter_fp32` + Phase C 4×8-bit radix on candidates + Phase D row fallback)
- **Harness:** `bench/correctness.py` (5 gates), `scripts/measure_s0.py`, frozen `.evo/config.yaml`
- **Defaults:** `SAMPLE_S=4096`, `sample_rank=90`, `g_cf_gmult=16`, wave64 ballot compression, `FP32_SMEM_CAP=2048`

## Correctness (S1)

All gates pass on MI355X:

1. Value multiset vs `torch.topk` (bitwise after sort)
2. Index uniqueness
3. Valid k/k, in-range indices
4. `input[index]` gather match
5. Distributions: uniform, gaussian, equal, +inf, adversarial (+ inject-fault returns rc≠0)

Phase D fallback rate on production shape: **1 row / 4096** (uniform random fill).

## Performance (S2 + S3)

**S0 frozen baselines** — see `knowledge/s0_baseline.json`.

**S2 anchor (x_0, 5×100 iters):** 2.940 ms wall, stddev 0.087% (< 2% gate).

**S3 tier results (prefill_main, 3-run quick sweep):**

| Variant | wall_ms | vs x_0 |
|---------|---------|---------|
| x_0 (g_cf_gmult=16) | 2.928 | — |
| + sample_rank=70 | 4.967 | +69% ✗ |
| + one_block_per_row | 3.812 | +30% ✗ |

**rocprofv3:** kernel trace captured at `log/rocprof_s2/` (single DB artifact). Qualitative bottleneck: **~20 discrete kernel launches per top-k call** (sample + 4-pass threshold + filter + 4-pass candidate select + remap + fallback), plus **wave atomic scatter** in Phase B — effective HBM efficiency ~**11×** worse than streaming floor (2.94 ms vs 0.253 ms read-only bound).

## Why 760 µs remains out of reach

1. **Launch tax:** Multi-kernel pipeline dominates vs fused gluon-style implementation.
2. **Phase B atomics:** Per-wave `atomicAdd` into global candidate buffers prevents pure streaming.
3. **Phase C radix:** Four global histogram passes per row on candidate buffers (correct but not free).
4. **No verified NT/cache bypass:** ROCm 10 gfx950 rejected planned `glc/slc` asm; vectorized `uint4` only.

Estimated floor with current architecture (1.05× full read @ measured BW): **~410–450 µs** assumes fused single-pass filter — we are **~6.5×** above that optimistic bound, indicating missing fusion not just tuning.

## Distilled actions for a follow-up run

1. **Fuse Phase B+C** into one persistent kernel per row (eliminate candidate round-trip).
2. **Register-tile top-2048** from streaming pass (no global candidate buffer).
3. **Fix NT loads** with gfx950-valid cache modifiers (ISA-verified).
4. **Do not** tighten sample rank without measuring fallback rate on adversarial gate first.

## Lineage

- Seed: `evo/topk_indices_kernel` @ `a339645`
- No additional commits accepted (no perf regression passed gluon bar).
