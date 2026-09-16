# topk-prefill AVO — known-bad directions

Append-only. Evidence from MI355X gfx950 seed run (Sep 2026).

## fp32 two-pass 8-bit radix (MSB@24 + LSB@0)

**Symptom:** M=64 direct path: 63/64 rows fail multiset vs torch.topk.  
**Cause:** Two 8-bit passes cover only 16 of 32 sortable bits; middle bytes [23:8] unconstrained.  
**Fix:** Four passes at shifts `{24,16,8,0}`.

## radix_scan grid `(M+255)/256` blocks with `row = blockIdx.x`

**Symptom:** Only rows `0..sb-1` get threshold/gather when M > sb (e.g. M=64 → 1 row).  
**Fix:** Launch `<<<M, 256>>>` for row-per-block scan/build kernels.

## `--sample-rank 70` (Tier 2 tighten)

**Symptom:** wall_ms 4.97 vs 2.93 baseline (+69%) on prefill_main.  
**Cause:** More Phase D fallbacks + extra full-row radix work; not a free lunch.

## `--one-block-per-row 1` on coarse_filter (Tier 1 grid)

**Symptom:** wall_ms 3.81 (+30%) vs default.  
**Cause:** Under-occupancy on 256-CU MI355X; single block/row cannot feed HBM.

## NT load inline asm `global_load_dwordx4 ... glc slc`

**Symptom:** hipcc/rocm10.0 compile error on gfx950.  
**Status:** Reverted to vectorized `uint4` load; `--nt-load 1` currently aliases cached load.

## HIP graph capture (not attempted this run)

Listed in plan as neutral/worse from gemm-topk; skipped after launch-overhead was not dominant vs multi-pass read/atomic tax.

## Extending DeepSelect survivor-buffer HIP port

Pre-AVO evidence: bf16 k=512 @ M=4096 N=131072 → 1874 µs, 118 KB LDS → 1 WG/CU. Out of scope for fp32 k=2048.
