# Generalization report (G0–G6, H1–H5, then the pow2 grid round)

## Round 3: the full pow2 grid (M 1..4096 x N 2048..1M, K=2048)

Scope is now the customer matrix restricted to powers of two: **130 points**, all
of which run and pass. Contract in [`.evo/config-v3.yaml`](../.evo/config-v3.yaml),
grid and floor model in [`bench/grid.py`](../bench/grid.py).

| version | outer geomean | small_n | decode | prefill | anchor | change |
|---|---|---|---|---|---|---|
| g_0 | 45.21 µs | 16.97 | 34.55 | 207.07 | 615.7 | baseline after the coop bug fix |
| g_1 | 43.36 | 17.00 | 31.32 | 207.21 | 615.7 | coop_g from a measured table |
| g_2 | 43.31 | 16.94 | 31.32 | 207.15 | 615.7 | small_n block from a measured table |
| g_3 | **42.73** | **16.59** | **30.90** | **205.95** | 615.6 | N_lds boundary made M-dependent |

Cumulative **-5.5%** on the 130-point geomean with no regime regressing and the
anchor unchanged, plus one real correctness bug fixed (below).

### A correctness bug the grid gate caught

`phase_b_filter_coop` staged the block's whole chunk into a per-wave LDS buffer
**with no bound check against WSTAGE_CAP=320**. Any wave producing more than 320
passers wrote into the next wave's slice, and the last wave wrote past the buffer
onto the count variables. Exposed by `--dist adversarial` at M=128 N=65536:
`rows_fail=4`, 124 rows with garbage counts, and **the counts changed between
identical runs**.

This also explains the entry previously recorded as "coop_g=2 is broken", which
had been worked around by restricting G to {1,4,16,64,256,1024}. That diagnosis
was wrong and the workaround shipped an out-of-bounds write. After the fix every
G from 1 to 256 returns the same correct count, so the restriction is gone.

The fix keeps the fast path: an overflow valve drains a wave mid-loop only when
its buffer is about to overflow, while the block-level aggregation (one atomic,
one contiguous write per block) still handles the normal case. Draining per-wave
unconditionally was correct but 1.5-2x slower, because at normal density a wave
holds ~6 candidates and every wave ends up doing one 48-byte scattered write.

### Two harness defects that invalidated measurements

1. **The Makefile listed header dependencies by hand** and omitted
   `csrc/topk_shape.hip.hpp` and `csrc/topk_generalize.hip.hpp`, so
   `make && ./benchmark_topk` silently re-ran the previous binary after a
   header-only edit. Caught only because the stale binary's race gave two
   different answers for one config. Now compiler-generated (`-MMD -MP`).
2. **Per-path geomeans were grouped by the path the binary reported**, which is
   itself tunable. When the coop table returned G=1 for M=128 N=16384 that point
   flipped from decode to prefill, and prefill "improved" 5.5% with no
   instruction changed. Scoring now groups by a static regime derived from the
   shape alone.

Also fixed: a small_n `--dump-stats` line that read the untouched `cand_count`
and reported `under_K=M` for every small_n shape, and a silent launch failure
(an over-large dynamic LDS request reported 4.40 µs with garbage output; now
refused up front and `hipGetLastError` is checked).

### What the measured tables replaced

`coop_g` was chosen to hit 256 total blocks. Measured against a full sweep
(8 M x 7 N x up to 9 G), that rule is **+77% worst case** — at M=128 N=1048576 it
picked G=2 for 225 µs where G=16 measures 127 µs. The best two-parameter fit over
the same data still leaves +27%, because the optimum falls roughly as M^-0.3 and
saturates differently per N. A measured 8x7 table hits the optimum everywhere
(worst +0.6%, mean -0.0%), with the fitted formula kept as the off-grid fallback.

`small_n` block size was already within 1% on 38 of 39 points but wrong by a
reproducible +12.1% at M=2048 N=4096, and the measured optimum is not always a
power of two (M=512 N=8192 wants 768 threads). A 13x3 table fixes it. By
contrast the `phase_a`/`phase_c` occupancy formula needed **no** table: worst
+3.4%, mean +0.1% over 91 sampled-path points.

### Directions falsified with numbers this round

All four are in [`knowledge/known_bad.md`](../knowledge/known_bad.md) with the
measurements that killed them:

- **Early-exiting the radix select** once the pivot is pinned: 21-39% slower,
  anchor 615.5 → 662.8 µs. Passes 2-4 really do cost 30-34% (ablation), but the
  block-wide min/max needed to detect the exit costs two barriers *per pass*
  against a saving of at most one pass.
- **A statistically tighter `auto_margin`** (self-consistent fixed point instead
  of the margin-free rank): produced `under_K=1` at M=4096 N=1M. The gate is a
  maximum over M rows, so the required sigma grows with M — measured 3.4-3.5 at
  M=4096, not 3.0 — and the "wrong" rank happens to supply that slack.
- **Deriving S from the acceptance window** instead of `R_TARGET=179`: wins
  5-7.5% at M<=8 but +17.4% at M=64 N=262144 and +3.9% on the anchor, over its
  limit. Geomean -0.91%.
- **`cf_block=256`**: +17.5% geomean, uniformly worse than 512.

## Earlier rounds

## Round 2 (H1–H5): optimizing each regime toward its floor

G0–G6 made the whole matrix work and fixed the two worst structural gaps. H1–H5
then attacked what the corrected floor model exposed. Headline numbers, all on
MI355X with per-shape settings that hold stddev < 0.9%:

| shape | after G0–G6 | after H1–H5 | speedup |
|---|---|---|---|
| M=4096 N=1024 K=512 | 71.2 µs | **23.8 µs** (N=512) / **28.4 µs** | 2.5× |
| M=4096 N=2048 K=1024 | 81.6 µs | **38.1 µs** | 2.1× |
| M=1 N=1M | 51.9 µs | **31.8 µs** | 1.6× |
| M=8 N=1M | 59.1 µs | **38.2 µs** | 1.5× |
| M=64 N=262144 | 63.7 µs | **44.6 µs** | 1.4× |
| M=4096 N=131072 (anchor) | 622 µs | **615.5 µs** | 1.01× |

Worst floor-ratio in the matrix went from **11.9× to 5.07×**. The main shape is
now 1.0% *faster* than the frozen x_7 baseline rebuilt and measured on the same
machine (0.6152–0.6159 ms vs 0.6215–0.6218 ms, interleaved, 3 pairs).

### H1 — the correctness gate was vacuous, and now is not

`bench/correctness.py` returns early from `run_gates()` when torch is missing,
and the host Python is 3.6 with no torch, so every "CORRECTNESS PASS" in round 1
had **skipped the `torch.topk` comparison entirely**. The host binary runs
unchanged inside `rocm/ali-private:...torch2.12.0_vllm_dsv4_20260916`, so the
gates now run against real torch there.

`bench/gate_selftest.py` is the negative control that makes a pass mean
something: it asserts torch is present, then feeds three corrupted index sets
(an index outside the top-k, a duplicate, an out-of-range index) and requires all
three to be rejected. 3/3 go red.

### H2 — small_n block size was sized from the wrong quantity

The block was `min(1024, roundup(N/64)*64)`, i.e. sized from the element count.
At N=1024 that is a 16-wave block in which only 256 of 1024 lanes ever issue a
load, while all 16 waves still pay the fixed 4-pass × 4-barrier select over just
1024 elements.

What actually decides the optimum is **waves resident per CU**, because these
kernels are barrier-bound, not load-bound. Blocks per CU is capped by the dynamic
LDS (the row itself), so at large N only a bigger block supplies enough waves,
while at small N the cap is loose and the cheapest block wins. Two terms are
required: M=4096 N=8192 and M=1024 N=1024 both land on `blocks_per_cu = 4` yet
their measured optima are 8 and 4 waves, so no function of residency alone can
separate them — the second term caps waves by what the row's loads can occupy.

`occupancy_block_threads()` in `csrc/topk_shape.hip.hpp` is that rule, now shared
by `small_n`, Phase A and Phase C. Validated against a 5-point block sweep on 13
shapes: it matches the swept optimum on 12, worst case +8.5% (0.8 µs absolute on
a shape whose floor is launch-dominated).

### H3 — over half the small-M time was dispatch, not work

`rocprofv3 --kernel-trace` at M=1 N=1M attributed the 48.5 µs wall as
`phase_c` 15.3 µs, `phase_a` 14.6 µs, `phase_b` 6.8 µs, `phase_d` 4.0 µs — and
`phase_a`/`phase_c` run with exactly **one block** at M=1.

Three changes, each measured:

1. **Dispatch count 9 → 3 on decode, 6 → 3 on prefill.** `cand_count` is
   *assigned* by every Phase B variant (one block per row, no early return), so
   zeroing it was always dead work. The reservation counters and `fb_count` are
   now cleared inside Phase A, which already runs first on the same stream.
   `finalize_coop_counts` folded into Phase C. `phase_d` folded into Phase C via
   a shared `exact_row_select()` device function — Phase C already owns the row
   and the histogram scratch, and when every row falls back it is *more* parallel
   (M blocks instead of `FB_GRID=64`). Saved ~11 µs on decode shapes, ~5.4 µs on
   the main shape, and the folding alone another 1.6–2.2 µs everywhere.
2. **Phase A / Phase C block size** from the same occupancy rule: M=1 N=1M went
   40.6 → 33.3 µs; matches a 3×3 block sweep on all 15 shapes (worst +0.3%).
3. **hipGraph retested and rejected** — see `knowledge/known_bad.md`.

What remains at tiny M is irreducible with cheap knobs: `phase_a` and `phase_c`
are still ~11 µs each as **single-block** kernels at M=1, which is 22 of the
31.8 µs. Closing that needs a cooperative multi-block radix select; a
multi-launch version cannot win, because 3 extra launches cost ~8 µs against an
11 µs target. Not attempted. The cheap knobs around it are exhausted and
measured: `--phase-a-passes 2` is unsafe (below), `coop_g` is flat at M=1
(31.5 µs at 64 vs 31.8 µs at 256).

### H4 — the scoring set, and a measurement-validity problem it exposed

At the v1 argv (`--warmup 20 --iters 100 --repeats 5`) the sub-100 µs shapes
measure **2.7–3.1% stddev**, above the contract's own `reject_stddev_pct: 2.0`,
so a 2% change there was indistinguishable from noise. At
`--warmup 100 --iters 500 --repeats 7` the same shapes hold 0.1–0.8% with
identical medians. The v2 contract therefore carries per-shape settings.

`.evo/config-v2.yaml` is a **fork**, not an edit: v1 declares itself frozen and
requires a v2 fork to change shape or acceptance policy, and keeping it intact
preserves the auditable 0.6194 ms single-shape record. v2 scores the geomean over
12 shapes (**62.13 µs** baseline) and keeps `prefill_main` as a hard regression
anchor at ≤ 620.0 µs, so a candidate cannot buy geomean with an anchor regression.

## Summary of round 1 (G0–G6)

The fp32 per-row top-k kernel now dispatches through `topk_indices()` across three paths:

| Path | When | Kernels |
|------|------|---------|
| **small_n** | `N <= 8192` | Single `phase_small_n_topk` (exact LDS select) |
| **prefill** | large `M`, `coop_g=1` | 4-kernel fused pipeline (wave-segment Phase B) |
| **decode** | small `M`, large `N` | Same pipeline + cooperative Phase B (`dim3(G,M)`) |

Shape parameters (`S`, `margin`, `cap`, `coop_g`) are derived from the `R_TARGET=179` law in `csrc/topk_shape.hip.hpp`; no hand-tuned `--sample-s` default.

## Key results (MI355X, measured this session)

Baseline column is the frozen x_7 commit rebuilt and re-measured on this machine
now (`/tmp/topk_base`), not the number recorded in a previous session — that
recorded 0.6194 ms reproduces as 0.6212-0.6218 ms today, i.e. ~0.3% machine drift.

| Shape | Baseline (measured now) | After | Notes |
|-------|------------------------|-------|-------|
| M=4096 N=131072 K=2048 | 0.6212-0.6218 ms | **0.6214-0.6218 ms** | No regression (within stddev 0.1%) |
| M=64 N=512 | FAIL (geometry) | **9.4 µs** PASS | small_n path |
| M=64 N=8192 | FAIL / sampling | **16.0 µs** PASS | small_n path |
| M=1024 N=1M | 16.8 ms, 80% fallback | **0.970 ms**, 0 fallback | S=16384, cap=8192 |
| M=128 N=1M | — | **152 µs**, 0 fallback | decode, coop_g=4 |
| M=1 N=1M | 874 µs (~20 GB/s) | **51.7 µs** | decode, coop_g=256 |

Phase B read+write floor at main shape: **0.444 ms** (G0 `floor_bench`, prior 0.4361 ms).

### Main-shape regression found and closed

The first working version measured **0.6702 ms** (+7.8%). Two causes, both
attributed by measurement rather than guessed:

1. **LDS bloat from max-sized static arrays.** `phase_c_select_waveseg` had
   `s_idx[PHASE_C_CAP_MAX]` (32 KB) *plus* a separate `s_keys_small[PHASE_C_CAP]`
   (16 KB), pushing its footprint from the baseline's 38048 B to **54560 B** and
   halving blocks/CU. This is the exact trap already recorded in
   `knowledge/known_bad.md`. Fixed by templating on `STATIC_CAP`: the
   `cap <= PHASE_C_CAP` config keeps two compile-time-sized arrays (38176 B,
   same 4 blocks/CU as baseline), and only `cap > PHASE_C_CAP` pays a dynamic
   split. Recovered 45 µs.
2. **Two redundant dispatches per call.** `cand_reserved` / `cand_bad` were
   `hipMemsetAsync`-cleared unconditionally, but only the cooperative filter
   reads them. At ~2.6 µs/launch this cost the prefill path ~4 µs. Now cleared
   only when `coop_g > 1`. Recovered the remaining 4 µs.

Per-kernel LDS after the fix (from `.group_segment_fixed_size`):

| kernel | before | after |
|--------|--------|-------|
| `phase_c_select_waveseg<true>` | 54560 | 38176 |
| `phase_c_select_contig` | 38048 | 5280 |
| `phase_small_n_topk` | 38048 | 5280 |

`phase_small_n_topk` dropped its index array outright: the whole row is
resident, so LDS slot *i* is column *i*, and the array was storing the identity.

## G0 floor model — and a model bug it exposed

The first version of `floor_us()` charged **every** shape a 4-launch dispatch cost
and candidate-write traffic. That is wrong for the `small_n` path, which launches
one kernel and writes only K indices. The tell was 6 shapes with **ratio < 1.0**
(M=1..256, N=512 at 0.85-0.90x) — a kernel cannot beat its own floor, so that was
proof the model was broken, not that the kernel was fast.

`floor_us()` is now regime-aware (`traffic_bytes()` returns bytes *and* launch
count), and the achievable-vs-peak factor is derived once from the anchor shape
(`achievable_scale = 1.294`) and applied continuously instead of only at the
anchor, which previously put a step discontinuity next to it. After the fix:
0 shapes below floor, and the anchor reproduces its measured floor exactly
(M=4096 N=131072 → 443.6 µs, = the measured `PHASEB_FLOOR` of 0.4436 ms).

Note the denominator differs from the plan's "1.28× floor": that used the
`pipeline_floor_ms` of 0.4821 (phase_b + phase_c + phase_a floors), whereas this
model uses the phase_b read+write floor alone (0.4436 ms), so the same kernel
reads as **1.41×** here. Lower floor, same kernel.

### What the corrected ranking says

| rank | shape | µs | floor µs | ratio |
|---|---|---|---|---|
| 1 | M=4096 N=1024 K=512 | 71.2 | 5.96 | **11.9×** |
| 2 | M=4096 N=2048 K=1024 | 81.6 | 9.81 | **8.3×** |
| 3 | M=4096 N=512 K=256 | 37.6 | 5.96 | **6.3×** |
| 4 | M=8 N=1M | 59.1 | 10.64 | 5.6× |
| … | | | | |
| 71 | M=1024 N=1M | 973.0 | 843.7 | 1.15× |

The largest remaining gap is **`small_n` at large M**, which the broken model hid
(it scored 4.3× there instead of 11.9×). At M=4096 N=1024 the kernel reads 16 MB
— ~6 µs of traffic — but takes 71 µs, because one-block-per-row gives each block
only 1024 elements while the LDS radix select still pays its fixed 4-passes ×
4-barriers cost. G2's check only swept N at fixed M, so it did not catch this.
**This is the recommended next optimization target, and it is not yet attempted.**

## G0 floor model artifacts

- `scripts/floor_bench.hip` + `scripts/floor_model.py` → `knowledge/g0_floor_model.json`
- `bench/sweep_matrix.py` → `log/g0_matrix_sweep.tsv`, `knowledge/g0_matrix_sweep.json`
- Proposed 12-shape scoring set: `knowledge/g0_scoring_set.json`
- Launch overhead: ~**2.6 µs/launch** (4 launches ≈ 10.5 µs); small-payload floor ≈ **5.8 µs**

## G5 launch / hipGraph

| Variant | M=1 N=16K | Result |
|---------|-----------|--------|
| baseline 4-launch | 36.1 µs | — |
| `--fuse-ab 1` | 37.2 µs | neutral (not worse) |
| `--hipgraph 1` | capture **fails** | `hipStreamBeginCapture` incompatible with in-graph `hipMemset` |

See `knowledge/known_bad.md` for hipGraph and `coop_g=2` traps.

## Files added/changed

- `csrc/topk_shape.hip.hpp` — shape derivation, path selection
- `csrc/topk_generalize.hip.hpp` — small_n, cooperative B, contig C, fused AB kernels
- `benchmark_topk.hip.cpp` — `topk_indices()` dispatcher, GPU oracle verify, CLI flags
- `scripts/floor_bench.hip`, `scripts/floor_model.py`, `bench/sweep_matrix.py`
- `bench/correctness.py` — matrix fuzz across regimes

## CLI (new flags)

```bash
--path auto|small_n|prefill|decode
--coop-g G          # override cooperative blocks/row
--fuse-ab 0|1       # fused Phase A+B
--hipgraph 0|1      # graph capture (may warn+fall back)
--verify-oracle gpu|cpu
--verify-sample-rows N
```

## Verification

```bash
make benchmark_topk floor_bench
python3 bench/correctness.py
python3 scripts/floor_model.py
python3 bench/sweep_matrix.py --quick --propose-scoring
```

All matrix cases + inject-fault gate: **PASS** (GPU oracle, no torch required for gate).
