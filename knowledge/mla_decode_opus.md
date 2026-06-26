# MLA decode OPUS (`mla_decode_opus`) — notes

## Implemented behavior (v1)

- gfx950 OPUS, absorbed **D=512**, bf16/fp16 Q/KV/O, fp32 accum + optional fp32 per-head sink.
- One paged KV pool + CSR `kv_indptr` / `kv_indices` per batch row.
- **QLEN** speculative positions per batch row: compile-time template `1..17`, host `switch(qlen)`.
- Causal prefix: position `p` uses the first `max(0, L - QLEN + p)` KV rows of that batch’s CSR row (`L = row length`).
- **QLEN on the grid z-dim:** `p = block_id_z()`, launch `grid(B, num_h_blocks, QLEN)`. Each block computes one (batch, head-block, position). Same total work as the serial loop but exposes `B x QLEN` blocks for the scheduler. Each block re-reads its position's KV (data reuse across positions is NOT done — see below).

## Profiling findings (gfx950 / MI355X, container `pa_bench_mh`, D=512 bf16)

Baseline (before grid-z: serial QLEN loop in block), B=256 L=4096 dense, event-timed:

| H | variant | QLEN 1/2/4/8 lat (us) | matrix % of 2.5PF |
|---|---|---|---|
| 16 | 16mx1_16nx4 | 133 / 253 / 500 / 995 | ~11% |
| 128 | 16mx8_32nx1 | 253 / 492 / 981 / 1951 | ~45% (matches case-study prefill ~48%) |

Latency was **linear in QLEN** — the serial in-block loop gave no amortization.

Derived PMC (rocprofv3 `MfmaUtil/LdsUtil/OccupancyPercent`, B=256):
- **H=128:** `MfmaUtil == LdsUtil`, `MeanOccupancyPerActiveCU ~= 2.0` waves/SIMD -> **LDS-capped, latency/occupancy bound** (same regime as the `attention-kernel-design` case-study; 132 KB LDS -> 1 block/CU).
- **H=16:** `MeanOccupancyPerActiveCU = 1.0` -> only **1 wave/SIMD**. Grid was `B x 1 = 256` blocks on 256 CUs = 1 block/CU, but 16mx1 LDS (~68 KB) allows 2 blocks/CU -> **grid-occupancy-limited**, not LDS-limited. This is why moving QLEN to the grid helps H=16.

NOT HBM-bound: dense `unified_kv` is 4 MB (L2-resident); the high apparent "GB/s" is L2/feed bandwidth, not DRAM (consistent with the case-study L-sweep falsification).

**Key correction vs the original plan:** the QLEN positions are *different speculative tokens* -> different query vectors. Only the KV *data* is shareable, not the attention output. True tile-major reuse needs QLEN live fp32 O accumulators (O = 128 VGPR/wave for 16mx8 at H=128; QLEN x -> spill past 256 VGPR), so it is register-bound to small QLEN x H. A "shared-prefix" trick does NOT work (queries differ).

## Optimization applied: QLEN on the grid z-dim

Correctness re-verified (64/64 pytest). Clean A/B once the box was idle, using **min-of-many**
repeat windows (DVFS/contention only ever slow things down, so the min is the reproducible
peak-clock truth). grid-z (1 launch) vs serial (N x qlen=1 launches, faithful proxy for the
old in-block loop: both keep B blocks = 1 block/CU). B=256 L=4096, reproduced to 3 sig-figs
across 3 runs:

| H | QLEN | grid-z us | serial us | speedup |
|---|---|---|---|---|
| 16 | 1 | 127.5 | 127.5 | 1.00x (control) |
| 16 | 2 | 138.2 | 254.6 | **1.84x** |
| 16 | 4 | 272.3 | 508.5 | **1.87x** |
| 16 | 8 | 540.6 | 1016 | **1.88x** |
| 128 | 1 | 231.4 | 232.0 | 1.00x (control) |
| 128 | 2 | 460 | 463 | 1.01x |
| 128 | 4 | 915 | 927 | 1.01x |
| 128 | 8 | 1826 | 1876 | 1.03x |

- **H=16: ~1.88x** — grid-z fills the idle 50% CU capacity (1->2 waves/SIMD). Note qlen=1->2 is
  nearly free (127->138 us): the second position's blocks soak up previously-idle CUs.
- **H=128: neutral** — LDS-capped at 1 block/CU, so extra blocks don't raise per-CU occupancy.
- qlen=1 control = 1.00x both variants confirms the method (grid.z=1 is identical to the old path).

## Benchmarking caveat (this box)

`pa_bench_mh` is shared and `--setperflevel high` / `--setsclk` do NOT stick in-container; idle
GPUs sit at low DPM (518-981 MHz, not 2.4 GHz) and short bursts ramp erratically. Under
contention the qlen=1 control swung 0.68-1.31x (±30% floor) and per-dispatch GPU time >10x.
The fix that gave reproducible numbers: **wait for an idle box** (`rocm-smi --showuse` all 0%)
and take **min over many repeat windows**, not mean. Use rocprofv3 derived metrics, not raw
`SQ_*`/`SQ_BUSY_CYCLES` ratios.

## Split-KV (flash-decode) — IMPLEMENTED, the big small-batch win

**Why:** latency is linear in L (61->756 us for L=1024->16384) and FLAT across batch
(B=1 ~= B=128 ~= 200 us at H=128) -> a single block's serial KV walk is the critical path, and
any batch below ~256 leaves most of the 256 CUs idle.

**What:** two-kernel flash-decode.
- Stage-1 (`mla_decode_splitkv_s1_16mx8/16mx1`): grid `(B, num_h_blocks, QLEN*num_splits)`,
  each block does one (batch, h-block, position, split) over a **tile-aligned** KV sub-range,
  writes UNNORMALIZED fp32 partial O + (m,l) to workspace (no sink, no normalize).
- Stage-2 (`mla_decode_splitkv_s2`): grid `(B*QLEN, H)`, 128 threads/block, online-softmax
  merges the num_splits partials, applies the sink, normalizes -> out.
- Host `mla_decode_opus_splitkv_fwd`; Python `mla_decode_opus_splitkv(..., num_splits=None)`
  auto-picks splits and allocates fp32 workspace `[B*QLEN*num_splits, H, D]` + `[...,H,2]`.

**Correctness:** 72/72 pytest (num_splits {2,4,8} x QLEN {1,2,4} x H {16,128} x dtype x sink)
match the reference.

**Measured (gfx950, L=4096, qlen=1, min-of-many, kernel-only):**

| H | B | single | split-KV | speedup |
|---|---|---|---|---|
| 128 | 1 | 199 us | 47 us (split 16) | **4.24x** |
| 128 | 4 | 200 us | 54 us (16) | 3.73x |
| 128 | 16 | 200 us | 56 us (16) | 3.54x |
| 128 | 32 | 197 us | 62 us (8) | 3.18x |
| 128 | 128 | 200 us | 145 us (2) | 1.38x |
| 128 | 256 | 231 us | 231 us (1) | 1.00x (auto fallback) |
| 16 | 4 | 126 us | 30 us (16) | **4.21x** |

**num_splits heuristic:** fill the CUs (`ceil(num_cu / (B*qlen*num_h_blocks))`) but **cap at 16** --
beyond ~16 the stage-2 reduction over partials + fp32 workspace traffic outweighs the extra
parallelism (B=1: split=16 -> 47 us vs split=64 -> 93 us). Min ~256 KV tokens of work per split.
`num_splits=1` routes to the single-pass kernel (no workspace).

## Comparison vs aiter production asm MLA decode (bf16)

Compared against `aiter.mla.mla_decode_fwd` (loads `mla_dec_stage1_bf16_a16w16_subQ16/128_*.co`
+ its own split-KV reduce). Matched (B, nhead, ctx=4096, qlen), bf16, min-of-many.
**Caveat:** asm computes the real MLA with RoPE (qk=576 = 512 nope + 64 rope, v=512); OPUS is
absorbed-512, no RoPE (~5% less total FLOPs) -> NOT a drop-in replacement, treat as ballpark.
`opus/asm < 1.0` = OPUS faster. Script: `op_tests/cmp_asm_vs_opus_mla_decode.py`.

- **nhead=128: OPUS 1.1-2.85x faster** across batch/qlen (16mx8 absorbed + split-KV beats the
  asm `subQ128_mqa128` decode). Gap >> the 5% RoPE difference, so it is genuine kernel efficiency.
- **nhead=16:** after the heuristic fix below, OPUS wins/ties everywhere EXCEPT qlen=4 at B>=64
  (1.34-1.36x loss). Root cause: the asm uses `mla_a16w16_qh16_m16x4_n16x1` -- it packs the 4
  query positions into the **MMA M-dim** (16 heads x 4 qlen = 64 M-rows), one fused QK over the
  shared KV. OPUS grid-z runs each qlen position as a separate block re-reading KV. Closing this
  needs a qlen-into-M packed 16mx1 kernel (feasible at small nhead; register-bound at H=128).

### Heuristic fix: 16mx1 holds 2 blocks/CU
`_pick_num_splits` originally stopped splitting at `base_blocks >= num_cu` (1 block/CU). The
16mx1 (H<=32) kernel (~68 KB LDS, 256-thread block) actually fits **2 blocks/CU**, so it should
target `2*num_cu` blocks. Passing `blocks_per_cu=2` for H<=32 flipped the nhead=16 large-batch
cases from loss to win (e.g. B=128 qlen=1: 1.10x -> 0.96x; B=64 qlen=2: 1.27x -> 0.95x).
16mx8 (H>32, 132 KB LDS) stays at 1 block/CU. Measured on gfx950, ctx=4096.

## qlen-packed decode (qpack) — IMPLEMENTED, closes nhead=16 qlen=8

To share the KV load across qlen positions (the asm `m16x4` trick), reuse the **efficient
16mx8 layout with NUM_WARPS == qlen**, so **each warp owns one speculative position's
W_M(=16) heads** and all warps share the KV LDS tile. For H==16 this is memory-contiguous
(`q[B,QLEN,16,D]` position-stride `16*D == W_M*head-stride`), so the existing T_M warp layout
places warp w on position w with no custom Q layout. Register-identical to the H=128 path
(each warp = 16 rows). New piece: each warp masks with its own `valid_kv(warp_id)` while the
KV loop bound stays the uniform `full_kv_len` -> added a separate `mask_kv_len` param to the
16mx8 accum (existing callers pass their own length; 136/136 pytest unchanged).

Kernels: `mla_decode_qpack_16mx8_kernel` (1 block/batch) and `mla_decode_qpack_s1_16mx8_kernel`
(+ split-KV: block = (batch, split), shares KV split across qlen warps, writes per-(position,
split) partials reduced by the existing stage-2). Python: `mla_decode_opus_qpack(...)`.

**Valid qlen: {4, 8}.** qlen=12 breaks the cooperative KV load (`warps_d=3` doesn't divide
`smem_d_rpt=8`); qlen=16 breaks the pipelined `stagger=warp/4` (4 groups -> divergent
barriers). qlen=4/8 give 1/2 stagger groups (8 = the proven config).

**Measured vs asm (nhead=16, ctx=4096, qlen=8, best OPUS = qpack-split):**

| B | asm | split-KV | qpack-split | best/asm |
|---|---|---|---|---|
| 1 | 79 | 35 | 84 | 0.45 |
| 32 | 96 | 89 | 97 | 0.93 |
| 64 | 140 | 183 | 132 | **0.95** (was 1.32 loss) |
| 128 | 250 | 406 | 194 | **0.78** |
| 256 | 465 | 926 | 289 | **0.62** |

### Size dispatch: split-KV vs qpack (both built, picked by B*qlen)
The two kernels are complementary and cross over at the SAME point for qlen=4 and qlen=8:
- **split-KV** puts qlen on the grid (blocks = B*qlen*splits) -> best parallelism when the GPU
  is starved (small batch).
- **qpack** puts qlen in the warps (1 KV read shared by all qlen positions) -> best once batch
  alone fills the CUs, so cutting KV traffic dominates.

Measured crossover = total query rows `B*qlen ~= 1.5*num_cu` (=384 on 256-CU gfx950), ctx=4096:

| qlen | split wins | qpack wins | crossover B |
|---|---|---|---|
| 8 | B<=32 | B>=64 | ~48 |
| 4 | B<=64 | B>=128 | ~96 |

`mla_decode_opus_splitkv` auto-routes H==16 && qlen in {4,8} to qpack when `B*qlen >= 384`,
else split-KV. (Threshold tuned at ctx=4096; longer KV moves the crossover lower since qpack's
traffic saving grows. A ctx-aware threshold is a possible refinement.)

### Hot path (zero alloc / zero sync) — IMPLEMENTED
The convenience auto path allocates the fp32 workspace and does a `kv_indptr.max().item()` GPU
sync per call (measured ~38-39us overhead, i.e. 25-60% of a 50-150us kernel). Decode runs this
per token, so the hot path avoids both:
- `mla_decode_opus_workspace(B, qlen, H, D, num_splits, device)` allocates the partial buffers
  once (size for worst-case; the fwd accepts any buffer with `>=` rows).
- `mla_decode_opus_splitkv(..., num_splits=N, partial_o=, partial_ml=, use_qpack=)` and
  `mla_decode_opus_qpack(..., num_splits=N, partial_o=, partial_ml=)` skip the `.item()` sync
  (only taken when `num_splits is None`) and the `torch.empty` (when buffers are passed).
- The split-vs-qpack dispatch test is `B*qlen` (sync-free), exposed as `use_qpack` so the caller
  can pin the choice. `num_splits` semantics differ per kernel: split-KV blocks = B*qlen*ns,
  qpack blocks = B*ns.

Measured (idle gfx950): auto vs hot, same kernel/result -- qlen=4 B=16 split 80.7->42.4us;
qlen=8 B=16 split 93.2->54.4us; qlen=8 B=64 qpack 130.9->92.2us; qlen=4 B=128 qpack 153.3->114.2us.
~38us removed across the board. 160/160 pytest pass.

**Still open: nhead=16 qlen=4 large batch (~1.25-1.35x vs asm).** qpack here uses NUM_WARPS=4 =
256-thread block = **1 wave/SIMD** (132 KB LDS caps the 16mx8 at 1 block/CU). The asm's m16x4
avoids this. Fix needs a shallower-LDS 16mx8 pipeline (2 blocks/CU -> 2 waves/SIMD) or a
dedicated m16x4-style kernel -- not done (deep-pipeline rewrite, narrow benefit).

## qpack-h8 (nhead=8, 2 positions/warp, NO padding) — IMPLEMENTED

Closes most of the nhead=8 qlen>2 asymmetry (nhead=8 was slower than nhead=16 because qpack
KV-sharing was H=16-only). Design: warp owns 16 rows = **2 positions x 8 heads** (NUM_WARPS=qlen/2),
so the MFMA is fully used (no padding -- the customer's no-waste form). Per-half causal mask:
M-row = lane%W_M; rows [0,8) use valid(2w), [8,16) use valid(2w+1) -- implemented as a per-lane
`last_valid` in `attn_mask_oob_kv_tile_2pos`, threaded via a trailing `mask_kv_len2` param on the
16mx8 accums (existing callers default -1 -> unchanged). Sink head = (lane%W_M)%H.

**Valid only for qlen=8** (NUM_WARPS=qlen/2=4 -> valid 16mx8 smem layout). qlen=4 -> NUM_WARPS=2
breaks the layout (would need KV_TILE=16); stays on split-KV.

Correct per-position (max|d|=0.0002, all positions/seeds/sink/ctx; 240/240 pytest). Wired into
`mla_decode_opus_splitkv`: H=8 qlen=8 B>=96 -> qpack-h8, else split-KV.

Measured (nhead=8 qlen=8 ctx=4096):
| B | split-KV | qpack-h8 | asm8 | OUR nhead16 |
|---|---|---|---|---|
| 64 | 180 (best) | 221 | 354 | 142 |
| 128 | 404 | **240** | 668 | ~164 |
| 256 | 946 | **270** | 1316 | ~250 |

-> nhead=8 qlen=8 now beats asm everywhere (was already; now larger margin and beats asm16 at
large B). The asymmetry vs OUR nhead=16 is **narrowed (2.5x -> 1.46x) but not eliminated**:
qpack-h8 runs at NUM_WARPS=4 = 1 wave/SIMD, vs nhead=16 qpack's NUM_WARPS=8 = 2 waves/SIMD.
Full parity would need a 2-waves/SIMD nhead=8 variant (NUM_WARPS=8, e.g. 1-pos/warp padded) -- not done.

## Is the H=8->16 padding latency cost real? Measured: ~0 (isolation)

Same 16mx1 kernel, nhead=8 (padded to W_M=16, does the identical 16-row MFMA) vs nhead=16
(full), single-pass ns=1, qlen=1, ctx=4096, idle gfx950, min-of-many:

| B | 16 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| padding delta (8 vs 16) | -0.8% | -7.4% | -0.8% | -0.1% | -1.6% |

nhead=8 is **never slower** (slightly faster: writes half the output). If padding cost wall-time,
nhead=8 would be slower than nhead=16 (it does the SAME MFMA) -- it isn't. MfmaUtil = 3-7%
(matrix unit ~94% idle), identical for 8 and 16 -> the padded rows run in idle matrix cycles
while the wave waits on the KV feed. So **padding is latency-free** here; its only cost is
matrix FLOPs/energy (16 rows computed, 8 used), spent in idle capacity. This is WHY qh8 has
little latency upside (KV-bound floor) and why TP8/nhead=16 (or disaggregated decode) is the
GPU-efficient choice when TP isn't forced by model-weight memory.

## nhead=8 (TP16: DeepSeek/Kimi/GLM) — base path also works, also beats asm

nhead=8 runs **correctly today** on the 16mx1 path (verified max|d|=0.0005, qlen 1/2/4): the
kernel does a 16-row MFMA with 8 valid heads (heads 8-15 are OOB->0, masked, discarded). It is
"H=16 cost" but **still beats aiter's qh8 asm everywhere** (measured opus/asm 0.45-0.94 across
qlen 1/2/4 x B 1/16/64/128, ctx=4096) -- aiter's qh8 (gqaratio8) kernels are less tuned than
qh16, and the 16mx1 runs at 2 waves/SIMD. So nhead=8 needs no new kernel; it is in the test
matrix. (A 2-positions-per-tile packing could remove the 8-of-16-row waste, but it is not
needed to beat asm.)

## FAILED experiment: single-buffer qpack for nhead=16 qlen=4 (reverted)

Hypothesis: qpack qlen=4 is occupancy-starved (NUM_WARPS=4 = 1 wave/SIMD, 132 KB pipelined LDS
caps 1 block/CU); a 1x-LDS single-buffer (le2-style loop) would give 2 blocks/CU = 2 waves/SIMD
and beat asm. Built it (SINGLE_BUFFER traits flag + constexpr branch reusing `le2_tiles`),
verified correct, **measured: 1.20-1.54x SLOWER than asm** (B=128: sb 231us vs pipelined-qpack
195us vs asm 156us). Losing the pipeline's load/compute overlap over ~128 tiles (ctx=4096)
outweighs the occupancy gain. Reverted the dispatch to pipelined qpack (the SINGLE_BUFFER
infra is left in place, dead/constexpr-guarded, documenting the result).

### Resolution: it was a DISPATCH bug, not a kernel limit
After the single-buffer dead-end, sweeping num_splits revealed qpack qlen=4 with the RIGHT ns
(which the heuristic already picks: B=64->4, 128->2, 256->1) ties/beats asm: B=64 87us vs asm 86,
B=128 159 vs 156, B=256 265 vs 279 (win). The "loss" was the size-dispatch threshold: the old
unified `B*qlen>=384` routed B=64 qlen=4 (=256) to split-KV (115us) instead of qpack (87us).
The crossover is per-qlen, NOT a single B*qlen (measured ctx=4096): **qlen=4 -> B>=24, qlen=8 ->
B>=48**. Fixed `_QPACK_MIN_B = {4:24, 8:48}` in the dispatch.

**Result: nhead=16 now ties or beats asm across the whole board** (OPUS/asm, ctx=4096):
qlen=1 0.49-0.98, qlen=2 0.55-0.96, qlen=4 0.54-1.03 (B=64/128 are 1.02-1.03x = ties within
noise, B=256 0.95 win), qlen=8 0.45-0.92. The single-buffer rewrite was NOT needed.

### (a) num_splits is already optimal at the tie cells (verified)
Fine ns sweep (qlen=4, ctx=4096): B=64 -> ns=4 (87us) is best (ns=3:99, ns=5:126); B=128 -> ns=2
(159us) is best (ns=3:206, ns=4:190). The heuristic already picks the CU-filling ns, which is
exactly asm-tie (86/156). No headroom via num_splits -> the B=64/128 qlen=4 cells are genuine
ties, not a tuning miss.

### (b) ctx-aware crossover — IMPLEMENTED
The split-vs-qpack crossover B drops as KV grows (qpack shares 1 KV read across qlen positions,
so its saving scales with KV length). Measured crossover B (gfx950):

| | ctx=1024 | ctx=4096 | ctx=8192 |
|---|---|---|---|
| qlen=4 | ~48 | ~24 | ~24 |
| qlen=8 | ~64 | ~48 | ~16 |

`_qpack_min_b(qlen, kv_len)` encodes these as coarse buckets (qlen=4: <2048->48 else 24;
qlen=8: <2048->64, <8192->48, else 16). The auto path reuses its single `.item()` for this
threshold AND the split heuristic -> no extra sync; the hot path passes `use_qpack` explicitly
so it stays sync-free. Verified the routing matches the measured crossovers and is correct
across ctx (e.g. ctx=8192 qlen=8 B=16 now correctly routes to qpack instead of split). 228/228
pytest pass.

## Phase 3 — remaining levers (not implemented)
- **nhead=16 qlen=4 B=64/128 at ctx=4096:** ~1.02-1.03x of asm (tied, within noise; ns already
  optimal). Not worth chasing.
- **nhead=128 B>=256:** ~48% peak, LDS-capped; cut LDS/block for >2 waves/SIMD.

- **H=128 large-batch (B>=256):** still ~48% peak, LDS-capped at 2 waves/SIMD. Lever: cut
  LDS/block (smaller KV tile / shallower buffering; trades pipeline depth) for >2 waves/SIMD.
- **Multi-accumulator QLEN tile reuse:** register-bound to small QLEN x H (fp32 O wall), not done.

Bench scripts: `op_tests/bench_mla_decode_opus.py` (latency/TFLOPS/roofline %),
`op_tests/ab_mla_decode_opus.py` (grid-z vs serial), `op_tests/sweep_mla_decode_opus.py`
(batch/L regime), `op_tests/bench_splitkv_mla_decode_opus.py` (single vs split-KV).

## RoPE / D=576 (asymmetric QK=576 / V=512) — IMPLEMENTED (H<=32 path)

Real MLA decode contract (matches aiter asm `mla_decode_fwd`): `q`/`unified_kv` last
dim = **576** (512 latent/nope || 64 rope, pre-rotated upstream), score = `q·k` over
all 576, **V = kv[:, :512]** (latent), `out` last dim = **512**. The kernel does NOT
rotate; it handles only the asymmetric QK(576)/PV(512) contraction. Executable
contract + harness: `op_tests/test_mla_decode_opus_rope.py` (`_ref_mla_decode_rope`).

### Key structural finding: build the 16mx1 (H<=32) path first, NOT 16mx8
The original "single template D_QK/D_V split across both variants" plan has a real
blocker on the **16mx8** path: its cooperative KV load (`make_layout_gkv` @~475)
splits `smem_d_rpt` across `warps_d = NUM_WARPS/smem_n_rpt = 2` warp-groups. With
bf16, `D_128B_SIZE = 64`, so `smem_d_rpt = D/64`: D=512 -> 8/2=4 (clean), but
D=576 -> **9/2 (non-integer)** -> the load tiling breaks (same class as the qpack
qlen=12 `warps_d=3 ∤ smem_d_rpt=8` break). So 16mx8 needs a load-distribution
rework for the odd 9 chunks; deferred.

The **16mx1** (H<=32) path is clean for 576: `make_layout_gkv/skv/rk` (@~1982/2002/2020)
use `smem_d_rpt` **directly** (no `warps_d` split), so `smem_d_rpt=9` tiles fine and
all derived inst counts stay integer (`kv_buffer_load=18`, `k_ds_read=18`,
`v_ds_read=32`). V latent sits at d[0,512) = smem d-blocks 0..7 (gkv loads global d
in order, so block i = global d[64i,64i+64)); rope = block 8. The PV read uses
`GEMM1_E_N=8` so it reads blocks 0..7 = latent and **never touches block 8** —
correct by construction. The 16mx1 path is also the asm `subQ16`/`qh16`/`qh8`
comparison target, so it's the right first deliverable.

### Implementation (reuses the existing 16mx1 kernels via a trait, no kernel dup)
- Traits `pa_prefill_16mx1_16nx4_traits` got `D_QK_TILE`/`D_V_TILE` (default = `D_TILE_SIZE`
  so all existing 512 instantiations are byte-identical): `GEMM0_E_K <- D_QK`,
  `GEMM1_E_N <- D_V`, `smem_d_rpt <- D_QK` (physical tile = 576 wide). 16mx8 traits got
  the same aliases (= D_TILE_SIZE) so the shared stage-2 can read `D_V_TILE`.
- New trait `mla_decode_rope_16mx1_traits<QLEN,dtype>` = `<...,D_QK=576,D_V=512>`.
- `make_layout_q` (16mx1) Q row spans `D_QK`. Kernel `VQ <- D_QK`, `v_o <- D_V`,
  stage-1 `partial_o` offset `<- D_V`, stage-2 `D <- D_V`.
- `kargs` gained `stride_o_h` (out head stride = D_V = 512) separate from `stride_qo_h`
  (q head stride = D_QK = 576); all existing host fns set `stride_o_h = out.stride(2)`
  (== `stride_qo_h` for non-RoPE -> no behavior change).
- Host: `mla_decode_opus_rope_fwd` (single-pass) + `mla_decode_opus_rope_splitkv_fwd`
  (reuses `mla_decode_splitkv_s1_16mx1_kernel` + the shared stage-2). Python:
  `mla_decode_opus_rope(..., num_splits=None)` (out alloc'd [B,QLEN,H,512]; auto-split
  via the same `_pick_num_splits`, `blocks_per_cu=2`).

### Correctness + regression
- **288/288** rope pytest (qlen{1,2,4} x H{16,8} x ctx{48,64,200,4096} x dtype x sink x
  num_splits{None,1,4}); max|diff| = 0.00195 (bf16). Covers single-tile, multi-tile,
  split-KV, auto-splits.
- **240/240** existing 512 pytest still pass (shared traits/stage-2/kargs/host edits are
  no-ops when D_QK==D_V==D_TILE). Module rebuild ~150s.

### Latency sanity (gfx950, ctx=4096, bf16, wall-time incl. wrapper, min-of-many)
| B | qlen | H | ns | us |
|---|---|---|---|---|
| 1 | 1 | 16 | 1 | 140 |
| 1 | 1 | 16 | 16 | 35 |
| 16 | 1 | 16 | 8 | 38 |
| 128 | 1 | 16 | 1 | 135 |
| 64 | 4 | 16 | 2 | 89 |
| 1 | 1 | 8 | 16 | 30 |

Same ballpark as the absorbed-512 path (the extra 64 rope dims add ~12.5% QK MFMA +
LDS, hidden under the KV-feed-bound floor). NOTE: build/run uses `PYTHONPATH=/aiter`
in `pa_bench_mh` (container's dist-packages aiter has no source ops); the stale
`module_mla_decode_opus.so` must be `rm`'d to force a rebuild after .cu/.h edits.

### Apples-to-apples vs asm (RoPE, true drop-in: both qk=576/v=512) — RUN
`op_tests/cmp_asm_vs_opus_mla_decode_rope.py` feeds IDENTICAL q(576)/kv(576) to asm
`mla_decode_fwd` and our rope kernel (same FLOPs, unlike the absorbed-512 ballpark).
Cross-validated numerically: asm≈opus (max|d|~0.01-0.03 bf16 reduction-order noise).

**Accuracy: OUR kernel is 10-250x closer to the fp32 reference than asm.** Same inputs,
each vs `_ref_mla_decode_rope`: opus max ~2e-4 (uniform across ctx/qlen); asm max
6e-4 to **6e-2** (h8 ctx4096 q4). We keep an fp32 accumulator (`D_ACC=float`) the whole
way; asm accumulates in lower precision (esp. the `qh8` path). So the asm↔opus 1e-2 is
almost entirely asm's own error — we're the accurate one.

**Latency (ctx=4096, bf16, `opus/asm`<1 = we win; asm timings vary ±20% on this shared box):**
- Small/mid batch (B<=16): **OPUS wins everywhere** (0.49-0.81), split-KV fills idle CUs.
- nhead=8: **win almost everywhere** (0.50-0.83); asm `qh8` is under-tuned. Only B>=64 qlen=1 ties (~1.04).
- nhead=16 qlen=1/2 B>=64: ~1.06-1.11 near-tie (no qpack for qlen<4; asm m16x4/single well-matched).

### RoPE qpack (qlen-into-warps, qlen=4 only) — IMPLEMENTED, closes the qlen=4 loss
The one real loss was nhead=16 qlen=4 B>=64 (split-KV 1.4-2.4x slower than asm `m16x4`,
which packs the 4 positions into the MMA M-dim with one shared KV read). Ported qpack to
RoPE: warp==position, 4 warps share the KV LDS tile, asymmetric QK(18 slices)/PV(16 slices).

**Why qlen=4 only:** qpack uses the 16mx8 cooperative load (`warps_d = NUM_WARPS/smem_n_rpt`).
qlen=4 -> NUM_WARPS=4, warps_d=4/4=1, so `smem_d_rpt(9)/warps_d(1)=9` is integer. qlen=8 ->
warps_d=2 ∤ 9 (the same 16mx8 break). smem 4x9-chunk tile ~148.5KB -> 1 block/CU (fits 160KB).

**16mx8 decoupling (shared accum, both le2 + pipelined):** added `NUM_D_SLICES_QK=D_QK/SLICE_D`
(18) and `NUM_D_SLICES_V=D_V/SLICE_D` (16) to the 16mx8 traits (default both = `NUM_D_SLICES`
-> non-rope instantiations compile byte-identical). `compute_qk` loops `NUM_D_SLICES_QK`,
`compute_pv` loops `NUM_D_SLICES_V`. `skv_slice(s)=(s/2)*chunk + (s%2)*32`, so QK slices 0..17
span chunks 0..8 (576), PV slices 0..15 span chunks 0..7 (latent 512), rope chunk 8 loaded but
unread by PV. All `NUM_D_SLICES` uses are confined to the two lambdas; the pipelined
prologue/epilogue only touch slots 0/1 (D-independent). make_layout_q->D_QK, make_layout_o->D_V.

**Stable internal win (qpack vs split-KV, nhead=16 qlen=4 ctx=4096):**
| B | split-KV | qpack | asm |
|---|---|---|---|
| 64 | 134 | **98** (ns4) | 85-96 |
| 128 | 494 | **175** (ns2) | 156-204 |
| 256 | 1366 | **678** (ns1) | 847 |

qpack is **1.4-2.8x faster than split-KV** at these cells -> the asm loss becomes a tie/win
(B=128 ~0.91-1.12, B=256 0.80). Auto-route: H==16 && qlen==4 && B>=`_qpack_min_b(4,ctx)`
(=24 at ctx>=2048) -> qpack, else split-KV. B<=16 stays split-KV (0.66-0.77, better there).
Correctness: 120 qpack pytest (B{4,64} x ns{1,2,4} x ctx{40..8192} x dtype x sink), worst
max|d|=0.00195. 648/648 total pytest (288 rope + 120 qpack + 240 regression).

## RoPE H>32 (16mx8, nhead=64/128) — IMPLEMENTED via NUM_WARPS=4 (the warps_d=1 trick)

The 16mx8 cooperative load tiles the KV as `smem_n_rpt x smem_d_rpt` chunk-tiles over
`NUM_WARPS` warps; `warps_d = NUM_WARPS/smem_n_rpt`. With the default **NUM_WARPS=8**,
`warps_d=2` and D_QK=576 -> `smem_d_rpt=9`, `9/2` non-integer AND `kv_buffer_load =
32*576/(512*8) = 4.5` -> the load can't tile 576. (This is fundamental, not just
`warps_d`: 36 chunk-tiles don't divide 8 warps.)

**Two dead-ends found (verified):**
- **KV_TILE=64** (makes `warps_d=8/8=1`): load tiles cleanly, BUT the 16mx8 *compute*
  is numerically WRONG -- a full-tile case (no masking) gives max|d|=0.13. The accum
  was written for KV_TILE=32 (`GEMM0_E_N=2`, `GEMM1_E_K=1`, `smem_n_rpt=4`); KV_TILE=64
  changes all three and the score/PV layout no longer matches. Error dilutes with tile
  count (per-tile systematic), confirming a layout mismatch. Reverted.
- **Split load** (latent 512 + rope 64, keep NUM_WARPS=8): correct compute, but the rope
  64-wide sub-region (4 chunk-tiles) doesn't tile over 8 warps either, needs a custom
  layout matched to the intricate physical smem placement. High-risk, deferred.

**The fix that worked: NUM_WARPS=4.** `warps_d = 4/smem_n_rpt(4) = 1`, so `smem_d_rpt/1=9`
and `kv_buffer_load = 32*576/(256*8) = 9` both tile cleanly -- and this is the SAME config
the qpack-rope path already uses (proven correct), so the existing 16mx8 kernels +
pipelined/le2 accum are reused as-is (`mla_decode_rope_16mx8_traits` = `<16,32,512,4,...,576,512>`).
Only fix needed: the shared 16mx8 single-pass out-offset and split-s1 partial-O offset had
`stride_qo_h`/`D_TILE_SIZE`; changed to `stride_o_h`/`D_V_TILE` (no-op for non-rope, needed
for rope `h_block>0`). Host routes H>32 -> `mla_decode_16mx8_32nx1_kernel` /
`mla_decode_splitkv_s1_16mx8_kernel<rope trait>`. 936/936 pytest (incl. 288 H>32, 240 regression).

**Dispatch by H (warp count chosen so the KV tile is read once per 128-head block):**
- **nhead=64** -> NUM_WARPS=4 (64 heads = 1 h_block, single read). `mla_decode_rope_16mx8_traits`.
- **nhead % 128 == 0** (128/256) -> NUM_WARPS=8 (128 heads/block, single read).
  `mla_decode_rope_16mx8_nw8_traits`. NUM_WARPS=4 here would be 2 h_blocks = 2x KV reads.

### NUM_WARPS=8 + the "shifted load" trick (the warps_d=2 fix) — IMPLEMENTED
NUM_WARPS=8 -> `warps_d = NUM_WARPS/smem_n_rpt = 8/4 = 2`, and 9 chunks don't divide
(36 chunk-tiles ÷ 8 warps = 4.5; `36 ≡ 4 mod 8` so the rope always leaves 4 un-tileable).
**Key insight:** the EXISTING gkv/skv with `smem_d_rpt=9` integer-truncates `9/2=4` and
`9*4/8=4`, so it cleanly loads EXACTLY the latent 512 (chunks 0..7, verified: group0->chunks
0-3, group1->chunks 4-7). The rope chunk is then brought in by a **second async_load shifted
+1 chunk** (source d+`D_128B`, dest +`smem_n_rpt*(linwave+pad)`): the same proven layout loads
chunks 1..8 (chunk 8 = rope) into lines 4..35; chunk 8 lands at lines 32..35 where `skv_slice(16/17)`
reads it. Chunks 1..7 reload redundantly (harmless). In-bounds (d 64..575), no OOB, no custom
layout. Gated by `ROPE_SHIFT_LOAD = (D_QK!=D_V) && (NUM_WARPS>smem_n_rpt)` (false for NUM_WARPS=4,
which loads all 9 cleanly).

### Root cause FOUND (deep opus study + bisection): the overlap, not the addressing
Studying `opus.hpp` (make_layout / unfold_x_stride / unfold_p_coord / async_load) revealed the
exact gkv/skv mapping: thread reads global chunk `c=a*2+warp/4` (a = y-iter 0..3), writes physical
chunk `c`, rowgroup `warp%4`; the **row is per-thread via `kv_page=kv_indices[(lane/8)*4+warp%4]`**
(NOT in gkv), and the within-line is the buffer_load lane-spread `lane -> [row=lane/8][d=lane%8]`.
The `+1-chunk shifted` load reads chunks 1..8 (overlap 1..7 with latent). Bisection (debug overrides
`NUM_D_SLICES_QK=16` latent-only ± shifted load) proved: **smem_d_rpt=9 pipelined latent is correct
WITHOUT the shifted load (0.001); the shifted load's OVERLAP corrupts it (0.15).** The overlapping
`buffer_load...lds` to chunks 1..7 (identical data) hits a merge/concurrency hazard in the tight
pipeline that le2 tolerates.

### Fix: no-overlap chunk-8 load (`make_layout_gkv_rope8`/`skv_rope8`)
Custom layout that loads ONLY chunk 8 (d in [512,576)) with `warps_d` stride **0** (all warps read
chunk 8, not chunk 9) + `warp%smem_n_rpt`->rowgroup, dest `+rope_chunk8_skv`. Same within-line as
latent (same gkv d-pattern + buffer_load spread). **No overlap** with chunks 0..7. Replaces the shift
at all 8 sites. Result: **le2 fully correct for all ctx (936/936)**, AND the big deterministic
corruption is gone in the pipelined too (small/mid ctx: 0.15 -> 0.027).

**chunk-8 load PROVEN correct; pipelined residual is NOT the schedule (sched-hints hypothesis
DISPROVEN this session).** Decisive test (`FORCE_LE2=true`, tight threshold): le2 + chunk-8 load
gives **max|d| = 0.0002 at ctx=4096** (identical to the absorbed path) -- the no-overlap chunk-8 load
is exactly correct, within-line and all. The pipelined path with the SAME load gives ~0.027 + NaN.

Experiments run to localize (all on the pipelined NW8 path):
- **sched hints OFF** (gated `sched_compute_qk<0>()`/`s_setprio` behind `!ROPE_SHIFT_LOAD`, kept
  `sched_barrier(0)` fences + `s_barrier`): **NO change** (still 0.0284 + NaN). -> the schedule
  hints are NOT the cause. (Supersedes the old "re-tune sched hints" conclusion.)
- **compiler memory fence** (`asm volatile("":::"memory")` after each chunk-8 `async_load`): **NO
  change**. -> not compiler DCE/reorder of the LDS write.
- **q_rope=0 vs k_rope=0** (the sharp diagnostic, ctx=4097 pipelined):
  `q_rope=0` -> **0.0002 (correct)** => latent path fully correct, chunk-8 values are finite.
  `k_rope=0` -> **0.5749 (garbage)** => with chunk 8 loaded as 0, compute STILL reads non-zero
  garbage at the chunk-8 read location. So `compute_qk`'s chunk-8 read (slices 16/17) returns
  **deterministic garbage**, not the loaded value, in the pipeline only.
- smem sizing verified correct (`smem_kv_tile_elems` = 9 chunks incl. chunk 8; `smem_size_bytes`
  = 4 tiles); le2/pipelined dispatch verified correct (pipelined only for >2 tiles).
- **Exhaustive static trace** of prologue + main loop (across `std::swap(s_kv[0],s_kv[1])`) +
  both epilogues: EVERY `compute_qk` reads a tile whose chunk-8 was loaded into that exact
  buffer/sub-tile, with write offset == read offset (`rope_chunk8_skv == skv_slice(16) == 16896`,
  `kv_slot_offset` added to both for sub1). The pairing is provably correct and byte-identical to
  the le2 write/read that works.

**Conclusion: source addressing is provably correct yet runtime reads garbage -> a runtime/HW-level
corruption** (most likely the extra divergent `if(warp_id<smem_n_rpt)` chunk-8 `async_load` + the
576/18-slice register pressure perturbing the tightly hand-balanced 10-cluster pipeline's
LDS-visibility timing). Source-level levers (sched hints, fences, vmcnt(0), overlap, redundancy,
guard) are ALL exhausted. Going further requires GPU instrumentation (ATT trace / direct LDS dump
of chunk 8 at runtime) -- a separate, larger effort. **NW8 stays on le2** (proven correct, clean
no-overlap load, a win over NW4). The pipelined-overlap (qlen=1 mid-batch ~1.3x) is the only
remaining lever and it is NOT recoverable from source alone.

OLD (superseded) notes:
The shifted load is correct in the single-buffer le2 accum (max|d|=0.00195 all ctx). In the
2-buffer pipelined accum it fails, and the failure is **nondeterministic** (same build/seed:
ctx=97 gave 0.15 in one run, NaN in another) -> a race, NOT a logic/offset bug. Evidence chain:
(1) NOT a vmem race — converting every partial `vmcnt(N)`/`vmcnt(1)` to `vmcnt(0)` (full drain)
did not fix it. (2) Isolated to the rope chunk via **q_rope=0** test (zeroing q[...,512:]): with
q_rope=0 the rope contributes 0 regardless of load, yet small ctx still NaN'd -> chunk 8 holds
inf/NaN garbage (0*inf=NaN), i.e. the shifted load's chunk 8 is **not reliably visible** to
`compute_qk`'s cross-warp read before it executes. (3) Fails even at the minimal 3-tile case
(prologue+epilogue), and is OK-ish at large ctx (repeated sub-tile reloads mask the stale read).
Every load/read offset was traced and matches le2's working pattern exactly (chunk 8 write at
`u_skv [+slot] + rope_shift_skv` -> physical chunk 8; read at `u_rk [+slot] + skv_slice(16)` ->
same), so the cross-warp LDS-visibility race lives in the 10-cluster pipeline's barrier schedule
interacting with the extra `async_load`, and resolving it needs LDS-dump instrumentation beyond a
reasonable budget. **NW8 is therefore forced to single-buffer le2** (`FORCE_LE2` trait flag, set
on `mla_decode_rope_16mx8_nw8_traits`; trait overrides smem to 1 tile). The pipelined shifted-load
code is left in place (constexpr-dead for le2-forced NW8 / non-rope) documenting the attempt.

**Measured (nhead=128 ctx=4096, NW8 le2 vs asm, idle GPU but asm timing noisy):**
| qlen | B16 | B32 | B64 | B128 | B256 |
|---|---|---|---|---|---|
| 1 | **0.94** | 1.34 | 1.39 | 1.23 | 1.11 |
| 2 | **0.95** | 1.20 | 1.06 | 1.06 | **0.96** |
| 4 | **0.57** | **0.77** | **0.88** | **0.84** | 1.01 |

vs the old NUM_WARPS=4 (qlen=1: B16 1.14, B32 1.63, B64 1.52): NW8 single-read **closes ~half
the gap and flips small batch to a win**. The residual qlen=1 mid-batch ~1.3-1.4x is now **le2's
missing load/compute overlap** (not the KV traffic). 936/936 pytest (H=64 NW4, H=128 NW8, regression).

## Split-KV heuristic: trend-fit closed form (2026-06, dense sweep)

`_pick_num_splits` originally hard-capped at **16** (and the rope wrapper hardcoded `_blocks_per_cu=2`
for all H -- wrong for nhead=128 16mx8 which is 1). A dense `num_splits` sweep (nhead x B x qlen x ctx,
argmin latency per cell) gave a clean trend that replaces all the ad-hoc caps:

```
W = num_cu * blocks_per_cu          # 256*2=512 (16mx1), 256*1=256 (16mx8)
base = B * qlen * num_h_blocks
ns = max(1, min(W // base, kv_len // W, 64))
```
- `W//base` = blocks/ns to fill the device. **floor, not ceil**: keep `base*ns <= W` so the launch
  is one resident wave -- overshooting W by even a few blocks spills a near-empty 2nd wave whose tail
  ~doubles latency (measured nh128 base10 L64K: ns24=315us, ns26 tails; ceil picked 26 -> 1.22, floor
  picks 25 -> **0.74**). floor==ceil whenever base divides W (all the clean cells), so floor only helps
  the odd-batch / non-dividing cases.
- `kv_len // W` = per-split work floor (>= ~W KV tokens). The heavier 16mx8 (W=256) tolerates ~2x more
  splits than light 16mx1 (W=512); longer ctx affords proportionally more. (Confirmed nh16 B1 floor
  ~ctx/512, nh128 B1 ~ctx/256.)
- 64 = kernel's hard max num_splits.
- `base` folds qlen/heads cleanly: equal `base` gives the same optimum across qlen in {1,2,4} (verified).

This single formula matches the measured argmin across the whole grid and removed the sqrt/gate/`/256`
cruft. Wins (opus/asm): nh=128 q1 B1 ctx64K **0.85->0.33**, ctx32K 0.89->0.58; nh=16 q1 B1 ctx64K
**1.57->0.87**; nh=16 B16 ctx64K 1.27->1.19 (also better-split than the old gated cap). After this +
qpack + s2-tiling, the only residual losses >1.15 (asm>40us) are **nh16 q1 B64 ctx5200 (1.23)** and
**nh16 q1 B16 ctx16384 (1.16)**.

### qpack crossover drops at long ctx (fixes nh=16 qlen=4 B=16)
`_qpack_min_b(4)` was 48/24/24 for ctx 1K/4K/8K; the crossover keeps dropping with KV length
(qpack shares 1 KV read across qlen positions). Added a **>=16384 -> 16** bucket: B=16 qlen=4 now
picks qpack instead of split-KV. Measured B=16 (split->qpack us): 16K 132->105, 32K 240->180,
64K 451->302 -> opus/asm **1.37/1.51/1.53 -> 1.09/1.13/1.02**. B=8 stays split (split<qpack), so
threshold 16 (not lower) is correct. Correctness <=0.0001.

### Stage-2 reduce: tile D across grid.z when (B*qlen*H) under-fills the GPU
Profiled the nh=16 q1 B1 ctx=32K loss (rocprofv3 --kernel-trace): **s1=57us, s2=31us** (asm total
62us) -- the stage-2 reduce was the gap. Root cause: s2 grid was `(B*qlen, H)` = **16 workgroups**
for nh=16 B=1 (~6% of 256 CU) -> latency-bound on the strided per-split `po` loads. (nh=128 had 128
blocks -> fine, which is why nh=128 B=1 already won.) Fix: add a **grid.z = d_tiles** dim, set in
`launch_splitkv_stage2` to `ceil(256/(B*qlen*H))` capped at `D_V/blockDim` (=4), and the kernel
d-loop starts at `blockIdx.z*nthreads+tid` stride `gridDim.z*nthreads`. Each (bp,head,d) still owned
by one block (no atomics); d_tiles=1 once the device is full (no large-batch change). **s2 31->11us.**
Wins (opus/asm, before->after this fix): nh=16 q1 B1 ctx32K 1.45->1.12, ctx64K 1.15->0.87 (win!),
ctx16K 0.89->0.66; nh=16 q1 B5 ctx32K 1.50->1.17; **nh=128 q1 B1 ctx64K 0.41->0.33** (its s2 also
tiles, d_tiles=2). Correctness <=0.0005, no large-batch regression.

Follow-up: the s2 d-tile target was raised from `num_cu` (256) to `4*num_cu` (the reduce kernel is
featherweight -> fits many blocks/CU), so mid-batch (B=16, base=256=1/CU) also tiles: **B=16 nh=16 q1
s2 24->9us -> 1.34->1.23**. Large batch (base>=1024) stays d_tiles=1 (no change).

### Stage-1 double-buffering: TRIED, REVERTED (made s1 slower)
Profiled s1 (the bigger component): nh=16 q1 B16 ctx32K s1=129us, HBM-streaming-bound (TCC L2 hit
2.3%, SQ_WAIT_INST_ANY dominant, ~4.7TB/s = 59% of 8TB/s peak). The `pa_prefill_16mx1_16nx4_pipeline`
loop is actually **single-buffered + serial** (`vmcnt(0)` BEFORE compute -> load fully exposed).
Tried a double-buffered variant (prefetch tile N+1 into a 2nd LDS buffer during compute N). Result:
**s1 129->139us (SLOWER), reverted.** Why: (a) the rope KV tile is 76KB (smem_d_rpt=9), so 2x =152KB
forces 1 block/CU; (b) the per-tile compute (16 heads, tiny MFMA) is far shorter than the 76KB HBM
load, so overlap hides almost nothing while the occupancy/bookkeeping cost is real. Lesson: **s1 is
already near its achievable BW for this access pattern** (more splits/blocks also don't help ->
confirmed not latency/occupancy-starved); the asm edge is a better KV memory layout, not prefetch.

### Other qh coverage (asm bf16 ships qh 8/16/32/64/128)
Benchmarked all asm bf16 head counts (was 8/16/128; added 32/64). Findings (clean recheck):
- **nh64 (16mx8 NW4): ~par** with asm, 1.0-1.16 at large B, wins at small B. NW4 is the clean
  pipelined path (not the forced-le2 NW8), so it holds up. asm qh64 only ships qseqlen1.
- **nh32: real weakness at large B + qlen>=2 (up to 3.0x slower).** opus routes H<=32 to **16mx1**
  (small 16-head tiles, T_N=4); at B256 q4 ns=1 -> ~2048 tiny blocks (8 waves) each walking the full
  ctx -> far slower than asm's dedicated `qh32_qseqlen4_gqaratio32` kernel. nh32 still *wins* at small
  B (q4 B1 0.64). Fix would need a 16mx8-style H=32 path (pack 32 heads into warps) -- new kernel,
  not a heuristic. (Measured nh32 q4 B256: 8192 3.02, 65536 2.55; q2 B256 8192 2.61.)
- Fixed a heuristic bug exposed by H=32: `num_h_blocks` was `ceil(H/128)` (=1 for H=32) but 16mx1
  packs 16 heads/block -> should be `ceil(H/heads_per_block)` with heads_per_block = 16 (H<=32) / 64
  (NW4) / 128 (NW8). Wrong value under-counted base -> over-split at large ctx.

## Still open (remaining asm-favored cells)
- **nhead=16 qlen=1, B=16-32, ctx>=16K: ~1.19-1.34x (stage-1 HBM-layout-bound).** s2 now tiled
  (B16 1.34->1.23). Residual is s1 streaming BW (~59% peak) vs asm's hand-tuned layout -- double-
  buffering doesn't help (above); needs a different KV load layout. Deep, low ROI. qlen=2/4 same B
  win via qpack.
- **nhead=128 qlen=1 mid-batch (B=16-32) mid-ctx (5K-16K): ~1.05-1.2x:** le2-bound (no pipeline
  overlap). Full win needs **pipelined NW8**, blocked on the cross-warp LDS race above (needs
  LDS-dump instrumentation). Small batch at this nhead is now a strong win via the split fix.
- **qpack qlen=8 at D=576:** warps_d=2; could try the shifted-load trick instead of split-KV. Untried.
- **asm `qh64` faults** on the cmp harness (nhead=64) -> no asm A/B for nhead=64.
- **FP8 KV:** dtype template axis + scale contract (`contract-first-playbook`).

## Disassembly study of aiter asm decode (2026-06, gfx950 bf16) — verified

Round-tripped the bf16 a16w16 decode `.co`s (`hsa/gfx950/mla/`) with `llvm-objdump`
(`disasm_to_s.py`, gfx950). Common structure across ALL of them:
- **Fixed 4 warps (256 thr), T_M row-parallel, T_N=1.** Each warp owns M/4 rows; M=nhead*qseqlen.
  In-register online softmax (`v_cmp/v_max/v_exp/v_add` inline between QK/PV MFMA), **1 `s_barrier`
  per KV tile** (double-buffer swap) -- no cross-wave P/reduction round-trip. Verified by reading the
  qh64 and qh16 main loops.
- **M = nhead x qseqlen is the real tiling knob.** Kernels pick qseqlen so M in [64,128]: qh64xqlen1
  (M64), qh16xqlen4 (M64), qh32xqlen4 (M128), qh16xqlen8 (M128). When M<64 they fall back to a
  split-KV stage1 (subQ16/subQ128).
- **Two generations, same skeleton:** legacy (qh16/qh8/subQ) = `v_mfma_f32_16x16x16_bf16` + MANUAL V
  transpose (`ds_write`/`ds_read_b128`) + 64KB LDS (2 blk/CU). gfx950-native (only qh32-qseqlen4,
  qh64-qseqlen1) = `v_mfma_f32_16x16x32_bf16` (double-rate) + `ds_read_b64_tr_b16` (free transpose)
  + 160KB LDS (1 blk/CU). All use 256 arch + 256 AGPR (accumulator in AGPR).
- **OPUS already matches the matrix/feed primitives**: traits use `W_K=32` (16x16x32), `tr_load`,
  `D_ACC=float`. The qpack path already does the M=nhead*qseqlen packing (qh16 qlen4, qh32 qlen2/4).
  So the asm's design is *already in use*; it was not a new idea waiting to be applied.

## FAILED experiment: qh32 qlen1 -> 16mx8 NW4 head-packed (reverted, 2026-06)

Hypothesis: route qh32 qlen1 off the T_N-split 16mx1 onto a head-packed T_M path (the only viable
head-pack is NW4-padded: 32 heads + 32 OOB rows in the 64-row tile, since NW2 breaks the 16mx8
cooperative load -- `warps_d = NUM_WARPS/smem_n_rpt = 2/4 = 0`). Changed dispatch `H<=32`->`H<32`
(C++ rope_fwd/splitkv + Python heuristic). **Correctness 40/40 pass. Perf: uniformly 1-13% SLOWER
than 16mx1** (clean 20-rep A/B, ctx 5200-32000, B 16-256). Reverted.

Root cause: padding 32->64 wastes **2 of 4 whole warps**, each doing a full KV walk -- NOT free
(unlike qh8->16 intra-MFMA padding, where the matrix unit was idle). Plus NW4 is 1 blk/CU vs 16mx1's
2 blk/CU -> worse occupancy in this KV-feed-bound regime. The saved T_N reduction doesn't pay for it.
**Lesson: T_M head-pack only wins when M fills the warps WITHOUT padding** (qh64=64 exact; qh32
qlen2/4=64/128 exact via qpack). For qh32 qlen1 (M=32) there is no padding-free T_M at NW>=4, so the
T_N-split **16mx1 (2 blk/CU) is OPUS's best available** option.

### Reframing: qh32 qlen1 large-B gap is NOT algorithmic
Disassembled the asm kernel that actually serves qh32 qlen1 (`MLA_A16W16_..._16mx4_32nx1_QH16.co`,
symbol `mla_a16w16_qh16_m16x4_n32x1`). Its compute loop is **identical to qh64 v3** (1176 MFMA / 1248
tr_load / 554 b128 / 15 barrier / 153 exp / 160KB / NW4); it differs only by +553 lines of prologue
head-bound masking (`s2<<6`, `min(.,64)`, `s2*0x12000` head-block stride). **So asm ALSO pads 32->64
and masks -- same padded-T_M(NW4) algorithm as the failed experiment.** Therefore the ~1.2-1.4x
qh32-qlen1 large-B gap is **OPUS's NW4/16mx8 stage-1 being ~1.3x less feed-efficient than asm's**
(same root cause as the qh64-qlen1 and qh16-qlen1 large-ctx gaps: OPUS s1 ~59% peak BW vs asm's
hand-tuned KV LDS layout). A new *layout* kernel will not close it (proven: all layout variants are
equal/worse). The real large-batch lever is the **shared stage-1 KV-feed pipeline / LDS layout**,
which needs profiling (LdsUtil / bank conflicts / BW) to pinpoint, not blind kernel carving.

### Profile of the qh64 qlen1 large-B gap (rocprofv3 PMC, 2026-06) — it's scheduling, not memory
Profiled qh64 qlen1 B256 ctx8192 (both 1 block/CU), OPUS NW4 vs asm qh64:
| metric | OPUS NW4 (4x pipe) | asm qh64 |
|---|---|---|
| FetchSize (HBM bytes) | 1.185M | 1.193M (**identical**) |
| LdsBankConflict | 0 | 0 |
| MemUnitStalled % | 0.02 | 0.00 |
| MeanOccupancy/CU | 1.00 | 1.00 (**both 1 wave/SIMD**) |
| MfmaUtil % | 21.6 | 29.0 |
| LdsUtil % | 21.6 | 29.2 |

Conclusion: **NOT memory/BW/bank-conflict/coalescing bound** (identical FetchSize, 0 conflicts, ~0
mem stall). The gap is pure pipeline utilization -- asm keeps MFMA+LDS ~34% busier at the SAME
1 wave/SIMD; **util ratio 29/21.6 = 1.34x ~= the perf gap.** So the earlier "asm has a better KV LDS
layout" hypothesis is WRONG -- both stream identically; asm just **hand-schedules its single wave's
load/LDS/MFMA tighter** (fewer exposed waitcnt gaps). (This corrects knowledge above that attributed
the s1 gap to KV memory layout.)

### FAILED experiment #2: NW4 single-buffer (FORCE_LE2) for 2 waves/SIMD (reverted, 2026-06)
asm is LDS-locked at 1 block/CU (160KB) so 2 waves/SIMD looked like an OPUS-only lever (256 VGPR ->
VGPR allows 2 waves; cut LDS to 1x tile ~37KB -> 2 blocks/CU). Routed qh64 (NW4) through FORCE_LE2
single-buffer. **Correctness 5/5 pass. Perf: uniformly WORSE -- opus/asm 1.0-1.64 (4x pipe) ->
1.5-1.83 (single-buf)** (clean 20-rep, ctx 5200-32000, B 16-256). Reverted.

Root cause (confirms the older single-buffer-qpack loss): **overlap depth dominates occupancy here.**
Single-buffer = load->wait->compute serial per tile; over ~256 tiles the exposed per-tile load
latency far outweighs the 1->2 wave/SIMD gain. The 4x-deep pipeline at 1 wave is already OPUS's right
structure. A 2x double-buffer (medium overlap + 2 waves) is the only untested middle, but given how
badly 0-overlap lost, reducing pipeline depth below 4x is unlikely to net positive.

### FAILED experiment #3: remove compute-cluster s_barriers (reverted, 2026-06)
ISA-diff (extract OPUS GPU code object via `llvm-objcopy --dump-section=.hip_fatbin` +
`clang-offload-bundler --unbundle` + `llvm-objdump`) showed OPUS qh64 qlen1 emits **48 s_barrier
/ 530 s_waitcnt** vs asm **15 / 121** -- so "OPUS over-syncs" looked like the lever. The
pipelined accum is an 8-cluster 2-deep ping-pong; the 4 compute clusters (compute_qk/compute_pv,
no cooperative LDS write) end with `s_setprio(0); sched_barrier(0); s_barrier(); sched_barrier(0)`.
Removed all 12 such `s_barrier`s (main loop + epilogues, the whole 16mx8 family). **Correctness
144/144 pass** (qh16/32/64/128 x ctx{200..32000} x 3 seeds x sink -- the barriers were genuinely
NOT load-bearing). **But perf was UNCHANGED** (qh64 qlen1 opus/asm 0.89-1.40, == baseline within
noise). Reverted (no benefit + unmotivated change to a shared accum).

**Critical lesson: the static sync COUNT was a red herring.** Removing 48->~24 barriers cost
nothing and gained nothing. At 1 wave/SIMD the cross-warp `s_barrier` over 4 lockstep warps is
cheap; the real stall is the **exposed VMEM->LDS load latency at tile boundaries + waitcnt**, which
the compiler's instruction *schedule* (not the barriers) fails to hide as tightly as asm's hand
schedule. **Test the lever by removing it; never infer the bottleneck from an instruction count.**

## Split-D experiment: the 1-wave/SIMD wall IS breakable (2026-06) — feasibility PROVEN, fast kernel BLOCKED

Pushed past "use asm here" on a free-rein attempt to beat asm in HIP/OPUS. Key chain:
- **Root cause of the 1-wave trap is VGPR, not LDS:** OPUS NW4 qh64 qlen1 uses **vgpr_count=512**
  (256 arch + 256 AGPR) -- the 64-row x 512-dim fp32 O accumulator alone is ~128 VGPR/warp ->
  512 total -> 1 wave/SIMD. asm is in the identical bind. (This is WHY FORCE_LE2 failed: it cut LDS
  but VGPR still pinned 1 wave.)
- **The escape asm can't follow: split the output D.** Each block computes the FULL QK+softmax but
  only a 256-wide V-half (O ~= 64 VGPR). **Measured `vgpr_count = 186`** for the D_V=256 kernel ->
  `floor(512/186)=2` blocks/CU = **2 waves/SIMD is VGPR-feasible** (the 2nd wave fills the 78%
  LDS-wait stall). The softmax denominator is identical across halves -> each block normalizes
  independently, no stage-2 merge. Launched as 2 compile-time `D_SPLIT` instantiations (cols [0,256)
  / [256,512)). asm is LDS+VGPR-locked at 1 wave and CANNOT do this.
- **Correctness PROVEN:** the split-D kernel with a single-buffer (le2) accum is numerically correct
  (48/48 vs fp32 ref, qh64 qlen1, ctx 40..32000, bf16/fp16, sink, 2 seeds).
- **But le2 (single-buffer) is 3-20x SLOWER** (no load/compute overlap over hundreds of tiles; the
  2-wave occupancy does NOT compensate for the lost overlap -- same lesson as FAILED #2). So the win
  needs a PIPELINED (overlap) accum at the low VGPR.
- **The 2-deep double-buffer (`pa_prefill_accum_db2`) NaNs** (all ctx/seeds). The single-tile path is
  logically identical to the proven le2; the bug is purely the cross-tile double-buffer (prologue
  load + prefetch s_kv[1] + std::swap). Tried: std::swap (vs runtime index), page-index hoisted to a
  local, `sched_barrier(0)` fences around prefetch/barrier -- none fixed it. The ping-pong WAR/RAW is
  provably correct by source reading, yet runtime reads garbage -> **a buffer_load_lds -> ds_read
  cross-tile visibility race that needs ATT/LDS-dump instrumentation to resolve** (the exact class the
  NW8 shifted-load notes above hit; not source-readable).

### db2 resolved + split-D DEFINITIVELY a perf dead-end (2026-06, deeper push)
The db2 NaN was NOT a race: localized via single-tile test (ctx=33, 1 tile, no prefetch/swap) which
ALSO NaN'd -> bug was in setup, not double-buffering. Root cause: `__launch_bounds__(BLOCK_SIZE, 2)`
on the split-D kernel forced the compiler to fit 2 blocks/CU (<=256 VGPR), but **db2's natural
`vgpr_count = 272`** -- squeezing it to <=256 produced *corrupt* code (NaN), not just a spill.
Relaxing to `__launch_bounds__(...,1)` -> **db2 correct** (all ctx, max|d|<=0.002).

But the perf is the verdict: db2 at lb=1 (272 VGPR -> 1 block/CU -> **1 wave/SIMD**) is **2.3-16x
SLOWER** than asm (547-3544us). Two compounding penalties: (a) single/shallow buffer loses the
4-deep pipeline's load/compute overlap (the dominant cost, ~10x), (b) split-D duplicates the full QK
across both D-halves. **The 2-wave occupancy goal is unreachable AND wouldn't help:** db2 needs 272
VGPR (can't fit 2 blocks/CU at all), and even a hypothetical 2-wave (<=2x) can't recover the ~10x
overlap-loss penalty.

### Inline-asm scheduling REFUTED by microbench (2026-06) — the compiler is already optimal locally
Tested the "hand-schedule the 1-wave pipeline with inline asm to match asm" hypothesis with two
standalone hipcc microbenches (1 wave/SIMD, 3.2M `v_mfma_f32_16x16x32_bf16`):
- **Interleaving independent MFMA chains: 0 gain** (1.00-1.01x for 2/4/8 accumulators). The MFMA
  unit is **throughput-bound at ~6.78 ns/MFMA** at 1 wave regardless of dependency -- the `s_nop 7`
  after each MFMA is the throughput limit, NOT a fillable bubble.
- **MFMA fed from LDS, compiler-scheduled: only 1.08x over the pure-MFMA floor**; hand-overlapping a
  2nd chain does NOT beat it (1.09x). So the compiler ALREADY hides ds-read latency under MFMA.

=> There is **no local scheduling slack** for inline asm to capture (corrects the earlier "compiler
schedules worse than asm" framing -- that was wrong). The ~1.3x gap is the **non-MFMA work that runs
while the MFMA unit idles** (the softmax/VALU phase between QK and PV, plus OPUS's `v_accvgpr`
accumulator shuffling). Hiding it needs overlapping tile-N softmax under tile-(N+1) MFMA across the
whole loop = hand-scheduling the entire pipeline = *being* the asm kernel. No localized inline-asm
tweak does it; inline asm only "matches asm" by reproducing asm's full hand-written schedule.

### ROOT CAUSE positively measured (2026-06): LDS-read stall (WAIT_LDS), OPUS 2.27x asm
PMC cycle+wait breakdown (qh64 qlen1 B256 ctx8192, OPUS single-pass vs asm; counters summed over
CUs so absolute % inflated, but cross-kernel RATIOS valid -- identical workload/dispatch count):
| | GUI_ACTIVE | MFMA_busy | VALUBusy | WAIT_LDS |
|---|---|---|---|---|
| opus | 1.428e9 | 3.993e10 | 23.3% | 1.034e10 |
| asm  | 1.056e9 | 3.993e10 | 23.8% | 4.548e9 |
| ratio | **1.35x (=perf gap)** | 1.00x | ~same | **opus 2.27x more** |

- MFMA work identical; VALU work identical (refutes the earlier "accvgpr/softmax VALU overhead"
  inference -- that was wrong, VALUBusy is the same).
- Both latency-bound at 1 wave/SIMD (MfmaUtil==LdsUtil==VALUBusy~22%, no unit near 100%, MemStalled~0).
- **OPUS stalls 2.27x more on LDS reads (lgkmcnt) than asm** -- disproportionate to its 1.35x overall
  slowdown -> LDS-read stall is THE dominant contributor.

### Lever A TESTED + refuted: deeper ds_read prefetch is register-bound (2026-06)
Tried the obvious fix -- deepen the intra-tile slice prefetch in the pipelined accum from 2-deep
(`v_k[2]`/`v_v[2]`, prefetch slice idx+2) to 3-deep (warmup-seed slice 2, prefetch idx+3, relax
in-loop `lgkmcnt` to 2x to allow 2 slices in flight). Did K alone, then K+V.
- Correctness 7/7 (qh64/128/qpack).
- **Perf: NO gain** -- qh64 q1 B256 opus/asm 8000/16000/32000 = 1.25/1.12/1.20 vs baseline 1.23/1.11/1.19
  (slightly worse).
- **vgpr_count pinned at 512 both times; private_segment (scratch) grew 100 -> 108 (K) -> 120 B (K+V)**
  -> the compiler SPILLS the extra slice buffers instead of realizing deeper prefetch, because the
  kernel is already at the 512-VGPR ceiling.

=> **Confirms (now measured, not inferred) that the shallow prefetch is register-bound.** You cannot
"just move the ds_read forward" -- moving it forward needs more registers to hold the in-flight read
results, and at 512 VGPR the compiler spills instead -> no benefit. asm fits the deeper prefetch only
via hand register allocation the compiler cannot replicate. Reverted (no gain + spill). Lever A dead.

### Prefetch distance + the AGPR-vs-VGPR register strategy (ISA, 2026-06) — the deepest root
Static dataflow over the disasm (ds_read dest reg -> first consuming instruction, distance in insts):
| LDS read | asm (median/p90/max) | OPUS (median/p90/max) |
|---|---|---|
| `ds_read_b128` (K) | 59 / 193 / 361 | **11** / 17 / 27 |
| `ds_read_b64_tr_b16` (V) | 42 / 203 / 219 | 32 / 42 / 150 |

asm issues K reads a median **59 insts ahead** vs OPUS **11** (5.4x deeper) -> asm's LDS latency is hidden,
OPUS's is exposed (the 2.27x WAIT_LDS). WHY asm can prefetch that deep -- register strategy (ISA-verified):
| | ds_read dest | MFMA inputs | accumulator | v_accvgpr copies |
|---|---|---|---|---|
| asm  | **AGPR** (1770 reads->a, 64->v) | AGPR | VGPR | **0** |
| OPUS | **VGPR** (1200 reads->v, **0->a**) | VGPR | AGPR | **4242** |

asm parks K/V in the 256 AGPRs (deep prefetch, no shuffles); OPUS reads into VGPR + accumulator in AGPR
(needs 4242 v<->a shuffles), filling the 512 VGPR budget -> shallow prefetch.

**Is the compiler INCAPABLE or just doesn't? -> POLICY gap, not capability gap (measured):**
- Compiler-default microbench (MFMA fed from LDS, low pressure): **17/17 reads->VGPR, 0 AGPR, 0 accvgpr**.
  Under pressure (OPUS) it uses AGPR ONLY as accumulator spill, **never** ds_read->AGPR (0/1200).
- Inline-asm capability probe (hipcc gfx950): forcing `ds_read -> AGPR` (`"=a"` constraint) + MFMA from
  AGPR COMPILES and yields `v_mfma_f32_16x16x32_bf16 v[6:9], a[0:3], a[4:7]` = asm's exact layout. So
  ds_read->AGPR **IS expressible** in HIP; the compiler/OPUS-DSL just never auto-chooses it.

**Root cause (fully measured): exposed KV-tile `ds_read` latency at 1 wave/SIMD, caused by the
compiler's register-allocation POLICY** (LDS reads -> VGPR, AGPR = accumulator spill) vs asm's
(reads -> AGPR, accumulator -> VGPR). The policy consumes VGPR -> shallow prefetch -> 2.27x WAIT_LDS
-> 1.35x slower. Matching asm needs the AGPR-read path hand-written (inline asm) or the OPUS `mma`
operand placement re-architected -- i.e. doing the reg-alloc by hand. Not a localized fix. Floor stands. asm issues the LDS reads far
enough ahead of the consuming MFMA to hide the latency (deep hand-scheduled prefetch); the OPUS
compiler issues them closer to use. Deeper prefetch needs more in-flight read results = more VGPR,
and OPUS is already at the 512 ceiling -> compiler can't, asm (hand reg-alloc) fits it. Consistent
with the microbench (MFMA-dense -> compiler hides reads fine; the real kernel is LDS-read-heavy so
prefetch DEPTH decides it). Not MFMA-sched, not VALU, not BW, not occupancy, not barriers.

**DEFINITIVE conclusion: qh64/qh32 qlen1 cannot beat asm from OPUS source.** OPUS faces a hard
trilemma it can't escape: the 64x512 fp32 accumulator forces 512 VGPR -> 1 wave/SIMD; getting 2
waves needs <=256 VGPR (split the accumulator) which forces a shallow pipeline that loses the
overlap worth ~10x; keeping the overlap (4-deep) forces 1 wave where only asm's hand schedule
extracts the last ~1.3x. **overlap >> occupancy** here, conclusively. asm wins by hand-scheduling
the 1-wave deep pipeline -- not reproducible from HIP/OPUS source. This is the floor; route pure
qlen1 large-B to asm.

The split-D / db2 / dv256 scaffolding is left as dead code (NOT dispatched; qh64 qlen1 on the
correct normal pipelined path; shared le2 `V_SKV_OFF=0` default verified safe, 11/11 regression).

### Bottom line on qh32/qh64 qlen1 large-B (levers exhausted)
The residual ~1.2-1.7x vs asm is **asm's hand-scheduled single-wave efficiency** (29% vs 21.6%
MfmaUtil), NOT occupancy, layout, memory, OR barrier count. **Four source-side levers now
empirically falsified**: (1) head-padding T_M, (2) single-buffer occupancy, (3) compute-cluster
barrier removal, (4) NW8 sched-hints (prior session) -- none moved it. The bottleneck is the
COMPILER's instruction schedule of the waitcnt/VMEM-latency-bound 1-wave pipeline; asm's hand
schedule hides the tile-boundary load latency tighter. So this is **not cheaply closable from OPUS
source** (would need inline-asm-level scheduling inside the HIP kernel, or a deeper-prefetch
pipeline rewrite). This is precisely WHY aiter ships asm for these hot decode shapes. Pragmatic options:
(a) accept it and route pure qlen1 large-B decode to the asm kernel (OPUS already wins qlen>=2 via
qpack and small-batch via split-KV); (b) a deep hand-scheduling pass on the 4x pipeline (high effort,
low confidence per the NW8 precedent). qh16/qh32/qh64 qlen1 all share this same scheduling-bound gap.
