# topk-prefill: falsified directions and traps

Append-only. Every entry carries the number that killed it. MI355X / gfx950,
ROCm 10.0, shape M=4096 N=131072 fp32 K=2048 unless stated otherwise.

## Correctness traps (each one produced wrong results)

### fp32 two-pass 8-bit radix covers only half the key
**Symptom:** 63 of 64 rows failed the multiset check against `torch.topk`.
**Cause:** ported the fp16 structure (MSB@8 + LSB@0) to fp32 as MSB@24 + LSB@0,
leaving bits [23:8] unconstrained.
**Fix:** four passes at shifts {24,16,8,0}.

### row-per-block kernel launched with `(M+255)/256` blocks
**Symptom:** only the first few rows had a threshold; the rest were garbage.
**Cause:** the scan/threshold kernels index rows by `blockIdx.x` but were
launched with a thread-count-style grid.
**Fix:** launch `<<<M, ...>>>`.

### Phase B silently truncating an overflowing candidate area
**Symptom:** none on the shipped distribution -- this is the dangerous one.
When a row's passer count exceeded the per-row candidate area, Phase B dropped
the surplus, so Phase C selected from an arbitrary subset and could return a
wrong top-k while every count still looked plausible.
**Fix:** a row is unusable if `cand_count < K` **OR** `> capacity`; both route to
the exact fallback. The adversarial distribution exercises this path (all 4096
rows overflow and fall back, and still pass).

### static `__shared__` sized to the maximum knob value
**Symptom:** 0.774 -> 0.852 ms regression from a change that should have been
free.
**Cause:** `__shared__ uint32_t s_keys[SAMPLE_S_MAX]` costs 64 KB of LDS
regardless of the runtime sample count, crushing occupancy.
**Fix:** dynamic LDS sized at launch.

## Performance directions that did not work

### Raising Phase B's blocks-per-row
gx=2 0.925, gx=4 1.253, gx=8 1.497, gx=16 2.025, gx=32 2.641, gx=64 3.422 ms.
Monotonically worse. All those blocks contend on the same `cand_count[row]`
global atomic. A pure read at gx=1 reaches 6.08 TB/s, so this was never a
bandwidth question. **One block per row, owning that row's candidate area, needs
no global atomic at all.**

### Non-temporal loads in the streaming filter
`__builtin_nontemporal_load` on the filter's dwordx4: 0.7930 vs 0.7916 ms.
Neutral. The pass is already pure streaming and L2 is not the constraint.
(Inline `global_load_dwordx4 ... glc slc` asm does not assemble on gfx950/ROCm
10 -- use the builtin, not asm.)

### Block-aggregated global atomics
One atomic per block-iteration instead of per wave removed the gx sensitivity
(flat 0.905-0.915 ms for gx=2..8) but was still slower than gx=1 with per-wave
atomics (0.858 ms). Reducing atomic *contention* is not as good as removing the
atomic entirely.

### Ballot-based gather in Phase C
Replacing the per-element `atomicAdd(&s_wgt,1)` with one LDS atomic per wave:
102.1 -> 104.2 us. No effect. The single-address LDS atomic was not Phase C's
bottleneck -- its radix passes are.

### Common-prefix pass skipping in Phase A
Phase C -6.0 us, Phase A **+8.0 us**. Phase C's candidates all sit above the
threshold so they share a high prefix and passes really are skippable; Phase A's
samples span the whole row, so no pass is ever skipped and the min/max reduction
is pure overhead. Shipped for Phase C only.

### `HIST_REPLICAS=8`
R=1 0.667, R=2 0.640, R=4 0.626, R=8 0.648 ms. Past 4 the extra LDS costs more
than the reduced atomic conflict buys.

### Phase A with 2 radix passes
0.6144 vs 0.6194 ms, so 0.8% faster, but the candidate spread runs to max=3919
against a capacity of 4096 (4.3% headroom, vs 9.6% at 3 passes). Overflow is
correct-but-slow rather than wrong, so this is a robustness trade, not a
correctness one. Not shipped: 0.8% is not worth the tail risk.
Note 3 passes give **bit-identical** candidate counts to 4, so the 4th byte
never moves the bucket at fp32 precision.

### Reducing the sample count back to S=4096 after the pipeline got faster
0.7043 vs 0.6246 ms. The trade-off did not flip: the 4 rows that fall back cost
far more than the extra 67 MB of sampling.

## Carried over from /home/mh/gemm-topk/knowledge/known_bad.md (not retried)

- HIP graph capture to remove launch overhead: neutral-or-worse there, and here
  the pipeline is 4 launches with the GPU busy, so launch is not the constraint.
- Collapsing parallel multi-CU work into a single block: 4.2% regression there;
  reproduced in spirit here by `one_block_per_row` on the old multi-block filter
  (+30%).
- Computing both `>=` and `>` per element: pure ALU tax.

## Structural facts worth keeping

### A read-only floor understates a filter kernel
Read 2.147 GB = 0.3516 ms (6.11 TB/s), but read 2.147 GB **+ write 94 MB** =
0.4361 ms. The 94 MB of candidates cost 87 us on their own. Judging Phase B
against the read-only number said 1.37x off; against the real floor it is 1.10x
and essentially finished. Always measure the floor with the write traffic in it.

### Candidate-count spread is estimator noise, and it scales as 1/sqrt(rank)
The count of elements above the rank-R value of S samples has std ~ count/sqrt(R).
At S=8192, margin 1.4, measured min/mean and rows under K (of 4096):
K=2048 R=179 -> 0.74, 0 rows; K=1024 R=89 -> 0.67, 4 rows; K=512 R=44 -> 0.52,
82 rows. A constant margin is therefore overfitted to one K. Solving
`mean*(1 - 3/sqrt(R0)) > K` gives 1.36 at K=2048 and 2.13 at K=512 and removed
the fallback at every K (K=512 guard: 0.7326 -> 0.5725 ms).

### A sortable fp32's top byte is sign+exponent, so histograms collide hard
On uniform[-1,1] roughly half of all positive values land in ONE of the 256
buckets, and the per-element LDS `atomicAdd` serialises. Replicating each bucket
across 4 adjacent counters (adjacent => different LDS banks) and summing before
the scan: 0.667 -> 0.626 ms.

### Small scattered stores, not the compare, are what a filter pass pays for
Ablation of the filter kernel: full 500.7 us, stores removed 361.9 us, all
compaction removed 361.1 us, read floor 349.0 us. So the load + compare +
ballot/popcount compaction cost 13 us, and the candidate stores cost 139 us for
94 MB. A wave produces ~5.6 passers per iteration, so each burst was ~45 B
against a 128 B line. Staging in LDS and flushing at >= 64 entries (~520 B
contiguous) recovered part of it (500 -> 480 us).

### Strided sampling would cost a full pass
Sampling one element in 32 fetches a whole 128 B line per useful float. Use
contiguous chunks (64 floats = 256 B) so DRAM traffic equals useful bytes, and
buy sample independence with more chunks rather than with stride.

### Superseded architecture: DeepSelect survivor buffer
Pre-AVO measurement: the bf16 k=512 HIP port of DeepSelect on this exact
M,N was 1874 us with 118 KB of LDS, pinning residency at 1 workgroup/CU. Not a
tuning problem -- keep the per-block LDS small enough for real occupancy.

### The max-sized-static-LDS trap, hit a second time (generalization round)
Same root cause as "static `__shared__` sized to the maximum knob value" above,
so it is now twice-measured. Generalizing Phase C to `cap in {4096, 8192}` with
`__shared__ int s_idx[PHASE_C_CAP_MAX]` plus a separate `s_keys_small[PHASE_C_CAP]`
put the kernel at **54560 B** LDS vs the baseline **38048 B**, halving blocks/CU:
main shape 0.6215 -> **0.6702 ms** (+7.8%), stddev 0.05%, reproduced 4x.
Fix: template on a compile-time `STATIC_CAP`; the common config keeps
compile-time-sized arrays, only the rare large cap pays a dynamic-LDS split.
Making the *common* config pay the runtime split instead costs +0.65%
(0.6254 vs 0.6215 ms) because `s_dyn + cap` is not a constant base offset —
measured, not assumed. Corollary: an identity index array is pure LDS waste
(`phase_small_n_topk` 38048 -> 5280 B by using slot==column).

### Unconditional memsets for a path-specific buffer cost a dispatch each
Clearing `cand_reserved`/`cand_bad` on every call, when only the cooperative
filter reads them, added 2 `hipMemsetAsync` dispatches per iteration and cost
the prefill path ~4 us (0.6216 -> 0.6257 ms). At ~2.6 us/launch on this box,
two stray dispatches are 0.65% of a 622 us kernel. Gate per-path scratch clears
on the path that uses them.

### A correctness gate that silently skips its oracle
`bench/correctness.py` returns early from `run_gates()` if torch is unimportable,
and the host Python is 3.6 with no torch. So every "CORRECTNESS PASS" printed on
the host had **never run the torch.topk comparison** -- the gate was vacuous and
still green. Found only by asserting `HAS_TORCH` in a separate self-test.
Fix: `bench/gate_selftest.py` asserts torch is present, then requires 3 injected
corruptions (non-top-k index, duplicate index, out-of-range index) to be
rejected; run the suite inside a torch image. A gate you have not seen go red is
not evidence.

### Sizing a barrier-bound block from its element count
small_n used `block = min(1024, roundup(N/64)*64)`. At N=1024 that is 16 waves
where only 256 of 1024 lanes issue a load, and all 16 waves still pay the fixed
4-pass x 4-barrier select over 1024 elements. M=4096 N=1024: 69.5 us at block
1024 vs **29.8 us at block 256**.
The deciding quantity is waves RESIDENT PER CU, capped by the kernel's dynamic
LDS -- not the load count. And one term is not enough: M=4096 N=8192 and
M=1024 N=1024 both give blocks_per_cu=4 yet want 8 and 4 waves, so residency
alone cannot separate them; a second term caps waves by what the row's loads can
occupy. See `occupancy_block_threads()`. Sizing purely by load count picks 512
at M=4096 N=2048 (50.2 us) where 256 measures 39.7 us.

### Zeroing a buffer that is assigned, not accumulated
`hipMemsetAsync(cand_count)` ran on every call, but every Phase B variant
ASSIGNS `cand_count[row]` with one block per row and no early return, so the
memset was always dead. Same for per-path scratch: clearing `cand_reserved` /
`cand_bad` unconditionally cost the prefill path ~4 us (0.6216 -> 0.6257 ms)
for buffers only the cooperative filter reads.
At ~2.6 us per dispatch on this box, each stray memset is 0.4% of a 622 us
kernel and ~8% of a 33 us one. Decode was running 9 dispatches of which 5 were
memsets; folding the clears into Phase A (already first on the stream) and
folding `finalize_coop_counts` + `phase_d` into Phase C took it to 3.

### Trusting a knob swept at only one M
`--phase-a-passes 2` looks free at M=1 N=1M (31.4 vs 31.8 us, and candidate
stats clean). At M=1024 N=1M the same setting gives `over_Calloc=3` and the wall
goes **949 -> 1917 us**, because the coarser 2-pass threshold collects far more
candidates (min 6161 vs 4817) and overflows the cap. Sweep a
correctness-affecting knob across the axis that changes its statistics, and read
`--dump-stats`, not just the clock.

### Cooperative Phase B: unbounded LDS staging buffer (was misdiagnosed as "coop_g=2 is broken")
**The earlier entry here blamed a "reservation race when G=2" and worked around
it by snapping coop_g to {1,4,16,64,256,1024}. That diagnosis was wrong** and
the workaround hid a real out-of-bounds write that could fire at ANY G.

Real fault: `phase_b_filter_coop` accumulated the block's entire chunk into the
per-wave LDS staging buffer `buf[bcnt + ...]` with **no bound check against
WSTAGE_CAP=320**. The non-cooperative `phase_b_filter_wavestage` flushes at
`bcnt >= WAVE_SIZE`; the cooperative version was written without that flush.
Any wave producing more than 320 passers wrote into the next wave's slice, and
the last wave wrote past `wbuf` onto `s_local` / `s_block_total` / `s_off`.

Symptom that exposed it: M=128 N=65536 `--dist adversarial`, where the passers
concentrate in the last chunk so one wave sees ~382 > 320. Result
`rows_fail=4` with 124 rows showing garbage-inflated counts, and **the counts
changed between identical runs** because the corruption is race-dependent. The
true count is 3059 (< cap 4096) and no row should have fallen back at all.

Second bug in the same kernel: it looped `for (i = i0 + threadIdx.x; i < i1;
i += stride)`, which lets the tail iteration run with only some lanes active.
`__ballot` then sees only active lanes, so `bcnt` stops being wave-uniform and
every flush offset is wrong. Use the `iters` + `live` mask form.

Fix: reserve-then-flush per wave (one `atomicAdd` on `cand_reserved[row]` per
>=64 staged candidates, ~75 per row at cap 4785, negligible), plus the `live`
mask. After the fix G in {1,2,4,8,16,64,256} all report the same 3059 with
`over_Calloc=0 rows_fail=0`, so the snap restriction was removed.

**Lesson: two separate instances of "this config is broken, avoid that config"
turned out to be one unchecked buffer.** Route around a correctness failure only
after you have the root cause; a G-value blacklist looked like a fix and shipped
an out-of-bounds write.

### One achievable-bandwidth anchor over-states the floor at large N
The floor model scaled ideal-at-peak by a single `achievable_scale` measured at
M=4096 N=131072 (1.294). That produced **0.94x at M=4096 N=1048576** -- below its
own floor, which is impossible, so it proves the model wrong rather than the
kernel fast. This is the self-check the model already documented, and it fired.

Cause: the fraction of peak a shape reaches depends on how much contiguous data
ONE block streams, i.e. N*4 bytes, not on the total payload. Measured with the
read+write floor kernel:

| N | bytes per block | achieved | scale vs 6.63 TB/s peak |
|---|---|---|---|
| 16384 | 64 KB | 4.87 TB/s | 1.246 |
| 131072 | 512 KB | 5.06 TB/s | 1.290 |
| 1048576 | 4 MB | 5.33 TB/s | 1.241 |

Fix: anchor at three N and interpolate in log2(N). That took the bad point from
0.94x to 0.979x. The residual is a weak dependence on GRID size as well -- at the
same N=1M, the M=1024 anchor gives 1.241 while M=4096 actually achieves 1.217,
because more blocks hide memory latency better. Modelling that second axis was
judged not worth it for a ranking column; the model is stated as +/-3% and a
reading at 0.98x is reported as "at the floor", not as beating it.

Second defect found in the same place: **`scripts/floor_model.py` rebuilt its
output dict by hand and silently dropped both `bw_sweep` and the anchor list**,
so the parser could collect several anchors and the model could still only ever
see one. A hand-rebuilt dict next to a parser that returns more than it needs is
how a model loses the data it was measured from.

### Cooperative grid.sync() costs 7-196 us, which kills multi-block select
`arch_scope: gfx950`, ROCm 10.0, MI355X. Measured with
[`scripts/coop_spike.hip`](../scripts/coop_spike.hip): a kernel doing 36 bare
`cg::this_grid().sync()` calls and nothing else, minus the same kernel doing
zero, divided by 36.

| block | grid | us per grid.sync() |
|---|---|---|
| 256 | 64 | **7.38** |
| 256 | 256 | **24.96** |
| 256 | 2048 (max) | **196.30** |
| 512 | 1024 (max) | **101.43** |

Cooperative launch itself is fine: `cooperativeLaunch=1`, max co-resident grid
is 2048 blocks at 256 threads / 1024 at 512 / 512 at 1024, an oversized grid is
rejected with error 720 rather than deadlocking, and grid.sync() gives correct
results.

It is the price that kills it. Spreading phase_a's 3-pass radix select over many
blocks needs ~2 grid syncs per pass = 6 total, against an ~11 us target. Even at
the cheapest grid=64 that is 44 us -- 4x more than the work being parallelized.
**Do not plan a multi-pass cooperative kernel on this stack for a target under
~100 us.** One kernel launch per pass (2.63 us each) is far cheaper than a grid
sync at any useful grid size.

### More histogram replicas do not help the radix select
`HIST_REPLICAS` exists to spread LDS atomic contention, since a sortable fp32's
top byte is low-entropy. Raising it from 4 is a loss:

| | R=4 | R=8 | R=16 |
|---|---|---|---|
| geomean over 10 shapes | 75.38 us | 77.13 (+2.3%) | 80.87 (+7.3%) |
| M=4096 N=8192 | 97.1 us | 113.7 | 119.3 |
| M=1 N=1048576 | 32.0 us | 31.8 (-0.6%) | 32.0 |

Small-M shapes gain 0.5% at best while large-M shapes lose badly, because
HIST_SLOTS = 256*R grows the static LDS and costs blocks/CU.

More useful than the tuning answer: the 0.5% ceiling at M=1 **rules out atomic
contention as phase_a's bottleneck.** Combined with phase_a measuring ~12 us at
both M=1 (1 block) and M=64 (64 blocks), its cost is the serial depth of one
row -- 3 passes x ~4 barriers over S keys in LDS -- not throughput. Adding
blocks per row cannot fix a latency-bound chain.

### Early-exiting the radix select once the pivot is pinned (small_n)
Priced first, which is why the attempt was made: an ablation with
`--small-n-passes 1..4` shows passes 2-4 cost **30-34% of phase_small_n_topk**
on all 7 shapes tried, while carrying almost no real work -- they only
histogram the elements matching the current pivot prefix.

Attempt: accumulate min/max of the ACTIVE set inside the histogram loop (those
LDS reads already happen, so the accumulation looked free) and break out once
they agree, because the pivot is then fully determined.

Result: **21-39% SLOWER**, and the anchor 615.5 -> 662.8 us.

| shape | before | with early exit |
|---|---|---|
| M=1024 N=8192 | 31.4 us | 43.6 us (+38.9%) |
| M=2048 N=4096 | 37.3 us | 49.8 us (+33.5%) |
| M=4096 N=8192 | 99.2 us | 130.2 us (+31.3%) |

The accumulation is free; the REDUCTION is not. Each check needs a block-wide
min/max (two barriers plus a shuffle butterfly), paid on every pass, against a
saving of at most one pass. On uniform fp32 the active set only collapses after
pass 3, so the exit fires at most once while the check is paid three times.
Lesson: pricing the thing you want to remove (the ablation) is not the same as
pricing the mechanism that removes it.

### A statistically tighter margin fails the gate: the sigma must grow with M
`auto_margin()` derives the over-collection factor from the margin-FREE rank
`R0 = K*S/N`, while the estimator really runs at `R = margin*K*S/N`. That looks
like an off-by-one in the model, and fixing it is WRONG.

Solving the self-consistent fixed point (`x^2 - (3/sqrt(c))x - 1 = 0` with
`c = K*S/N`, `margin = x^2`) gives a smaller margin -- 2.129 -> 1.839 at
N=1048576 -- and immediately produced **under_K=1 at M=4096 N=1048576**, a row
short of K, where the original gives under_K=0.

The reason: the gate is "no row of M undershoots", i.e. a MAXIMUM over M draws,
so the sigma that matters grows with M, roughly sqrt(2 ln M). Back-solved from
the measured spread at the main shape, the deepest of 4096 rows sits **3.4-3.5
sigma** below the mean, not 3.0. Using R0 instead of R inflates the noise
estimate by about the right amount, so the "wrong" rank silently supplies the
M-dependence. Any tighter formula has to put that dependence back explicitly.

### Deriving S from the acceptance window instead of R_TARGET=179
`R_TARGET = 179` is reverse-engineered from ONE shape (cap/K = 2) and provably
does not transfer: at N=1048576 (cap/K = 4) it asks for 43000 samples and gets
clamped. Replacing it with "smallest S whose 3-sigma candidate window still fits
under cap" is cleaner and **still loses**:

| shape | R_TARGET | derived | delta |
|---|---|---|---|
| M=1 N=262144 | 28.1 us | 26.0 us | -7.5% |
| M=8 N=524288 | 35.9 us | 33.2 us | -7.5% |
| M=64 N=262144 | 40.9 us | 48.0 us | **+17.4%** |
| M=4096 N=131072 (anchor) | 615.5 us | 639.4 us | **+3.9%, over the 620 us limit** |

Geomean over 21 points: -0.91%. A smaller S forces a larger margin and a larger
cap, so it trades phase_a work for candidate volume: at small M phase_a's
single-block cost dominates and the trade pays, at large M the extra ~52% of
candidates that Phase B writes and Phase C selects costs more. An M-dependent
switch between the two rules would capture ~0.3% of the outer geomean, which
does not pay for the complexity. Kept behind `--s-rule 1` for reproduction only.

Also learned here: **S is quantized to powers of two.** The sampling geometry
needs `(N / (S/64)) % 4 == 0`, so for a power-of-two N the chunk count must be a
power of two too; `--sample-s 12288` and `6144` are silently rejected and fall
back to SAMPLE_S_MAX.

### A hand-written header list in the Makefile silently measured a stale binary
`benchmark_topk: benchmark_topk.hip.cpp csrc/topk_common.hip.hpp` did not list
`csrc/topk_shape.hip.hpp` or `csrc/topk_generalize.hip.hpp`, so editing either
of those and running `make && ./benchmark_topk` re-ran the PREVIOUS binary with
no warning. Caught only because the stale binary's race gave two different
answers for one config. Fixed with compiler-generated deps (`-MMD -MP -MF`),
which cannot fall out of date. Verified: `touch csrc/topk_generalize.hip.hpp`
now triggers exactly one rebuild.

### hipGraph on the 4-kernel pipeline (launch-bound regime retest)
Retested in the regime the plan called out (launch-bound small M), and it is
**measurably worse**, so the direction stays dead — but for a different reason
than the first attempt reported.

First attempt blamed `hipMemsetAsync` inside the captured region for error 900.
**That attribution was wrong.** Capture still failed at 900 after every memset
was removed from the pipeline. The actual cause is capturing on the **legacy
default stream**: `hipStreamBeginCapture(0, ...)` returns 900 "operation not
permitted when stream is capturing" whatever the region contains. Capturing on
a `hipStreamCreateWithFlags(hipStreamNonBlocking)` stream succeeds immediately.

With capture actually working, graph-on vs graph-off (warmup 10, iters 50,
repeats 3):

| shape | off | on | delta |
|---|---|---|---|
| M=1 N=1M | 40.4 us | 46.8 us | **+15.8%** |
| M=8 N=1M | 48.0 us | 55.2 us | +15.0% |
| M=64 N=131072 | 38.4 us | 45.5 us | +18.5% |
| M=4096 N=131072 | 617.8 us | 624.1 us | +1.0% |

The penalty is a roughly fixed +6 to +7 us at every shape, i.e. one
`hipGraphLaunch` costs more than the 4 individual launches it replaces (4
sequential empty launches measure 10.5 us total on this box). 4 nodes is too few
to amortize graph launch. Keep `--hipgraph 0`.

### Sampling geometry: one predicate was answering two questions
Servability ("can the sampler read this (N,S)?") and preference ("is this the S
we want?") were the same function, `sampling_geometry_ok`. Relaxing it so that
non-pow2 widths become servable therefore also changed which S the pow2 grid
picks: shapes the old rule bumped up to `SAMPLE_S_MAX` via the repair kept the
smaller S the R_TARGET law asks for, and the **decode geomean went 30.90 ->
31.15 us (+0.82%)**, reproduced on 3 consecutive outer-tier runs with per-point
sd 0.04-0.08%, then confirmed by an interleaved old/new binary A/B on the same
machine (old 30.90/30.90, new 31.15/31.15). The S the old rule forced was simply
the better one. Fix: `sample_stride_exact` keeps deciding the choice,
`sampling_geometry_ok` only decides what can be served.

Worth stating because the first re-measurement looked like drift and was not:
the within-run sd is ~0.05% while the between-run spread on this box is also
small, so a 0.8% regime shift is real and has to be attributed, not averaged
away.

### chunk_stride belongs on the host, not in the sampler
`phase_a_threshold` computed `chunk_stride = N / chunks` itself -- an integer
division in device code for a value the host already had. Adding the alignment
mask there cost **0.8% of the decode geomean** for an arithmetically identical
result on every pow2 shape. Passing it as a kernarg instead removed both the
division and the mask and came out **faster than before the change** (outer
geomean 42.66-42.67 us vs 42.71-42.72 us old, decode 30.84 vs 30.90, two
interleaved A/B rounds). Generalizing a kernel is not automatically a cost: the
host knows more than the kernel and should be made to say it.

### aiter's JIT does not rebuild when a generated source changes
After `scripts/export_aiter_op.py` rewrote `topk_per_row_avo_kernels.cu`, the
next `import aiter` reused `aiter/jit/module_top_k_per_row.so` from 42 minutes
earlier and reported the pre-change verdict (`topk_avo_supports(256, 131328,
2048) == False`) for code that had already been fixed. The `.so` mtime being
older than the `.cu` mtime is the check. Delete
`aiter/jit/module_top_k_per_row.so` and `aiter/jit/build/module_top_k_per_row`
(from inside the container -- they are root-owned) after every export. Same
failure class as the Makefile header-dependency bug above, one layer out.

### `--fuse-ab 1` faults on any coop_g > 1 shape (pre-existing, not fixed)
`./benchmark_topk --mode verify --m 64 --n 131072 --topk 2048 --fuse-ab 1` dies
with HIP error 700 (illegal memory access). `phase_ab_fused` does not initialize
`cand_reserved` / `cand_bad` -- only `phase_a_threshold` does -- and
`phase_c_select_contig` then reads them on the coop path. Confirmed pre-existing
by running the same command on a pre-change binary, which faults identically;
coop_g=1 shapes pass on both. The knob is a diagnostic, defaults off, and is
left broken rather than half-fixed, but it must not be trusted for ablations.

### Ragged rows need per-row extent, not just pitch
aiter passes `stride0` as row pitch and `rowEnds[row]` as the exclusive end.
Selecting over the full pitch on a triangular matrix (`row_len = row + 1`) pulls
tail garbage into the candidate set and emits indices outside `[0, row_len)`.
Rows shorter than K must emit `min(K, row_len)` indices then `-1` padding.
Fix: `template <bool RAGGED>` plus `row_ends` kernarg; uniform launches pass
`nullptr` and compile the old path unchanged. Short rows (`row_len < max(S,K)`)
route through `phase_a_threshold` to the exact path via `threshold_f = +inf`.
2010/2010 expanded-grid verifies green (402 shapes x 5 distributions); inner
geomean 64.53 us vs 64.81 us baseline (-0.43%).

### A short row needs no select at all -- copy aiter, do not out-think it
First ragged version routed every `row_len <= K` row through the exact radix
select. Correct, but aiter does not: both of its kernels special-case
`row_len <= k` and emit the columns in index order with a `-1` tail
(`topk_per_row_kernels.cu:398` mb path, `:2241` ob path), because when every
element is selected there is nothing to rank. Adopting the same identity emit
took the triangular 4096-row k=2048 case **46.21 -> 42.40 us** and k=512
**44.34 -> 40.98 us**, against aiter's 47.36 / 67.04.

The convention is load-bearing, not cosmetic: aiter picks between its two
kernels with a perf heuristic (`should_use_mulblocks`), so both write the SAME
padding, and a third implementation that ordered short rows differently would
give the same call a different meaning at a batch-size boundary.

### `k > stride0` is servable, and aiter has no guard against it
`topk_avo_supports` refused `k > stride0`, so aiter's own default prefill config
declined at num_rows 64/256/1024 (`unsupported geometry`, stride0 = num_rows
there). aiter's `top_k_per_row_prefill` has no such check -- with ragged rows
`k > stride0` just means EVERY row is the identity case above. Fix: the geometry
is sized by `geometry_k_ragged(K, N) = min(K, N)` (one definition in
`topk_shape.hip.hpp`, used by both the harness dispatcher and the aiter entry),
because the sampler only ever has to serve `min(K, row_len) <= min(K, N)`.
Passing the raw K instead asked `derive_shape_params` for a candidate capacity
that cannot exist, which is what produced the refusal.

### the ragged prefix decides WHICH path runs, so prefix 0 tests half the kernel
`row_ends[r] = r + 1` bounds every extent by M, so at the S the prefill path
derives (8192) every row is under `max(S, K)` and takes the identity/exact
route: `M=512 N=131072` reported `fallback_rows` for all 512 rows, meaning the
SAMPLER never saw a ragged row. The ragged gate was green while the ragged
sampler path was entirely unexercised.

aiter's real prefill config is `num_prefix=131072`, where every extent is long
(131073..131328) and ragged by up to num_rows. Those are two disjoint paths, so
`bench/grid.py` `RAGGED` carries the prefix per shape and runs both; with
prefix 131072 `fallback_rows == 0`, which is the check that the sampler is the
thing being tested.

Verified failable before trusting a pass: `--inject-fault 1` gives
`rows_fail=1` on all three regimes (prefix-0 identity, prefix-131072 non-pow2
sampler, prefix-131072 large sampler).

### the GPU oracle is not independent on a ragged row
`phase_d_fallback` shares `exact_row_select` -- including the `row_len <= K`
identity emit -- with the kernel under test, so both sides use the same
`row_len`. If `row_len_dev` itself were wrong they would agree and the check
would pass. Ragged points therefore run `--verify-oracle cpu`, which recomputes
the extent on the host from its own `row_ends`.

### fb_rows overflowed when two phases both appended the same row
Phase A's ragged routing appended the row to `fb_rows` AND Phase C appended
every row it routed, so a short row was counted twice against an M-entry
buffer: `M=512` triangular reported `fallback_rows=1024` and wrote 512 ints past
the end (at `M=4096` the count came back 3269, i.e. already corrupted). Silent,
because `fb_rows` is diagnostics-only and the clobbered region gets rewritten.
Fix: Phase A marks the row with `threshold_f = +inf` and nothing else -- that
alone starves Phase B, drops cand_count under k_out and makes Phase C route it.

Related: Phase C now routes `len <= K` unconditionally instead of relying on
`cand_count < k_out`, because under the `inf` distribution a row of +inf values
passes the +inf threshold and can push cand_count above k_out, which would send
an identity row down the candidate path and order it differently to aiter.

### triangular test data must clamp row_ends to the pitch
`row_ends[r] = r + 1` faults with HIP 700 as soon as `M > N` (M=4096 N=512: row
512 onward claims an extent past the allocation). aiter's
`create_row_boundaries` cannot hit this because it sizes the matrix at
`max(row_ends)`. A harness bug, not a kernel bug -- but it presents as an
illegal access inside the kernel, so check the generator first.
