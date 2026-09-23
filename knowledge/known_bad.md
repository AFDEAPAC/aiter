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
After `scripts/export_aiter_op.py` rewrote `topk_per_row_sampled_kernels.cu`, the
next `import aiter` reused `aiter/jit/module_top_k_per_row.so` from 42 minutes
earlier and reported the pre-change verdict (`topk_sampled_supports(256, 131328,
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
`topk_sampled_supports` refused `k > stride0`, so aiter's own default prefill config
declined at num_rows 64/256/1024 (`unsupported geometry`, stride0 = num_rows
there). aiter's `top_k_per_row_prefill` has no such check -- with ragged rows
`k > stride0` just means EVERY row is the identity case above. Fix: the geometry
is sized by `geometry_k_ragged(K, N) = min(K, N)` (one definition in
`topk_shape.hip.hpp`, used by both the harness dispatcher and the aiter entry),
because the sampler only ever has to serve `min(K, row_len) <= min(K, N)`.
Passing the raw K instead asked `derive_shape_params` for a candidate capacity
that cannot exist, which is what produced the refusal.

### the "M=256 is only 1.02x" figure was a measurement artifact
Chasing it cost several dead ends, all of them from comparing across harnesses.
The 1.02x came from the op test's `us` column, which is `@perftest()`; the
standalone benchmark reported 69.1 us for what @perftest called 81.5 us, and
while the harness, the timer AND the data generator all differed, no comparison
between them meant anything. Two specific wrong turns: `--dist gaussian` was
used as a stand-in for `torch.randn` and is not one (M=4096 measured 797 us
against aiter's 578, i.e. the hash-based Box-Muller in `fill_random_fp32`
produces a different candidate distribution); and two attempts to profile the
aiter path with rocprofv3 failed, once on a stale trace and once because
importing `op_tests/test_topk_per_row.py` RUNS ITS WHOLE SWEEP at import time
(it has top-level `parse_args()`), which also makes `sys.argv` tricks useless --
inline the helpers instead.

Fixed by `bench/aiter_ab.py`: same harness, same data, one timer, times only the
op. With that, M=256 is **1.19x**, not 1.02x.

The residual dip is real but second-order, and it is NOT a path switch.
Measured ratio vs aiter across rows at width ~131k: 1.58x (64), 1.52x (128),
1.31x (192), **1.19x (256)**, 1.25x (320), 1.41x (512), 1.48x (1024), 1.65x
(2048), 1.46x (4096) -- a shallow V centred on 256. `should_use_mulblocks`
(topk_per_row_kernels.cu:2648) shows why it cannot be a dispatch effect: on a
256-CU part it takes the multi-block path only for `batch_size <= 64` with
`seq_len >= 131072`, and `batch_size > 128` returns false outright, so every
point from M=128 up is one-block-per-row on BOTH sides. Both are amortizing
fixed per-row cost and aiter's curve happens to fall faster over 192->256
(+8.9% time for +33% rows, against avo's +19.2%). avo's own per-row cost is
monotone with no knee at the CU count: 0.525, 0.298, 0.266, 0.268, 0.244,
0.206, 0.183, 0.170, 0.158, 0.140 us/row at M=64..2048, so there is no
1-block-per-CU cliff to find.

### one unused kernarg cost 1% of the small_n geomean
Adding `values` as `template <bool WRITE_VALUES>` compiles the stores out of the
no-values instantiation, and the resource remark confirmed it: `<false,false>`
of `phase_small_n_topk` kept VGPRs 43, occupancy 8, zero spills. It was still
**+1.0% slower** (small_n geomean 28.51/28.54 -> 28.83/28.86 us, two runs each).

Attributed by experiment, not by reading: adding a SINGLE unused
`float* dummy_val` kernarg to `phase_small_n_topk` on the pre-values tree, with
no other change, reproduced the whole regression (28.82/28.85). So the cost is
the kernarg itself -- small_n runs one short block per row, so its prologue is a
real share of the kernel -- not code size, not registers, not the stores.

Fix: bundle the outputs in `TopkOut<WRITE_VALUES>` (`topk_common.hip.hpp`), a
struct holding one pointer when false and two when true. `TopkOut<false>` is
8 bytes, exactly the `int* out_idx` it replaces, so every later argument keeps
its old offset. SGPRs went back to 74 for `<false,false>` (76 for `<false,true>`)
and small_n to 28.54/28.64/28.58, inside the band on 3 runs.

**The general lesson: a template parameter makes the BODY free, not the
SIGNATURE.** On a kernel whose cost is dominated by per-block fixed work, check
the kernarg layout too, and keep the disabled instantiation byte-identical
rather than merely branch-free.

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

### nonzero rowStarts needs RowExtents and costs ~1.5% on uniform small_n
Implementing rowStarts as absolute column indices (`out = rowStart + local`) required
`RowExtents<RAGGED>`: 8 B `{nullptr}` on the uniform path (same slot as the old
`row_ends` pointer) and 16 B `{starts, ends}` on ragged. Even with `if constexpr
(RAGGED)` on the gather lambda and `len = pitch` on the uniform body, inner-tier
small_n geomean moved 28.55 -> 28.98 us (+1.5%, 3 runs) while rowStarts shapes
pass the CPU oracle. Treat as the price of closing the silent-wrong-answer hazard;
refresh the inner baseline at g_10 rather than leaving every future candidate
 fighting a false reject.

### N=4K-8K small_n shortfall is LDS-full-row staging at fixed block geometry
At M=1024 the target-grid gap concentrates at N=4096 and N=8192 while N=65536+
is on the prefill path and passes. Measured on this box (5 repeats, warmup 20,
iters 100): M=1024 N=4096 wall 21.2 us path=small_n, N=8192 31.5 us path=small_n,
N=16384 48.0 us path=prefill (sampler). Both shortfall points stage the entire row
into dynamic LDS (`pitch * 4` B) then radix-select with one block per row; there is
no chunking or bandwidth overlap like the sampler path gets. N=8192 is 1.5x N=4096
while work is 2x, so the curve is sublinear but still ~1.5-1.7x over the aiter
oracle at these widths. Next lever: block-size table for small_n at N in {4096,
8192} (currently sized mainly for N<=2048), not Phase B/C tuning.
Implementing rowStarts as absolute column indices (`out = rowStart + local`) required
`RowExtents<RAGGED>`: 8 B `{nullptr}` on the uniform path (same slot as the old
`row_ends` pointer) and 16 B `{starts, ends}` on ragged. Even with `if constexpr
(RAGGED)` on the gather lambda and `len = pitch` on the uniform body, inner-tier
small_n geomean moved 28.55 -> 28.98 us (+1.5%, 3 runs) while rowStarts shapes
pass the CPU oracle. Treat as the price of closing the silent-wrong-answer hazard;
refresh the inner baseline at g_10 rather than leaving every future candidate
 fighting a false reject.

### outer tier run-to-run noise exceeded POINT_REGRESS_PCT on decode N=65536
Two consecutive outer-tier runs of the same binary differed by >5% on the decode
M=1024 N=65536 cell, tripping the per-point limit and making sub-10% changes
unscoreable. `bench/grid.py` `LARGE_ARGV` repeats raised 5 -> 9 so the median
stabilizes before the per-point compare.

### topk_select backend registration is blocked on missing sweep tooling
`aiter/ops/topk_select.py` references `topk_backend_sweep.py` and
`topk_backend_fit.py` (lines 90 and 367) but neither file is in the aiter-topk
tree. The fitted `_dispatch` rule covers a 565-cell sweep; AVO cannot enter until
that sweep is re-run. Classify AVO as nondeterministic-tie like `plain`:
`block_gather_topk` breaks ties by `atomicAdd` arrival order, so it belongs in
`_NONDETERMINISTIC`, not the `tie='low'` or `'high'` sets.

### a template parameter creates a separate instantiation per value, so the gate must cover each one
`WRITE_VALUES` is a compile-time flag on the four OUTPUT kernels only
(phase_a/phase_b never touch the output). The uniform gate at 8 values shapes
exercised `phase_small_n_topk`, `phase_c_select_waveseg<STATIC_CAP=true>`,
`phase_c_select_contig`, and `exact_row_select` -- but NOT
`phase_c_select_waveseg<STATIC_CAP=false>` (only reachable at cap > 4096 with
coop_g == 1, i.e. M >= 256 at N in {524288, 1048576}) nor
`phase_c_select_contig` on the ragged path (every other ragged point derives
coop_g == 1). A green gate on the 8-shape set therefore said nothing about
those two instantiations with values on.

Fix: add three explicit shapes to `bench/grid.py` `VALUES` -- (256, 524288),
(256, 1048576, prefix=131072), (8, 524288, prefix=131072) -- and re-run
verify_grid; 2130/2130 green. Same lesson as the unused-kernarg trap: a template
makes the body free, not the obligation to test every instantiation that ships.

### Phase C with 3 radix passes is WRONG, and a partial-dist gate certified it
Shipped in g_11 and reverted in g_12. `g_phase_c_passes = 3` measured anchor
M=4096 N=131072 615 -> 609 us (-1%), rocprof phase_c 69.7 -> 64.2 us, inner
geomean -1.66%. All of that gain was invalid: at M=4096 N=131072 the 3-pass
Phase C gives `rows_fail=1` on `--dist gaussian` and `--dist inf`, against
`rows_fail=0` on all five dists at 4 passes (isolated A/B, both oracles).

Two independent mistakes, both worth remembering:

1. **A documented invariant was overwritten instead of read.** The line directly
   above the edit says "Phase C must use all 4 passes to be exact. Fewer is a
   TIMING ABLATION ONLY" (`benchmark_topk.hip.cpp:57`), and
   `block_select_lds`'s own header says Phase A may use fewer passes *because*
   its pivot is only a filter threshold, while "Phase C and the fallback MUST
   use all 4 (their result is the answer)". The "3 passes are bit-identical to
   4" note is about **Phase A's candidate counts**, not about a pivot that is
   returned as the answer. Carrying a note across kernels whose pivots mean
   different things is how a correct fact produces a wrong change.
2. **The gate that passed did not cover the axis the change moved.** Only
   `verify_grid.py --dist adversarial --inner` (24 points) was re-run; it was
   green while gaussian/inf were broken. `--dist all` (2140 points) fails on
   the 3-pass build and passes at 2140/2140 after the revert. A pass count is a
   distribution-sensitive knob, so the dist axis was exactly the one that had to
   be swept, and the skipped check was flagged "unverified" yet still shipped.

Negative control that caught it: the SAME `rows_fail=1` appeared on the shipped
default, which briefly looked like a pre-existing failure. Re-running the
baseline binary with `--phase-c-passes 4` separated them -- without that A/B the
regression would have been mis-attributed to the distribution instead of the
change. Always price a suspected pre-existing failure against the specific knob.

Still true after the revert: the block-size sweep (256..1024 for phase_a/c),
`--fuse-ab 1` on prefill, and phase-a 2-pass do not beat the occupancy default,
and S=8192 remains optimal vs 4096/16384 on the anchor.

### Folding the HIST_REP reduction into the pivot scan: one barrier fewer per pass
**Accepted, g_13.** A radix pass in `block_select_lds` held 5 block barriers:
clear `s_hist`, histogram, reduce the HIST_REP replicas into `s_red`, and two
inside `block_find_pivot_bucket`. The reduction exists only because the scan
reads one value per bucket, so `block_find_pivot_bucket_rep` folds the replica
sum in as it reads, and the separate loop plus its barrier disappear -- 5
barriers to 4, no extra LDS, no arithmetic added (the same 256xHIST_REP adds
happen either way, just in the 256 threads that need them).

Won on every shape tried, largest where the select is the biggest share of the
kernel, which is what a barrier-bound select predicts:

| shape | before | after | delta |
|---|---|---|---|
| M=4096 N=131072 (anchor) | 0.6132 ms | 0.6090 ms | -0.68% |
| M=4096 N=1048576 | 3.1741 ms | 3.1680 ms | -0.19% |
| M=1 N=1048576 | 0.0316 ms | 0.0309 ms | -2.2% |
| M=1024 N=65536 | 0.0793 ms | 0.0781 ms | -1.5% |
| M=4096 N=8192 (small_n) | 0.0995 ms | 0.0947 ms | **-4.8%** |
| M=2048 N=4096 (small_n) | 0.0372 ms | 0.0355 ms | **-4.6%** |

Per-kernel at the anchor: `phase_a` 65.7 -> 62.2 us, `phase_c` 69.8 -> 68.1 us,
`phase_b` unchanged (476-478 us). Inner geomean 64.72 -> 62.89 us (-2.83%),
small_n regime -5.1%. `verify_grid --dist all` 2140/2140 with the same 409
warnings, and the gate was proven failable on the shipped fused path first
(`--inject-fault 1` -> rows_fail=1, `--inject-fault 2` -> values ok=0).

`s_red` is now dead for `block_select_lds` callers but is still declared
`__shared__` in those kernels and still used by `block_select_stream`. Removing
those declarations would return ~1 KB of LDS per block, which changes
occupancy and so needs `PHASE_A_STATIC_LDS` / `PHASE_C_STATIC_LDS` re-read off
`.group_segment_fixed_size`. Deliberately left as a separate change.

### SHIPPED g_16 (v4 Stage 1): coop_g table extended to M=4096
The OPEN lever above was picked up 2026-09-17. Full sweep in `log/coop_sweep.tsv`
(5 M x 4 N x 6 G). Table rows M=256..4096 added; `ni<=2` columns forced to 0
(coop_g=1) because the sweep only measured N>=131072. **Bug fix required:**
`phase_b_filter_coop` used `n4_per_row=pitch/4` instead of `n4_cover(len)` for
ragged rows, causing HIP 700 on rowStarts M=256 N=131072 at coop_g=8 (aiter always
uses ragged rowStarts). Fixed in `topk_generalize.hip.hpp`. verify_grid 2140/2140;
inner per-cell ACCEPT (-1.1% geomean); anchor 609.4 -> 581.0 us.

### FALSIFIED (v4 Stage 2): a per-region scan form buys nothing once coop_g lands
The rep-vs-wave0 scan trade that justified g_14 and g_15 **does not survive g_16.**
Priced on ONE fresh binary via a temporary `--scan-wave0` override, warmup 20 /
iters 100 / repeats 9, wall ms:

| shape | wave0 everywhere | per-region table | rep everywhere |
|---|---|---|---|
| M=4096 N=131072 (anchor) | 0.5840 | 0.5846 | 0.5843 |
| M=4096 N=262144 | 1.0440 | 1.0456 | 1.0456 |
| M=1024 N=1048576 | 0.8546 | 0.8552 | 0.8559 |
| M=2048 N=262144 | 0.5317 | 0.5315 | 0.5329 |
| M=256 N=1048576 | 0.2161 | 0.2163 | 0.2169 |
| M=4096 N=8192 (small_n) | 0.0946 | 0.0944 | 0.0942 |

Three independent rounds on the anchor, the one cell the table existed for:
wave0 0.5838 / 0.5838 / 0.5842 against rep 0.5843 / 0.5851 / 0.5853 (stddev
0.07-0.12%). Rep is consistently the SLOWER of the two now -- the opposite sign
to g_14's -0.35%.

**Why the old evidence expired:** g_14/g_15 measured the anchor at `coop_g == 1`,
where Phase B/C are `phase_b_filter_wavestage` + `phase_c_select_waveseg`. g_16
gives that shape `coop_g = 8`, so it runs `phase_b_filter_coop` +
`phase_c_select_contig` with a different block/LDS shape, and the barrier
structure the scan form trades against is no longer on the critical path. The
plan's own rule ("coop_g first, it changes WHICH PATH a shape takes") applies to
the *evidence* as well as to the sweep order: a pre-coop measurement cannot be
reused to justify a post-coop knob.

Reverted in full rather than shipped with an all-wave0 table: the runtime form
put a branch inside `block_select_lds` (the innermost select, where the `#if`
lets the compiler drop one side entirely) and a `hipMemcpyToSymbol` in the
dispatch path. The A/B is reproducible from `-DSELECT_WAVE0_SCAN=0`, which is
how g_15 documented it, so the knob earned nothing it did not already have.

**Process trap that nearly shipped this:** the first `ACCEPT` for the scan table
was measured on a STALE binary. `make` reported "Nothing to be done for 'all'"
right after an edit to `csrc/topk_shape.hip.hpp`, so `score_grid` timed the
previous build and the saved inner baseline recorded that config. Caught by
comparing mtimes (`benchmark_topk` 23:24:29 vs the header at 23:24:46) plus
`make -n`. Check both before trusting any number, per config `build_integrity`.

### v4 Stage 3 NOT DONE: per-region S rule
`g_s_rule` stays the global `0` (R_TARGET law). A regional form was drafted
(`effective_s_rule`, rule 1 at M<=8) and reverted UNMEASURED, because the same
expiry that killed Stage 2 applies: the "wins 5-7.5% at M<=8" claim predates
g_16, and M<=8 shapes now take coop_g 8-64 from the extended table. It needs its
own sweep on a post-g_16 binary before it can be judged, so nothing here is
evidence either way.

### CORRECTED: those nine v4 cells were not drift, the per-point KEY was wrong
The entry below diagnosed nine outer-tier cells as baseline decay. **That
diagnosis is wrong.** The interleaved A/B in it was sound and did establish that
the change was not responsible; the error was the next step, concluding
"therefore the baseline decayed" without testing the other candidate, "therefore
the comparison is wrong".

`compare()` keyed its per-point map on `(m, n)`, but `all_shapes()` emits the
same `(m, n)` at k = 512, 1024 and 2048. In the v4 outer baseline **134 of 134
distinct `(m, n)` pairs collide**, so the map kept whichever k happened to come
last and every k's measurement was scored against it.

The test that settles it needs no GPU and no second binary: compare a baseline
against **itself**. Under the old key the v4 baseline flags **43 of its own 402
points** as regressing beyond 5%, including all nine cells called drift here --
`M=1024 N=16384 K=2048` 46.50 us scored against 40.90 us from a different k,
`M=4096 N=16384 K=2048` 163.00 against 139.60, `M=4096 N=65536 K=512` 348.30
against 316.60. A gate that fires when compared with itself is not measuring the
candidate.

Fixed in v5 Stage 3: the key is `(m, n, topk)` and `score_grid.py --self-test`
asserts both tiers compare clean against themselves. Under the corrected key the
same v5 outer run that reported the entire N=65536 column at +5.4% to +12.7%
reports **0 cells regressed and 23 improved**.

**Generalises to:** before believing either "my change regressed it" or "the
baseline drifted", check that the comparison itself is sound. Self-comparison is
free, instant, and would have saved both of these investigations.

### SUPERSEDED (see above): "a baseline cell can drift 8-13% with no code change"
Nine outer-tier cells (M in {1024, 2048, 4096} x N in {16384, 32768, 65536})
scored +5.9% to +13.0% against the g_15 outer baseline and tripped
`POINT_REGRESS_PCT`. None of it was the change. Interleaved, same-session A/B of
a pre-v4 binary (`git archive 726a9e60 | tar -x -C /tmp/pre_v4 && make`) against
the shipped one, two rounds each:

| shape | pre-v4 | shipped | g_15 baseline |
|---|---|---|---|
| M=1024 N=16384 | 0.0465 / 0.0465 | 0.0466 / 0.0467 | 0.0428 |
| M=4096 N=16384 | 0.1630 / 0.1630 | 0.1632 / 0.1630 | 0.1443 |
| M=4096 N=32768 | 0.2432 / 0.2432 | 0.2433 / 0.2432 | 0.2217 |
| M=2048 N=65536 | 0.1898 / 0.1895 | 0.1895 / 0.1896 | 0.1837 |

Both binaries agree to 0.1-0.3% on all nine, and all nine derive `coop_g == 1`,
where `coop` is false and `phase_b_filter_coop` is never launched -- so the only
two shipped kernel diffs cannot reach them even in principle. The baseline is the
thing that moved.

Method worth reusing, and it is still worth reusing -- it is only the conclusion
above that was wrong: when a per-point REJECT lands on cells whose code path you
can show is untouched, do NOT tune against it. Build the OLD commit into a
separate directory and interleave the two binaries in one session. Then, before
blaming the machine, run the baseline against itself. A baseline is a
measurement and can decay -- this repo has recorded one contention event writing
a bogus 78.30 us into a baseline -- but a broken comparison looks exactly the
same and is far cheaper to rule out.

### SHIPPED (v5 Stage 7): an odd row pitch needed four deleted guards, not a port
`stride0 % FP32_EPT != 0` was refused by `sampling_geometry_ok`,
`sample_stride_exact`, `topk_sampled_supports` and the harness, so aiter silently
kept such widths on its own mb/ob path. All four guards existed for the
misalignment Stage 2 proved is a non-issue, and the kernels already handled an
odd pitch: `sample_chunk_stride` masks the chunk spacing to a multiple of 4 so
chunk starts stay 4-aligned RELATIVE to the base, the RAGGED instantiation counts
vectors with `n4_cover(len)`, and `load_row_f4<true>` loads the final partial
vector element-wise.

So: four relaxed guards, plus the harness routing an odd pitch through RAGGED
with full-row extents -- what the aiter entry always instantiates anyway.
`RAGGED=false` stays byte-identical, because it truncates its vector count and
carries no per-lane bound, and adding one would charge the scored pow2 grid a
compare per element for a case it never sees.

Worth 1.27x to 2.81x measured through the real Python dispatch against the aiter
path these widths used to fall back to (M=64 N=65537 73.40 -> 57.63 us;
M=256 N=1048573 693.48 -> 246.69 us).

The positive control that makes the tail claim checkable: plant each row's
maximum at index N-1, which is ONLY reachable through the clamped partial vector.
Found in 8 of 8 rows on small_n (M=8 N=12289) and 8 of 8 on the sampled path
(M=8 N=131073, coop_g=16, under_K=0). All three residues are in `grid.ODD_NS`
because the clamped tail is 3, 2 and 1 elements long respectively;
`verify_grid --dist all` 3085/3085.

**The estimate was wrong because the diagnosis was.** This was scoped as a
feature -- port aiter's `vectorized_process` head/middle/tail to seven load
sites -- and it was four deleted `if` statements. A wrong root cause does not
just produce a wrong fix, it produces a wrong estimate of the work.

### SHIPPED (v5 Stage 5): occupancy_block_threads truncated twice, so it undershot
`occupancy_block_threads` computed `TARGET_WAVES_PER_CU / min(lds_blocks, g)`
with integer division and then rounded the result DOWN to a power of two. Two
truncations in a row, and they cost up to 40% of the target wave count whenever
the quotient was not already a power of two -- which phase_a hits as soon as S
leaves {4096, 8192, 16384}, i.e. only at non-pow2 N. Measured best block against
what the old form picked, M=1024..4096:

| LDS-limited blocks/CU | S range | old pick | measured best | gap |
|---|---|---|---|---|
| 7 | 4096..4544 | 256 | 512 | -0.0% .. -2.2% |
| 6 | 4608..5504 | 256 | 512 | -0.1% .. -1.1% |
| 5 | 5568..6848 | 256 | 512 | +0.1% .. -3.0% |
| 4 | 6912..8896 | 512 | 512 | agree |
| 3 | 8960..12352 | 512 | 1024 | -0.3% .. -3.0% |
| 2 | 12416+ | 1024 | 1024 | agree |

Ceiling division plus rounding UP to a power of two reproduces the measured
optimum in every class: all 14 shapes re-checked land within +/-0.3% of their own
best. Both halves are needed -- with truncating division, `32/7` becomes 4, which
is already a power of two, so rounding up afterwards cannot recover it (+2.2% at
M=2048 N=49152).

**Shipped as a formula fix, not the per-class table the plan called for.** The
table would have matched these six classes just as well and then been silent
about every S nobody measured; one systematic flaw explaining all four gaps is
the cheaper and more general answer. Gate: 2885/2885, inner -0.26%, outer -0.05%
with 38 cells improved and 0 regressed -- a small geomean move because the wins
sit on the non-pow2 N and dilute across 547 points.

### SHIPPED (v5 Stage 6): the S rule is regional, and its own search was broken
The v3-era note said rule 1 "wins 5-7.5% at M <= 8 but regresses the anchor by
+3.9% ... for an overall geomean of just -0.91%", and concluded FALSIFIED. Two
things were wrong with that conclusion.

**The region is bigger than M <= 8.** Re-measured on g_22, rule 1 against rule 0:
M=1 -5.8/-6.9/-6.1%, M=2 -3.0/-7.7/-6.2%, M=4 -1.2/-7.2/-5.9%,
M=8 -1.6/-7.7/-7.7%, M=16 -1.5/-10.8/-6.8%, M=32 -2.8/-9.5/-4.0% at
N=131072/262144/524288. M=64 turns: it wins at two of the three N and loses
**+17.2%** at the middle one, so the boundary is M <= 32. The anchor measures
+1.7% under rule 1, not +3.9%, and keeps rule 0 either way.

**Rule 1's own search had the Stage 3 bug.** It stepped `S *= 2`, so it only ever
considered powers of two, and a non-pow2 N has no exact stride at those -- the
loop ran to SAMPLE_S_MAX and rule 1 asked for MORE sampling than rule 0, the
opposite of its purpose. At M=1 that cost +21.5% at N=65532 (4160 -> 16384),
+13.2% at N=131068 (8256 -> 16384) and +8.1% at N=32832 (4608 -> 8192). Stage 3
fixed exactly this in the repair path and left this search behind. Stepping by
SAMPLE_CHUNK_ELEMS instead zeroes all three and finds wins the doubling form
could not reach at all: N=524288 16384 -> 5440 (-5.0%), N=1048572 16384 -> 10816
(-4.6%).

Result: outer per-cell **-1.71% with 130 cells improved and 0 regressed**, the
largest single move of the v5 run. And fallback pressure went DOWN where the
rule changed, not up: `under_K > 0` lines at M <= 32 fell from 37 to 6, because
a smaller S comes with a larger `auto_margin` and so a wider candidate window.

### THIRD instance this run: a table fitted on a subset of the domain it serves
Stage 4 picked a coop_g cell from the argmin at the one N it was measured at, and
that cell was wrong by 12.6% at another N in the same bucket. Stage 4's
tie-breaker did it again one level down. Stage 6 then fitted the S-rule REGION on
three pow2 N and shipped +21.9% on a non-pow2 N inside it.

Same error three times, each time one level up: a cell fitted on one N, a
tie-break checked at one N, a region fitted on one class of N. The rule that
catches all three: **whatever range a decision serves, measure at both edges of
that range before believing the middle.** For a table cell that means >= 2 N per
bucket; for a region it means the non-pow2 N as well as the pow2 ones; for a
tie-break it means the same worst-case test as the selection it is overriding.

### SHIPPED (v5 Stage 4): kCoopLog2G at half-octave N, fitted minimax not argmin
Two separate things, and the second one is the transferable part.

**The table needed finer N resolution.** Indexing N by `ilog2_floor(N) - 14`
gives one column per octave. Refitting the same 2093-point sweep with octave
columns instead of half-octaves costs up to **+5.42%** (M=64 over
[16384, 32768)) and more than 1% on 12 of the (M, octave) pairs, worst in
[524288, 1048576) at large M. Splitting each octave at 1.5x fixes it.
Filling the v4 hole at the same time (M >= 256, N in [32768, 131072), where I
had set the columns to 0 because the v4 sweep never measured them) is worth
1.01x to 1.28x at M=256, growing with N, and 1.00x to 1.07x above that.

**A table cell must be fitted on the WORST case over the range it serves.**
The first fit took the argmin at the single N each bucket had been measured at,
and that is how M=1024 column 11 came out as G=1: at N=786432 G=1 led G=16 by
0.2%, and at N=1048572 -- same bucket -- G=1 costs **+12.6%**. It shipped into an
outer-tier run as a real +8.2% regression on that cell, the only one in 547.
Refitting as minimax over every measured N in the bucket picks G=16, which costs
0.2% and 0.1% at the other two N in the bucket.

The same flaw hid in the tie-breaking pass, which nudges ties toward
monotone-in-N: checked at one N it moved M=64 column 1 to G=16, free at N=24576
and **+9.4%** at N=28672. Both passes now evaluate across the bucket, which
needed a second and third N measured per bucket (`coop_sweep_{mid,base16k}`).
After that no fitted cell is worse than +3.4% against any N measured inside it,
down from +9.4%.

**Watch the units on a G probe.** The v4 note that justified this work said
"M=256 N=32768 has no win: default 32.5 us, G=2 36.0, G=4 43.0", and I carried
that forward as "G=8 costs +32% there". It does not -- that was G=4. G=8 is
32.5 us against G=1's 32.2, a 0.9% tie, because the v4 probe never tried G=8 at
that shape. A sparse G probe is not a statement about the G values it skipped.

### SHIPPED (v5 Stage 3): search for the nearest exact sampling stride
`derive_shape_params`'s exact-stride repair offered exactly ONE candidate: `N/64`
chunks, which `align_sample_s` then clamps to `SAMPLE_S_MAX`. So at
`N = 2^k + 64` -- aiter's own `num_prefix + num_rows` pattern -- the only exact
choice on offer was the largest one, and S went 4096 -> 16384: four times the
phase_a sampling for 0.2% more data. Measured on the shipped tree: M=4096
N=32768 243.0 us against N=32832 **313.6 us (+29%)**, M=1024 +32.5%, M=256
+18.3%, while N=33024 keeps S=4096 and costs 253.3 us. 1.33% of all N in
[32768, 1048576] were inflated >= 2x, in two bands (1536 values in the 32768
octave, 1848 in the 65536 one).

Searching upward for the smallest exact-stride S instead finds one very close by:
N=32832 takes **4608** (72 chunks of 456), N=49216 takes 4800, N=65600 takes
4352, N=92332 takes 5952. Measured **-10.8% to -22.2%** across M=64..4096 on
those shapes. Every pow2 N and the other aiter widths are untouched, because an
exact stride means the repair block never runs at all. N=131328 moves 8256 ->
8576, trading a masked stride for an exact one at no cost (+0.0% / +0.5%).

**Capping the repair's growth instead is the wrong fix, and measurably so.** A
growth cap keeps the law's S with a MASKED stride, and the mask is not free: at
M=4096 N=65600 it produced `under_K=760` on `--dist inf` where the uncapped S
gives 0, while the neighbouring pow2 N=65536 at the SAME S=4096 also gives 0. So
the fallback pressure came from the masked stride, not from the sample count.
The search keeps exactness and pays only the growth exactness actually costs:
across the whole 547-point grid, `verify_grid --dist all` warnings went 673 ->
668 and `under_K > 0` lines 113 -> 114. The single residue is M=4096 N=92332 on
`inf`, 19 rows of 4096, which the exact fallback covers.

### v4 Stage 4: block-size tables for phase_a/phase_c
The +12.1% hole at M=2048 N=4096 is on **phase_small_n_topk** and is already
covered by `kSmallNWaves` since g_2. Sampled-path `occupancy_block_threads` stays
the fallback (+3.4% worst / +0.1% mean over 91 points per config-v3); no new table
shipped.

### OPEN, LARGEST KNOWN LEVER: the coop_g table stops at M=128 on a false premise
`choose_coop_g` (`csrc/topk_shape.hip.hpp:341`) returns 1 for every `M >= 256`,
and `kCoopLog2G` only covers M=1..128. The justification in the comment above
the table is "M >= 256 has enough rows to fill the GPU without splitting any of
them". **That is measurably wrong.** 256 rows at one block per row is 256 blocks
on 256 CUs, which fills the CUs but leaves 1 block per CU and so no latency
hiding inside one.

Measured with `--coop-g` (warmup 20 / iters 100 / repeats 9), which bypasses the
early return, so no code change was needed to price it:

| shape | default (coop_g=1) | best override | speedup |
|---|---|---|---|
| M=256 N=1048576 | 410.8 us | **215.5 us** (G=16) | **1.90x** |
| M=256 N=524288 | 227.1 us | 126.4 us (G=8) | 1.80x |
| M=256 N=131072 | 64.4 us | 47.0 us (G=8) | 1.37x |
| M=512 N=1048576 | 598.8 us | 448.4 us (G=8) | 1.34x |
| M=1024 N=1048576 | 944.8 us | 885.3 us (G=8) | 1.07x |
| M=2048 N=1048576 | 1847.6 us | 1738.1 us (G=8) | 1.06x |

Correct at M=256 N=1048576 G=16 on all five distributions (rows_fail=0,
under_K=0; the `over_Calloc` on equal/adversarial is the usual sampling-off
warning the default path also raises).

Scope, so the table extension is not overfitted: the win needs BOTH large M and
large N. At **M=256 N=32768 there is no win at all** -- default 32.5 us is
already the best, G=2 is 36.0 and G=4 is 43.0 -- so the boundary is a 2D region
like the existing table, not a single raised threshold. G is not monotone
either: at M=256 N=1048576, G=16 is 215.5 us but G=32 is 220.8 and G=64 is
237.2, so each cell has an interior optimum and must be swept, exactly as
M=1..128 already was.

Note the reported `path=` stays `prefill` for M >= 512 because `PATH_DECODE`
additionally requires `M <= N_LDS_SMALL_M_LIMIT`; the cooperative filter is
selected by `coop_g > 1` alone, so it applies on either path and the label is
only for reporting.

Next step when this is picked up: sweep G over M in {256, 512, 1024, 2048,
4096} x N in {131072, 262144, 524288, 1048576}, extend `kCoopLog2G` with the
measured rows, drop the `M >= 256` early return, then re-run
`verify_grid --dist all` (the gate has never exercised coop_g > 1 above M=128)
and `score_grid`. Expect the inner tier's M=256 N=32768 point to be unaffected,
which is a useful sanity check that the new rows did not over-reach downward.

### 2 barriers per radix pass is reachable, but only by confining the scan to one wave
**Accepted, g_15.** After g_13 and g_14 a pass held 3 block barriers: one after
the histogram, and two inside `block_find_pivot_bucket_rep`. Those two exist for
different reasons, and only one of them is structural:
- the first publishes `s_wavetot`, the cross-wave partial sums, which exist
  ONLY because 256 buckets span 4 waves at 64 lanes;
- the second publishes `s_scan` to the block, and since g_14 also publishes the
  histogram clear.

`block_find_pivot_bucket_wave0` gives the whole scan to wave 0 -- 256 buckets at
4 per lane fit one wave, so the suffix scan is shuffles only -- and the first
barrier disappears with the partials. The second stays, and CLEAR still rides on
it for free, because wave 0 is now the histogram's only reader and zeroes each
slot as it reads. Zero extra LDS, no slot read twice.

Measured as a near-mirror of g_14's trade -- that one bought the anchor and cost
small_n, this one buys small_n and the latency-bound small-M shapes and costs the
anchor: small_n M=4096 N=8192 -2.2%, M=1 N=1M -1.6%, M=2048 N=4096 -1.4%,
M=4096 N=1M neutral, M=1024 N=65536 neutral, **anchor +0.4%** (607.2 -> 609.4
us). Inner geomean 62.66 -> 61.96 us (-1.1%) with small_n -2.3%, decode -1.3%
and prefill neutral, so the aggregate is a clear win and the single regressing
point sits far inside POINT_REGRESS_PCT.

Two routes to 2 barriers that do NOT work, so they do not need re-measuring:
- **Every wave scans redundantly.** Removes both of find_pivot's barriers, but
  with all waves reading all slots no wave may zero anything until all have
  finished, so the clear needs its own before-and-after pair and the pass is back
  to 3 -- now at 4x the LDS reads.
- **Double-buffer the histogram.** Genuinely reaches 2 (clear the idle buffer
  during the scan), but costs +4 KB, which takes phase_a from 4 to 3 blocks/CU at
  S=8192 (163840/42008 against 163840/37912) to buy a barrier that g_14 measured
  at -0.35% on the anchor. This is the max-sized-static-LDS trap in a new
  costume.

Barrier count per pass across this session: 5 (pre-g_13) -> 4 (g_13) -> 3 (g_14)
-> 2 (g_15), and the per-step gains are diminishing and increasingly
regime-split, which is the signal that this lever is close to spent.

### Wave-aggregating the LDS histogram atomic costs 6-38x what it saves
**Falsified, kept as `-DHIST_AGG_ROUNDS=n` so it reproduces.** The per-element
`atomicAdd` into `s_hist` is the one atomic in these kernels that is NOT
aggregated, and it serialises: a sortable fp32's top byte is sign+exponent, so
~half of uniform[-1,1] lands in one of the 256 buckets. `hist_add_aggregated`
elects the lowest outstanding lane each round, groups the lanes sharing its
bucket, has the leader add the group count, and lets the remainder fall back to
individual atomics, so it is exact regardless of how well it groups (verified
PASS on all five distributions at the anchor for rounds 1, 2 and 3).

Priced the ceiling first with `-DABLATE_HIST_ATOMIC=1` (drops atomicity, same
addresses, wrong results): `phase_a` 61.7 -> 53.0 us at the anchor, **-14.1%**,
i.e. about 8.7 us or 1.4% of the 607 us wall. Then measured the mechanism:

| shape | rounds=0 | rounds=1 | rounds=3 |
|---|---|---|---|
| M=4096 N=131072 | 0.6074 ms | 0.6476 (+6.6%) | 0.7297 (+20%) |
| M=4096 N=8192 | 0.0965 ms | 0.1329 (+37.7%) | 0.1954 (+102%) |
| M=2048 N=4096 | 0.0353 ms | 0.0445 (+26%) | 0.0559 (+58%) |
| M=1 N=1048576 | 0.0305 ms | 0.0382 (+25%) | 0.0537 (+76%) |
| M=1024 N=65536 | 0.0777 ms | 0.0868 (+11.7%) | 0.1028 (+32%) |

Monotone in rounds, so it is the aggregation machinery itself: one `__ballot` +
`__shfl` + `__ballot` + `popcount` per element, on the innermost loop, against a
single LDS atomic that the LDS unit resolves in hardware. **LDS atomics on
gfx950 are cheap enough that no cross-lane scheme pays for itself here**, which
retro-explains the older "one LDS atomic per wave for s_wgt: 102.1 -> 104.2 us,
no effect" note as the same result rather than a coincidence.

Do NOT reach for aiter's LDS-histogram-then-global-flush
(`topk_per_row_kernels.cu:490`) as a model either. That exists because aiter's
multi-block path puts several blocks on one row and must combine histograms in
GLOBAL memory; these kernels are one block per row, so the histogram never
leaves LDS and the global stage it aggregates does not exist. The global atomics
that DO remain here (`cand_reserved`, `s_wgt`/`s_weq`) are already bulk
per-wave, and the Phase B filter has no atomic of any kind.

### Third instance: an ablation prices the COST, never the MECHANISM
Three separate attempts this session found a real, measured inefficiency and
then failed to reclaim any of it, because what the ablation prices is the work
you want to remove, not the machinery that removes it:

| measured waste | mechanism tried | outcome |
|---|---|---|
| passes 2-4 are 30-34% of small_n and carry almost no work | block-wide active-set min/max early exit | 21-39% SLOWER |
| same | wave-private in-place compaction | phase_a -0.5%, wall worse on 3 of 4 shapes |
| histogram atomic is 14.1% of phase_a | wave-level leader-election aggregation | +6.6% to +37.7% |

The two changes that DID work this session removed a **barrier** instead
(g_13 fold-the-reduction, g_14 clear-on-read), which is the resource this select
is actually short of. Before costing out another idea here, ask which of the
three it removes -- reads, atomics, or barriers -- and only the third has a
track record.

### Clearing the histogram on read removes a second barrier, but it is a regime trade
**Accepted, g_14.** With the reduction already folded in (g_13), the remaining
per-pass clear loop only exists to zero buckets the scan has just finished
reading, so `block_find_pivot_bucket_rep<CLEAR=true>` zeroes each bucket as it
reads it and both the loop and its barrier disappear: 4 block barriers per radix
pass, down from 5 before g_13. No LDS cost, which is what makes it preferable to
double-buffering the histogram (+4 KB, enough to cut phase_a from 4 to 3
blocks/CU at S=8192 -- the max-sized-static-LDS trap this repo has already hit
twice).

Exact rather than opportunistic, which is the part to preserve if this is ever
refactored: `HIST_SLOTS == 256 * HIST_REP` and thread `t` owns exactly
`[t*HIST_REP, (t+1)*HIST_REP)`, so the 256 threads that read the histogram cover
every slot once, and blocks here are always >= 256 threads because the scan
indexes buckets by `threadIdx.x`. Drop either property and slots silently stop
being cleared, which would corrupt the NEXT pass rather than this one.

It is a trade, not a free win: anchor -0.35%, M=1 N=1M -0.8%, decode -0.8%,
prefill -0.8%, against **M=4096 N=8192 +1.8%** (small_n regime geomean +0.40%,
inside the 0.5% band, so the scoring rule accepts it). Taken because large N is
the target and N <= 32768 is routed to aiter's own prefill by the
`stride0 >= 32768` dispatch. [unverified hypothesis] for the small_n point: the
clear now runs on the 256 threads that also carry the wave-scan, where the old
loop spread it over all `blockDim.x` threads (512 at that shape), so the work
moved onto the critical path rather than disappearing.

### The @perftest cell for M=4096 N=131072 has ~1% spread, and one 2% outlier got reported
A canvas built 2026-09-17 recorded the AVO op at **562.6 us** for the dense
M=4096 N=131072 k=2048 cell of `bench/topk_select_grid.py`. That number does not
reproduce, **not even on the commit it was taken from**:

| aiter-topk csrc | AVO us | topk_select us (same cell, same run) |
|---|---|---|
| `bb1413ed2` (the build the canvas used) | 574.6 | 958.9 |
| `4bb5c2634` (g_12) | 577.2 | 961.6 |
| `c291d154c` (g_15, current) | 571.3 | 960.0 |
| canvas, 2026-09-17 16:27 | **562.6** | 964.2 |

The control is the `topk_select` column: 958.9-964.2 across all four, a ~0.5%
spread, so the harness and the machine are steady. The AVO column across three
structurally different builds is 571-577, also ~0.5%, and the canvas's 562.6
sits 2% below every one of them including its own. It is measurement spread in
that pipeline, not a code change -- and the barrier work of g_13..g_15 is in
fact marginally net-POSITIVE here (571.3 against 574.6).

Two lessons. First, do not read a single `@perftest` cell as a baseline; it needs
the same repeat-and-compare discipline as `score_grid`. Second, and this is the
one that actually cost time: **that cell is not comparable to the standalone
benchmark's wall time at all**, so seeing 562.6 next to `benchmark_topk`'s ~609
us invites the conclusion that something regressed by 8%. The two differ in
three ways at once -- `@perftest` reports DEVICE time while `--mode time` reports
wall; the aiter op always instantiates `RAGGED=true` while the benchmark's
default is `RAGGED=false`; and the data comes from `torch.randn` rather than the
benchmark's own generator. `bench/aiter_ab.py` exists precisely because that
comparison already went wrong once, at 18% on M=256. Quote one pipeline or the
other, never one number from each.

### A transient contention artifact reported +24% across every regime at once
During g_14's baseline save, `score_grid --tier inner` reported geomean 78.30 us
with decode 39.79 / prefill 254.65 / small_n 37.57 -- about +24% on all three at
once, against 62.69 us measured minutes earlier on the same binary. Three quiet
re-runs gave 62.62 / 62.68 / 62.66, and `rocm-smi` showed no other KFD process
and 39-42 C junction temperatures afterwards, so it was another tenant's job
overlapping the run, not the change.

The tell is that a real kernel change does not move decode, prefill AND small_n
by the same large factor -- those paths share almost no code. Treat a uniform
multi-regime shift as a machine event and re-measure before recording it. The
damage here was that the bad number had already been written to
`knowledge/grid_baseline_inner.json` by the same command that measured it;
`--save-baseline` commits whatever it happens to see, so on a shared box do a
quiet re-run BEFORE saving, not after.

### Compacting the radix select's active set between passes buys nothing
**Falsified, kept as `--phase-a-compact 1` so the measurement reproduces.**
The filter-rescan select reads all `c` keys every pass and discards the ~255/256
that miss the pivot prefix, and those later passes were measured at 30-34% of
`phase_small_n_topk`. Compaction attacks exactly that: wave-private, in-place,
zero extra LDS and zero extra barriers (wave w compacts its own segment, a
survivor lands at or below the index it came from, so in place is safe), taking
three passes from ~3c key reads to ~2c.

It is correct -- identical `under_K` / `over_Calloc` / `rows_fail` to the
baseline on all five distributions at the anchor, i.e. the same pivot -- and it
is worth **nothing**: `phase_a` 65.7 -> 65.4 us (-0.5%), and wall time got
WORSE on three of four shapes (M=4096 N=1M +0.30%, M=1024 N=65536 +1.0%,
M=1 N=1M +2.2%; anchor -0.08%, inside noise).

Why, and this is the reusable part: **the discarded reads were never the cost.**
`phase_a`'s cost is the serial depth of one row -- passes x barriers over the
keys in LDS -- as already recorded under "Early-exiting the radix select once
the pivot is pinned". Compaction removes read volume and leaves the barrier
count untouched, so it cannot move a barrier-bound kernel; the extra
ballot/popcount in the critical path is what makes it net negative at small M.
The 30-34% ablation figure measures what those passes COST, not what mechanism
can reclaim it -- the same trap that section already warns about. The barrier
fusion entry above is the version of this idea that works, because it removes a
barrier instead of a read.

### decode large-N full-op BW is launch-bound, not filter-bound
rocprof at M=128 N=65536: phase_a+b+c sum ~27.6 us vs 32.5 us wall; 3x launch
floor dominates. coop_g table re-sweep (G=4/8/16/32) confirms G=8 at N=65536;
do not chase 5 TB/s full-op on decode via AVO -- route small-M decode through
topk_select FlyDSL instead.

### top_k_per_row_prefill dispatches stride0 >= 32768 to AVO when supported
`aiter/ops/topk.py` routes non-stable calls with stride0 >= 32768 through
`top_k_per_row_prefill_sampled` when `topk_sampled_supports()`; `stable=True` and
`AITER_DISABLE_TOPK_SAMPLED=1` force the original mb/ob path. Verified by
`op_tests/test_topk_prefill_dispatch.py`.

## Robustness traps (found by bench/stress_topk.py, 2026-09-18)

### A bounds bug that does NOT fault is the dangerous one
**Symptom:** `rowEnds = pitch + 64` at M=64 N=131072 returned indices 131134 and
131126 -- past the pitch -- with no error, no fault and no warning.
**Cause:** the kernel honoured the caller's unclamped extent, and the caching
allocator had the over-read backed by mapped memory, so nothing complained.
**Why it survived until now:** every earlier test relied on a fault to notice an
over-read. The v5 Stage 2 HIP 700 only faulted because that slice ran to the end
of the allocation; move it 1000 floats back and the identical bug is silent.
**Fix:** clamp in `RowExtents<true>`, and stop relying on faults -- poison the
out-of-window region by VALUE so a bad read shows up in the answer.

### NaN is the wrong poison for this kernel
**Symptom:** filling everything outside `[rowStart, rowEnd)` with NaN detects
nothing.
**Cause:** the ordering key at `csrc/topk_common.hip.hpp:258`,
`(u & 0x80000000u) ? ~u : (u ^ 0x80000000u)`, sends -NaN below -inf, so a NaN
poison is read and then discarded -- invisible.
**Fix:** poison with `+inf`, which that same key ranks above everything, so an
out-of-range read cannot fail to appear in the output. (vLLM's
`test_deep_select_topk` can use NaN only because DeepSelect ships
`abort_when_nan_found=True`. Copy the idea, not the constant.)

### AITER_CHECK aborts the process; it does not raise
**Symptom:** `k = 8193`, `stride1 = 2`, `stride0 = 0`, `k <= 0` and a short
workspace each killed the calling Python process outright.
**Cause:** `csrc/include/aiter_hip_common.h` throws only when
`g_aiter_can_throw` is set, and only the `aiter_safe_call` ctypes bridge
(`aiter_ctypes_error.h`, used by exactly one other kernel) sets it. The AVO
entry does not go through it, so every `AITER_CHECK` is a `std::abort()`.
**Fix:** validate in `top_k_per_row_prefill_sampled` (`aiter/ops/topk.py`) and raise
`ValueError`. Adopting the `aiter_safe_call` C-ABI instead would mean changing
the entry's return type and its binding -- disproportionate for argument checks.
**Generalises:** before assuming a vendor library's check macro raises, read it.

### An unmemoised binding call cost more than the validation it enabled
**Symptom:** adding `topk_sampled_supports()` to the Python wrapper regressed the
ragged path by +12.5% at M=64 N=65537 and +7.5% mean over 12 shapes.
**Measured cause:** `topk_sampled_supports` is **4.86 us/call** and
`topk_sampled_workspace_size` is **4.80 us/call** through the `@compile_ops`
binding, against a 43 us kernel. The tell was that the absolute delta was a
constant ~5 us that did not grow with the work -- host overhead, not kernel cost.
**Fix:** `functools.lru_cache` on a pure `(numRows, stride0, k)` query. Safe:
`params_for -> derive_shape_params` reads no device state (`CU_COUNT` is a
`constexpr`). Residual after memoising: +0.39% mean, which is the price of never
aborting.
**Still on the table (measured, not done):** `top_k_per_row_prefill` calls
`topk_sampled_supports` unmemoised on every dispatch and
`top_k_per_row_prefill_sampled` calls `topk_sampled_workspace_size` unmemoised, so the
production path pays ~9.7 us of binding overhead per call -- 22% of the 43 us
shape. Memoising both is a free win but changes the perf baseline, so it wants
its own measured commit.

### grep for the accessor, not for the helper
**Symptom:** patching the extent accessors compiled after 4 call sites were
updated and still had two unclamped ones: `extents.row_len(row)` in
`phase_small_n_topk`, and five more in `benchmark_topk.hip.cpp`.
**Cause:** the first search was `\.row_start(\|row_len_of(\|RowExtents<` over
`csrc/` only -- it missed the direct `.row_len(` form and the whole benchmark
translation unit, which also contains kernels.
**What caught it:** the patch script re-read the files afterwards and failed if
any unclamped accessor remained. Write the read-back check, not just the edit.

### Pairing a width with `width + 1` does not measure parity when the width is a power of two
**Symptom:** a first even/odd sweep compared stride0 = B against B+1 for
B in {65536 ... 1048576} and reported odd widths costing up to **+5.71%**
(M=256 B=1048576), with per-shape spreads under 1.1% -- i.e. not noise.
**Cause:** B is a power of two and B+1 is not, so the pair moves TWO things at
once. Leaving the power of two changes the sampling stride from exact to masked
and moves the `kCoopLog2G` and phase_a S lookups; all of that was being charged
to parity.
**Fix:** add B+2 -- even, and also not a power of two -- to the SAME pass, and
read two comparisons instead of one:
`(B+1) vs (B+2)` is parity, `(B+2) vs B` is power-of-two-ness.
**And then the control itself was wrong, twice over.** The three widths were
still timed one after another -- build, warm, time, free, next -- which charged
run-order drift to whichever width came first. That reported the pow2 boundary
at **+0.82% mean, max +5.60%**. The tell: in the loud cells B+1 and B+2 came out
nearly equal to each other (513.08 and 513.08 at M=1024 B=524288; 1956.29 and
1956.25 at M=4096 B=524288) while both sat the same distance above the B timed
before them. Parity survived only by luck -- B+1 and B+2 are adjacent, so the
drift between them cancels, while the pow2 comparison spans the whole triple.
Separately, the hand-written timing loop reused one input and therefore measured
a warm L2, where aiter's own @perftest rotates arguments to defeat it (234.42 us
against 217.56 us for AVO at M=256 stride0=1048577).
**Final method:** aiter @perftest, all three widths built and warmed before any
of them is timed, then 4 interleaved rounds.
**Result over 30 (M, B) cells:** parity **mean +0.29%** (-0.76% to +2.17%),
pow2 **mean +0.13%** (-6.93% to +2.72%). Judged against each cell's own 4-round
spread, **exactly one of the 12 cells with a delta over 1% survives**: M=4096
B=524288, where the POWER OF TWO is 7.4% SLOWER than either neighbour (1766.99
against 1645.20 and 1644.58 us, spreads 0.09-0.13%). Every parity cell is inside
its own noise. An odd pitch costs nothing anywhere in the sweep.
`bench/parity_sweep.py`, `reports/parity_sweep.json`.
**Generalises, three ways:** (1) when a one-unit step also crosses a structural
boundary, add the third point that crosses the boundary without the property
under test; (2) sequential timing charges drift to run order -- interleave, and
warm every arm before timing any of them; (3) use the project's own perf
harness, not a freshly written loop, or the number is not comparable to anything
else in the tree.

### The dispatch threshold is one octave too low: AVO is slower than aiter at N=32K
**Measured 2026-09-18**, aiter `op_tests/test_topk_per_row.py` with
`--prefill_backend aiter avo` and `AITER_DISABLE_TOPK_SAMPLED=1`, so both backends
go through one `@perftest` on one dataset, k=2048, fp32, 79 shapes.
`reports/avo_vs_aiter_sweep.tsv`.

`aiter_us / sampled_us`, below 1.00 meaning AVO is the slower one:

```
 M \ N     2K    4K    8K   16K   32K   64K  128K  256K  512K 1024K
     1   1.01  0.62  0.76  0.62  0.87  1.20  1.41  1.81  1.82  1.48
     2   0.90  0.62  0.73  0.73  0.92  1.24  2.00  2.32  2.24  1.31
     4   0.76  0.87  0.76  0.75  0.90  1.34  1.99  2.37  2.20  1.95
     8   0.90  0.87  0.93  0.77  0.88  1.25  1.76  1.97  1.93  1.85
    64   1.08  0.87  0.87  0.86  0.97  1.28  1.79  1.85  2.24  2.54
   256   1.09  0.84  0.93  0.99  0.85  1.27  1.58  1.90  2.62  2.90
  1024   1.17  1.23  0.92  0.99  1.27  1.52  1.77  2.28  2.70  3.08
  4096      -  1.12  1.07  0.92  1.02  1.32  1.58  2.31  2.54  2.93
dispatch  aiter aiter aiter aiter  AVO   AVO   AVO   AVO   AVO   AVO
```

**The N <= 16K columns do not matter**: `aiter/ops/topk.py:418` requires
`stride0 >= 32768`, so production never sends those to AVO.

**The N=32K column does matter, and AVO loses it for M <= 256** -- 0.85x to
0.97x, i.e. 3% to 15% slower than the path it replaced. Only M=1024 (1.27x) and
M=4096 (1.02x) win there. At N >= 64K AVO wins every cell, 1.20x to 3.08x.

**Why no gate caught it:** the evolution log and `bench/score_grid.py` compare
AVO against ITS OWN earlier baseline, never against aiter. A cell where AVO is
correct, is not regressing, and is simply worse than the op it replaced is
invisible to every gate in this repo. `bench/aiter_contract_audit.py` compares
against aiter but only for CORRECTNESS. N=32768 is in the scored grid and has
been green the whole time.

**FIXED** in `aiter/ops/topk.py`: the floor is now `SAMPLED_MIN_STRIDE0_WIDE = 32768`
when `numRows >= 1024` and `SAMPLED_MIN_STRIDE0_NARROW = 49152` below that.

49152, not the 65536 that first looked right. The original grid jumps an octave
between 32K and 64K, so the crossover had to be filled in before picking a
number (`log/crossover/`, aiter_us / sampled_us, below 1.00 meaning AVO is slower):

```
 M \ N     32K   40K   48K   56K   64K
 1        0.87  1.05  1.15  1.31  1.22
 8        0.88  1.05  1.22  1.09  1.27
 64       0.94  0.95  1.15  1.11  1.28
 256      0.81  0.84  1.04  1.04  1.23
 512      0.91  0.92  1.03  1.02  1.19
 1024     1.25  1.26  1.40  1.45  1.52
 2048     1.14  1.17  1.28  1.30  1.38
 4096     1.02  1.04  1.19  1.24  1.34
```

A flat 65536 would have thrown away the 1.15-1.22x that `numRows <= 64` earns at
48K. Verified through the dispatcher with both backends under one @perftest:
M=256 N=32768 now routes to mb/ob at 30.47 us instead of AVO at 36.92 (a 22%
loss avoided), M=256 N=49152 routes to AVO at 39.55 against mb/ob's 42.22, and
M=1024 N=32768 keeps AVO at 66.63 against mb/ob's 88.23.

**Generalises:** a no-regression gate measured against your own history cannot
see "worse than the thing you replaced". If an op is dispatched in place of
another, the comparison against that other op has to be a gate, not a one-off.

### hipEvent-per-launch timing is not kernel time, and it faked a whole plateau
**Symptom:** the ideal-selector floor (scripts/select_grid.hip) came out as a flat
6.2 us across the entire small-M half of the grid, unchanged whether N was 2K or
1024K, and our own kernel measured *faster* than that "floor" in 13 cells.
**Cause:** select_grid brackets every launch with a hipEvent pair (its lines
462-475), so its number is kernel time plus the command-processor bubble around a
single dispatch. Our side comes from aiter `@perftest`, which reads kernel
duration out of a profiler trace. Two different rulers, and the difference is a
roughly constant 3 us that is invisible at 300 us and is the entire measurement
at 3 us.
**Measured head to head** at m=128 n=16384 g=4 mlp=4 tb=1024: event timing says
read 5.96 / select 6.16 us, a rocprofv3 kernel trace of the same run says
3.44 / 3.64. Across the 390-cell grid the event ruler sits a median 2.99 us above
the trace (p10 2.16, p90 4.06) and 3.66 us on the cells that formed the plateau.
**Fix:** `bench/floor_grid.py --trace` (now the default) runs select_grid under
rocprofv3 and attributes dispatches to spec lines by order. The pattern is fixed
by select_grid's own loop -- one select for the hit counts, then 3 passes of 5
warm-up (read, select) pairs and `iters` timed pairs -- and the driver aborts if
the trace does not match it, because a drift there would silently mis-assign
every later cell. On kernel time the floor scales with N properly: M=4096 goes
11.26 us at N=2K to 2517.62 us at N=1024K, and the dispatch probe is 1.92 us
rather than 6.20.
**Generalises:** before comparing two numbers, check they are the same
measurement. A per-launch event pair includes dispatch; a profiler trace does
not. Whoever reads the report will notice a kernel that beats its own lower
bound.

### More samples, not fewer, even when the profiler times every dispatch
**Symptom:** cutting `iters` from 50 to 5 in traced mode -- reasoning that a
profiler timestamps every launch so a handful is enough -- made the floor
irreproducible: re-running moved the median cell 12.27% and put 258 of 390 cells
over 5%, against 1.86% median in event mode. The dispatch probe swung 3.76 -> 1.80 us.
**Fix:** same sample count in both modes. Re-running then moves cells over 20 us
by a median 0.29%, 5-20 us cells by 1.77%, and cells under 5 us by 5.06% -- and
that last number is 0.14 us of absolute jitter over a very small value, not drift.
**Generalises:** short kernels need the samples whichever instrument you point at
them. A better timer does not replace repetition.

### FALSIFIED: the small-M gap to the floor is not coop_g being too low
**The lead looked good.** select_grid picks g so that m*g is exactly 512
workgroups -- its stated rule, 256 CUs at 2 blocks/CU -- while our `coop_g` at
M=16 N=32K is 8, i.e. 128 blocks. That cell is 9.6x the floor, and the 7-10x band
in the report sits exactly where select_grid splits rows hardest (g=32 at M=16,
g=16 at M=32, down to g=2 at M=256).
**Killed by:** `./benchmark_topk --mode time --m 16 --n 32768 --topk 2048
--coop-g G`, wall_ms by G: 4 -> 0.0284, **8 -> 0.0230**, 16 -> 0.0239,
32 -> 0.0240, 64 -> 0.0240. The auto choice of 8 is already the best available,
more splitting is slightly worse, and asking for 32 or 64 still reports
coop_g=16 -- the shipped `kCoopLog2G` table (csrc/topk_shape.hip.hpp:392) tops
out there for these shapes. So block count is not what costs us 9.6x.
**What is left:** the floor is one kernel doing one pass; we run three dispatches
(sample, filter, select) whose durations the profiler sums. At a 2.64 us floor
the per-phase fixed cost is the whole story. That is a structural difference, not
a tuning knob, and it wants a profile before anyone guesses further.

### What actually sets phase A and phase C cost: latency per radix pass, not block count
Measured 2026-09-18 while scoping the red zone of `reports/ceiling_report.html`.
`rocprofv3 --kernel-trace` per-phase medians, M=16 N=32768 k=2048, floor 2.64 us:

```
phase_a_threshold    5.68 us   27.8%
phase_b_filter_coop  6.30 us   30.5%
phase_c_select_contig 8.52 us  41.7%
```

**Phase B is the only phase that scales with N**, and it is already at peak:
17.2 GB in 2802 us at M=4096 N=1M is 6.13 TB/s. A and C are flat in N because
they touch only `S` samples and `cap` candidates. At M=16 N=64K they are 68% of
the time; at M=4096 N=1M they are 7.4%. The red zone is exactly where A+C
dominate B.

**The obvious diagnosis is block count, and it is wrong.** A and C launch
`<<<M, block>>>`, one workgroup per row, against B's `<<<dim3(coop_g, M)>>>` --
16 workgroups against 256 CUs at M=16. But phase A costs 6.80 us at M=16 and
7.06 us at M=256: sixteen times the workgroups for 4% more time. The machine
absorbs 16x more blocks for free, so blocks are not the constraint.

**Sweeping the passes shows what is:** at M=16 N=32768,
`--phase-a-passes` 1/2/3/4 gives a_thresh 4.62 / 4.72 / 5.70 / 6.76, and
`--phase-c-passes` 1/2/3/4 gives c_select 5.40 / 6.84 / 7.74 / 8.52. Both are
about **4 us of fixed cost plus 1.0 us per radix pass**, and a pass moves 4
elements per thread at S=4096 with 1024 threads -- so that 1.0 us is barrier and
LDS-scan latency, not data. Splitting a row across blocks does not remove per-pass
barrier latency; it adds a cross-block sync to every pass. That is the same
reason `coop_g` and `--fuse-ab` do not help here.

**Consequence for any "close the gap at small M" plan.** Even if splitting made
every pass free, M=16 N=32768 would go 20.5 -> ~14.3 us (A to its 3.6 us fixed,
C to 4.4, B unchanged at 6.3), which is 6.7x the floor rather than today's 9.6x.
The floor is one kernel; we run three, each with ~4 us of fixed cost. **Below
roughly 5x at small M needs fewer kernels, not better ones.**

### Re-falsified, and one of them nearly shipped again
Everything cheap in phases A and C was already tried and recorded in this file;
these were re-measured on 2026-09-18 before that entry was read, which is the
process failure worth remembering -- read this file before probing, not after.

- **`--phase-a-passes 2`** wins 1.2-3.6% in the red zone and is catastrophic
  outside it: M=256 N=1M +320%, M=1024 N=262144 +127%, M=4096 N=1M +29.6%. The
  coarser threshold lets the candidate count past `cap` and rows drop into the
  exact fallback. Any use of it needs a measured per-region table, for 3%.
- **`--phase-c-passes 2`** looked correct on three shapes under
  `--dist adversarial` and is a 1.7 us win. It is WRONG, and the invariant is
  written directly above the variable
  (`benchmark_topk.hip.cpp:57`: "Phase C must use all 4 passes to be exact.
  Fewer is a TIMING ABLATION ONLY"). The g_11/g_12 entry above records the same
  mistake with 3 passes: green on `--dist adversarial`, `rows_fail=1` on
  gaussian and inf. A pass count is a distribution-sensitive knob, so testing it
  on one distribution proves nothing.
- **`--fuse-ab 1`**: M=16 N=32K 43.4 us against 23.0 (+89%) with
  `fallback_rows=8` of 16; M=4096 N=131072 2764 against 581 (+376%).
- **`coop_g`** at M=16 N=32K: 4 -> 0.0284 ms, 8 -> 0.0230 (the auto pick),
  16 -> 0.0239, 32 -> 0.0240, 64 -> 0.0240; 32 and 64 report 16 because
  `kCoopLog2G` tops out there.
- **Block sizes**: `--phase-c-block` 256/512/1024 gives 14.92 / 10.28 / 8.56 us
  and `--phase-a-block` shows no win either. The occupancy default is best.

### FALSIFIED: fusing phase A into phase B, even done properly
Chased because reducing the kernel count is the only route below ~5x the floor at
small M (see the phase A/C characterisation above). It does not pay, and the
reason is not the one the existing `--fuse-ab 1` failure suggests.

**The existing failure is a launch-config artefact, not a verdict on fusion.**
`phase_ab_fused` launches `<<<M, a_block>>>`, so the streaming half loses the
row splitting that `phase_b_filter_coop` gets from its `dim3(coop_g, M)` grid.
The penalty tracks `coop_g`, not the LDS footprint as first assumed:

```
M=1024 N=65536   S=4096  coop_g=1   78.1 -> 98.1 us    +26%
M=4096 N=65536   S=4096  coop_g=2  317.3 -> 1113.5     +251%
M=4096 N=131072  S=8192  coop_g=8  580.0 -> 2764.0     +377%
M=2048 N=131072  S=8192  coop_g=8  270.8 -> 1355.7     +401%
```

At `coop_g == 1` the fused grid is the grid phase B would have used anyway and
the cost is +26%; every `coop_g > 1` shape pays four times over for the lost
split. So a fused kernel has to keep the `dim3(G, M)` grid and let each of the G
blocks recompute the threshold redundantly -- which is affordable only at small M
where the machine is idle, i.e. exactly the red zone.

**What kills it is the zeroing, not the fusion.** `phase_a_threshold` also clears
`cand_reserved[row]`, `cand_bad[row]` and `fb_count`
(`benchmark_topk.hip.cpp:497-501`). Under a `dim3(G, M)` grid those G blocks run
concurrently, so block 0 zeroing the counter while block 3 is already
`atomicAdd`-ing it is a race. Removing phase A therefore means replacing that
clear, and both ways cost more than the kernel they remove:

- **A memset.** Measured in the same trace as the phases themselves:
  `__amd_rocclr_fillBufferAligned` is 1.84-3.16 us, against phase A's 5.68 us at
  M=16 N=32768, of which about 1.9 us is the bare-kernel floor and 3.8 us is
  work that the fused kernel still has to do. So 3.16 + (3.8 + 6.3) = 13.3 us
  against today's A + B = 11.98 us. A net loss before any of the fusion risk.
- **Per-block private candidate regions**, so no shared counter needs clearing.
  The shared atomic reservation exists precisely because candidates cluster
  unevenly across a row's chunks; partitioning the buffer G ways would turn that
  clustering into spurious `cand_bad` overflows and push rows into the exact
  fallback, which costs far more than the kernel saved.

**Generalises:** a kernel that also initialises shared state is not just its own
cost. Before costing a fusion, find what else the kernel being removed was doing.

### FALSIFIED: one cooperative kernel with grid.sync instead of three launches
The last untried structural idea for the red zone, killed by its own gate in
under an hour. `scripts/gridsync_probe.hip` prices both halves of the trade on
the same box, same timer, same run: a cooperative kernel whose body is N grid
syncs, and N back-to-back empty ordinary launches.

```
blocks   us per grid.sync   coop-launch intercept | us per ordinary launch  intercept
    64          8.877              20.73          |        2.897              0.85
   128         17.695              14.10          |        3.017              0.04
   256         31.991              15.10          |        3.028              0.37
   512         48.513              19.40          |        3.033              0.42
```

**A grid sync costs 3x to 16x an ordinary kernel launch, and unlike a launch it
gets worse with grid size.** At the size this would have used -- 128 blocks, from
M=16 with coop_g=8 -- one sync is 17.7 us against a 3.0 us launch. The fused
kernel would have traded two launches (6.0 us) for two syncs (35.4 us) inside a
pipeline whose entire wall time is about 23 us.

It is dead twice over: `hipLaunchCooperativeKernel` itself has a 14-21 us
intercept against 0.04-0.85 us for an ordinary launch, so the design loses ~17 us
before the first sync executes.

**And the launches were never costing what the microbenchmark says.** Back to
back with nothing else running, a launch is 3.0 us. Inside the real pipeline the
CPU enqueues ahead of a busy GPU and the cost mostly disappears -- wall time minus
the summed kernel durations, on a quiet box:

```
M=16  N=32768   wall 23.00 us  kernels 21.60  gap 1.40 us over 3 launches, 0.47 each
M=64  N=131072  wall 30.20 us  kernels 27.88  gap 2.32 us                   0.77 each
M=256 N=65536   wall 36.80 us  kernels 34.56  gap 2.24 us                   0.75 each
```

So the entire launch overhead of the three-kernel pipeline is 1.4 us of 23.0 at
M=16, six percent. Fusing to three-into-one could recover that plus two kernel
prologues, about 4.7 us of 23 at the very best, against a zeroing replacement
that costs 3.16 us (`fillBufferAligned`) or a CAS loop under 128-way contention.
**That is the ceiling on the whole reduce-the-kernel-count direction, and it is
small.** Price a launch inside the pipeline that will actually run, not in a loop
that does nothing else.

**Generalises:** on this hardware a grid-wide barrier is not a cheaper kernel
boundary, it is a much more expensive one. Any design that reaches for
`grid.sync()` to avoid a launch should price both first -- it is a twenty-line
microbenchmark. Note the shape of the cost too: launch cost is flat in grid size
and sync cost is linear in it, so the bigger the grid the worse the trade.

### At small M this op is host-bound, and no gate or report could see it
Measured 2026-09-18 after the red-zone kernel work ran out of directions. The
enqueue cost of one `top_k_per_row_prefill` call -- the caller's CPU time before
it can do anything else -- against the same call's end-to-end time, 200 calls
with one synchronise at the end:

```
                  BEFORE memoising          AFTER
shape           enqueue   e2e   host%    enqueue   e2e
M=16  N=32768    25.29   25.43  99.4%     25.61   25.75   (routes to mb/ob now)
M=64  N=65536    32.05   32.19  99.6%     25.47   27.65
M=256 N=65536    32.25   37.45  86.1%     25.54   37.93
M=64  N=131072   32.73   32.90  99.5%     25.80   32.51
M=4096 N=131072  33.45  636.78   5.3%     25.68  636.24
```

**At M <= 64 the call was 99% host.** The GPU work -- 21.6 us of kernel at
M=16 N=32768 -- was entirely hidden behind 25.3 us of CPU. Every kernel-side
number in `reports/ceiling_report.html` and every `score_grid` cell is GPU time,
so none of them can show this, and a kernel improvement in that region would not
have reached the caller at all.

Two unmemoised binding lookups were 9.7 us of it: `topk_sampled_supports` at
4.86 us/call in the dispatcher and `topk_sampled_workspace_size` at 4.80 us/call in
the wrapper. Both are pure functions of (numRows, stride0, k); `lru_cache` took
them to 0.089 and 0.092 us, 53x. Fixed in aiter-topk 6a71f2f33.

**What is left, decomposed at M=64 N=65536 (enqueue us):**

```
raw binding _top_k_per_row_prefill_sampled        18.07
+ public wrapper                              19.99
+ dispatcher                                  20.52
  get_module (lru-cached)                      0.063
  get_topk_scratch_workspace                   1.064
```

So ~18 us sits inside the binding itself: three `hipLaunchKernel` calls at about
3 us of CPU each, plus marshalling four tensors and five scalars. `get_module`
is already cached and is not the cost. That is aiter framework territory rather
than this op, and it is the ceiling on anything done at the Python layer here.

**Note the asymmetry that makes this easy to get wrong.** A kernel launch costs
about 0.5 us of GPU gap when the queue is full (wall minus summed kernel time)
but about 3 us of HOST time regardless. The two are different resources and the
binding constraint at small M is the host one.

## Structural facts worth keeping (2026-09-18 additions)

- `RowExtents` (`csrc/topk_common.hip.hpp`) is the ONLY place `rowStarts[]` and
  `rowEnds[]` are dereferenced. `row_len_dev()` just below it has no callers.
- Kernels live in BOTH `csrc/topk_generalize.hip.hpp` and
  `benchmark_topk.hip.cpp`; a change to the extent contract has to touch both,
  and `scripts/export_aiter_op.py` then carries it into aiter.
- 16 of the 34 device kernels are `RAGGED=false`. Their instruction streams are
  byte-identical across the clamp, which is why the scored pow2 grid cannot
  regress and the ragged A/B is the only measurement that can see the cost.

## The customer spec reframes what is worth optimising (2026-09-19)

The spec is: fp32 logits `[M, N]`, arbitrary M/N, `M` in 1, 4, 8, ... 4k, `N` in
512, 1024, ... 1M, `topk = 2048`. Two consequences that should stop work rather
than start it, and one that started work:

- The `N > 1.5M` cliff is **out of spec**. Do not spend time on it.
- The `k = 512` crossover is **out of spec**; the spec fixes `topk = 2048`.
- `N = 512` and `N = 1024` are **in spec and had essentially no coverage**.
  `bench/grid.py` bottoms out at `N = 2048`, `verify_grid.py` inherits that
  floor, and `stress_topk.py` touched `N = 512` in one boundary case. Closed by
  `bench/spec_low_n.py`.

Those two columns are not a smaller grid point. At `topk = 2048` they are the
`k >= N` regime -- every element of the row is selected, so the answer is a
permutation of the row plus a `-1` tail -- and the dispatch lands on aiter's
one-block path, not AVO, because AVO needs `stride0 >= 32768`.

### The unmemoised-binding bug was on the mb/ob path too, and it was bigger

`6a71f2f33` fixed `topk_sampled_supports` and `topk_sampled_workspace_size`. The same
mistake sat one branch over, on the path every small shape actually takes.
Measured through the binding on this box:

```
topk_use_mulblocks      6.344 us
topk_ob_workspace_size  6.740 us
topk_mb_workspace_size  5.059 us
```

The one-block dispatch calls the first two on every call: 13.1 us of host time
to pick a path and size a workspace, in front of a kernel that rocprofv3 times
at 2.36 us. Fixed in aiter-topk `76e94f4df`, which took the spec's low-N corner
from 24.7-26.2 us of wall time to 15.8-16.3 us, 32/32 still correct.

**The control is what makes that number trustworthy.** `M=4096 N=4096` is the
one cell in that sweep whose time is real GPU work, and it did not move: 70.13
-> 70.11 us. Every cell that was waiting on the host dropped about 9.5 us. A
uniform drop across all 32 cells would have been much weaker evidence, because
it is also what a measurement artefact looks like.

Note the saving realised (9.5 us) is less than the two lookups measured in
isolation (13.1 us); the per-call microbenchmark of a binding includes overhead
that is shared once both are on the same call. Trust the end-to-end number.

**Generalise this before it bites a third time:** any `@compile_ops` binding
that is a pure function of its arguments costs about 5-7 us per call through the
binding layer alone. On this op that is two to three times the kernel. Grep for
`@compile_ops` shape queries on any hot path and assume each one is 6 us until
measured otherwise.

Still 16 us of wall time around a 2.4 us kernel, so the low-N region remains
host-bound. What is left, from cProfile at `M=64 N=512`: `torch_to_aiter_pybind`
5x per call, the `compile_ops` wrapper 3x, `torch.empty` 5x.

### A green sweep that measured the wrong module

The first post-memoisation run of the low-N sweep reported no change, then
`aiter has no attribute top_k_per_row_prefill`. Cause: `python /script.py` sets
`sys.path[0]` to the **script's directory**, not the cwd, so `import aiter`
resolved to the copy in site-packages instead of the `/aiter` mount. The run was
green against a module we had not touched.

`-e PYTHONPATH=/aiter` fixes it, but the durable fix is the assertion now at the
top of `bench/spec_low_n.py`: print `aiter.__file__` and abort unless it is
under the mount. This is the same failure class as the g_11 gate that was green
because it never ran the changed instantiation -- a check that verifies the
wrong object is worse than no check, because it produces confidence.

## Profiling the anchor, 2026-09-22 (see reports/anchor_profile_2026-09.md)

### The read-only floor AND the gx=1 read+write floor were both the wrong yardstick

`scripts/bw_kernel.hip` measures `PHASEB_FLOOR_read+write` at `dim3(1, M)` with a
grid-stride walk. `phase_b_filter_coop` runs `dim3(8, M)` with a contiguous per-block
chunk. Judged against the gx=1 floor (435.84 us) phase_b looks like 1.05x; against a
floor measured at its OWN geometry (`scripts/bw_gx_floor.hip`, G=8 wg512, 423.56 us) it
is 1.077x. Small difference here, but the habit matters: `bandwidth_g26.md` drew its
"streaming is already at peak" conclusion from the mismatched shape.

### The write stream, not the write GRANULARITY, is what costs

Same kernel, same geometry, only argv[1] changes: read-only 339.88 us, read + 93.98 MB
of 8 B candidate writes 423.56 us. The 94 MB costs **83.68 us = 1.123 TB/s effective**,
against 6.318 TB/s on the read. PMC says the writes are already clean -- 95.7% of
`TCC_EA0_WRREQ` are full 64 B, `TCC_EA0_RDREQ_32B` is 0, `TCC_EA0_WRREQ_STALL` is
negligible. So the old note "small scattered stores are what a filter pass pays for" is
now only half true: the LDS wave-staging fixed the granularity, and what remains is the
flat cost of mixing any write stream into a read stream on this HBM. **The only lever
left on that 83.7 us is writing fewer bytes** (narrower candidate records, or a tighter
margin), not writing them better.

### phase_a + phase_c are 21% of wall on 10% of the traffic

63.20 + 59.48 = 122.68 us moving 262.9 MB, i.e. 2.13 and 2.16 TB/s. Charged at the
read+write floor rate that traffic is 49.7 us, so ~73 us is not memory time. Both are at
full occupancy (LDS 4,608 / 5,632 B, VGPR 16 / 24), so it is not occupancy. ATT per-wave
critical path on phase_c: `lgkmcnt` + `s_barrier` = 35-38%, against 0.3% aggregate stall
on `ds_read` itself -- the LDS radix select's barrier/scan dependency chain, not LDS
bandwidth and not bank conflicts. This is the largest remaining structural target, and
it is the same conclusion `.evo/session_checkpoint.md` reached for small M ("S3:
parallelize phase_a and phase_c"), now shown to hold at the anchor too.

### phase_b has no load pipelining depth: every vmcnt wait is vmcnt(0)

ATT aggregate: `s_waitcnt vmcnt(0)` holds 70.9% of stall, `buffer_load` itself 0.0%.
Per-wave critical path: 66.8-74.5%. One single static `s_waitcnt vmcnt(0)` accounts for
67.5M of the 68.4M vmcnt stall cycles. The wave runs only **8 loop iterations**
(`SQ_INSTS_VMEM` = 9.14 per wave), so there is no room to software-pipeline inside a
wave; occupancy is what hides it, which is why the kernel is still within 7.7% of its
floor. **Not yet falsified**: the causal weaken-the-wait A/B (`att.md` 3b) needs a source
edit and was not run.

### phase_b spends 51% of VALU issue capacity, at 17 lane-ops per element

`SQ_INSTS_VALU` = 143,595,723 per dispatch (confirmed twice, by hand-rolled rocprofv3 and
by rocprof-compute, agreeing to 0.1 ppm). Against 256 CU x 4 SIMD x 2.4 GHz / 4 that is
51%. The header comment claims "ONE integer compare per element"; the real cost is 17
VALU lane-ops per element once the 4x `__ballot` + `__popcll` + conditional LDS
addressing is counted. Not the binding constraint at 4.92 TB/s, but it is the reason
`v_cmp_ngt_f32` and `v_cmp_ne_u32` appear in the top-15 ATT stalls.

### summary.txt in log/large_n_profile is stale and names kernels that no longer dispatch

Dated 2026-09-17 21:41 against a 2026-09-18 14:00 binary. It reports
`phase_b_filter_wavestage` / `phase_c_select_waveseg` at `coop_g=1`. The anchor now
dispatches `phase_b_filter_coop` / `phase_c_select_contig` at `coop_g=8`. Anything that
cites those kernel names as "the anchor breakdown" is citing a superseded build.
Related: `log/large_n_sweep.tsv`'s 609.40 us for this cell is not stale drift, it is the
`coop_g=1` configuration -- re-measured warm as 609.6 us. The coop_g sweep at the anchor
is g=1 609.6, g=2 582.2, g=4 591.8, g=8 581.3, g=16 645.2, g=32 780.0 us.

### FALSIFIED: splitting the exact fallback select across blocks with a hand-rolled barrier
`arch_scope: gfx950`, measured 2026-09-22 on aiter `7d9c2d128` + the sampled
routing widening.

**The cost being attacked is real and worth restating.** One row whose candidate
set comes out unusable costs a flat ~350 us, whatever M is. Measured at
N=524288, gaussian, by seed: m=32 goes 39-40 us at `fb_count=0` to 384-391 us at
`fb_count=1`; m=4096 goes 1881 us to 2243 us. It is flat because the select is
`RADIX_PASSES` full re-reads of the row plus the gather -- `block_select_stream`
filters by pivot prefix inside the loop rather than compacting -- and ONE
workgroup does all five while its 31 or 4095 neighbours have already retired.

**Where it happens is not where it looks.** `phase_d_fallback` is only ever
launched from `run_fallback`, which precedes it with `fill_identity_rows` and so
means "exact select for EVERY row". The per-row fallback is done inline by
`phase_c_select_contig`, which both appends the row to `fb_rows` and calls
`exact_row_select` itself. A first attempt parallelised `phase_d_fallback` and
changed nothing at all, because that kernel never runs on this path.

**The barrier is not the reason it fails, which is the surprising part.**
`scripts/spin_barrier_spike.hip` prices a hand-rolled sense-reversing barrier
over G blocks sharing one row at 0.92 / 0.81 / 1.33 / 1.43 / 1.64 / 2.19 us for
G = 2 / 4 / 8 / 16 / 32 / 64 -- all under the 2.63 us of the extra kernel launch
that is the alternative, and far under the 7.38 us this file records for
`cg::this_grid().sync()` at grid=64. That measurement stands; it is a different
mechanism from the cooperative sync and it is genuinely cheap in isolation.

**What kills it, with Phase C deferring to a split `phase_d_fallback`
(FB_SPLIT=16, 3 barriers per radix pass):**

| case | before | after |
|---|---:|---:|
| m=32 n=524288 seed 0, `fb_count=1` | 391 us | **468 us** |
| m=32 n=524288 seed 1, `fb_count=0` | 39 us | **46 us** |
| m=32 n=262144 all-equal, all 32 rows | 600 us | **2112 us** |
| m=128 n=262144 all-equal, all 128 rows | 761 us | **3814 us** |
| m=128 n=1048576, no fallback | 124 us | **132 us** |

Three separate losses, and the third is the one that generalises:

- **+7 us on every call**, fallback or not, for the extra dispatch plus the
  512 B `hipMemsetAsync` the barrier needs to start from a known state.
- **The split does not repay even at one row.** 391 -> 468 us with 16 blocks on
  the row. Spreading the scan 16 ways did not beat the 13 barriers and the
  per-pass fold of HIST_SLOTS counters into a global histogram.
- **The barrier price is per block AND per concurrent group.** The same spike
  measures G=16 at 1.43 us with one group and **4.22 us with eight**, G=64 at
  2.19 us against **10.39 us**. `all-equal` runs 32 or 128 groups at once, and
  the exact select there was ALREADY fully parallel -- one block per row across
  M blocks -- so the split replaced a perfect arrangement with a contended one.

**Generalises:** a barrier priced in isolation is not priced. The number that
matters is its cost at the concurrency the kernel actually reaches, and the
regime where a split is most tempting (few rows) is the opposite of the regime
that sets the barrier's worst case (many rows). Reverted; the ~350 us
characterisation above is the part worth keeping.

**Still open.** The row falls back because its sampled threshold missed, and
that is width-specific rather than data-luck: over 5 seeds x ~30000 rows,
N=524288 is the ONLY width that trips it, at 1.2% of rows under
`topk_shape.hip.hpp`'s small-S rule (M<=32, `S_RULE1_M_MAX`) against 0.005%
above it. Making the threshold not miss at that width would remove the cost
without touching the select at all, and is untried.

### The N>=128K multiplier is the candidate WRITE, and both ways out are now closed
Measured 2026-09-23 on gfx950 over the published PR 5686 pipe101 floor, with the
profiler-trace ruler that divides by captured events (bench/select_ab_sweep.py).

Per-kernel share of `topk_select` routed to `sampled`, seed 0:

| cell | total | phase_b | phase_a | phase_c |
|---|---:|---:|---:|---:|
| m=16 n=131072 | 26.94 us | 34.4% | 25.4% | **40.1%** |
| m=128 n=524288 | 67.03 us | **66.8%** | 17.9% | 15.3% |
| m=4096 n=262144 | 1088.76 us | **82.6%** | 11.7% | 5.6% |

Two things that redirect any future attempt:

**At small M the biggest kernel is phase_c, not phase_a.** Every "make the
threshold cheaper" idea targets phase_a, which is 25% of an m=16 call. Zeroing it
outright takes m=16 n=131072 from 26.94us to 20.1us, which is 18.7% of the floor
against 13.9% -- still deep red. The phase_a serial-depth lead is real (see the
entry above) and it is not where the time is.

**At large M the excess over the floor is the candidate write, and it is
structural.** phase_b's achieved bandwidth, m=4096:

| N | phase_b | read | TB/s | vs pipe101 floor |
|---|---:|---:|---:|---:|
| 131072 | 486.89 us | 2.147 GB | 4.41 | 1.435x |
| 262144 | 899.27 us | 4.295 GB | 4.78 | 1.391x |
| 524288 | 1672.73 us | 8.590 GB | 5.14 | 1.310x |
| 1048576 | 2812.80 us | 17.180 GB | 6.11 | 1.108x |

The ratio falls monotonically with N and is FLAT in M (1.418 / 1.391 / 1.391 at
M=1024 / 2048 / 4096, N=262144). That is the signature of a per-row cost against
a read that grows with N -- the `margin * K` candidates each row writes, at the
1.123 TB/s effective write rate this box gives a mixed read/write stream. It is
not a streaming inefficiency, so coop_g cannot reach it (swept: the shipped
table is already optimal on every cell tried, including the non-monotone dip at
M=512/1024 N=131072, where coop_g=2 really is best at 0.0739 against 0.0780 at 8).

Both ways to write less are now measured and closed:

- **Fewer bytes per candidate** is arithmetically dead. At the anchor the
  candidate density is 2867/131072 = 2.19%, so a 128 B line has a 51% chance of
  holding one; dropping the key and re-reading it in phase_c touches
  0.51 x 4096 lines x 128 B x 4096 rows = 1.09 GB to save 47 MB of writes.
- **Fewer candidates** is bounded below by the estimator. Safety needs
  `margin * (1 - 3/sqrt(R)) >= 1`, and the joint (S, margin) sweep -- the first
  one, both knobs had only ever been moved alone -- confirms it: at m=4096
  n=262144, margin 1.20 gives 54 fallback rows and 1.453x, 1.25 gives 7 and
  1.290x, 1.30 gives 1 and 1.261x. The shipped auto point is the optimum, and
  the one cell where something beat it (m=512 n=262144 at margin 1.30, S=11520,
  0.970x) regresses m=4096 n=262144 to 1.261x, which is a per-cell fit.

**Consequence.** Of the 42 red pow2 cells at N >= 128K, the 20 at M <= 16 are
dispatch-bound (floor 3.66-5.5us against a three-kernel pipeline) and the rest
are held by a write volume that neither knob can lower. Closing them needs a
different candidate representation or a different pipeline shape, not a tuning
pass.

### The candidate WRITE is the whole compaction cost -- the arithmetic is free
`arch_scope: gfx950`, 2026-09-23, m=4096, k=2048, dist gaussian, phase_b timed
alone with `rocprofv3 --kernel-trace`.

Three ablations, two of which give wrong results and exist only to price a half
(kept as `-DABLATE_COMPACT=n` so they reproduce, same convention as
`ABLATE_HIST_ATOMIC`):

| variant | n=131072 | n=262144 |
|---|---:|---:|
| shipped | 465.87 us | 869.18 us |
| `ABLATE_COMPACT=1` fixed-slot write, no prefix arithmetic | 465.19 | 870.82 |
| `ABLATE_COMPACT=2` all arithmetic, no `ds_write` | 464.55 | 873.58 |
| tiny margin so nothing passes the threshold | **363.52** | **717.40** |

The first three are the same to within noise. **Removing the ballot-prefix
arithmetic changes nothing; removing the LDS write changes nothing; removing the
CANDIDATES saves 100.35 / 151.84 us.** So the cost is the 94 MB of global
candidate writes and their drain, not the compaction instructions.

That kills a plausible-looking lead, recorded here because it looked strong:
the compaction cost is 100 us at n=131072 and 152 us at n=262144 for the SAME
candidate volume, which reads as "per-iteration overhead" and is not. The extra
52 us is the read/write mixing penalty getting worse as the read stream grows --
a hardware property, not something the kernel can schedule around.

Confirmed independently by PMC on the same kernel: `TCC_EA0_RDREQ = 16783874`
x 128 B = 2.148 GB, exactly the input, with **0% 32-B requests**;
`TCC_EA0_WRREQ = 1526620` at **95.7% full 64 B**; `TCC_EA0_WRREQ_STALL` 1.4e4
against 1.5e6 requests; TCC hit 5.6% (pure streaming). phase_b wastes no bytes.

**Consequence.** Writing fewer bytes per candidate is arithmetically dead (see
the density entry above) and writing fewer candidates is bounded below by the
estimator (see the joint (S, margin) entry). With the compaction instructions
now priced at zero, there is nothing left in phase_b at large M.

Also measured and closed while looking: a wave-uniform `if (b_k)` guard on each
of the four compaction slots -- 24% of slots are empty across all 64 lanes at
2.19% density -- is correct on all five distributions but 0.9% (n=131072) to
1.8% (n=262144) SLOWER. The scalar branch costs more than the skip saves.
`--cf-block` was swept for the first time and `auto` (512) is already optimal:
64 costs 2.2x, 128 1.39x, 256 1.04x, 1024 1.26x.

### Ceiling: phase_a and phase_c free still leaves 15 of 40 cells red
`arch_scope: gfx950`, 2026-09-23, k=2048, dist gaussian, seed 0, per-phase times
from `rocprofv3 --kernel-trace`, efficiency against the published PR 5686
pipe101 floor. The "free" column is `phase_b measured + 2 x 2.64us` of dispatch
(the empty-launch marginal from knowledge/g0_floor_model.json), i.e. what the
pipeline would cost if the threshold and the select were instantaneous.

```
            N=131072      N=262144      N=524288      N=1048576
   M=1    18.1 -> 34.9  16.0 -> 33.5  14.2 -> 30.3  13.9 -> 30.0
   M=4    15.7 -> 31.0  14.8 -> 29.9  14.1 -> 27.5  18.1 -> 37.3
   M=16   14.3 -> 26.5  13.5 -> 25.3  15.0 -> 28.3  24.5 -> 44.0
   M=64   19.6 -> 35.8  28.5 -> 50.6  40.0 -> 63.3  55.1 -> 77.1
   M=128  31.7 -> 53.1  42.4 -> 67.0  55.6 -> 76.5  60.3 -> 72.6
   M=256  42.7 -> 63.8  55.9 -> 76.6  59.3 -> 71.0  70.0 -> 78.9
   M=512  55.5 -> 77.5  55.5 -> 69.5  68.1 -> 78.8  71.5 -> 77.7
   M=1024 52.2 -> 67.7  62.5 -> 79.6  65.9 -> 75.5  73.0 -> 79.4
   M=2048 59.4 -> 79.7  59.3 -> 73.1  67.6 -> 77.2  74.2 -> 80.1
   M=4096 56.4 -> 72.3  60.7 -> 73.9  68.6 -> 77.5  83.4 -> 90.3
```

**12 of 40 are at or above 60% now; 25 would be if phase_a and phase_c cost
nothing.** The other 15 -- every cell at M <= 16, and M=64 at N <= 262144 --
stay between 25% and 51% in a limit that cannot be reached. Chasing 60% there is
chasing a number this pipeline shape cannot produce, because phase_b alone plus
two dispatches already exceeds the floor/0.6 budget.

This bounds every remaining direction at once, and it should be the first thing
read before opening a new one:

- phase_b is at 94% of its streaming floor (ablation: 363.52us for 2.147 GB =
  5.91 TB/s against 6.3) and its remaining excess is the candidate write, closed
  three independent ways above.
- phase_a and phase_c are flat in N (6.6 -> 13.7us and 11.0 -> 13.5us across
  N=131072..1048576 at M<=128) and their block width is already optimal --
  swept for the first time here, and SMALLER is strictly worse: phase_a at
  m=16 n=131072 reads 6.64us at 1024 threads, 8.30 at 512, 9.59 at 256, 13.88
  at 128. The barrier-cost-per-wave argument for a narrower block is wrong;
  fewer threads means more elements each, and that dominates.
- `coop_g` was re-swept at small M and LARGE N, which the 2026-09-18 sweep never
  covered (it was at N=32768): auto is within 0-3.7% of the best value at
  m=16 n=131072, m=16 n=1048576, m=64 n=524288 and m=128 n=1048576.

### OPEN LEVER: rule 0's sample count is too large at N <= 262144, and the
### neighbourhood is cliff-edged
`arch_scope: gfx950`, 2026-09-23, k=2048, dist gaussian, seed 0, full-pipeline
wall from benchmark_topk, warmup 20 / iters 100 / repeats 3.

Not falsified and not shipped: a real effect that needs more work than one pass.

`derive_sample_s_for_n` rule 0 is `S = R_TARGET * N / (margin * K)` with
R_TARGET = 179. Swept S per cell over M = 64..4096, N = 131072..1048576:

| N | S auto picks | measured best | ratio vs auto |
|---|---|---|---|
| 131072 | 8192 | **6144** | 0.981 - 0.995x, and 6144 wins at ALL SEVEN M |
| 262144 | 16384 | **8192 - 12288** | 0.952 - 0.986x |
| 524288 | 16384 | 16384 | auto already best |
| 1048576 | 16384 | 16384 | auto already best |

At N >= 262144 the law is saturated -- it asks for 16365 at N=262144 and 65462
at N=1048576 and gets SAMPLE_S_MAX either way -- so R_TARGET stops being a law
there. The file already says R_TARGET "does not transfer" and that rule 1 was
written to derive the requirement instead; rule 1 is gated to M <= 32 by
S_RULE1_M_MAX, so M >= 64 never gets the derived form.

The win reproduces tightly where it was checked properly. Interleaved A/B/A/B,
six pairs in one clock state, auto against S=12288:

    m=4096 n=262144   auto 1.0474 (1.0453-1.0482)   S=12288 1.0266 (1.0256-1.0276)   0.9802x
    m=512  n=262144   auto 0.1399 (0.1395-0.1402)   S=12288 0.1377 (0.1376-0.1382)   0.9843x
    m=4096 n=131072   auto 0.5884                    S=12288 0.6193                   1.0526x

Per-arm spread is 0.3% and the arms do not overlap, so these are real at 2%.
Note the third line: the same S is 5.3% WORSE at N=131072, so any rule has to be
a function of N, not a constant.

**Why this is not shipped.** The neighbourhood is cliff-edged and the cliff is
not understood. S=10240 at N=262144 sends rows to the exact fallback at
m=256 (3.28x), m=512 (2.20x), m=1024 (2.20x), m=2048 (1.53x) and m=4096 (1.24x)
while 8192 and 12288 next to it are clean. S <= 8192 at N=1048576 is 1.45x to
**13.62x**. It is not simple stride exactness -- both 10240 and 12288 take the
masked (non-exact) stride at N=262144 and only one of them falls off. Before any
of this ships it needs: the cliff explained, a distribution gate (gaussian only
so far, and knowledge/known_bad.md is emphatic that sampling knobs are
distribution-sensitive), and a full-grid A/B.

**What it would be worth.** At N=262144 the gains are 0.952-0.986x, which moves
m=2048 from 59.3% of the pipe101 floor to 60.6% and m=4096 from 60.7% to 61.8%.
At N=131072 the gains are real but 0.5-2%, and flip nothing. So this is worth
perhaps two pow2 cells, not a regime change -- weigh that against the cliff
before spending on it.

## phase_b's 100us is the epilogue's candidate write, and neither alignment nor
## the thread map can reach it (gfx950, 2026-09-23)

`arch_scope: gfx950`. m=4096 k=2048 --dist gaussian --seed 0, phase_b device time
from rocprofv3 --kernel-trace, upper three quartiles of 20 launches.

Two earlier readings in this file are WRONG and are corrected here.

**Correction 1 -- the ABLATE_COMPACT prices above are void.** `scripts/price_compact.sh`
ran `EXTRA="-DABLATE_COMPACT=$A" make -j`, and the Makefile never reads `$EXTRA`.
make succeeded, so the `||` hipcc fallback never fired and all three arms built the
SAME binary. "prefix arithmetic free, compaction ds_write free" was three runs of
the shipped kernel. Rebuilt with hipcc carrying the define, the staging ds_write is
24.8us at n=131072 and 29.6us at n=262144 -- real, not free.

**Correction 2 -- ABLATE_EPI=3 is not an alignment ceiling.** It points every wave's
dst at `row_base`, which shrinks the footprint eightfold and lets the eight stores
overwrite each other. Its 56.0us/128.9us prices a smaller write. The honest version
(ABLATE_EPI=6: full footprint, same line count, head rounded down to 128B) is worth
NOTHING -- 459.81 against 457.14 at n=131072.

**Where the time is.** The cost lives in the epilogue, not in the filter loop. Every
ABLATE_DRAIN variant missed it because COOP_DRAIN_WAVE is `#undef`'d before the
epilogue and the epilogue carries its own inline copy loop.

    variant                                     n=131072   n=262144
    shipped                                       461.28     868.39
    no epilogue copy loop        (ABLATE_EPI=1)   362.99     715.25
    no epilogue at all           (ABLATE_EPI=2)   332.79     678.22
    th=+inf, nothing passes      (ABLATE_TH=1)    340.63     694.65

The copy loop is 98.3us and 153.1us. The rest of the epilogue -- two __syncthreads,
the serial prefix over waves, one block atomicAdd -- is 30.2us and 37.0us.

`--margin 0.02` is NOT a valid "nothing passes" control: derive_shape_params feeds
margin into `rank` (:130) and `cap` (:131), so it moves the configuration as well as
the branch. ABLATE_TH=1 sets `th = INFINITY` inside phase_b and leaves every
host-side parameter alone; it reads 340.63 where --margin 0.02 read 363.42.

**Two levers tried on the copy, both nearly worthless.**

    one block-wide contiguous walk (ABLATE_EPI=5, CORRECT, 5-dist PASS)  -6.0us  -5.1us
    per-wave parallel copy         (ABLATE_EPI=4, CORRECT)             -12.8us  -8.2us
    head forced to a 128B boundary (ABLATE_EPI=6, wrong results)         +2.7us  -2.4us

s_off is a prefix sum, so `row_base + s_base + s_off[w]` over consecutive w is
already ONE contiguous run; the eight-chunk walk only restarts the thread-to-address
map inside it. Collapsing that to a single walk is correct and measurable but small.

**What is left is bytes.** ~2867 candidates per row (margin 1.4 x K=2048) x 4096 rows
x 8B = 93.9MB in 98.3us = 0.96 TB/s, against 5.93 TB/s for phase_b's 2.148GB of
reads. That is the same write-costs-5.6x-read-per-byte the anchor ledger recorded,
so the copy is not mis-issued -- it is paying the box's write price for every byte.
Alignment, the thread map, and the number of passes are all closed. Only fewer bytes
move this: a narrower candidate record, or fewer candidates.

## The candidate write costs what it DISTURBS, not what it writes (gfx950, 2026-09-23)

`arch_scope: gfx950`. Follows the entry above, same harness.

m=4096 k=2048 auto shape, from scripts/sp2.cpp:

    N         S      margin   rank    cap   coop_g   expected candidates/row
    131072    8192   1.400    179     4096  8        2867
    262144    16384  1.400    179     4096  8        2867

The two widths plan IDENTICAL candidate counts, so phase_b's epilogue writes the
same 2867 x 4096 x 8B = 93.9MB at both. It costs 98.3us at N=131072 and 153.1us at
N=262144 -- 1.56x for the same bytes. The only thing that changed is the read
stream around it, 2.148GB against 4.295GB.

So the epilogue write is not paying its own bandwidth. It is paying to interleave
with the read stream, and the bill scales with the reads it interrupts. Three
measurements agree and none of them make sense under a pure-bandwidth model:

    lever                                    N=131072   N=262144
    halve the record to 4B (ABLATE_EPI=7)      -5.9us     -48.4us
    --margin 1.40 -> 1.08, 23% fewer cands     -8.9us     -21.5us
    head forced to a 128B boundary (EPI=6)      +2.7us      -2.4us

Halving the bytes buys 6% of the copy at N=131072 and 31% at N=262144. Under a
bandwidth model both would be ~50%. Under an interference model the win tracks how
much read traffic there is to protect, which is what the numbers do.

Occupancy is NOT the mechanism. wbuf is WSTAGE_WAVES=8 x WSTAGE_CAP=320 x 8B =
20KB, and at 512 threads per block gfx950 admits 4 blocks/CU on wave slots against
7 on LDS, so the staging buffer is not the binding constraint and shrinking it
cannot raise occupancy.

**What this closes.** Every in-place fix to the copy is capped by the interference,
not by the copy: alignment 0, thread map -6.0us, per-wave parallel -12.8us, halving
the record -5.9us at the width where the read stream is smallest.

**What it opens.** The write only has to exist because phase_c is a separate kernel
that reads the candidates back. cap=4096 x 8B = 32KB fits in LDS, so a phase_b/c
fused at coop_g=1 would never put a candidate in global memory at all: it removes
98.3-153.1us from phase_b AND the same 93.9MB read from phase_c, and it keeps the
read stream pure. coop_g=1 means one block per row, which is 4096 blocks at M=4096
(16 per CU) but ONE block for the whole row at M=1, so it has to be M-gated. NOT
attempted -- recorded as the direction the pricing points at.

## Non-temporal loads: the implementation is the whole result (gfx950, 2026-09-23)

`arch_scope: gfx950`. This EXTENDS "Non-temporal loads in the streaming filter"
near the top of this file, which measured 0.7930 against 0.7916 ms and closed the
direction. That reading is right for what it tested and wrong as a general answer.

The knob it tested, `--nt-load` / `g_use_nt_load`, flips a `__constant__` read
inside `load_f4` (csrc/topk_common.hip.hpp:285). Two consequences: the branch sits
in the innermost load, and `load_f4` is shared, so phase_a and phase_c's exact
fallback get non-temporal loads too -- and those DO reuse what they read.

Re-measured, three-kernel device total, k=2048 --dist gaussian --seed 0, all four
arms back to back on one card:

    M     N        cached   --nt-load 1   phase_b template   size gate
    4096  131072   580.52     576.23        555.18            553.95
    4096  1048576 3068.06    3084.03       2776.12           2765.43
    128   1048576  129.11     127.31        114.12            114.51
    128   131072    35.13      35.77         36.49             35.81
    64    262144    39.49      38.72         41.11             38.70

The runtime knob is 0.981x to 1.018x -- neutral, as the original entry found. The
same intrinsic as a compile-time template parameter on phase_b_filter_coop alone
is 0.884x to 0.956x on the wide cells.

**It is not free everywhere.** Unconditional, it LOSES at small work: m=64
n=262144 1.050x, m=128 n=131072 1.042x, m=16 n=1048576 1.031x, m=64 n=131073
1.029x, m=1 n=1048576 1.020x. The winners and losers separate cleanly on input
size and nothing else:

    win  (0.884x-0.969x)   M*pitch >= 2^27   = 512MB and up
    lose (1.020x-1.050x)   M*pitch <= 2^26   = 256MB and down

2^27 elements is 512MB, the first size that cannot sit in gfx950's 256MB MALL.
Below it the input CAN stay resident across calls and the caching is the whole
benefit; above it nothing survives anyway and the cache line only evicts what the
other blocks are still reading. Shipped as that gate, which is `--nt-gate` on
benchmark_topk (-1 = gate, 0 = off, 1 = on).

Measured over 27 shapes with the gate: every cell at or above 2^27 is 0.884x to
0.969x, and every cell below it compiles the SAME device code as the gate-off arm
(the same phase_b_filter_coop<RAGGED,false> instantiation, only the host-side
branch differs), so the 1.045x at m=128 n=131072 and 1.033x at m=16 n=131072 are
run-to-run spread, not regressions. That also sets the noise band at these sizes:
+-4.5% at 35us, which is worth remembering before reading a small-cell A/B.

--mode verify is VERDICT PASS on all 245 of 7 M x 7 N x 5 distributions.

**The lesson to carry.** "Tried the intrinsic, it did nothing" is not a result
about the intrinsic. Where the branch lives and which kernels inherit it decided
the sign here.

## The same intrinsic, one kernel later, is a 17% regression (gfx950, 2026-09-23)

`arch_scope: gfx950`. Follows the entry above, which shipped non-temporal loads in
phase_b behind a size gate. The obvious next step -- phase_c reads the candidate
array once and never again, and phase_b now writes it non-temporally so it is not
in cache anyway -- is WRONG.

Three-kernel device total and phase_c alone, k=2048 --dist gaussian --seed 0:

    M     N          phase_c cached   phase_c non-temporal
    4096  131072        72.64us          72.85us
    4096  1048576      125.81us         141.48us    +12.5%
    1024  524288        31.18us          36.55us    +17.2%
    128   1048576       13.95us          15.48us    +11.0%
    128   131072        10.88us          10.53us

Kept as NT_CAND, defaulting to 0.

[unverified hypothesis] for the sign flip, and it is worth checking before reusing
either result: phase_b's streaming load is a dwordx4, so a 64-lane wave covers
1024B and every line it touches is fully consumed whether or not it is cached.
phase_c's candidate read is 8B per lane, so a wave covers 512B and a cached fetch
of a 128B line serves 16 lanes at once. Bypassing that is giving up the sharing,
not avoiding pollution. Request counts would settle it; they were not measured.

Either way the rule "this read is streamed, so make it non-temporal" does not
survive contact: it won by up to 11.6% one kernel earlier and lost by up to 17.2%
here, in the same pipeline, on the same data.

## Fusing phase_b into phase_c: the prize is real and the entry price is higher
## (gfx950, 2026-09-23)

`arch_scope: gfx950`. Closes the direction the entry above pointed at. k=2048
--dist gaussian --seed 0, three-kernel device total, upper three quartiles of 20
launches, measured on top of the shipped non-temporal loads.

**The prize, priced from both ends.** ABLATE_EPI=1 removes phase_b's candidate
write; ABLATE_CREAD=1 removes phase_c's read of the same array. Both give wrong
results and exist only to price the pair a fused kernel would delete:

    M     N         shipped   no write   no read    neither
    4096  131072    556.93     464.11     536.74     453.88   -103.1us  0.815x
    4096  262144    974.84     852.62     954.56     840.78   -134.1us  0.862x
    1024  524288    433.82     412.33     427.13     401.17    -32.7us  0.925x

**The price.** Keeping candidates in LDS means one block owns the whole row, so
coop_g=1. phase_b alone, against the shipped coop_g=8 at 512 threads:

    M     N         g8 b512   g1 b512   g1 b1024   g2 b1024
    4096  131072     417.22    505.72      n/a       447.97
    4096  262144     778.65    982.00      n/a       852.06
    1024  524288     365.89    546.36      n/a       391.08

coop_g=1 costs +88.5us at N=131072 and +203.4us at N=262144, against prizes of
103.1us and 134.1us. It is a wash at the first width and a clear loss at the
second. 1024 threads with WSTAGE_WAVES raised to 16 does not launch at coop_g=1
(it does at coop_g=2, so it is not the staging buffer); coop_g=2 is cheap
(+30.8us, +73.4us) but splits a row's candidates across two blocks, which is
exactly what fusion cannot have.

Worse at small M: m=128 n=1048576 goes 86.54us to 351.84us at coop_g=1, 4.07x,
because 128 rows give 128 blocks for 256 CUs.

[unverified hypothesis] why coop_g=1 is slower when occupancy is full either way
(4096 blocks is 16 per CU, and only 4 fit at once): granularity. coop_g=8 puts
32768 blocks through 1024 concurrent slots, 32 waves of blocks, so the ragged
last wave is 1/32 of the run; coop_g=1 gives 4 waves and a tail worth 1/4. Not
measured -- a block-start/end timestamp histogram would settle it.

**Conclusion.** Fusion is closed at these widths. It would need a coop_g=1 read
that costs less than 100us more than coop_g=8, and nothing tried here gets close.

## phase_a is half read and half select, and the read is not the slow half
## (gfx950, 2026-09-23)

`arch_scope: gfx950`. ABLATE_PA=1 keeps phase_a's sample load and its LDS fill and
drops the radix select. k=2048 --dist gaussian --seed 0, phase_a device time:

    M     N         S       full    read only   select   read GB/s
    4096  131072    8192     65.82     33.78     32.04     3973
    4096  262144   16384    124.52     60.16     64.36     4462
    4096  1048576  16384    116.36     57.29     59.07     4686
    1024  524288   16384     35.95     15.80     20.15     4247
    128   1048576  16384     14.13      5.91      8.22     1419

Corrects an estimate made earlier in this session: dividing phase_a's whole 124us
by its 268MB gave 2.16 TB/s and the conclusion that the sampler reads badly. Half
that time is the select. The read alone is 4.46-4.69 TB/s against phase_b's 5.93
on the same card, so the headroom there is about 25%, or ~15us at m=4096
n=262144 -- not the 79us the bad arithmetic suggested.

The select is 64.36us at m=4096 n=262144, 6.6% of the 975us three-kernel total.
It is a multi-pass radix over S=16384 keys in LDS per row, and it is the larger
half at every shape measured. OPEN as a direction; not attempted.

m=128 n=1048576 reads at 1419 GB/s because 128 rows give 128 blocks, which is the
same not-enough-blocks floor coop_g=1 hits in the fusion entry.

## The sample-count cliff is derive_cap halving, and rule 0 was never checked
## against CAP_SAFE_FILL (gfx950, 2026-09-23)

`arch_scope: gfx950`. Closes the OPEN item recorded earlier in this session, which
said a forced sample count of 10240 at N=262144 cost up to 3.28x for reasons that
were "unexplained".

There is nothing unexplained. scripts/sp3.cpp prints the plan `derive_shape_params`
produces for a forced S at M=4096 K=2048 (cand_hi is the three-sigma upper edge of
the candidate count, sigma is how many of those fit between the expected count and
the cap):

    N=262144    S    margin   rank    cap   cand_hi  hi/cap  sigma
              4096   2.129     68    8192     5945   0.726   7.25
              6656   1.712     89    8192     4622   0.564  12.61
              8192   1.600    102    8192     4248   0.519  15.18
             10304   1.502    120    4096     3916   0.956   3.64
             12480   1.436    140    4096     3688   0.900   4.64
             14656   1.400    160    4096     3547   0.866   5.43
             16384   1.400    179    4096     3510   0.857   5.74   <- shipped

The cap HALVES between S=8192 and S=10304. A smaller S forces `auto_margin` up,
and a larger margin is what buys the bigger cap; once the margin settles to 1.4-1.5
the cap drops to PHASE_C_CAP and the headroom goes with it. S=10304 lands at 0.956
of the cap with 3.64 sigma, which is the same corner CAP_SAFE_FILL was introduced
to remove at N=524288 (3.09 sigma, 1.2% of rows to the exact fallback at ~350us
flat). The 3.28x is that fallback, not a mystery.

`sample_stride_exact` is NOT the discriminator -- all seven S above are exact --
and neither is align_sample_s, which only rounds to a multiple of
SAMPLE_CHUNK_ELEMS=64, so a request for 10240 is served as 10432.

**What that leaves of the lever.** The only S below the shipped one that keeps the
cap is 12480-14656. phase_a scales linearly with S and is 124.52us at this shape
(60.16 read + 64.36 select), so S=12480 buys about 30us, and the margin it forces
costs 2.6% more candidates for phase_b to write and phase_c to select. That
matches the 0.9802x measured earlier by interleaved A/B. It is a 2% win bought with
5.74 -> 4.64 sigma of fallback headroom, and one fallback row costs ~350us flat.
Not taken.

**Separate finding worth acting on.** The shipped rule-0 plans at N=131072 and
N=262144 both sit at 0.857 of the cap -- ABOVE the CAP_SAFE_FILL = 0.85 that this
repo already applies elsewhere. That constant was added to rule 1's acceptance in
`derive_sample_s_for_n` only; rule 0, which serves every M > 32, has never been
checked against it. bench/fbrate.py found no fallback rows at those widths, so
this is a latent margin rather than a live bug, but the two paths disagree about
what "safe fill" means and only one of them is enforced.

## phase_a's radix select is near its floor at 3 passes, and compaction cannot
## pay for itself there (gfx950, 2026-09-23)

`arch_scope: gfx950`. Extends the phase_a read/select split above. k=2048 --dist
gaussian --seed 0, phase_a device time, `--phase-a-passes` walked up from the
ABLATE_PA=1 floor (load + LDS fill, no select):

    m=4096 n=131072        floor 33.50
      plain    1 pass 45.81   2 57.13 (+11.32)  3 64.72 (+7.59)  4 72.61 (+7.89)
      compact  1 pass 45.61   2 61.14 (+15.53)  3 67.29 (+6.15)  4 70.88 (+3.59)
    m=4096 n=262144        floor 60.12
      plain    1 pass 85.91   2 108.15 (+22.24) 3 123.23 (+15.08) 4 138.22 (+14.99)
      compact  1 pass 95.55   2 119.51 (+23.96) 3 129.79 (+10.28) 4 137.73 (+7.94)

Two things to take from this.

**The passes cost the same whether or not they are filtered.** Passes 2 and 3 only
touch keys whose high digits match the pivot -- about 1/256 of them -- and still
cost 15.08us against pass 1's 25.8us. So the cost is re-reading all c keys out of
LDS and evaluating the filter, not the histogram atomics.

**Compaction works and still loses.** `block_select_lds_compact` gets the later
passes down (pass 4 marginal 7.94 against 14.99) but pays +9.6us on pass 1 to do
the compacting. At the shipped npasses=3 it is a net LOSS, 129.79 against 123.23,
and only breaks even at 4 passes. With only two passes after the first there is
not enough left to amortize the compaction. `--phase-a-compact 1` measuring
"neutral" earlier was this, not an inert flag.

Note the default is npasses=3, not RADIX_PASSES=4: `--phase-a-passes 4` measures
72.61us where the default measures 64.72us.

**How much is left.** At m=4096 n=262144 the three kernels total 975us and phase_b
alone is 778us of it, reading 4.295GB at 5.52 TB/s against the 5.93 TB/s the
load-only ablation reaches -- 93% of its own read floor. Against a pure
read-the-input-once floor of 724us the pipeline is at 74%, and the 251us of
difference is phase_a's read (60) and select (63), phase_c (72), and phase_b's own
overhead (54). There is no large single item left at this shape.

### Correction to the rule-0 / CAP_SAFE_FILL flag above

The entry above flagged that the shipped rule-0 plans at N=131072 and N=262144 sit
at 0.857 of the cap, above the CAP_SAFE_FILL = 0.85 that rule 1 enforces, and
called it a latent margin. That framing is wrong and the direction is backwards.

Fill is a proxy for the quantity that actually decides whether a row falls back,
which is how many standard deviations of the candidate count fit between its
expected value and the cap. The same fill maps to a different sigma at a different
operating point, so the two rules cannot share the constant:

    plan                                    fill    sigma
    rule 0, N=131072, S=8192   (shipped)    0.857    5.74
    rule 0, N=262144, S=16384  (shipped)    0.857    5.74
    rule 1, N=524288, after CAP_SAFE_FILL   0.849    4.94
    rule 1, N=524288, before  (1.2% fell back) 0.991  3.09

Rule 0's two points are SAFER in sigma than the rule-1 point CAP_SAFE_FILL
explicitly accepts. Applying the constant to rule 0 would force the cap from 4096
to 8192 -- doubling phase_c's LDS -- to fix a margin that is already wider than
the one the constant was written to produce. bench/fbrate.py finding no fallback
rows at those widths agrees.

What this means for anyone touching `derive_cap` or `CAP_SAFE_FILL`: the invariant
to preserve is the sigma, and 0.85 is only the fill that happens to produce ~4.9
sigma at rule 1's operating point. Do not port the constant across rules.

## A stale JIT module in ONE arm of a pair, and it is not the one you clear
## (gfx950, 2026-09-23)

`arch_scope: gfx950`. aiter's topk backends do not all live in the same JIT
module. `top_k_per_row_prefill_sampled` is `module_top_k_per_row`; the `plain`
backend is `module_topk_plain`. A sweep harness that clears only the first will
happily measure a months-old `plain` kernel.

How it showed up: after rebasing onto six new upstream commits, a paired sweep
reported 41 cells at N <= 32769 running 1.30x to 1.47x SLOWER on the
with-changes arm than on the baseline arm -- on a backend the changes do not
touch. Both arms reported `pick=plain`, so it was not routing.

It was the build. `module_topk_plain.so` in the with-changes worktree was dated
23:15 the previous night; the baseline worktree had been recreated that afternoon
and so built it fresh from the new upstream source. Three of the six upstream
commits are exactly that kernel ("Hold short rows in registers instead of
re-reading them each pass", "Stage pass 1 candidates during the pass 0 scan",
"Find the crossing bucket without a block-wide prefix scan"), and they are worth
1.3x-1.4x at those widths. The with-changes arm was running the code from BEFORE
them while being compared against the code AFTER them.

`csrc/kernels/topk_per_row_kernels.cu` was byte-identical between the two
worktrees, which is what makes this one nasty: the source diff is clean and the
binary is not.

Two rules that follow:

- Clear EVERY module the sweep can dispatch to, not the one under test:
  `rm -rf $AITER/aiter/jit/build/module_top*k* $AITER/aiter/jit/module_top*k*.so`.
  `ls` what is left afterwards and put it in the log.
- A regression on a backend the change cannot reach is a harness bug until
  proven otherwise. The tell here was that every run of the night, including all
  three pre-change baselines, agreed with each other and ONLY the freshly built
  worktree disagreed.

This is the same class as the `.evo/config-v5.yaml` note about `make` reporting
"Nothing to be done" and a stale binary being measured and then stored as a
baseline. Different door, same room.

## Unrolling phase_c's candidate read is worth nothing (gfx950, 2026-09-23)

`arch_scope: gfx950`. An ATT trace of phase_c at m=4096 n=131072, decoded with
line tables, attributes 20.8% of the kernel's latency to `s_waitcnt vmcnt(0)`
feeding `s_keys_ext[i] = fp32_to_sortable_bits((uint32_t)(p >> 32))` -- the read
of the candidate array. The same lever that paid in phase_a's sampler (PA_UNROLL,
0.9875x) does nothing here:

    PC_UNROLL      1       2       4       8
    m=512  n=131072   14.24   14.27   14.40   14.31   (phase_c us)
    m=256  n=262144   10.61   10.50   10.58   10.53
    m=4096 n=131072   71.87   71.28   72.14   71.98

Why the two differ: phase_a's sampler had ONE load per thread, so the block waited
out a single round trip with nothing else issued. phase_c's loop already shows
four distinct load sites in the trace, and c is about 2867 against a 1024-thread
block, so it runs three iterations that the compiler has already overlapped. The
20.8% is the latency of the read, not a missing overlap.

Narrowing phase_c's block does not help either -- the auto width already matches
the best explicit one at every shape measured, and 256 threads is much worse:

    phase_c us       auto     256     512    1024
    m=512  n=131072  14.33   20.13   14.96   14.36
    m=64   n=1048576 12.84   26.70   17.17   12.85
    m=4096 n=131072  71.86   92.28   72.02   92.92

The remaining 23.6% of phase_c is `s_barrier`, 16.1% of it the one inside
`block_find_pivot_bucket_wave0` where fifteen of sixteen waves wait while wave 0
walks 256 buckets. That scan is already four buckets per lane plus a six-step
shuffle; the cost is the block-wide synchronisation, not the scan.

## The gate that killed the multi-block plan, with the numbers (gfx950, 2026-09-24)

`arch_scope: gfx950`. The N>=128K red cells are starved, not slow: at m=8 n=524288
`phase_a` runs at 1.38% CU utilisation (rocprof-compute) because one block per row
means 8 workgroups on 256 CUs. The obvious answer is to give each row more blocks.
Three measurements say not to.

**A kernel launch is cheap here.** `scripts/null_kernel_ramp.hip` times a kernel
that stores one byte per block, swept over grid, block width and dynamic LDS:

    blocks   wg256   wg512   wg1024      (us, and flat across 0/8/32/64 KB of LDS)
         8   1.404   1.387    1.396
       128   1.440   1.432    1.499
       512   1.507   1.595    1.787
      4096   2.261   3.099    4.797

1.39us at 8 blocks, and the LDS reservation costs nothing -- 64 KB reads the same
as 0. So three kernels are 4.2us of ramp at m=8, 15% of that shape's 28.55us, and
"fewer kernels" is not where the time is.

**Half of each selecting kernel is the select, and the other half does not split.**
ABLATE_PA and ABLATE_PC keep the loads, the LDS fill and the gather and drop
`block_select_lds`:

    shape             phase_a  no-select  select | phase_c  no-select  select
    m=8   n=524288       7.42      3.81    3.61  |  12.59      5.32     7.27
    m=32  n=1048576     10.21      6.09    4.12  |  13.87      5.75     8.12
    m=128 n=131072       9.17      4.99    4.18  |  10.10      4.44     5.66
    m=512 n=131072         --        --      -- |  14.23      5.99     8.24

**That arithmetic closes the two-stage split.** A `_stream_split_parts`-shaped
split of phase_a pays the 1.39us ramp AND the ~2.4us non-select fixed cost TWICE,
which is 7.6us of new floor, to save part of a 3.6-4.2us select. It cannot win at
any of these shapes, and the barrier-based variant was already closed for a
different reason (`knowledge/known_bad.md`: the spin barrier's price is 1.43us at
G=16 with one concurrent group and 4.22us with eight, and the regime where a split
is tempting is the one with many groups).

For phase_c the split is not even arithmetically available: the stream pattern
needs `width >= parts * k` and phase_c has `c ~= 1.4 * k_out`.

**And aiter's own multi-block path is slower here.** `should_use_mulblocks` on
MI355X selects it for `batch <= 128` at these widths, and `plain` measures 247.11us
at m=8 n=1048576 where `sampled` measures 34.99.

**What the gate pointed at instead.** The select is the addressable half, and its
cost is front-loaded: phase_c at m=512 n=131072 costs 4.38us for pass 1 and 1.62 /
1.16 / 1.12 for passes 2, 3 and 4, because pass 1 is the only unfiltered scan of
all c keys. Counting pass 0's digits during the read that already has the keys in
registers removes most of it -- shipped for phase_a and then phase_c, worth 0.9611x
to 0.9916x of the three-kernel total with no shape slower.

## phase_b wants 1024 threads at small M, and --coop-g 1 is not how to test it
## (gfx950, 2026-09-24)

`arch_scope: gfx950`. Two things, one of them a trap.

**The trap.** `--coop-g 1 --cf-block 1024` fails with HIP 719 and the sweep that
uses it silently records the configuration as slow. It is not a coop_g=1 problem:
`derive_shape_params` has `if (p.path == PATH_PREFILL) p.coop_g = 1`, so forcing
coop_g to 1 reports `path=prefill`, and the prefill path runs
`phase_b_filter_waveseg` instead of `phase_b_filter_coop`. That kernel's
`seg_stride` layout assumes at most 8 waves. `--coop-g 2 --cf-block 1024` on the
same shape runs fine and reports `path=decode`. Check the `path=` field the
benchmark prints before concluding anything from a coop_g override.

**The finding.** `scripts/bw_gx_floor.hip` prefers wg1024 at small M, and the
shipped phase_b is capped at 512 threads by `WSTAGE_WAVES = 8`. With
`WSTAGE_WAVES_OVERRIDE=16`, phase_b device time / three-kernel total:

    m=8 n=1048576    shipped auto 12.8 / 35.7
      coop_g=32      cf512 13.3 / 34.9      cf1024 10.8 / 32.4
    m=32 n=1048576   shipped auto 26.4 / 49.4
      coop_g=16      cf512 26.5 / 49.0      cf1024 23.5 / 46.5
    m=128 n=131072   shipped auto 18.9 / 37.4
      coop_g=8       cf512 19.0 / 37.3      cf1024 17.8 / 36.4

0.908x, 0.941x and 0.973x on the total. It does NOT generalise upward: at m=4096
n=131072 the same build measures 447.97us at coop_g=2 cf1024 against 417.22 at
coop_g=8 cf512, so this has to be M-gated.

**Why it is not shipped as-is.** `wbuf` is `__shared__ uint64_t[WSTAGE_WAVES *
WSTAGE_CAP]`, so raising WSTAGE_WAVES to 16 reserves 40 KB in every block whether
or not it runs 1024 threads -- a 512-thread block would pay 20 KB it cannot use.
Doing this properly means a template parameter on the block width and the
`choose_coop_g(M, N, n4, 512, ...)` rule re-derived for 1024, since that 512 is
written into the coop_g choice as an assumption.

**And it flips nothing on its own.** The three shapes above need 22.2, 14.8 and
15.1us to reach 60% of the pipe101 floor; this is worth 3.3, 2.9 and 1.0.
