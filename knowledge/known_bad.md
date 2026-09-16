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
