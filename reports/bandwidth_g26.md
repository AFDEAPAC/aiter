# AVO effective read bandwidth at g_26, against a 4 TB/s floor

Measured 2026-09-18 on MI355X / ROCm 10.0 through aiter's own `@perftest`
(`op_tests/test_topk_per_row.py --prefill_backend avo`), which rotates arguments
to defeat L2 and reads GPU kernel time from a profiler trace. fp32, k=2048,
ragged rows (row r ends at `num_prefix + r + 1`). Every row of every run
reported `all_close = True`.

Bandwidth here is **effective read bandwidth**: `4 * sum(row_len) / kernel_time`.
Rows are ragged, so `M * N * 4` overstates the traffic by up to 25% at large M
and small N; the per-row sum is used instead.

## The hardware reference, measured on this box

`knowledge/g0_floor_model.json` `bw_sweep`, produced by `scripts/bw_kernel.hip`,
is a pure streaming read:

| bytes | time | bandwidth |
|---|---|---|
| 4 KB .. 33 MB | **5.8 us, flat** | 0.0007 .. 5.75 TB/s |
| 64 MB | 12.96 us | 5.18 TB/s |
| 128 MB .. 2 GB | 22 .. 364 us | 5.84 .. 6.61 TB/s |

Below 33 MB this GPU is latency-bound at a flat 5.8 us no matter how few bytes
you ask for. **4 TB/s therefore requires at least `4e12 * 5.8e-6` = 23.2 MB of
traffic, for any kernel whatsoever.** That single fact disposes of most of the
grid before AVO is even considered.

## Bandwidth, TB/s

| M \ N | 2K | 4K | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1024K |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.02 | 0.05 | 0.08 | 0.13 |
| 2 | 0.01 | 0.00 | 0.01 | 0.01 | 0.01 | 0.02 | 0.05 | 0.09 | 0.15 | 0.24 |
| 4 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.05 | 0.09 | 0.16 | 0.28 | 0.46 |
| 8 | 0.02 | 0.01 | 0.02 | 0.02 | 0.04 | 0.08 | 0.14 | 0.27 | 0.48 | 0.84 |
| 64 | 0.19 | 0.10 | 0.14 | 0.19 | 0.29 | 0.54 | 0.95 | 1.49 | 2.49 | 3.39 |
| 256 | 0.63 | 0.33 | 0.52 | 0.71 | 0.93 | 1.65 | 2.48 | 3.33 | **4.16** | **4.73** |
| 1024 | 1.49 | 0.69 | 0.97 | 1.21 | 1.90 | 2.83 | 3.45 | 3.95 | **4.69** | **5.44** |
| 4096 | - | 0.80 | 1.16 | 1.33 | 2.01 | 3.02 | 3.68 | **4.52** | **4.85** | **5.62** |

7 of 79 cells clear 4 TB/s.

## Why the other 72 do not

### R1 -- 51 cells: not enough bytes for ANY kernel (not an AVO issue)

Traffic under 23.2 MB. `bw_kernel` cannot reach 4 TB/s at those sizes either,
because its time is pinned at 5.8 us. This covers the whole M <= 8 block and
everything left of the diagonal. Reporting these as an AVO shortfall would be a
category error.

### R2 -- M <= 8: occupancy, unreachable at any N

Fitting `t = F + bytes/BW` over the largest shapes, including extra points at
N = 2M, 4M, 8M measured specifically to give these rows enough bytes:

| M | asymptotic BW |
|---|---|
| 1 | 0.72 TB/s |
| 2 | 0.57 TB/s |
| 4 | 1.16 TB/s |
| 8 | 3.09 TB/s |

The asymptote itself is below 4, so no shape reaches the floor. The kernel is one
workgroup per row; `coop_g` splits a row across blocks and is what lifts M=8 to
3.09, but 8 rows cannot fill 256 CUs. **4 TB/s is not an achievable target for
M <= 8 and should not be held against the kernel.**

### R3 -- M >= 64: the streaming part is already at peak; a fixed cost is not amortised

Marginal bandwidth, `d(bytes)/d(time)` between adjacent N:

| M | 32K->64K | 64K->128K | 128K->256K | 256K->512K | 512K->1024K |
|---|---|---|---|---|---|
| 64 | 3.66 | 3.62 | 3.47 | 7.64 | 5.32 |
| 256 | 7.12 | 4.97 | 5.09 | 5.55 | 5.47 |
| 1024 | 5.40 | 4.43 | 4.60 | 5.79 | 6.46 |
| 4096 | 5.77 | 4.65 | 5.83 | 5.24 | 6.67 |

Every extra byte moves at 3.5-6.7 TB/s, i.e. at or above what `bw_kernel`
achieves. **There is no streaming inefficiency left to recover.** What keeps the
average down is a fixed cost `F` that does not scale with N:

| M | F (us) | asymptotic BW | traffic needed for 4 TB/s | N needed at k=2048 |
|---|---|---|---|---|
| 64 | 30.8 | 5.57 TB/s | 437 MB | 1669K -- past the cliff below |
| 256 | 30.4 | 5.46 TB/s | 456 MB | 434K |
| 1024 | 83.6 | 6.01 TB/s | 1000 MB | 238K |
| 4096 | 261.4 | 6.07 TB/s | 3070 MB | 183K |

The model predicts the observed crossings: it says M=1024 needs 1000 MB, and the
measurements are 1072 MB -> 3.95 and 2145 MB -> 4.69.

`F` is roughly 0.064-0.12 us per row and is NOT launch overhead -- three
dispatches at the measured 2.63 us each is only 7.9 us of the 261 us at M=4096.
[unverified] A traffic estimate accounts for about a quarter of it: the
candidate buffer is `cap` entries per row written in phase B and read in phase
C (134 MB each way at M=4096 cap=4096), plus the phase A sample (67 MB) and the
index output (34 MB), around 61 us at 6 TB/s. The remaining ~200 us is not
explained by traffic and wants a profile before anyone guesses further.

### Not the reason: selectivity

`k` was the obvious suspect and it is wrong. Holding (M, N) fixed and sweeping
k over 512 / 1024 / 2048 moves bandwidth by under 6%:

| M | N | k=512 | k=1024 | k=2048 |
|---|---|---|---|---|
| 256 | 32768 | 0.96 | 0.99 | 0.91 |
| 4096 | 262144 | 4.62 | 4.47 | 4.34 |

The apparent correlation with `N/k` in the first pass was N moving, not k.

## A cliff above N ~ 1.5M, outside the tested grid

Found while giving the small-M rows enough bytes. At M=64, N=1048576 is 79 us
and N=2097152 is **2134 us** -- twice the data, 27x the time. Still correct.

`benchmark_topk --dump-stats` localises it exactly:

```
N=1048576  margin=2.129 cap=8192  mean=4423   over_Calloc=0  fallback_rows=0
N=1572864  margin=2.853 cap=8192  mean=5769   over_Calloc=0  fallback_rows=0
N=2097152  margin=3.000 cap=8192  max=4294967295 over_Calloc=1 fallback_rows=1
```

`margin` saturates at its 3.000 clamp, so past roughly N=1.5M the sampled
threshold can no longer keep the candidate set inside `cap`, a row overflows,
and `phase_d_fallback` runs an exact select over the full row for it. The
fallback is doing its job -- `rows_fail=0` throughout -- but one fallen-back row
out of 64 costs 27x on the whole call.

The scored grid stops at N=1048576, so nothing in the repo covers this. It is a
correctness-safe but performance-fatal regime, and it is the reason M=64 cannot
reach 4 TB/s in practice: it would need N=1669K, which is past the cliff.

## Summary

- 4 TB/s is met at M >= 256 once traffic passes roughly 450 MB (M=256),
  1000 MB (M=1024) or 3070 MB (M=4096).
- For M <= 8 the target is unreachable by construction -- occupancy, not code.
- For M = 64 it is unreachable in the usable N range, blocked by the cliff.
- Nothing in between is a streaming problem; the marginal bandwidth is already
  at hardware peak everywhere. The lever is `F`, and the first step on `F` is a
  profile, not a guess.
