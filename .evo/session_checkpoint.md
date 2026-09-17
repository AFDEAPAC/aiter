# Session checkpoint — pow2 grid optimization round

State as of 2026-09-17. Read this before resuming; it separates what is
VERIFIED from what is merely changed.

## Where things stand

Accepted lineage is in [`log/grid_evolution.tsv`](../log/grid_evolution.tsv):
g_0 (45.21 µs) -> g_1 -> g_2 -> **g_3 (42.73 µs outer geomean)**, -5.5%
cumulative, no regime regressed, anchor 615.6 µs (limit 620.0).

Working tree == g_3 == the saved baselines. Nothing is half-applied.

## How to reproduce the gates (correctness FIRST, always)

```bash
make benchmark_topk                       # auto header deps; do not trust a stale binary
python3 bench/verify_grid.py --dist all   # 650 verifies, ~3 min, must be 0 failed
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size 16G -v $PWD:/work -w /work \
  rocm/ali-private:ubuntu22.04_rocm10.0.0_cp313_torch2.12.0_vllm_dsv4_20260916 \
  bash -c 'python bench/gate_selftest.py; python bench/correctness.py'
python3 bench/score_grid.py --tier inner  # 24 points, ~20 s, drives decisions
python3 bench/score_grid.py --tier outer  # 130 points, ~95 s, commit gate
```

`gate_selftest.py` is not optional. `correctness.py` returns early from
`run_gates()` when torch is missing and the host python is 3.6 with no torch, so
without the selftest a whole round can report PASS having never called
`torch.topk`. That already happened once.

## Verified this round

- The 130-point grid all runs and passes, 5 distributions, 650 verifies.
- `coop_g` measured table: hits the swept optimum on all 56 decode shapes
  (worst +0.6%). Biggest single win M=128 N=1M 225 -> 127 µs.
- `small_n` block measured table: worst +1.8% vs swept optimum (was +12.1%).
- `phase_a`/`phase_c` occupancy formula needs no table: worst +3.4% over 91 pts.
- N_lds boundary is M-dependent: small_n takes N=16384 for M<=256.
- `phase_b_filter_coop` out-of-bounds LDS write fixed; all G values correct.

## Not attempted (the big one left)

**S3: parallelize `phase_a` and `phase_c` for small M.** At M=1 N=1M,
`rocprofv3 --kernel-trace` attributes 11.19 µs to phase_a and 11.26 µs to
phase_c of a 31.7 µs wall, and both run with exactly ONE block. That is 71% of
the time on 1 of 256 CUs. Cheap knobs around it are exhausted and measured:
block size (done, via the occupancy rule), `--phase-a-passes 2` (unsafe: M=1024
N=1M goes 949 -> 1917 µs), `coop_g` (flat at M=1).

Two designs, neither tried:
- (a) multi-block partial histograms in global memory + a small pivot kernel.
  Costs +1 launch (~2.6 µs) against an ~11 µs target, so at most ~5-6 µs/kernel.
- (b) `hipLaunchCooperativeKernel` with grid sync: one launch, but the grid is
  capped to co-resident blocks and ROCm 10 support is UNVERIFIED on gfx950.
Spike (b) first; if unavailable, (a) is the fallback. hipGraph is NOT a fallback
(measured 13-18% slower on launch-bound shapes).

## Also not attempted

- `cf_block=1024`: needs WSTAGE_WAVES 8 -> 16, doubling phase_b LDS from 20.5 to
  40 KB, and templating to avoid regressing the shipped 512 path. Judged low
  value because phase_b is already ~1.1x its own read+write floor at the anchor
  and cf_block=256 was 17.5% worse. This is a judgement, not a measurement.
- `phase_a_passes` full-grid sweep (only spot-checked; 2 is known unsafe).

## Traps that will bite again

See [`knowledge/known_bad.md`](../knowledge/known_bad.md). The ones that cost
the most time here:
1. Stale binary from hand-written header deps in the Makefile.
2. Per-path geomeans grouped by the reported path, which candidates can change.
3. `--sample-s` silently ignored unless S/64 is a power of two (pow2 N).
4. small_n `--dump-stats` reads an untouched buffer; its CANDSTATS is n/a.
5. A failed launch is instant and wrong, not slow.
