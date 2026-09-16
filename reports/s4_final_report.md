# TopK prefill fp32 on MI355X — AVO run report

**Target: beat 760 µs. Result: 619.4 µs, correctness green, measured 0 fallback rows.**

| | wall ms | vs target | vs this run |
|---|---|---|---|
| Measured pipeline floor (read+write) | 0.4821 | 0.63× | 0.78× |
| **Shipped (x_7)** | **0.6194** | **0.82×** | **1.00×** |
| Customer gluon (external, not reproduced here) | 0.760 | 1.00× | 1.23× |
| aiter `topk_select` (external, not reproduced here) | 0.950 | 1.25× | 1.53× |
| Seed kernel x_0 | 2.9587 | 3.89× | 4.78× |
| `torch.topk`, same shape | 4.8541 | 6.39× | 7.84× |

Shape: fp32 input `[4096, 131072]`, `topk=2048`, output `int32 indices [4096, 2048]`.
Hardware: MI355X / gfx950, 256 CU, `HIP_VISIBLE_DEVICES=0`, ROCm 10.0.
Measurement: 20 warmup + 100 measured iterations, 5 repeats, median of run
medians. Final stddev **0.067%**, well inside the 2% reject threshold.

---

## What changed, and what each change bought

All numbers are the full-pipeline wall time on the scored shape.

| Version | Change | wall ms |
|---|---|---|
| x_0 | seed: three-stage pipeline with global multi-pass radix | 2.9587 |
| x_1 | Phase A and Phase C fused into single in-LDS select kernels | 1.1345 |
| x_2 | parallel pivot scan; bounded fallback dispatch; zero-global-atomic filter | 0.7916 |
| x_3 | wave-private candidate regions; packed 64-bit candidate store | 0.7740 |
| x_4 | register-resident suffix scan; runtime sample count on dynamic LDS; vectorized Phase A load | 0.6997 |
| x_5 | LDS-staged contiguous candidate flush; replicated histogram; fewer barriers | 0.6251 |
| x_6 | skip radix passes on the candidates' common prefix (Phase C only) | 0.6202 |
| x_7 | over-collection margin derived from estimator noise instead of a constant | 0.6194 |

Kernel launches per call went from about 20 to 4.

## Where the time goes now

`rocprofv3 --kernel-trace`, µs per call, against each kernel's own measured
traffic floor:

| kernel | µs | floor | ratio | LDS | VGPR |
|---|---|---|---|---|---|
| `phase_b_filter_wavestage` | 478.4 | 436.1 | **1.10×** | 20992 | 24 |
| `phase_c_select_waveseg` | 69.7 | 24.0 | 2.90× | 38400 | 44 |
| `phase_a_threshold` | 66.9 | 22.0 | 3.04× | 5632 | 40 |
| `phase_d_fallback` | 3.8 | ~0 | — | 5632 | 48 |
| sum | 618.8 | 482.1 | 1.28× |

The streaming filter, which is where the budget was always going to live, is
within 10% of its own floor. The remaining 137 µs of theoretical headroom sits
in the two select kernels, in their radix passes.

## The three findings that actually moved the number

**1. A read-only floor understates a filter kernel, and that mis-ranked the work.**
Reading 2.147 GB takes 0.3516 ms (6.11 TB/s). Reading it *and writing the 94 MB
of candidates* takes 0.4361 ms — the writes cost 87 µs on their own, because HBM
writes cost more per byte (the copy benchmark runs at 4.48 TB/s versus 6.11
read-only). Judged against the read-only number the filter looked 1.37× off and
worth attacking further; against the real floor it is 1.10× and finished. I had
also mis-stated this floor in the previous session's report, because the
bandwidth microbenchmark counted a 1 GiB buffer as 4 GiB.

**2. The filter's cost was the scattered stores, not the compare.**
Ablating the kernel: full 500.7 µs, stores removed 361.9 µs, all compaction
removed 361.1 µs, pure-read floor 349.0 µs. So load + compare + ballot
compaction cost 13 µs, and the candidate stores cost 139 µs to move 94 MB. A
wave produces about 5.6 passers per iteration, so each burst was ~45 B against a
128 B line. Staging passers in LDS and flushing once a wave holds 64 of them
(~520 B contiguous) took the kernel to 480 µs.

Relatedly, raising the filter's blocks-per-row made it monotonically worse
(gx=2 0.925 ms through gx=64 3.422 ms) because every block contended on the same
`cand_count[row]` atomic — while a pure read at one block per row still reaches
6.08 TB/s. One block per row owns that row's candidate area outright, so the
shipped filter has **no atomic of any kind**: each wave has a private slice and
a wave-uniform register counter.

**3. The candidate-count spread is estimator noise, and a constant margin
overfits to one K.**
The number of elements above the rank-R value of S samples has spread
≈ count/√R. That spread, not the mean, decides whether a row lands inside
`[K, capacity]` and so whether it pays the exact fallback. Measured at S=8192
with a fixed margin of 1.4 (min/mean, rows under K out of 4096):

| K | rank R | min/mean | rows under K |
|---|---|---|---|
| 2048 | 179 | 0.74 | 0 |
| 1024 | 89 | 0.67 | 4 |
| 512 | 44 | 0.52 | 82 |

Solving `mean·(1 − 3/√R₀) > K` instead gives 1.36 at K=2048 and 2.13 at K=512.
That removed the fallback at every K and took the K=512 guard shape from
0.7326 ms to **0.5725 ms (−22%)**. This is also why raising the sample count from
4096 to 8192 was worth 67 MB of extra reads: at S=4096 four rows fell short of K,
and each fallback row is expensive because one block streaming a whole row is
latency-bound, not bandwidth-bound.

## Correctness

`python3 bench/correctness.py` → `CORRECTNESS PASS gates=5 inject_red=ok`.

1. Value multiset bitwise-equal to `torch.topk` (ties may pick any index).
2. Index uniqueness — the check that value-recall is blind to.
3. Exactly K outputs, all in range.
4. `input[index[i]]` reproduces the reference multiset.
5. Distributions: uniform, gaussian, all-equal, +inf-saturated, and an
   adversarial layout that concentrates the large values at the row tail.

Every gate was proven red by fault injection (`--inject-fault 1`) before being
trusted.

**The timed path is the verified path.** The fused fast path and the exact
fallback are verified separately, and `--pipeline direct` runs the fallback over
every row as an independent full-row oracle — it passes on the production shape
with all 4096 rows going through it. This was deliberate: the failure mode where
a gate certifies a slow path while the timed fast path is broken is exactly the
incident recorded in `gemm-topk/knowledge/known_bad.md`.

One correctness bug found and fixed during the run had no visible symptom on the
scored distribution: when a row's passer count exceeded the candidate area,
Phase B silently dropped the surplus and Phase C then selected from an arbitrary
subset. Rows are now routed to the fallback on `cand_count < K` **or**
`> capacity`. The adversarial distribution exercises it — all 4096 rows overflow,
fall back, and pass.

Guard shapes (correctness set, not scored): `K=512` 0.5725 ms and `N=32768`
0.2752 ms, both green.

## What you need to do

Nothing to merge — this is a standalone repo at `/home/mh/topk-prefill-avo`, not
integrated into aiter, as scoped. To reproduce:

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  -e HIP_VISIBLE_DEVICES=0 -v /home/mh:/home/mh -w /home/mh/topk-prefill-avo \
  rocm/ali-private:ubuntu22.04_rocm10.0.0_cp313_torch2.12.0_20260910 \
  bash -lc 'make && python3 bench/correctness.py && \
    ./benchmark_topk --mode verify_and_time --m 4096 --n 131072 --topk 2048 \
      --warmup 20 --iters 100 --repeats 5'
```

`./bw_kernel` reprints the measured floors.

## Unverified / limitations

- **The 760 µs and 950 µs references are external and were not reproduced on
  this machine.** Neither local checkout has `topk_select`; it lives in upstream
  main and needs a FlyDSL JIT build, and `/opt/aiter` is read-only to the
  unprivileged uid. `torch.topk` at 4.8541 ms is the locally measured anchor. It
  is also unknown whether gluon's 760 µs is a wall or kernel time; both are
  reported here (wall 0.6194 ms, kernel sum 0.6188 ms — the pipeline is 4
  back-to-back launches, so they nearly coincide).
- **`--nt-load` is a no-op in substance.** The builtin compiles, but it measured
  neutral (0.7930 vs 0.7916 ms), so the shipped path leaves it off. Inline
  `global_load_dwordx4 ... glc slc` asm does not assemble on gfx950/ROCm 10.
- **No PMC counter profile.** Attribution here is from `--kernel-trace` timings
  plus targeted ablations and synthetic floors, not from `SQ_*` counters. The
  ablations are the stronger evidence for the claims made, but a counter run
  would confirm the LDS-conflict story in §3 independently.
- **The fallback is correct but slow per row** (one block streaming a whole row
  is latency-bound). The shipped configuration has 0 fallback rows on the scored
  shape and all guards, but a distribution that defeats the sampler pushes rows
  onto it; the adversarial gate sends all 4096 there and still passes, at a large
  time cost. If a real workload looks like that, the fallback needs multi-block
  cooperation per row.
- **Decode (M=4) is out of scope** for this run, as agreed.

## Directions tried and rejected

Full list with numbers in `knowledge/known_bad.md`. The notable ones: more
blocks per row in the filter (monotonically worse, atomic contention);
non-temporal loads (neutral); block-aggregated atomics (better than per-wave
contention, worse than no atomic); ballot gather in Phase C (no effect — its
radix passes, not its atomics, are the cost); histogram replication beyond 4
(LDS cost exceeds the benefit); common-prefix skipping in Phase A (+8 µs, since
its samples span the whole row); Phase A with 2 radix passes (0.8% faster but
halves the overflow headroom).

## Remaining headroom, if this is picked up again

137 µs, all in the two select kernels at ~3× their traffic floors, and the
measured per-pass cost says it is the radix passes themselves — roughly 7-12 µs
of fixed cost per pass across 4096 blocks, mostly barriers and the histogram.
The untried lever is a wider radix digit (11-12 bits, 3 passes instead of 4) at
the cost of a larger in-LDS histogram, which trades directly against the
occupancy that `phase_c` already spends 38 KB of LDS on.

## Lineage

`evo/topk_indices_kernel`: `a339645` (seed) → x_1 `3ef1c47` → x_2 `9d510a4` →
x_3 `72d4836` → x_4 `dd69571` → x_5 `0f1f732` → x_6 → x_7 (HEAD).
Rejected attempts with their numbers are in `log/attempts.tsv`.
