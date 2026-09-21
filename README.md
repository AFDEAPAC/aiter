# topk-prefill-avo — per-row fp32 top-k on MI355X (gfx950)

Standalone HIP kernel + benchmark for `topk` over fp32 `[M, N]`, emitting int32
indices only.

> **This history is parked inside the `AFDEAPAC/aiter` repository, on a branch
> whose history is unrelated to aiter's.** It is a separate project that has no
> aiter files in it and builds on its own; the fork is only being used as a place
> to keep it. The part of this work that *is* aiter code lives in aiter's own
> history on `feat/topk-per-row-gfx950-avo`, as the `top_k_per_row_prefill_sampled`
> op, generated from here by
> [`scripts/export_aiter_op.py`](scripts/export_aiter_op.py).

> **On the name.** The kernel is called `sampled`. The repository, the branch
> `feat/topk-per-row-gfx950-avo` and the path `/home/mh/topk-prefill-avo` still
> say `avo`, and they keep that name on purpose: renaming the directory would
> break the absolute paths in the docker invocations recorded in every
> `bench/` docstring, and buys nothing the identifier rename did not.
>
> `sampled` describes what the kernel does — it estimates a threshold by
> sampling, filters a candidate set against it, selects exactly, and sends the
> rows whose candidate set came out unusable to a separate exact fallback
> (`phase_d_fallback`). The seed kernel was named `topk_fp32_sampled` for the
> same reason. AVO named the process that produced the kernel, not the kernel,
> which is why it is no longer used for it. AVO is still the right name for the
> evolution harness itself, so `AVO_USE_CUSTOM_HARNESS` and
> `AVO_ALLOW_STUB_BENCH` keep theirs.
>
> This kernel **references DeepSelect but is not a port of it.** It borrows two
> things, both cited where they are used: the "distort" fp32-to-monotone-uint32
> trick (`csrc/topk_common.hip.hpp`) and a lesson about a `__shfl_down` tree
> that cost 3% recall. The architecture differs — DeepSelect's `v3_fp32` is one
> kernel, this is nine — and our actual hipify of DeepSelect is a different tree
> (`csrc/hip_kernels/v3/`, bf16). See
> [`knowledge/deepselect_vllm_comparison.md`](knowledge/deepselect_vllm_comparison.md).

Originally tuned for the single shape M=4096, N=131072, K=2048, where it measures
**0.6194 ms** on MI355X (5 runs x 100 iters, stddev 0.067%) against a 0.760 ms
external target and a 0.4821 ms measured pipeline floor — `torch.topk` on the
same shape is 4.8541 ms. It now covers M=1..4096 x N=2048..1M with the shape
choices derived per shape in [`csrc/topk_shape.hip.hpp`](csrc/topk_shape.hip.hpp),
gated on 130 points x 5 distributions.

Start with [`reports/grid_report.html`](reports/grid_report.html) for the current
130-point picture and [`log/grid_evolution.tsv`](log/grid_evolution.tsv) for the
perf lineage. [`reports/s4_final_report.md`](reports/s4_final_report.md) is the
earlier single-shape run, and [`knowledge/known_bad.md`](knowledge/known_bad.md)
is everything that did not work, with the number that killed each one.

Note that the phase table and the "no atomic of any kind" claim further down
describe the original single-shape pipeline. The cooperative decode path added
since does use atomics, with a reservation valve and an exact fallback for rows
that overflow.

## Build and run

Needs ROCm 10 and a gfx950 device. On this host:

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  -e HIP_VISIBLE_DEVICES=0 -v /home/mh:/home/mh -w /home/mh/topk-prefill-avo \
  rocm/ali-private:ubuntu22.04_rocm10.0.0_cp313_torch2.12.0_20260910 \
  bash -lc 'make && python3 bench/correctness.py && \
    ./benchmark_topk --mode verify_and_time --m 4096 --n 131072 --topk 2048 \
      --warmup 20 --iters 100 --repeats 5'
```

`./bw_kernel` reprints the measured bandwidth floors the report compares against.

## Structure

- `benchmark_topk.hip.cpp` — the four phase kernels, host orchestration, CPU
  verifier, timing harness, and the AVO knob surface.
- `csrc/topk_common.hip.hpp` — sortable-key mapping, the register-resident pivot
  scan, the ballot-based gather, and the tunable constants.
- `bench/correctness.py` — five hard gates, each proven red by fault injection.
- `scripts/bw_kernel.hip` — bandwidth floors, including one with the same
  read:write ratio and grid shape as the filter kernel.
- `scripts/measure_s0.py` — records the baselines into `knowledge/s0_baseline.json`.

## How it works

Four kernels per call:

| phase | what | µs |
|---|---|---|
| A `phase_a_threshold` | per-row sampled threshold, selected entirely in LDS | 67 |
| B `phase_b_filter_wavestage` | streaming dwordx4 filter, one compare per element, ballot compaction, LDS-staged contiguous candidate flush | 478 |
| C `phase_c_select_waveseg` | exact 4x8-bit radix select on the candidate set in LDS + tie-correct gather | 70 |
| D `phase_d_fallback` | exact full-row select for rows the sampler mis-served | 4 |

Phase B owns one row per block, so each wave writes into a private slice of that
row's candidate area and the kernel needs **no atomic of any kind**. Phase D also
runs standalone via `--pipeline direct` as an independent full-row oracle, so the
timed fast path is never certified by a different path.

## Correctness

`--mode verify` compares against a CPU reference per row; `bench/correctness.py`
adds index uniqueness, output-count, gather-consistency, and five distributions
(uniform, gaussian, all-equal, +inf, adversarial) against `torch.topk`. Ties may
pick any index, so values are compared as a multiset.

Rows whose candidate count falls below K or overflows the candidate area are
routed to the exact fallback — both conditions, since a silent overflow produces
a wrong answer with plausible-looking counts.
