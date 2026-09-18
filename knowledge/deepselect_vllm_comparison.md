# DeepSelect in vLLM (PR #56464) vs our port, and what it means for AVO

Question asked: vLLM integrated DeepSelect for the DSA sparse indexer; is the
kernel they use the same one we ported, did they optimise it further, and does
their version show us anything to improve?

Short answer: **the kernel is identical, file for file. NVIDIA-side did no
kernel work at all.** The gap worth acting on is ours, not theirs, and it is in
a different place than expected -- our fp32 AVO kernel is already level with
DeepSelect fp32, while our own bf16 HIP port of DeepSelect is 6.1x off the CUDA
original.

All of the below was verified on 2026-09-18 against the GitHub API and the
checkouts on this machine; provenance is given per claim.

## 1. What the PR is

`vllm-project/vllm` PR #56464, "[Perf][Kernel] Integrate DeepSelect TopK for the
DSA sparse indexer", author `ZJY0516`, opened 2026-09-11, **merged into `main`
2026-09-13**, 22 commits, 8 files, +829/-52.

| file | change |
|---|---|
| `cmake/external_projects/deepselect.cmake` | new, +97 |
| `vllm/model_executor/layers/indexer_topk.py` | new, +339 |
| `tests/kernels/test_top_k_per_row.py` | +323 |
| `vllm/config/kernel.py` | +32 |
| `vllm/engine/arg_utils.py` | +16/-1 |
| `vllm/model_executor/layers/sparse_attn_indexer.py` | +17/-51 |
| `CMakeLists.txt`, `setup.py` | +1, +4 |

Every line is integration: a FetchContent hook, a backend registry with six
selectable top-k implementations (`deep_select` / `cooperative` / `persistent` /
`per_row` / `flashinfer` / `torch`) behind `--sparse-indexer-topk-backend`, a
shape-based `auto` heuristic, and tests. The PR body says so itself: "This PR
vendors DeepSeek's kernels unchanged and adds a shape-based dispatch, rather
than modifying existing kernels."

## 2. Which DeepSelect they pin, and how it differs from ours

`deepselect.cmake` does **not** fetch from `deepseek-ai/DeepSelect`. It fetches
`https://github.com/vllm-project/DeepSelect.git` at
`GIT_TAG d96d33afe1fab0d6066da49cdc91e64c2bee65ea`.

That is worth a second look, because a fork plus a pinned SHA is exactly the
shape of "they patched something". They did not:

- `d96d33af` is the head of the fork's `dev` branch. Its **parent is
  `0f03b68`**, which is simultaneously `deepseek-ai/DeepSelect` main, the
  `vllm-project/DeepSelect` main, and the commit our `/home/mh/DeepSelect`
  is sitting on. So the fork is one commit ahead of upstream and we are level
  with upstream.
- That one commit is "Migrate to PyTorch stable ABI (#1)". It touches
  `csrc/api.cpp` (+134/-79), `csrc/dispatch_utils.h` (+13/-10), a new
  `csrc/stable_tensor_checks.h` (+74), `deep_select/interface.py` (+5/-3) and
  `setup.py` (+6/-1). **Nothing under `csrc/cuda_kernels/`.**

So the compute kernels vLLM runs are byte-identical to the ones we hipified.
There is no NVIDIA-side kernel optimisation to copy.

### Where we genuinely differ from them

Not in kernel content, but in kernel **coverage**. `deepselect.cmake` globs
three families:

```
csrc/cuda_kernels/v3/instantiations/*.cu
csrc/cuda_kernels/v3_fp32/instantiations/*.cu
csrc/cuda_kernels/v3_cluster/instantiations/*.cu
```

We ported one: `csrc/hip_kernels/v3/` (bf16, 40 upstream instantiations, one
shipped config). `v3_fp32` (30 instantiations) and `v3_cluster` were excluded by
the porting plan and remain unported.

`v3_fp32` is not a dtype swap of `v3`: `v3_fp32/topk_select.cuh` is 603 lines
against `v3/topk_select.cuh`'s 149. Both include the same
`cuda_kernels/common_parts.cuh` (1517 lines), which we have already ported as
`common_parts_hip.cuh` (1381 lines) -- so the shared and hardest half of a
`v3_fp32` port is done, and what is missing is its own 603-line header.

Their build is also Blackwell-only: `deepselect.cmake` enables the arch only for
CUDA >= 12.9 and `"10.0f"`, i.e. SM100/SM103.

## 3. Performance, and what is and is not comparable

Their numbers (PR body): NVIDIA GB200, torch 2.13.0+cu130, CUPTI timing, CUDA
graph replay, cold L2, DeepSeek-V4.1-Flash Lightning Indexer shapes with
`index_topk=512`, where the indexer "vocab" is the KV context length.

Ours: `knowledge/grid_baseline_outer.json`, g_25, measured 2026-09-18 10:20 on
MI355X / ROCm 10.0, `--warmup 20 --iters 100` with run-median of run-medians.

**These two harnesses are not equivalent and the comparison below is
indicative, not a head-to-head.** They flush L2 and replay a CUDA graph; our
`bench/grid.py` has no flush or cold-cache mechanism at all (grepped, there is
none), and the hardware differs. Do not quote this table as a like-for-like
benchmark.

fp32, `topk`/`index_topk` = 512, kv = 1,048,576. TB/s is `rows * kv * 4 B` over
the reported time.

| rows | DeepSelect @ GB200 | our AVO @ MI355X | their next-best backend |
|---:|---|---|---|
| 256 | **191 us** (5.6 TB/s) | 208.20 us (5.16 TB/s) | FlashInfer ragged 595 us |
| 64 | 88 us | **64.90 us** (4.14 TB/s) | cooperative 132 us |
| 8 | 85 us | **30.60 us** (1.10 TB/s) | cooperative 40 us (their `auto` keeps it) |

Read carefully: at 256 rows DeepSelect is 9% ahead of us; at 64 and 8 rows we
are ahead, and at 8 rows DeepSelect loses to vLLM's own cooperative kernel so
badly that their `auto` heuristic routes around it. Our small-`M` weakness is
elsewhere -- at 8 rows we are at 1.10 TB/s, which is a 4.70x ratio to the
modelled floor, the worst cell in the whole 587-point grid. It is an occupancy
problem, not a DeepSelect problem.

These AVO points are uniform (`RAGGED=false`) grid cells, so they are unaffected
by the row-extent clamp added alongside this report.

## 4. Our own DeepSelect port is the actual gap

Their bf16 eager microbench: bs=256, kv=1M, **119 us (4.5 TB/s)**.
Our HIP port on the same shape and dtype
(`DeepSelect/reports/s7_mi355x_report.md`, 2026-09-16, kineto kernel time,
`tests/test.py --dtype bf16 --max-topk 512`): **730.88 us**.

That is **6.1x**. The s7 report already attributes it, and none of the causes
are dtype or numerics:

- 64 waves for the entire GPU (16 workgroups x 4 waves on 256 CUs), from one CTA
  per row plus a 118,784 B LDS footprint against 163,840 B per CU, which caps
  residency at one workgroup per CU regardless of batch;
- `VALU:VMEM = 136:1` -- VALU-bound, not bandwidth-bound (4.19 MB in 102.91 us
  on the profiled shape is 41 GB/s, a fraction of a percent of HBM);
- `scratch = 112 B`, i.e. register spilling, against upstream's
  `STACK_BASELINE = 8` -- and the ROCm build path never runs upstream's
  `SpillCheckBuildExtension`, so nothing caught it.

## 5. The blind spot this exposes

`reports/deepselect_compare.json` has 12 rows and **all 12 are
`RuntimeError: HIP build supports bfloat16 input only`**. We have never once
measured DeepSelect against AVO at the same dtype. Every DeepSelect-vs-AVO
statement available to us today is an inference across two different dtypes,
two different harnesses and two different vendors' hardware.

Porting `v3_fp32` would close that, and it is the cheapest remaining
epistemic win: the shared `common_parts` is already ported, so the work is its
603-line header plus instantiations, and the payoff is the first direct
measurement rather than another indirect one.

## 6. Recommendations, ordered by evidence strength

1. **Port `v3_fp32` to HIP** to get a same-dtype, same-machine, same-harness
   DeepSelect-vs-AVO number. Highest information gain per unit of work;
   removes the only comparison we cannot currently make.
2. **Do not chase fp32 kernel ideas from DeepSelect.** We are within 9% at 256
   rows and ahead at 64 and 8. There is no evidence of a technique to copy, and
   the one place they clearly win (large batch) is 9%, well inside the range a
   harness difference can explain.
3. **If the bf16 port matters, attack occupancy, not the inner loop.** The
   binding constraint is `target_occupancy=1` coupled to
   `reconstruct_threshold=4096` via the LDS plan; raising occupancy requires
   shrinking `reconstruct_threshold` first. s7 lists this as untried.
4. **Run upstream's spill check on the ROCm path.** `scratch = 112 B` is
   unexplained and upstream gates it at 8.
5. **Leave `v3_cluster` alone.** CDNA has no DSMEM/cluster equivalent, and the
   PR's own data supports the decision: at bs=8 their `auto` picks cooperative
   (40 us) over DeepSelect (85 us), so the regime the cluster kernel exists for
   is not one DeepSelect wins anyway.

## 7. One thing worth borrowing that is not a kernel

`tests/kernels/test_top_k_per_row.py` fills every position past each row's `end`
with NaN and relies on DeepSelect's `abort_when_nan_found=True` to turn an
out-of-range read into a loud failure. The idea transfers; the poison does not.
Our ordering key (`csrc/topk_common.hip.hpp:258`) sends -NaN below -inf and
drops it, so a NaN poison can be read and never seen. `bench/stress_topk.py`
uses `+inf` instead, which our key ranks above everything and which therefore
cannot be read silently -- that is what caught `rowEnds > stride0` returning
indices past the pitch with no error at all.

Note also that vLLM's test file already carries gfx950-specific top-k cases
(`test_top_k_per_row_decode_gfx950_long_c4a`), against the *decode* entry rather
than the prefill entry AVO replaces.
