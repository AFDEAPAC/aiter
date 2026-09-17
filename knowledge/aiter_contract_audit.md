# AVO vs aiter: differential contract audit

This op is dispatched in place of `top_k_per_row_prefill`, so every term of that
contract aiter honours and AVO silently does not is a functional bug waiting for
the first caller who uses it. The shipped HIP-700 fault on unaligned row bases
was found **by accident** while scoping an unrelated tuning stage; this audit
exists to find the rest by construction instead.

Reproduce with [`bench/aiter_contract_audit.py`](../bench/aiter_contract_audit.py)
(19 terms x 2 backends, ~170 s) and, for a row-level comparison,
[`bench/avo_vs_aiter_rows.py`](../bench/avo_vs_aiter_rows.py):

```
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size 16G \
  -v /home/mh/aiter-topk:/aiter -v /home/mh/topk-prefill-avo:/topk -w /aiter \
  rocm/ali-private:ubuntu22.04_rocm10.0.0_cp313_torch2.12.0_vllm_dsv4_20260916 \
  python /topk/bench/aiter_contract_audit.py
```

Measured 2026-09-18, MI355X, `g_18-stable`, against `aiter-topk` at `c15ebce2c`.

## Method

Each term is driven from the **Python entry point**, not from our own harness, so
the dispatch, the `supports()` gate and the C++ argument checks are all in scope.
For each term the same inputs go through AVO (`AITER_DISABLE_TOPK_AVO=0`) and
through aiter's own mb/ob path (`=1`), and the two outputs are compared to each
other and to `torch.topk`.

Two design points that the results depend on:

- **Every case runs in its own subprocess.** A memory fault poisons the HIP
  context, so a single failing term would otherwise take every later term down
  with it and report a wall of false failures.
- **The comparison key must be order-independent.** AVO and aiter emit the same
  multiset in a different order (0 of 64 rows had identical index *lists*; 64 of
  64 had identical value *multisets*), so a raw fp32 `sum` accumulates in a
  different order and lands on a different rounding. The first version of this
  audit used the plain sum and reported the **baseline shape** as diverging when
  all 64 rows were in fact identical. Sort first, accumulate in float64.

## Results

Three terms fail on AVO and are served by aiter. Everything else either matches
aiter exactly or is out of contract for both.

### FAIL: unaligned row base, `rowStarts[r] % 4 != 0`

`rowStarts = r * 1` and `r * 65` both crash AVO with a memory fault while aiter
serves them and matches `torch.topk`. From our own harness, at M=256 N=131072
prefix=131072 with the CPU oracle: stride 64 passes on uniform, gaussian and inf;
strides 65, 1, 3 and 7 all give `HIP error 700`.

**One cause, not two.** With bases aligned (strides 4, 8, 64) and `len`
deliberately not a multiple of 4 (prefix 130000), every distribution passes, so
the `n4_cover` over-read past the row end is *not* the fault. The fault is the
`reinterpret_cast<const vfloat4*>(input + row * pitch + row_start)` in the load
loops assuming 16-byte alignment.

Blast radius: the parameter is documented and accepted (g_10), and
`topk_avo_supports(numRows, stride0, k)` **cannot see `rowStarts`**, so the
dispatcher has no way to route around it -- the fix has to be in the kernel.
Production is not hit today only because aiter's `create_row_boundaries` returns
`row_starts = zeros`. Note that this also means upstream aiter has never
exercised its own arbitrary-start path, so "aiter tests this" is not available as
an argument; our gate has to.

`stride0 % 4 != 0` is the **same root cause** (an odd pitch misaligns every row
after row 0) but a different symptom: `supports()` returns false, the dispatch
routes to aiter, and the caller silently gets the slower path instead of a fault.
One fix closes both.

### FAIL (low severity): `stride1 != 1`

aiter's prefill ignores the parameter entirely (`int64_t /*stride1*/`,
`topk_per_row_kernels.cu:2850`); AVO asserts it and aborts. A caller passing
`stride1 != 1` on contiguous data is passing a wrong value, so aiter "working"
here only means it ignored the argument -- but an abort is still worse than a
decline, and `topk_avo_supports` does not take `stride1`, so Python cannot route
around it. Fix shape: either widen `supports()` or turn the abort into a decline.

### Matches aiter, diverges from `torch.topk`: NaN

- **+NaN**: both AVO and aiter select it. The multiset comparison against torch
  fails only because `NaN != NaN`; with NaN normalised the multisets are equal.
- **-NaN**: both AVO and aiter **drop** it, where torch ranks NaN highest. Our
  key `(u & 0x80000000) ? ~u : (u ^ 0x80000000)` sends -NaN below -inf, and
  aiter's twiddle does the same under its comparison direction.

Both are aiter's semantics, which is the contract we are held to, so neither is
an AVO bug. Recorded because a caller migrating from `torch.topk` would see it.

### Out of contract for both: `rowEnds > stride0`

Faults on both implementations. Not an AVO regression; nothing to fix.

### PASS, identical to aiter

`baseline_uniform`, `rowstart_aligned_4`, `stride0_odd` (declined by
`supports()`, routed to aiter, correct), `stride0_nonpow2` (`2^k + num_rows`),
`inf_mixed`, `k_small` (k=3), `k_at_cap` (k=8192), `rowlen_zero`
(`rowEnds == rowStarts`), `rowlen_negative` (`rowEnds < rowStarts`; both emit all
`-1` / `-inf`), `rowlen_short` (identity emit, `-inf` padding on both),
`numrows_one`, `workspace_exact` (`topk_avo_workspace_size` is sufficient),
`stable_true` (never routes to AVO).

Row-level confirmation on four shapes (M=64 N=65536, the same with
`rowStarts = 4r`, M=8 N=131072, M=256 N=65536): **every** row's value multiset is
identical between AVO and aiter and equal to `torch.topk`. Emit order differs,
which the contract does not constrain -- `stable=True` is the ordering-guaranteed
mode and it does not route here.

## Terms that are not applicable

- **`in_idx`** (`in_idx_buf`, `topk_per_row_kernels.cu:404`): the kernel accepts
  an input-index array but the prefill entry passes `nullptr`
  (`:2921`), and the Python signature has no such parameter. Not reachable.
- **`Phase::Decode` / `next_n`** (`:368`, `:1509`): a separate entry point
  (`top_k_per_row_decode`) with its own dispatch. The prefill entry always passes
  `Phase::Prefill` and `next_n = 0`.
- **`stable = true`**: excluded by the dispatch condition in `topk.py`, verified
  above.

## Harness limitation worth knowing

`benchmark_topk` refuses `K > N` on the **uniform** path
(`benchmark_topk.hip.cpp:1466`), so that combination cannot be scored there. It
is not a product gap: the aiter entry always supplies `rowStarts`/`rowEnds` and
therefore always instantiates `RAGGED=true`, where `geometry_k_ragged` handles it
-- confirmed by three ragged shapes with `k > pitch` passing (M=1024 pitch=1024,
M=64 pitch=512, M=4096 pitch=512).

## Bug list, ordered by blast radius

1. **Unaligned row base faults.** Documented parameter, no way for the caller to
   detect it in advance, hard GPU fault. Fix by porting aiter's `skip_cnt`
   head / aligned-middle / tail structure (`topk_per_row_kernels.cu:171-272`),
   which also makes `stride0 % 4 != 0` servable. Stage 2 of the v5 plan.
2. **`stride1 != 1` aborts instead of declining.** Needs an API decision, not a
   kernel change.
