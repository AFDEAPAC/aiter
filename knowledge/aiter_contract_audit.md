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

### FAIL: unaligned row base, `rowStarts[r] % 4 != 0` -- FIXED in v5 Stage 2

`rowStarts = r * 1` and `r * 65` both crash AVO with a memory fault while aiter
serves them and matches `torch.topk`. From our own harness, at M=256 N=131072
prefix=131072 with the CPU oracle: stride 64 passes on uniform, gaussian and inf;
strides 65, 1, 3 and 7 all give `HIP error 700`.

**The root cause is the over-read, not the misalignment.** An earlier version of
this document said the opposite, on the strength of an isolation experiment that
did not isolate anything: aligned strides with `len % 4 != 0` at prefix 130000
passed, but in that configuration the slice also ended ~1000 floats short of the
pitch, so the over-read stayed in bounds and the case could not have failed
either way.

The experiment that does separate them holds the stride fixed and moves only the
slice end:

- misaligned bases (strides 65, 1, 3 giving `base % 4` in {1, 3}) with the slice
  ending far from the pitch (prefix 100000): **all pass**, every distribution,
  on both the sampled and the small_n path;
- the same strides with the slice running to the pitch (prefix 131072):
  **HIP 700**;
- stride 64 with the slice running to the pitch: passes, because `len` is then a
  multiple of 4 and there is no over-read at all.

So gfx950 serves a 4-byte-aligned `global_load_dwordx4` natively, and declaring
the `vfloat4` typedef `aligned(4)` changed nothing -- tried, still faulted. What
faults is `n4_cover(len)` rounding the vector count UP: the last vector of a
slice whose length is not a multiple of 4 reaches up to 3 floats past it, which
lands in the next row for an interior row but past the **end of the allocation**
for the last one. Nonzero rowStarts are what make `row_start + len` land within 3
floats of the pitch, which is why only they expose it.

Fixed by `load_row_f4<RAGGED>` (`csrc/topk_common.hip.hpp`), which loads the
final partial vector element by element under `< len`. `RAGGED=false` keeps the
plain load: 20 of 34 kernels have byte-identical instruction streams after the
fix and every kernel that changed is a `RAGGED=true` instantiation, so the scored
uniform grid is untouched (inner tier +0.04%, 0 cells regressed). The ragged path
pays +0.63% to +0.83%, measured interleaved against the pre-fix binary.

Blast radius before the fix: the parameter is documented and accepted (g_10), and
`topk_avo_supports(numRows, stride0, k)` **cannot see `rowStarts`**, so the
dispatcher had no way to route around it -- the fix had to be in the kernel.
Production was not hit only because aiter's `create_row_boundaries` returns
`row_starts = zeros`. That also means upstream aiter has never exercised its own
arbitrary-start path, so "aiter tests this" was never available as an argument;
our gate now carries it (`grid.ROWSTARTS` strides 1, 3, 7, 65).

`stride0 % 4 != 0` was a **separate** question, and once alignment was known not
to be the problem it turned out to need no kernel change at all -- see the next
section.

### FAIL (low severity): `stride1 != 1` -- FIXED by routing, not by semantics

aiter's prefill ignores the parameter entirely (`int64_t /*stride1*/`,
`topk_per_row_kernels.cu:2850`); AVO asserts it and aborts. A caller passing
`stride1 != 1` on contiguous data is passing a wrong value, so aiter "working"
here only means it ignored the argument -- but turning a working call into an
abort purely because AVO became available is a regression we introduced.

Fixed by adding `stride1 == 1` to the dispatch condition in
`aiter/ops/topk.py`, so such calls keep going to mb/ob exactly as before. The
assert stays for direct callers of `top_k_per_row_prefill_avo`. Deliberately
NOT fixed by ignoring `stride1` the way aiter does: silently computing against
the wrong layout is worse than declining, and `topk_avo_supports` cannot express
the condition because it only takes `(numRows, stride0, k)`.

### RESOLVED: `stride0 % 4 != 0` is now served, and it needed no kernel change

Three guards were rejecting it -- `sampling_geometry_ok`, `sample_stride_exact`
and `topk_avo_supports` -- plus the harness's own `N % FP32_EPT` check. All four
were there for the alignment that Stage 2 proved is a non-issue. The kernels
already handled an odd pitch:

- `sample_chunk_stride` masks the chunk spacing to a multiple of 4, so chunk
  starts stay 4-aligned **relative to the row base** whatever the base is;
- the RAGGED instantiation counts vectors with `n4_cover(len)` and
  `load_row_f4<true>` loads the final partial vector element-wise;
- gfx950 serves the resulting 4-byte-aligned `dwordx4` natively.

So the change is four relaxed guards, and the harness routes an odd pitch through
RAGGED with full-row extents -- which is what the aiter entry always instantiates
anyway. `RAGGED=false` is left byte-identical: it truncates its vector count and
applies no per-lane bound, and giving it one would cost the scored pow2 grid a
compare per element for a case it never sees.

Evidence, beyond the multiset check against the CPU oracle:

- **Positive control on the tail.** Each row's maximum planted at index `N-1`,
  which is only reachable through the clamped partial vector: found in 8 of 8
  rows on the small_n path (M=8 N=12289) and 8 of 8 on the sampled path
  (M=8 N=131073, coop_g=16, `under_K=0`). If the tail were dropped the top-k
  would be missing its largest entry.
- **All three residues** are in the gate (`grid.ODD_NS`), since the clamped tail
  is 3, 2 and 1 elements long respectively. 40 new points, `verify_grid --dist
  all` 3085/3085.
- **Failable first**: `--inject-fault 1` and `2` both fire at N=131073 and
  N=196609.
- **Identical to aiter** on all four odd-N audit cases, including the
  tail-max control.

What it buys, measured through the real Python dispatch in the correctness image
(`AITER_DISABLE_TOPK_AVO` 1 vs 0), since these shapes previously fell back to
aiter's own mb/ob path:

| shape | aiter | AVO | speedup |
|---|---|---|---|
| M=64 N=65537 | 73.40 us | 57.63 us | 1.27x |
| M=256 N=131073 | 116.93 us | 80.66 us | 1.45x |
| M=1024 N=131075 | 290.94 us | 190.29 us | 1.53x |
| M=4096 N=131073 | 972.14 us | 661.30 us | 1.47x |
| M=64 N=196609 | 107.39 us | 66.22 us | 1.62x |
| M=256 N=1048573 | 693.48 us | 246.69 us | 2.81x |

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

1. **Unaligned row base faults.** FIXED in v5 Stage 2 by clamping the final
   partial vector, not by porting aiter's `skip_cnt` head/tail -- the head half
   of aiter's design turned out to be unnecessary here, because the hardware
   tolerates the misaligned wide load and only the over-read was fatal. The gate
   now carries the repro as a negative control: all four new `ROWSTARTS` entries
   fault on the pre-fix binary and pass on the post-fix one.
2. **`stride0 % 4 != 0` declined.** FIXED in v5 Stage 7 by relaxing four guards.
   No kernel change was needed once Stage 2 had established that the
   misalignment is not the problem; worth 1.27x to 2.81x against the aiter path
   these shapes used to fall back to.
3. **`stride1 != 1` aborts instead of declining.** FIXED by adding
   `stride1 == 1` to the dispatch condition in `aiter/ops/topk.py`.

After all three fixes the audit reports **no AVO-only failure**: 21 of 22 terms
identical to aiter (including four odd-`stride0` cases), 1
(`rowEnds > stride0`) out of contract for both, and the NaN pair matching aiter
while both differ from `torch.topk`.

The cheapest lesson in the list: fix 2 was scoped as a feature -- port aiter's
head/middle/tail to seven load sites -- and turned out to be four deleted `if`
statements. What made the difference was Stage 2 correcting the root cause first.
A wrong diagnosis does not just produce a wrong fix, it produces a wrong estimate
of the work.

## What generalises

The alignment hypothesis was wrong, and it was wrong in the way that is easy to
miss: a real fault, a plausible mechanism, and a confirming experiment that had
no power to falsify it. The load *was* misaligned; that simply was not why it
faulted. What caught it was re-deriving the isolation so that exactly one
variable moved -- same stride, different slice end -- rather than trusting the
first experiment that agreed with the theory. Also worth keeping: the fix that
followed from the correct cause is a third of the size of the one that followed
from the wrong cause.
