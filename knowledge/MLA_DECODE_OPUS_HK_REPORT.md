# MLA decode — OPUS + HipKittens (HK) HIP implementations: handoff report

Status snapshot of two **experimental gfx950 MLA-decode attention** implementations on this
`mh-hip-mla-dev` branch, for a colleague tracking progress. Both are HIP-source (no hand asm),
target MI350X/MI355X (gfx950), and are gated behind the experimental flag — they are NOT in
upstream aiter. Deep design notes + the full experiment ledger live in
[`knowledge/mla_decode_opus.md`](mla_decode_opus.md) and
[`knowledge/opus_deep_pipeline_project.md`](opus_deep_pipeline_project.md); this file is the map.

## 1. TL;DR

| impl | what it is | best shapes | perf vs hand-asm | status |
|---|---|---|---|---|
| **OPUS** `mla_decode_opus` | MLA decode written in aiter's OPUS single-header tile DSL. Absorbed **D=512** + a **RoPE D=576** variant; split-KV + qpack for small batch. | small/medium batch, qlen≥2 (qpack/split-KV that asm lacks); beats asm on many shapes | ~1.0–1.3× of asm; residual gap is **compiler-owned register allocation** (not algorithm/memory) | correct (pytest green); optimized to the compiler-regalloc floor |
| **HK** `hk_mla_decode_fwd` | MLA decode in the HipKittens tile DSL (manual AGPR register control). bf16 qh64 + fp8 qh64/qh128 decode. Any (qh,qlen) split of M (e.g. bf16 qh32/qlen2, qh16/qlen4) = MTP/qlen>1 supported. | the hottest qh64/qh128 single-position decode; also qlen>1 MTP | ~1.05–1.07× of asm (bf16 qh64); residual is asm's **holistic hand-schedule** | correct (staged-verified, incl. qlen>1 after the STAGE-34 split-output bound fix); at the HIP-source floor |

The two are complementary: OPUS = flexible scaffolding (split-KV/qpack/dispatch) but compiler regalloc;
HK = manual register pinning but shallower compiler-scheduled pipeline. The natural next step is a
**hybrid** (OPUS outer scaffolding + HK-style register control on the inner GEMM loop).

## 2. Source layout

### OPUS (`module_mla_decode_opus`)
| file | role |
|---|---|
| `csrc/include/mla_decode_opus.h` | the kernel (traits, layouts, QK→softmax→PV, split-KV/qpack, 3 pipeline variants) |
| `csrc/py_itfs_cu/mla_decode_opus_kernels.cu` | `#define MLA_DECODE_OPUS_IMPL` + include of the header → JIT TU |
| `csrc/pybind/mla_decode_opus_pybind.cu` | pybind entry |
| `aiter/ops/mla_decode_opus.py` | Python ops (`mla_decode_opus_fwd`, `_splitkv_fwd`, `_qpack_fwd`, `_qpack_splitkv_fwd`) + the `mla_decode_opus` wrapper + `_pick_num_splits` / `_qpack_min_b` heuristics |
| `aiter/__init__.py`, `csrc/include/rocm_ops.hpp`, `aiter/jit/optCompilerConfig.json` | op registration + JIT module config |

### HK (`hk_mla_decode_fwd`)
| file | role |
|---|---|
| `csrc/kernels/mla/hk/mi35x_v32_fwd_decode_m16x4_bf16_bf16.cuh` | the bf16 qh64 decode kernel (art register tiles, fused QK→softmax→bf16-P-pack→PV) |
| `csrc/kernels/mla/hk/hk_mla_buffer_managers.cuh` | LDS tile / KV buffer managers (no-pad layout, b128 async load) |
| `csrc/kernels/mla/hk/hk_mla_utils.cuh` | warp-reduce, helpers (includes `opus.hpp`) |
| `csrc/kernels/mla/hk_decode_fwd.cu` | `hk_mla_decode_fwd` host entry + arch/shape guards (gfx950, nhead·qlen ∈ {64,128}) |
| `aiter/mla.py` | dispatch: routes `mla_decode_fwd` to HK under the `use_hk` gate (§4) |

### Design notes (this branch, `knowledge/`)
- `mla_decode_opus.md` — OPUS behavior, profiling, the grid-z QLEN optimization, split-KV/qpack.
- `opus_deep_pipeline_project.md` — the full HK-vs-asm deep-dive ledger (STAGE log: correctness bugs, the operand-double-buffer / resource-injection findings, the per-wave≠wall trap).

> Excluded from this branch (scratch, intentionally not pushed): root `_*probe.cu`, `*.cand_*`,
> `archive_stage9_singlestage/`, `*.html`, `.rocprofv3/`.

## 3. Build

Both are **JIT-compiled** by aiter on first call (`@compile_ops(..., develop=True)`).

- **OPUS** module: `module_mla_decode_opus`. After editing `mla_decode_opus.h`, the JIT may not
  auto-detect the header change → force a rebuild: `AITER_REBUILD=1` (full) / `AITER_REBUILD=2`
  (module only), or `rm` the stale `.so`. `qlen` is a host `switch` over compiled templates (1..17);
  device body is `#if defined(__gfx950__)`.
- **HK** module: built from `csrc/kernels/mla/hk_decode_fwd.cu`; **requires HipKittens**, which aiter
  **auto-clones** at build time into `3rdparty/HipKittens` (`aiter/jit/core.py` `clone_3rdparty`,
  `HIP_KITTENS_DIR`, from `https://github.com/HazyResearch/HipKittens.git`). That is why HipKittens is
  **not** committed to this branch — a clean checkout fetches it on first build. HK is gfx950-only.
- Toolchain: ROCm 7.2.x, hipcc/clang for gfx950, on an MI350X/MI355X box.

## 4. How to run / dispatch

**OPUS** — call directly:
```python
from aiter.ops.mla_decode_opus import mla_decode_opus
out = mla_decode_opus(q, unified_kv, kv_indices, kv_indptr, attn_sink,
                      softmax_scale, qlen)   # absorbed D=512; rope variant = *_rope entry
```
Single-pass vs split-KV vs qpack is chosen by `_pick_num_splits` / `_qpack_min_b` (small-batch regime).

**HK** — dispatched through the normal `aiter.mla.mla_decode_fwd` when ALL hold (`aiter/mla.py:439`):
`get_gfx()==gfx950` **and** `nhead*max_seqlen_q ∈ {64,128}` **and** Q/KV dtype matches (bf16 for the
qh64 case; fp8 for qh64 / qh128) **and** `page_size ∈ {1,64}` **and** `AITER_ENABLE_EXPERIMENTAL=1`.
Otherwise it falls back to the production asm path.

## 5. Tests & benchmarks (`op_tests/`)

**Correctness (pytest, fp32 reference):**
- `test_mla_decode_opus.py` — OPUS absorbed-512 decode vs a per-batch CSR-prefix causal reference.
- `test_mla_decode_opus_rope.py` — OPUS RoPE (D=576) variant; contract matches aiter asm `mla_decode_fwd`.
- `test_mla_persistent.py` — HK persistent decode (builds the persistent metadata; `AITER_ENABLE_EXPERIMENTAL=1`
  → HK, `=0` → asm; both print "golden vs aiter_asm").

**Perf / comparison (timing only, min-of-many, idle GPU):**
- `bench_mla_decode_opus.py` — per-(QLEN,H) latency, matrix TFLOPS, HBM GB/s for OPUS.
- `bench_splitkv_mla_decode_opus.py` — single-pass vs split-KV (workspace pre-allocated, splits fixed).
- `sweep_mla_decode_opus.py` — B × KV-length sweep to locate the bound regime.
- `sweep_hk_vs_asm_decode.py` — qh64/qlen1 bf16 HK-vs-asm over a ctx×B grid (run twice, EXP=1/0).
- `cmp_asm_vs_opus_mla_decode.py` — asm vs OPUS absorbed-512 (caveat: asm does 576-dim QK → ~12% more FLOPs; ballpark).
- `cmp_asm_vs_opus_mla_decode_rope.py` — **apples-to-apples** asm-rope (576/512) vs OPUS-rope (same data, identical FLOPs).
- `ab_mla_decode_opus.py` — within-process grid-z QLEN vs serial qlen=1 launches (A/B for the grid-z win).

Run inside the dev container, GPU idle (`rocm-smi --showuse`), e.g.:
```bash
AITER_ENABLE_EXPERIMENTAL=1 python3 op_tests/test_mla_persistent.py -n 64,1 -d bf16 -kvd bf16 -b 16 -c 16384
python3 op_tests/test_mla_decode_opus.py        # OPUS correctness
python3 op_tests/cmp_asm_vs_opus_mla_decode_rope.py   # fair asm-vs-OPUS
```

## 6. Analysis / current status (full detail in `knowledge/*.md`)

**OPUS** (`mla_decode_opus.md`): correct (pytest green). Key results:
- QLEN moved onto the grid z-dim → **~1.84–1.88×** for H=16 (was occupancy-limited at 1 block/CU);
  H=128 already LDS-capped (~1.0×). Linear-in-QLEN serial loop removed.
- Bound (H=128): `MfmaUtil ≈ LdsUtil`, ~2 waves/SIMD → **latency/occupancy bound, LDS-capped** (132 KB
  LDS → 1 block/CU); **NOT HBM-bound** (dense KV is L2-resident). The residual vs asm is the
  **compiler-owned regalloc** (`ds_read→VGPR, accum→AGPR` shuffles + shallow prefetch), not algorithm.
- Small-batch covered by **split-KV** (`_pick_num_splits`, floor heuristic) + **qpack** (1 KV read shared
  across qlen positions) — a regime the asm decode does not have.

**HK** (`opus_deep_pipeline_project.md`): correct (staged dump-and-compare). Key results:
- bf16 qh64 decode ≈ **1.05–1.07×** of the hand-asm `.co` (3-rep back-to-back).
- Bound proven by **resource injection** = **matrix-unit + HBM, overlapped**; feed/LDS-read is hidden →
  operand double-buffer + deeper ring are **wall-neutral** (verified). The residual gap is asm's
  holistic hand-schedule (compact code, fine-grained ring), not a resource HK lacks → **closing it from
  HIP source is not expected**; route the hottest shape to asm, keep HK where it wins.
- Trap captured: at occupancy-1, **per-wave ATT stall / `wave_dur` is not a wall proxy** — trust 3-rep
  wall only.

**Open / next:** the OPUS+HK hybrid (OPUS scaffolding + HK register control on the inner GEMM); the
RoPE chunk-8 cross-warp visibility race on the NW8 pipelined path (currently forced to single-buffer le2,
~8–16% left on the table — see `mla_decode_opus.md`).

## 7. Caveats
- **Experimental**: gated behind `AITER_ENABLE_EXPERIMENTAL`; not upstream; APIs/heuristics may change.
- gfx950 only for HK; OPUS is gfx950-tuned (numbers are MI355X / ROCm 7.2.x).
- Perf numbers above are from an idle-box, min-of-many / 3-rep methodology; re-measure on your box
  (thermal/clock drift is ~5% over a long session).
