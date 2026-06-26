# Project plan: opus deep-pipeline kernels (match asm qlen=1 / fix NW8)

Status: PROPOSED (2026-06). Owner: TBD. Scope: gfx950 MLA decode (RoPE D_qk=576/D_v=512, bf16).

## 1. Problem
opus's general decode kernels trail aiter's hand-tuned per-qh asm kernels in two
regimes, and the cause is the same: a **shallow software pipeline**.

- **qlen=1, mid-batch (B16-128), nh in {8,32,64}: ~1.4-1.6x slower than asm.**
  qpack can't help (1 position -> no KV reuse); splits are already optimal; it is
  pure kernel efficiency.
- **NW8 (nh128 qlen>=2, nh32-qpack qlen=4): forced to single-buffer le2** (no
  load/compute overlap) because the hand-scheduled 10-cluster pipeline has an
  unresolved cross-warp LDS race on the rope chunk-8 read (~8-16% left on the table).

Real MLA models use nh 16/128 (already ~1.0-1.2). nh 8/32/64 are "other qh aiter
ships"; this project is what it takes to reach <=1.0-1.1 there.

## 2. Evidence (why it's a redesign, not a patch)
Disassembled `mla_a16w16_qh64_qseqlen1_gqaratio64_v3.co` (gfx950):
- `.group_segment_fixed_size = 163840` (**full 160 KB LDS**), `.vgpr_count = 512`
  (**max VGPR**), 1 block/CU. asm spends *all* per-block resources for depth, not
  occupancy.
- Steady-state `s_waitcnt vmcnt(10..20)` -> **keeps ~2 KV tiles of global loads in
  flight** (deep prefetch), 2 LDS buffers in the 160 KB.
- `a[72:207]` (~135 AGPRs) operand window: `ds_read_b64_tr_b16` issued far ahead of
  the `v_mfma_f32_16x16x32_bf16` that consumes it, interleaved ~1:1 -> LDS latency
  fully hidden.
- KV loop **unrolled ~8x**.

opus emits a shallow version (drain `vmcnt(0-1)`, modest register buffering, light
unroll). A prior double-buffer patch on the 16mx1 loop **regressed** (128->139us):
it added the cheap 2nd LDS buffer but not the deep register/prefetch pipeline, and
2x LDS (76KB rope tile -> 152KB) forced 1 block/CU with no compensating overlap.

Conclusion: the win lives in **codegen depth** (register-resident operand buffering
+ deep vmcnt + unroll at 1 block/CU). hipcc's AGPR/VGPR allocation for deep MFMA
pipelines is the weak link -> needs **explicit register/tile control**, not "hope
the compiler schedules it".

## 3. Goals / success criteria
- G1: qh{32,64} qlen=1 mid-batch (B16-128, ctx 1.2K-64K) **<= 1.1x vs asm**.
- G2: NW8 pipelined correct + overlapped -> nh128 qlen=1 mid-ctx mid-batch and
  nh32-qpack qlen=4 each gain back the ~8-16% le2 leaves (target <= 1.0-1.05x).
- G3: no regression on the shipped wins (nh16/128 qlen{1,2,4}, nh32 qpack qlen>=2).
- Measure with the existing `cmp_asm_vs_opus_mla_decode_rope.py` (warmup20/iters100/
  min10) on an **idle** GPU; gate on opus-vs-fp32 <= 5e-4.

## 4. Options (with tradeoffs)
A. **HipKittens -> opus** (preferred long-term). Tile DSL with explicit register-tile
   + MMA + async-pipeline primitives; structures the pipeline so the compiler can't
   mis-schedule the deep buffering. Pros: reusable across shapes, expresses deep
   prefetch/unroll naturally, path to also fix NW8. Cons: still via hipcc (register
   control must be explicit in HK), integration + a 2nd authoring layer, gfx950
   validation needed.
B. **Hand-asm round-trip** (`asm-kernel-build-patch` + `vendor-lib-dispatch` skills):
   clone/patch the vendor `.co`->`.s`, tune, register into opus dispatch. Pros: asm-
   class perf now, per shape. Cons: per-shape, brittle, no general solution.
C. **opus.hpp DSL deep-pipeline codegen**: make the DSL emit deep buffering/unroll.
   Pros: in-house. Cons: fights the exact hipcc-AGPR limit above; hardest.

## 5. Phased plan
- **Phase 0 (1-2 wk) - HK proof of concept.** Port ONE kernel (qh64 qlen=1) in
  HipKittens on gfx950. Verify it emits ~512-VGPR / deep-`vmcnt` / unrolled schedule
  (check via `llvm-objdump` + `.vgpr_count`/`group_segment`), correctness <= 5e-4,
  and **<= 1.1x vs asm** at B16-128. GO/NO-GO on HK.
  - If NO-GO: fall back to Option B for the few hot shapes.
- **Phase 1 - generalize qlen=1.** qh{8,16,32,64,128} qlen=1 via HK; wire into the
  opus dispatch behind the existing wrapper; re-sweep.
- **Phase 2 - NW8 / qlen>=2 deep pipeline.** Rebuild the 16mx8 accum (or HK
  equivalent) with the deep pipeline; resolve the chunk-8 cross-warp visibility with
  explicit tile sync; target G2. (This also retires the `FORCE_LE2` workaround.)
- **Phase 3 - consolidate heuristics.** Fold the new kernels into `_pick_num_splits`
  / qpack routing; one clean full sweep; update the bench HTML + knowledge.

## 6. Risks
- HK may not emit the deep pipeline on gfx950 without manual scheduling -> Phase 0 gate.
- ~~D=576 LDS (76 KB/tile) caps double-buffering at 2 tiles in 160 KB; >2-tile prefetch
  needs a more compact KV layout (extra redesign).~~ **CORRECTED in STAGE 26 (2026-06):
  the 72 KB is HK's COARSE 64-KV-pos tile (kBlockN=64). asm uses 16-KV-pos FINE tiles
  (18 KB each) and rings ~8 of them in the same 160 KB. A deep ring does NOT need a more
  compact layout -- it needs FINER tile granularity (smaller kBlockN). Capacity is not the
  wall; granularity + compiler scheduling is.
- NW8 chunk-8 race is unresolved at source level (needs ATT/LDS-dump instrumentation);
  Phase 2 may require that diagnosis first.
- Effort vs payoff: nh 8/32/64 are non-standard; prioritize only if a workload needs them.

## 7. Out of scope (already shipped this session)
Trend-fit split heuristic, qpack ctx>=16K crossover, stage-2 D-tiling, qpack-H32
split-KV (qlen>=2 nh32 1.5-3x -> ~1.0-1.3). These are the non-deep-pipeline wins.

## 8. HK verification (2026-06) - Option A primitives confirmed, no longer "hope"
Pulled HK to `3rdparty/HipKittens` (clones from HazyResearch at pinned commit
`a5e308a`, see `aiter/jit/core.py:731`) and read the source. The §2/§4-A claim
"needs explicit register/tile control, not hope-the-compiler" is now VERIFIED at
the primitive level:

- **Register-file is a source-level choice via a flat index convention**
  (`include/common/macros.cuh`): index `< 256` -> emits `v[idx]` (VGPR); index
  `>= 256` -> emits `a[idx-256]` (AGPR).
- **`ds_read_b64<GPR>` / `ds_read_b128` / `ds_read_b64_tr_b8/b16` read LDS DIRECTLY
  into AGPR** when `GPR>=256` (`macros.cuh:221` -> `ds_read_b64 a[%0:%1], ...`).
  This is exactly the copy our root cause said hipcc can't skip (hipcc always does
  ds_read->VGPR then accvgpr_write VGPR->AGPR; HK emits ds_read->AGPR).
- **`mfma_f32_16x16x32_bf16<A,B,C,D>()` enumerates all 16 a/v operand combos**
  (`macros.cuh:586+`) - D/A/B/C each independently VGPR or AGPR. bf16 path exists,
  not just fp8.
- **Two modes**: `rt` (`types/register/rt.cuh`, compiler-allocated data array,
  portable/easy) vs `art` (`types/register/art_base.cuh`, "uses register ranges
  instead of data arrays for assembly-level register management" - the hot-loop
  asm-mode where you hand-assign indices + overlays). HK lets you choose per-kernel
  where to sit on the portable<->control spectrum.

HK already ships an MLA decode reference in THIS aiter tree (commit `57ed6a01a`,
`csrc/kernels/mla/hk/`): `mi35x_..._m16x4_fp8_fp8` is **qh64 qlen1** (blockM=64,
gfx950), `..._m16x8` is qh128; `mi3xx_...` is the gfx942 fork. They use `art` tiles
with hand-assigned ranges + `clobber<>` + deliberate register overlaying
(`mi35x_v32_fwd_decode_m16x4_fp8_fp8.cuh:62-119`), `amdgpu_num_vgpr(68)` on gfx950
vs `(72)` on gfx942, and ~80 hand-written `s_waitcnt` + `s_setprio`.

### Portability ceiling (concrete)
- HK README states support for **CDNA3 + CDNA4 only**. The `>=256 -> AGPR` scheme is
  CDNA-specific; **RDNA has no AGPR**, so the whole art/AGPR pipeline is structurally
  inapplicable on gfx12. HK is "write-once across gfx942<->gfx950 re-TUNE", not
  "write-once across families".
- Within CDNA the per-arch cost is a re-tune (num_vgpr budget, register layout,
  waitcnt), not an asm rewrite - algorithm + tile types + managers carry over.

### TRACED (2026-06): shipped qh64 fp8 kernel is ALL-VGPR, not AGPR
Resolved the open question. The shipped `m16x4` (qh64 fp8, gfx950) uses **zero AGPR**:
- All register constants are 68..255 (`mi35x_..._m16x4_fp8_fp8.cuh:49-75`); the macro
  convention maps `<256 -> VGPR`. `art.cuh:50` confirms `range::lo = L` is the raw
  index (no hidden +256 offset). So Q/K/V/P/output-accum all live in **v68..v255**.
- `amdgpu_num_vgpr(68)` reconciled: it's a deprecated clang **hint** (limits only the
  COMPILER's own VGPR allocations, "exact number not guaranteed, rounded up to satisfy
  allocation requirements" per clang AttributeReference). So compiler temporaries sit
  in v0..v67; the hand-managed art tiles occupy v68..v255 via inline asm + `clobber<>`;
  the backend rounds the real `.vgpr_count` up to 256 because the asm references v255.
  Net: full 256 VGPR, 0 AGPR, 1 wave/SIMD.
- Register budget fills exactly: operands q_nope32 + q_rope4 + kv 4+4 + p_comp16 = 60,
  output accum (fp32) = 128, total 188 = (256-68). fp8's small operand footprint is
  what lets it be VGPR-only.

### KEY INSIGHT: HK qh64 strategy != the asm .co strategy (different mechanism, same goal)
- asm `.co` (bf16, §2): AGPR-heavy - `a[72:207]` ~135 AGPRs, `ds_read_b64_tr_b16` ->
  AGPR. Uses AGPR as a 2nd register bank to hold KV operands + free VGPR.
- HK `m16x4` (fp8): VGPR-only + deliberate **register overlaying** (`p_mfma` overlays
  `p_comp[0..3]`, alt-V overlays `p_comp[8..15]`, lines 62-75). No AGPR.
- Both bypass hipcc regalloc (the real root cause); they just pick different register
  files. fp8 fits VGPR-only; **[extrapolation]** a qh64-**bf16** port roughly doubles
  the operand footprint (60 -> ~120 regs) while the fp32 accumulator stays 128, so it
  overflows the 188-reg window and would likely be FORCED to push K/V into AGPR
  (index >= 256, which the HK macros support) - i.e. reproduce the asm strategy, NOT a
  trivial dtype swap of the fp8 kernel.

### Next step before committing to a bf16 prototype
- Decide bf16 register plan first: confirm the operand-doubling overflow above by laying
  out the bf16 tile budget, then choose VGPR-overlay vs AGPR-spill (likely AGPR). The
  fp8 `m16x4` is the structural template (managers, schedule, softmax) but its register
  map does NOT carry over to bf16 unchanged.

### OPUS internals (traced 2026-06) - why "swap regalloc, keep pipeline" is NOT separable
- OPUS issues MFMA via `__builtin_amdgcn_mfma_*` (`opus.hpp:2243-2247`) and holds tiles
  in `opus::vector_t<T,N>` (`opus.hpp:2164`) -> **register allocation is 100% the
  compiler's**. OPUS DOES hand-schedule waitcnt (`s_waitcnt_lgkmcnt(number<k_ds_read_insts>)`,
  `mla_decode_opus.h:1032`) + CSE-break asm hints, but never assigns a register file/index.
- Consequence: you cannot "embed HK regalloc under an unchanged OPUS pipeline". In this
  kernel class regalloc == pipeline depth (deep prefetch == more in-flight tiles ==
  explicit residency). Replacing the builtin path means rewriting the inner gemm data
  path (loads+tiles+mfma) AND its schedule as one unit.
- FEASIBLE hybrid (mechanically proven: the shipped HK kernel already
  `#include "opus/opus.hpp"`, `hk_mla_utils.cuh:8`, mixing `opus::` + `kittens::` in one
  TU): keep OPUS outer scaffolding (split-KV, work dist, RoPE, softmax/epilogue, dispatch,
  LDS layout) + rewrite ONLY the QK/PV inner gemm loop in HK `art`. The boundary is the
  gemm loop, NOT "register allocation". This reuses the easy scaffolding, not the hard loop.

### Phase 0 execution (started 2026-06): build+bench the SHIPPED HK fp8 kernel first
Cheapest-information-first: before writing any bf16, build and benchmark the already-shipped
`mi35x_..._m16x4_fp8_fp8` (qh64) to (1) prove the HK toolchain builds/runs on this gfx950
box, (2) get a real HK perf + `.vgpr_count` + vmcnt-depth datapoint, (3) compare vs the
comparable asm MLA kernel. GO/NO-GO on HK rests on this before bf16 authoring cost.

#### RESULTS (2026-06, container pa_bench_mh = host /home/mh/aiter -> /aiter, MI355X gfx950, ROCm7.2)
Driver: `PYTHONPATH=/aiter AITER_ENABLE_EXPERIMENTAL=1 python3 op_tests/test_mla_persistent.py
-n 64,1 -d fp8 -kvd fp8 -b <B> -c 1200`. HK builds clean (module_hk_mla, 33s) + correctness PASS.

A/B same fp8 qh64 config (experimental ON=HK vs OFF=asm), single-run us @ ctx1200:
| B | HK | asm | HK/asm |
|---|----|-----|--------|
| 16 | 28.61 | 22.08 | 1.30x |
| 32 | 33.72 | 24.96 | 1.35x |
| 64 | 39.48 | 32.79 | 1.20x |
|128 | 60.78 | 46.09 | 1.32x |

ISA diff (qh64 fp8, disasm via clang-offload-bundler unbundle + llvm-objdump):
| | HK `m16x4` (page1) | asm `mla_a8w8_qh64...v3_ps` |
|---|---|---|
| steady vmcnt | **<=2** (shallow) | **10** (deep prefetch) |
| AGPR operands | **0** (all VGPR, max v255) | **1286** (max a[140]) |
| v_mfma | 2040 | 294 |
| disasm lines | 16976 | 6326 |

**KEY VERDICT: HK != asm perf for free.** AMD's own shipped HK fp8 kernel is 1.2-1.35x SLOWER
than the asm fp8 kernel, and the ISA shows WHY: HK runs the SAME shallow-pipeline pattern as
OPUS (vmcnt<=2, zero AGPR) despite HK's macros supporting deep-vmcnt + ds_read->AGPR. The HK
authoring layer gives you the *expressiveness* to write the deep pipeline; it does NOT do it
for you. A naive HK port lands at ~1.3x (the OPUS gap class), not asm parity.

#### Sharp next experiment (fp8, cheapest variable-isolating test of "can HK catch asm")
Both HK and asm fp8 qh64 kernels already exist -> modify the HK `m16x4` to (a) deepen prefetch
(issue async buffer_load_lds ahead, hold vmcnt ~10) and (b) move K/V tiles to AGPR (range index
>= 256, supported by the HK macros). Re-bench vs asm fp8.
- If it closes the 1.3x gap -> GO: HK can reach asm, and we have the method (then port to bf16).
- If deep+AGPR HK still trails -> NO-GO signal: HK has structural overhead; prefer asm round-trip
  (Option B) for the hot shapes.
This isolates the exact root-cause variable in the dtype where a clean A/B already exists, far
cheaper than authoring bf16 from scratch.

#### CORRECTION (2026-06): fp8 gap is MFMA-SIZE dominated; bf16 (our target) is NOT
Disasm of the two asm kernels shows the fp8 detour was the WRONG dtype to experiment on:
- fp8 asm `mla_a8w8_qh64...`: `v_mfma_f32_32x32x64_f8f6f4` x294 (BIG scaled fp8 MFMA).
- HK fp8 `m16x4`: `v_mfma_f32_16x16x32_fp8_fp8` x2040 (SMALL, 8x more instrs). 294*8~=2040.
- HK's asm-mode (`art`) macros only go to 16x16x32 fp8; the 32x32x64 exists ONLY in HK's
  rt/builtin path (`ops/.../register/tile/mma.cuh:104 mfma323264`, compiler-allocated), NOT
  in `art`. So matching asm in fp8 needs a NEW 32x32x64 art macro + re-tile 16x16->32x32 = big.

bf16 asm `mla_a16w16_qh64_qseqlen1_gqaratio64_v3_ps.co` (the actual target dtype):
- MFMA = `v_mfma_f32_16x16x32_bf16` x1176 -> SAME 16x16x32 size as HK's art bf16 macro
  (bf16 has no 32x32x64 instruction). So NO re-tile needed for bf16.
- vmcnt steady 10 (one 20) = deep prefetch; AGPR 2946 operands (max a[206]); ds_read =
  1248x `ds_read_b64_tr_b16` (transpose-on-read into AGPR) + 554x ds_read_b128.
- => bf16 gap = prefetch-depth + AGPR ONLY (our original root cause). HK art already has
  `ds_read_b64_tr_b16<GPR>` (AGPR when >=256) + `mfma_f32_16x16x32_bf16` all-a/v combos.

REVISED Phase-0 experiment (correct target): port `m16x4` fp8 -> bf16, authored WITH deep
prefetch (vmcnt ~10) + K/V in AGPR (index >=256) from the start; bench vs the bf16 asm. Note
bf16 doubles operand footprint (~60->~120 regs) vs fp8, overflowing the 188-reg VGPR window
=> the bf16 port is FORCED to use AGPR anyway (which is exactly the lever to test). Effort:
this is real kernel authoring (days), not a minimal edit. The fp8 `m16x4` is the structural
template (managers/softmax/pipeline skeleton/LDS layout); its register map does not transfer.

#### bf16 port worklist (contract traced 2026-06, before any code)
The fp8 `m16x4` is templated on Traits<q_t,kv_t,...> and the MFMA is the dtype-polymorphic
`hk::mma_ABt` (auto-selects `mfma_f32_16x16x32_bf16` for bf16 tiles) -> MFMA is FREE. But a
bf16 instantiation is NOT clean; concrete changes required:
1. **KvManager8bitsV3 / QManager8bitsV3 -> 16-bit variant.** Byte math IS sizeof(kv_t)-param'd
   (`kNumBytesPerSubBlock=4*32*sizeof(kv_t)`: fp8 128 / bf16 256; "264"->520). BUT fp8-baked:
   - `kNumThrPerSubBlockRow = kNumSubBlockCols / kNumBytesPerThrPerRnd = 32/4 = 8` treats elem
     count as bytes (only valid sizeof==1); bf16 needs 16 (32 elem * 2B / 4B-per-load).
   - ds_read transpose variant: fp8 path vs bf16 needs `ds_read_b64_tr_b16` (matches asm).
   So the thread->(row,col) load mapping + ds_read transpose must be re-derived for 2B elems.
2. **Register re-budget (the AGPR lever).** art_base: regs/thread = packed_per_thread *
   sizeof(dtype)/4. 16x16 tile: fp8 = 1 reg, bf16 = 2 regs (16x32 kv tiles double too). The
   fp8 map fills v68..255 exactly (188 regs); bf16 operands ~double -> overflow -> MUST place
   K/V (and likely P) tiles in AGPR (range index >= 256). The static_assert
   `register_range::size == registers_per_thread` will flag every wrong-sized range at compile
   time -> use it as the inner loop (cheap correct feedback).
3. **P pack fp8->bf16**: `pack_4f32_to_fp8` (8 calls) -> `v_cvt_pk_bf16_f32` (in hk_mla_utils);
   p_mfma footprint doubles (4 fp8 vgprs -> 8 bf16).
4. **Traits + dispatch**: add HkMlaDecodeFwdTraits<bf16,bf16,bf16,...> instantiation + wire a
   bf16 branch (currently dispatch is `if(q_is_fp8 && kv_is_fp8)`).
5. **Deepen prefetch** (separate measured step, AFTER correct): current m16x4 is 2-pong LDS,
   prefetch 1 tile ahead (vmcnt<=2). Deep (asm vmcnt~10) needs more LDS KV buffers; bf16 KV
   tile is 2x LDS so the 160KB cap bites harder -> may need >2 pong or a compacter layout.

Effort: real kernel authoring, NaN-prone (cf. prior db2 NaN history). Recommended execution
order: (a) bf16 Traits+dispatch+P-pack+register-map scaffold, build, let static_asserts
enumerate the manager fixes; (b) adapt 16-bit manager mapping till correct (gate vs ref); (c)
ONLY then deepen prefetch + push AGPR, bench vs `mla_a16w16_qh64_qseqlen1_gqaratio64_v3_ps.co`.
GO/NO-GO = does deep+AGPR bf16 HK reach <=1.1x of that asm.

#### Probe result (2026-06): compiler-enumerated worklist + computed bf16 register map
Probe = explicit-instantiate the m16x4 kernel with `HkMlaDecodeFwdTraits<hk::bf16,hk::bf16,
hk::bf16,64,4,2,64,1>` and compile (flags from module_hk_mla build.ninja + full torch isystem
paths; `--offload-arch=gfx950`). Result: 20 errors, ALL in the register-map layer:
- `register_range::size == registers_per_thread` x12 on ranges <104,105>/<108,109>/<112,113>/
  <114,115>/<120,121>/<124,125> (= kv_0/kv_1/kv_alt/p_mfma) -> each fp8 size must ~2x for bf16.
- `row_vec_layout`/`col_vec_layout` missing + `rows % ...` asserts are downstream of the size
  mismatch. (No manager/MFMA errors yet -- those surface after the register map is fixed.)
Confirms the design: art_base regs/thread for bf16 = 2x fp8 (16x32 tile 2->4, 16x16 1->2).

Computed bf16 register budget (kv_t tiles double; comp_t=float tiles unchanged):
- operands (bf16): q_nope 64, q_rope 8, kv_0/kv_1/kv_0_alt/kv_1_alt 8 each, p_mfma 8 ~= 112
- float: p_comp 16, oaccu 128 -> operand+float total ~256 regs > VGPR window.
Probe file: /home/mh/aiter/_bf16_probe.cu (scratch; reuse to re-validate the register map).

#### CORRECTION (2026-06): "oaccu -> AGPR" plan is WRONG; correct lever is q+kv -> AGPR
VERIFIED start state: `_bf16_probe.cu` compiles to exactly 20 errors, all in the register-map
layer (`register_range::size == registers_per_thread` on ranges 104/108/112/114/120/124, each
fp8 size-2 must become bf16 size-4/-2). Matches the prior probe note.

Register-sizing contract (art_base.cuh:78-83): `registers_per_thread = (elements_per_thread /
packing<dtype>::num()) * sizeof(dtype)/4`. 16x32 tile: fp8=2, bf16=4. 16x16 tile: fp8=1,
bf16=2. 16x16 FLOAT (p_comp/oaccu base) = 4 (dtype-invariant) -> p_comp/oaccu UNCHANGED at
16/128 regs.

The blocker the old PLAN missed -- gfx950 VALU CANNOT touch AGPR (cdna4-isa SKILL.md p.32
accessibility table: "Acc VGPR: VALU src/dst = via v_accvgpr_{read,write} only"). But the kernel
does VALU on the accumulator EVERY tile: softmax rescale `v_pk_mul_f32 v[...]` / `v_mul_f32
v[...]` on oaccu (fp8 file lines 736-744, 886-954) + softmax max/scale on p_comp + the
P-pack `v_cvt_pk_bf16_f32 %0,v[%1],v[%2]` (buffer_managers:1598, reads p_comp / writes p_mfma).
=> oaccu, p_comp, p_mfma are all VALU-touched -> ALL must stay VGPR. Putting oaccu in AGPR
would cost 128x(accvgpr_read+mul+accvgpr_write) per tile.

CORRECTED register plan (= the asm strategy: AGPR is the 2nd bank for KV/Q operands):
- AGPR (flat index >=256): q_nope 64 + q_rope 8 + kv_0/kv_1/kv_0_alt/kv_1_alt 32 = 104 AGPRs
  (a0..a103 = index 256..359). Loaded via buffer_load/ds_read/ds_read_b64_tr_b16 with GPR>=256
  (macros.cuh dispatches >=256 -> a[...]). Used only as MFMA A/B operands.
- VGPR (v68..): oaccu 128 + p_comp 16 + p_mfma 8 = 152 managed; + ~68 compiler temps (num_vgpr
  hint) = ~220 total -> rounds to 224 <= 256. No overlay needed (kv_alt no longer aliases
  p_comp since it moved to AGPR), simplifying the map.
- MFMA combos: QK `mma_ABt(p_comp[D=v], kv[A=a], q[B=a])` and PV `mma_ABt(oaccu[D=v],
  V[A=a], p_mfma[B=v])` are both in the 16 v/a combos HK enumerates (macros.cuh:566-647). OK.
- Rescale/pack/softmax stay `v[...]` (oaccu/p_comp/p_mfma all VGPR) -> body asm unchanged on
  that axis; AGPR room (104/256) leaves headroom to deepen KV prefetch later (asm uses ~a[206]).

split_many semantics (art.cuh:55-88, VERIFIED): `split_one<L,R,N>` cuts [L,R] into consecutive
size-N chunks; `split_many_t<range<L,R>,N>` = list of size-N ranges. The split-N == the
per-base-tile `registers_per_thread`. fp8 16x32 uses N=2; bf16 16x32 needs N=4; float 16x16
tiles (p_comp/oaccu) stay N=4. art<rows,cols,shape> makes height*width = (rows/16)*(cols/32 or
16) base tiles, and needs exactly that many ranges (art.cuh:229). mma_ABt_base (bf16) ->
`macros::mfma_f32_16x16x32_bf16<A::lo,B::lo,C::lo,D::lo>` auto-picks v/a by index>=256
(mma.cuh:33, macros.cuh:566-647) -> AGPR plan emits correctly.

Transformation rule fp8->bf16 (systematic): every kv_t art range (q_nope/q_rope/kv_*/p_mfma and
all their body sub-views) DOUBLES its register span AND its split-N (2->4); float tiles
(p_comp/oaccu) unchanged. Rebase indices: q+kv -> AGPR (>=256), oaccu/p_comp/p_mfma -> VGPR.

CONCRETE bf16 register map (m16x4 qh64):
- VGPR: p_mfma v68..75 (8); p_comp v112..127 (16, float, unchanged); oaccu v128..255 (128,
  float, unchanged). (gap v76..111 unused -- fine at occupancy 1.)
- AGPR (flat index): q_nope 256..319 (a0..63, 64); q_rope 320..327 (a64..71, 8);
  kv_0 328..335; kv_1 336..343; kv_0_alt 344..351; kv_1_alt 352..359 (8 each, a72..103).
- P-pack: 2:1 bf16 (v_cvt_pk_bf16_f32), p_mfma 4->8 regs, 8->16 pack calls reading p_comp.
- Managers: q/kv load helpers must emit ds_read into AGPR (GPR>=256, supported) and switch
  ds_read_b64_tr_b8 -> _tr_b16 + the 2-byte thread/col map (DEFERRED to post-register-map probe).
File to author: csrc/kernels/mla/hk/mi35x_v32_fwd_decode_m16x4_bf16_bf16.cuh (cp from fp8).

#### PROGRESS-2 (2026-06): bf16 kernel COMPILES + RUNS (no crash); correctness FAILS (load dist)
State of the new file + edits (all in tree, compiling, wired, runnable):
- Register map (done, verified earlier).
- 16-bit KV reg reads (hk_mla_buffer_managers.cuh, branched on sizeof(kv_t)==2):
  * load_k_to_gpr: ds_read_b64 -> ds_read_b128 (4 regs); offset folded into runtime addr
    (bf16 kFixedOffset exceeds 16-bit DS imm).
  * load_transposed_v_to_gpr: 1x ds_read_b64_tr_b8 -> 2x ds_read_b64_tr_b16 (GPR, GPR+2),
    2nd issue +kNumBytesPer2SubBlocksWithPadding; lane map row=(L%16)/4+(L/32)*8,
    col=(L%4)*4+16*((L%32)/16) (mirrors HK shared_to_register col-layout). offset folded.
  * finalize_load_transposed_v_to_gpr: no-op for bf16 (tr_b16 already col-major).
- P-pack: pack_2f32_to_bf16 (v_cvt_pk_bf16_f32) x8 (hk_mla_utils.cuh); body uses 8 calls.
- Body V-load GPR offsets +2 -> +4 (bf16 V tile = 4 regs).
- Dispatch: kOccupancy 2 -> 1 (bf16 needs 1 wave/CU: 2x KV LDS = 2*74880=149760 > 80KB@occ2;
  also metadata cluster_multiplier=1 for non-fp8 so grid must be num_cu*1).
- Wiring: hk_decode_fwd.cu routes bf16 (q dtype==BFloat16) to bf16 kernel; mla.py use_hk +
  bf16 m16x4 gfx950 clause; test = op_tests/test_mla_persistent.py -n 64,1 -d bf16 -kvd bf16.

BUILD NOTE (important): the JIT `cuda_compile` ninja rule has NO depfile -> editing .cuh does
NOT trigger recompile. After any .cuh edit: `rm hk_decode_fwd.cuda.o && ninja` in
/aiter/aiter/jit/build/module_hk_mla/build, then `cp module_hk_mla.so ../../..//module_hk_mla.so`
(i.e. to /aiter/aiter/jit/module_hk_mla.so). Run tests with PYTHONPATH=/aiter
AITER_ENABLE_EXPERIMENTAL=1 inside container pa_bench_mh.

RESULT: kernel runs (no memfault) at ~23-27us but output is ~all NaN (99.5% elems) -> KV in LDS
is garbage. Root cause = the VRAM->LDS async load (KvManager8bitsV3::async_load_k_tile +
get_kv_ld_row_base_idx/get_kv_ld_col_base) and QManager8bitsV3::load_q_to_gpr are fp8-byte-baked
and NOT yet adapted for bf16. These were left as the deferred "16-bit manager mapping" step.

#### REMAINING WORK (precise) -- the hard part, needs careful derivation + test-iterate
The blocker is the LDS WRITE distribution. `buffer_load_lds` packs consecutive lanes contiguously
(LDS[M0 + lane*size]) and supports <=4 bytes/lane. The fp8 read-side LDS layout
(get_block_lane_offset = (row/4)*kNumBytesPer2SubBlocksWithPadding + ((row%4)*32+col)*sizeof) is
satisfied, for a single buffer_load_lds, ONLY by a specific lane->(row,col) mapping:
- KEY FACT (derived): for bf16, get_block_lane_offset(row=L/16, col=(L%16)*2) == L*4 (contiguous)
  -> the correct bf16 write mapping is row_local=lane/16 (4 rows), col_local=(lane%16)*2 (2
  cols/lane). One warp (64 lanes*4B=256B) fills exactly ONE bf16 sub-block (4x32x2B).
- fp8 fills TWO sub-blocks per warp (lane/32 selects); bf16 fills ONE -> bf16 needs 2 buffer_load
  per (warp, pass) covering the 2 sub-block groups (pass rows 0-15 and 16-31), each with its own
  per-lane KV-row resolution (logical row +16 -> different p_kv_indices page). This ripples into
  the body's row_kv_ld[] resolution + all async_load_k / async_load_k_tile call sites.
Concretely to do:
1. bf16 get_kv_ld_col_base = (warp/kNumWarpsPerCol)*kNumSubBlockCols + (lane%16)*2 (ELEMENT col).
2. bf16 get_kv_ld_row_base_idx = (warp%kNumWarpsPerCol)*kNumSubBlockRows + lane/16 (+ sub-group*16).
3. async_load_k_tile bf16: 2 buffer_load_lds (sub-group 0/1); voffset = row*kQkHeadDim*sizeof +
   col_base*sizeof + kColOffset*sizeof; lds dest = warp slot + group*kNumBytesPerSubBlock; pass
   instruction offset = kColOffset*sizeof (and cancel -kColOffset*sizeof in lds base). NOTE offset
   adds to BOTH vram and lds (verified via the OOB ds_write path: lds = warp_base+kColOffset+lane*4).
4. body: resolve row for both sub-groups per pass (logical +16) and pass through.
5. QManager load_q_to_gpr bf16: buffer_load_dwordx4 currently strides by fp8 bytes; q is in AGPR
   now (GPR_NOPE_START=256). Re-derive the 4-byte/elem load+shuffle for 2-byte q, write into AGPR.
6. boundary OOB ds_write_b32 -> zero 8 bytes for bf16 (currently 4).
Inner loop: edit .cuh -> rm hk_decode_fwd.cuda.o -> ninja -> cp .so -> run test (B=1 c=64) ->
expect NaN%->0 and checkAllclose pass (atol/rtol 0.01). Then sweep B/ctx, then perf vs the bf16
asm. The cmp tool for perf: op_tests/cmp_asm_vs_opus_mla_decode_rope.py per G4 in this doc.

#### UPSTREAM CHECK (2026-06): no pure-bf16 m16x4 HK source exists; but V40 has the technique
Checked all aiter remotes/branches + the pinned HipKittens repo:
- `mmd/dev/mla_bf16_16mx4` + the "bf16 16mx4" commits only ship the ASM `.co`
  (mla_a16w16_qh64_qseqlen1_gqaratio64_v3_ps.co = our REFERENCE), NOT an HK source kernel.
  => the pure-bf16/bf16 m16x4 HK port is genuinely new (continue it).
- `origin/jruan/hk_mla_v4` = active newer HK MLA "V40 gen1":
  `mi35x_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1.cuh` + `hk_mla_v40_buffer_managers_gen1.cuh`
  + doc/hk_mla_v40_gen1_spec.md. It is fp8-NoPE(448) + **bf16-RoPE(64)**, kBlockN=32, 8 warps,
  m16x8 -- a DIFFERENT (quantized-NoPE) design, NOT pure bf16, so not a drop-in. But it is the
  source of truth for bf16->LDS loads.

KEY REUSABLE FACTS from V40 (verified by reading hk_mla_v40_buffer_managers_gen1.cuh):
- buffer_load_lds supports **size=16 (b128)**: `__builtin_amdgcn_raw_ptr_buffer_load_lds(rsrc,
  lds_ptr, /*size=*/16, v_off, s_off, i_off, aux)`; lane t -> LDS byte t*16 (HW-fixed stride;
  pass lds_ptr = base + lane_idx*16, it is v_readfirstlane'd to M0 = lane0 base). My earlier
  "<=4B/lane" assumption was WRONG -- b128 is the right tool and is far simpler.
- The imm `offset:` field advances BOTH vmem AND lds (V40 pre-subtracts on the LDS dst to cancel).
- V40's KV LDS layout is **no-padding**, plain contiguous 16x32 sub-blocks col-major
  (sub_block_byte_offset = (col_tile*kNumRowTiles + row_tile)*kSubBlockBytes). No bank-conflict
  padding -> b128 full-warp-contiguous write lands cleanly. (V40 adds a vmem-side XOR swizzle for
  bank conflicts, but writer+reader agree; CORRECTNESS-first can skip the swizzle on both sides.)
- For m16x4 bf16: lane t (0..15) -> row=t/4, col=(t%4)*8 (8 cols=16B) fills one 4-row sub-block;
  64 lanes fill 4 sub-blocks. t*16 == get_block_lane_offset(t/4,(t%4)*8) ONLY in a no-pad layout
  (the fp8 520-byte padded chunk breaks the t*16 linearity across the full warp).

REVISED PLAN for the load distribution (replaces the 4-byte 2-subblock derivation above):
1. Give the bf16 KV path a NO-PADDING contiguous LDS sub-block layout (drop kNumPaddingDw for
   bf16); update get_block_lane_offset/get_block_fixed_offset (read) to the no-pad form so
   load_k_to_gpr (ds_read_b128) + load_transposed_v (2x tr_b16) still match.
2. Rewrite async_load_k (bf16) with b128: lane t -> 16B, lds dst = sub_block_base + lane*16,
   v_off = per-lane row*kQkHeadDim*sizeof + col*sizeof; one b128 per sub-block (or per 2 if
   contiguous in the no-pad layout). Re-derive warp/lane->(row,col,subblock) for 4 warps.
3. Q load: same b128 idea (q -> AGPR; load q nope+rope as bf16 b128).
4. Re-add bank-conflict swizzle later (perf), after correctness.
Reference while implementing: hk_mla_v40_buffer_managers_gen1.cuh prefetch_kv_tile (b128 rope
direct load) + store_kv_tile_step + load_k_to_gpr on origin/jruan/hk_mla_v4.

#### PROGRESS-3 (2026-06): b128 no-pad KV+Q load IMPLEMENTED; compiles+runs+no crash; all-NaN
Implemented the no-pad b128 path (KvManager8bitsV3 / QManager8bitsV3 bf16 branches in
hk_mla_buffer_managers.cuh):
- get_lds_size_in_byte bf16 = kBlockN*kQkHeadDim*2 = 73728 (no pad); sub_block_byte_offset_16(rt,ct)
  = (ct*kNumRowTiles16(4)+rt)*1024.
- load_k_to_gpr bf16: ds_read_b128 at in_sb = row*64 + col*2 (row=lane%16, col=(lane/16)*8) +
  sub_block(kRowOffset/16,kColOffset/32) folded into addr.
- load_transposed_v bf16: 2x ds_read_b64_tr_b16 over the two 16-col halves (base, base+32), no
  swizzle, no finalize.
- async_load_kv_tile_bf16: one warp -> one 16-row row-tile, all 18 col-tiles; b128 (size=16,
  lane t -> LDS t*16) via opus::make_gmem<uint8_t>(...).cached_rsrc +
  __builtin_amdgcn_raw_ptr_buffer_load_lds; phys row via get_kv_ld_row (bounds-checked buffer
  load -- NOT raw p_kv_indices[] deref, which gets speculatively hoisted and faults OOB).
- load_q_to_gpr_bf16: VRAM bf16 -> per-warp LDS bounce (b128, warp offset in v_off, s_off=0) ->
  ds_read_b128 into q AGPR (nope k_q_nope_begin+ct*4, rope k_q_rope_begin+(ct-16)*4).
- Body: prologue + mla_main pass-0 prefetch wrapped in `if constexpr(sizeof(kv_t)==2)`; fp8 split
  prefetch sites guarded `if constexpr(sizeof(kv_t)==1)`.
Bugs fixed this round: kOccupancy 2->1 (LDS/grid), KV/Q OOB memfaults, the speculative
p_kv_indices deref fault (-> use get_kv_ld_row), Q soffset (-> fold warp into v_off, s_off=0).

STATUS: compiles clean, runs WITHOUT memfault, but output is ~all-NaN (99.5%). fp8 path STILL
PASSES (delta 0.02, within fp8 tol) -> the bug is isolated to the bf16 path.
DIAGNOSIS: NaN pattern = row_sum_e -> 0 then 1/row_sum_e -> inf -> nan, i.e. QK scores are
huge/inf -> the K or Q data in registers is garbage. BUT the b128 write<->read layout math was
checked on paper repeatedly and is self-consistent (write byte t*16 == read byte row*64+col*2 ==
get-block formula; col groups + rows align; AGPR clobber_gpr handles >=256). So the garbage source
is NOT obvious from the addressing -- likely a data-format / sync / MFMA-operand subtlety.
BLOCKER for fast iteration: device printf ABORTS here ("Hostcall: no handler found") -- cannot
printf-debug. Next step must be register-level: rocgdb breakpoint after the QK loop reading v112
(p_comp scores) for one wave, OR write a loaded-K LDS value into split_output and compare to the
torch reference K in Python (definitive load check). Likely suspects to verify with that:
(a) the b128 buffer_load_lds actually lands the bytes (disasm the emitted instr / check size=16);
(b) the tr_b16 V operand layout; (c) whether ds_read_b128 into AGPR returns correct data.
Build/test loop reminder: rm hk_decode_fwd.cuda.o && ninja && cp .so; PYTHONPATH=/aiter
AITER_ENABLE_EXPERIMENTAL=1 python3 op_tests/test_mla_persistent.py -n 64,1 -d bf16 -kvd bf16
-b 1 -c 64  (expect NaN%->0 + checkAllclose pass when fixed).

#### ROOT CAUSE FOUND (2026-06) via staged unit tests: AGPR operands are NOT reserved
Methodology (per user): dump each stage's intermediate to an output buffer and compare to a
torch reference, stage by stage, instead of expecting correct end-to-end output. Device printf
ABORTS here (hostcall) -- so dump via final_output / split_output and read back in Python. Added
env-gated DBG_LOAD blocks in op_tests/test_mla_persistent.py (test_absorb_decode_bf16) + #if
DBG_* dump blocks in the kernel. Run with -ms 1 (single split -> 1 work item, 1 tile).
Results:
- Stage-1  (raw KV LDS after buffer_load_lds):           PASS (max_abs_diff=0).
- Stage-1b (load_k_to_gpr LDS->AGPR, A-layout):          PASS.
- Stage-1c (load_q VRAM->LDS->AGPR, prologue):           PASS.
- Stage-1d (kv_0_top in-loop, right before the QK mma):  PASS (K not corrupted in-context).
- Stage-1d (q_0  in-loop, right before the QK mma):      *** FAIL: regs 0,1 OK, regs 2,3 GARBAGE
  (1e26..1e34). *** Q verified correct in the prologue (1c) but corrupted by the QK loop.
- Stage-2  (p_comp QK scores):                           garbage (consequence of corrupted q).

=> ROOT CAUSE: the q/kv AGPR operands (a0..a103) are not truly reserved. The kernel keeps the
fp8 `amdgpu_num_vgpr(68)` hint, which forces the compiler into 68 arch-VGPRs; under that pressure
the compiler spills/uses AGPRs a0..a103 for its OWN temporaries, clobbering q/kv. The HK `art`
`clobber<>` is a one-shot asm clobber at the decl point, NOT a permanent reservation; for VGPR
the num_vgpr hint bounds the compiler's range, but there is NO AGPR equivalent, so nothing keeps
the compiler out of the q/kv AGPRs. (The shipped fp8 m16x4 is ALL-VGPR, so it never hit this.)

FIX OPTIONS (next):
1. Reserve AGPRs from the compiler: find an amdgpu_num_agpr-style attribute / -mllvm flag, or
   place q/kv in high AGPRs and bound the compiler's AGPR use, so its temps can't land in a0..103.
2. Reduce VGPR pressure so the compiler doesn't need to spill into AGPR (raise num_vgpr if the
   managed-VGPR layout allows, or shrink compiler temps).
3. Re-evaluate the register plan: the AGPR operand approach is unproven in this kernel framework;
   may need the V40-style register management (which is also asm-mode but tuned).
Validate the fix with the SAME staged ladder (qin/kin -> pcomp -> full checkAllclose). The DBG
scaffolding (kernel #if DBG_* + test DBG_LOAD modes agpr/q/kin/qin/pcomp) is in-tree; remove the
#define DBG_* and the test DBG_LOAD block once correct.

#### FIX-1 (2026-06): AGPR corruption FIXED via num_vgpr 68->104; all-NaN gone
The compiler was reusing q/kv AGPRs (a0..103) for its own temps under the fp8 `num_vgpr(68)`
arch-VGPR cap (deterministic, address-like garbage in a2,a3 -- not random spill;
`-amdgpu-spill-vgpr-to-agpr=0` did NOT help). FIX: raise the arch-VGPR budget to the max that
still sits below the managed VGPR tiles, and move p_mfma up so the compiler owns a contiguous
low range:
- amdgpu_num_vgpr(68) -> 104; k_p_mfma_begin 68 -> 104 (p_mfma 104..111, p_comp 112..127,
  oaccu 128..255; compiler owns v0..103).
RESULT: Stage-1d qin now PASS (q not corrupted in-loop); end-to-end output is now FINITE
(max_abs_delta ~1.8, was NaN). fp8 path unaffected (separate num_vgpr in its own kernel).
NOTE: the build.ninja was also patched with `-mllvm -amdgpu-spill-vgpr-to-agpr=0` while
diagnosing -- harmless, can stay or be reverted; the real fix is num_vgpr=104.

#### STILL OPEN (2026-06): QK scores wrong (finite) -- next staged target
With q/kv verified correct in-loop (qin/kin PASS), the QK MFMA output p_comp is FINITE but wrong:
sorted-multiset(p_comp warp0) vs ref K.Q^T differs by ~71 (so NOT merely a layout mismatch --
the values themselves are wrong), and end-to-end checkAllclose fails ~95%. Only q tile 0 (idx0)
was verified in-loop; q tiles 1..15 and the headdim-tile accumulation (16 nope + 2 rope mmas into
p_comp_lo/hi) are unverified. Next: dump a higher q tile (e.g. idx4) + p_comp after JUST the
NoPE loop (vs NoPE-only ref) to localize whether it's a wrong q-tile load, the accumulation, or
the rope contribution. Then softmax -> PV -> output. (Build/test loop + DBG modes as above.)

#### PROGRESS (2026-06): bf16 register map AUTHORED + VERIFIED; only 16-bit managers remain
Created `mi35x_v32_fwd_decode_m16x4_bf16_bf16.cuh` (cp of fp8 + edits) and drove the probe in
stages. Edits applied:
- Symbol rename `m16x4_fp8_fp8` -> `m16x4_bf16_bf16` (8 sites).
- Register-map declaration block reworked per the CORRECTED plan: oaccu v128..255 / p_comp
  v112..127 / p_mfma v68..75 (VGPR); q_nope a0..63 / q_rope a64..71 / kv_0..kv_1_alt a72..103
  (AGPR, flat idx 256..359). All kv_t art ranges: span x2, split-N 2->4. Added explicit
  clobber<kv_0_alt_ranges>/<kv_1_alt_ranges> (no longer overlay p_comp). p_mfma_lo/hi -> +0..+3
  / +4..+7 (N=4).
- Body QK loops (NoPE+RoPE): per-iter stride 4->8 regs, num_*_iter divisor /4->/8, q_range_0/1
  spans +1/+3 -> +3/+7 (N=4), tile_idx divisor /2->/4. (This also fixed the load_k col-offset
  overflow -- same root cause.)
- Dispatch -> bf16 Traits (q_is_bf16/kv_is_bf16, HkMlaDecodeFwdTraits<bf16,bf16,bf16,...>).

VERIFIED via probe (`_bf16_probe.cu` instantiates the bf16 kernel; compile flags from
module_hk_mla build.ninja, --offload-arch=gfx950, in container pa_bench_mh):
- register_range::size==registers_per_thread asserts: 20 -> 0 (register map fully correct).
- col-offset overflow asserts: gone.
- Remaining: 144 errors, ALL one kind -- `load_k_to_gpr(): ds_read_b64 requires 2 consecutive
  registers` (range now 4 regs for bf16). i.e. the compiler has isolated the work to exactly the
  16-bit KV manager.

REMAINING WORK (next session):
1. 16-bit KV manager (the 144 errors): `KvManager8bitsV3::load_k_to_gpr` and
   `load_transposed_v_to_gpr` must load 4 regs/lane for bf16, not 2. QK K-load: `ds_read_b64`
   (2 regs) -> a 4-reg load (`ds_read_b128` or two b64) with the bf16 thread->(row,col) map (the
   fp8 map packs 8 elems/lane as 8 bytes; bf16 = 8 elems * 2B = 16B = 4 regs). V-load:
   `ds_read_b64_tr_b8` -> `ds_read_b64_tr_b16` (matches the asm) + re-derived finalize swap. The
   byte-math constants (kNumBytesPerSubBlock etc.) are already sizeof(kv_t)-param'd, but
   kNumThrPerSubBlockRow = kNumSubBlockCols/kNumBytesPerThrPerRnd = 32/4 treats elems as bytes
   (fp8-only) -> needs the 2B variant. Likely author a KvManager16bitsV3 (+QManager16bitsV3 for
   the Q LDS bounce / shuffle which is also 1-byte-baked).
2. P-pack 2:1 (NOT yet done; compiles but numerically wrong): body still calls
   `pack_4f32_to_fp8` x8 writing 4 fp8 regs, but p_mfma is now 8 bf16 regs. Replace with
   `v_cvt_pk_bf16_f32` (buffer_managers:1598) 16 calls (2 f32 -> 1 bf16 reg).
3. THEN: build the real module (add to hk_decode_fwd.cu + pybind + dispatch wrapper), gate
   correctness vs ref (<=5e-4), then perf vs mla_a16w16_qh64_qseqlen1_gqaratio64_v3_ps.co.
Probe loop (cheap inner feedback): keep editing + recompiling `_bf16_probe.cu` until 0 errors.

#### ROOT CAUSE FOUND (2026-06): QK wrong = LDS read address-register corruption (reg pressure)
Staged debug nailed it. Evidence chain (all GPU-verified):
- p_comp after NoPE ~= QK over first 2 headdim tiles only (multiset N2). So only NoPE idx0
  (col-tiles 0,1) contributes; idx1..7 (col-tiles >=2) give ~0.
- q tiles (idx0 AND idx4) verified correct in-loop; KV LDS write verified correct for ALL
  col-tiles (kvlds max_diff=0); curr col-tile 8 STILL correct in LDS at idx4 time, read via a
  provenance-kept C++ ptr (`lds[16384+...]`, max_diff=0). Timing/async ruled out.
- BUT `load_k_to_gpr<0,256>` (col-tile 8) -> kv_0_top AGPR = ALL ZERO (kin4).
- rocgdb (debugtrap at the col-tile-8 `ds_read_b128 a[72:75],v42`): the ADDRESS register held
  bf16 DATA values (0x3fb2.., varying per-lane in high bits), NOT a small LDS offset. => the
  ds_read went to a garbage address -> read 0.
Why: bf16 `load_k_to_gpr` folded the sub-block offset `kSbFixed` into the RUNTIME ADDRESS
(`ds_read(p_lds_kv + in_sb + kSbFixed, 0)`), unlike fp8 which puts the per-(row,col) selector in
the ds IMMEDIATE offset. Address-in-runtime forces a DISTINCT per-col-tile per-lane address; the
compiler hoists ~72 of these and, under this kernel's extreme VGPR pressure (oaccu v128..255 +
p_comp/p_mfma v104..127 leave only v0..103; q+kv in AGPR), spills the offsets into AGPRs that
OVERLAP the q_nope range (a0..63) -> the address reg for col-tile>=2 gets clobbered with q data.
Disasm confirms offsets stored via `v_accvgpr_write a18/a19/...` (= q tiles' AGPRs).

Attempted fix (match fp8): put low16 of kSbFixed in ds immediate, only high bits in address
(`ds_read(p_lds_kv+in_sb+kAddr, kImm)`). Logically correct + should cut offset registers, BUT it
RESHUFFLES global regalloc and now an EARLY prologue `buffer_load_dwordx4 v3,s[8:11],0 offen lds`
(offset +3576, rocgdb precise-memory) faults global -> SIGSEGV. So the kernel is register-pressure
CRITICAL: any change to the LDS-offset codegen corrupts a different address reg. NOT yet shippable.

NEXT (architectural, needs decision): reduce register pressure so a clean LDS-read addressing
fits. Options: (a) shrink oaccu live range / spill oaccu differently; (b) don't force q into AGPR
(or place q AGPRs where the compiler won't reuse them for offsets); (c) reduce prefetch/pipeline
state regs; (d) compute LDS read addr with fewer hoisted temps. Current file state: the immediate-
offset fix is IN `hk_mla_buffer_managers.cuh load_k_to_gpr` (faults); revert to `+kSbFixed,0` to
get back the runnable-but-N2 build. rocgdb workflow that worked: build JIT, then
`PYTHONPATH=/aiter AITER_ENABLE_EXPERIMENTAL=1 rocgdb -batch -ex "set amdgpu precise-memory on"
-ex run -ex "thread <gpu wave>" -ex "x/4i \$pc" --args python3 op_tests/test_mla_persistent.py
-n 64,1 -d bf16 -kvd bf16 -b 1 -c 64 -ms 1` (GPU waves show as "AMDGPU Wave"; __builtin_debugtrap
stops rocgdb, s_trap 2 just aborts).

#### UPDATE (2026-06): immediate-offset fix applied; conflict is SYSTEMIC (reg pressure)
Applied the fp8-style fix in `load_k_to_gpr` (bf16): `ds_read_b128(p_lds_kv+in_sb+kAddr, kImm)`
with kImm=kSbFixed&0xFFFF, kAddr=kSbFixed&~0xFFFF. Disasm CONFIRMS correct: col-tile 8 read is
now `ds_read_b128 a[72:75], v1 offset:32768` (shared per-lane base v1 + constant immediate, exactly
like fp8) -- the per-col-tile offset precompute (that spilled into q AGPRs) is GONE.
Result: clean build now RUNS to completion (was N2-wrong; now end-to-end = NaN, i.e. behavior
changed -> read fix took effect; downstream softmax/PV unverified = next staged target).
BUT the register conflict only MOVED, not gone. rocgdb on the clean immediate-fix build faults in
the PROLOGUE Q-load (hsaco ~0x4a5c): `v_accvgpr_read v1,a64; v_add v1,v42,v1; buffer_load_dwordx4
v1,s[8:11],0 offen lds` -- i.e. the compiler stored a Q-gather VRAM offset into a64 (= q_rope AGPR)
and uses it as the buffer offset. Same class as the K bug: HK inline-asm q/kv AGPRs don't express
liveness to the compiler, so under pressure it reuses them as address scratch. (NB: this Q-load
fault only triggers UNDER rocgdb -- the non-rocgdb run completes -> likely a debug single-step
artifact on predicated buffer_load_lds, NOT a real run fault; but it shows the same reuse pattern.)
INSTRUMENTATION BLOCKED: with the immediate fix, ANY added dump (even small kin) now faults
(regalloc tipped). So can't dump-verify QK on this build; rely on disasm + end-to-end.

SYSTEMIC ROOT: kernel is VGPR-critical (oaccu v128-255 = 128 regs, p_comp/p_mfma v104-127, compiler
only v0-103; q+kv forced to AGPR). The compiler reuses managed q/kv AGPRs (a0-71) for LDS/VRAM
address scratch because their liveness isn't expressed across the inline-asm boundary -> corruption
under pressure, and any codegen change just moves which address gets corrupted.
REAL FIX NEEDED (architectural, pick one): (a) reduce reg pressure -- shrink oaccu live range or
simplify the deep pipeline to single-buffer (route 4: "correct first, pipeline later"), freeing
v0-103 so the compiler never touches managed AGPRs; (b) make HK express q/kv AGPR liveness (clobber
lists / keep-alive) so the compiler won't reuse them; (c) move q out of AGPR. Recommend (a)/route 4:
get a register-roomy, CORRECT bf16 kernel first, then re-add the deep pipeline measured.
Current code state: immediate-offset fix is IN load_k_to_gpr (disasm-correct, runs, NaN e2e).

#### DEFINITIVE CONCLUSION (2026-06): why asm fits and we don't (asm disasm evidence)
Read the asm oracle `mla_disasm/mla_a16w16_qh64_qseqlen1_gqaratio64_v3.s`:
- asm metadata: `.amdhsa_next_free_vgpr 512`, `.amdhsa_accum_offset 256` => 256 arch VGPR + 256 AGPR.
- asm QK: `v_mfma v[34:37], a[72:75], a[0:3]` -> Q in AGPR (a0-71), K in AGPR (a72-119+),
  accumulator p_comp in VGPR (v34-49). PV output accumulator (oaccu) in VGPR (v50-177).
- asm max VGPR = 177 (so v178-255 FREE for scratch), max AGPR = 206.
Our kernel metadata (readelf notes on the hsaco): `.vgpr_count 360` (=256 arch + 104 agpr),
`.agpr_count 104`, `.vgpr_spill_count 0`. So we ALLOCATE only 104 AGPR (q/kv) and leave a104-255
(152 AGPRs) UNUSED; arch VGPR is maxed at 256 (oaccu v128-255 + p_comp v112-127 + p_mfma v104-111),
leaving the compiler only v0-103.
HARD CONSTRAINT (kernel comment line 44-46, confirmed): oaccu/p_comp/p_mfma are VALU-touched
(online-softmax rescale / max / cvt_pk_bf16) and **gfx950 VALU cannot read/write AGPR**, so they
MUST stay VGPR. => we canNOT move the 152-reg accumulator block to AGPR. The asm is the same
(accumulators in VGPR); it fits only because its hand-allocated inner loop needs ~34 VGPR scratch.
THE GAP: our COMPILER needs >104 VGPR scratch (HK abstraction + deep pipeline + addressing), far
more than asm's 34. With only v0-103 free, it reuses dead q/kv AGPRs (a0-71) as scratch -> the
LDS-read address regs get q DATA -> col-tile>=2 corruption (the original bug). The correct
ds-immediate addressing fix (K and V) is disasm-correct but ADDS net pressure and tips the kernel
into a NON-DETERMINISTIC global-load fault (fault VA varies per run). Even K-fix-alone is unstable.
TRIED + RULED OUT: `amdgpu_num_agpr` attr (doesn't exist), `-amdgpu-spill-vgpr-to-agpr=1` (no help;
the reuse is normal allocation of dead regs, not a memory spill), `amdgpu_num_vgpr(104)` (ineffective;
arch vgpr still 256). No flag/attr makes the compiler use the free a104-255 for scratch instead of
reusing dead q/kv.
=> ROOT ANSWER to "why does asm fit and we don't": not the register BUDGET (both use accumulators in
VGPR) but COMPILER SCRATCH EFFICIENCY -- our compiled inner loop needs ~3x the scratch of hand-asm,
which doesn't fit in the 104 free VGPRs, forcing q/kv AGPR corruption. Two real fixes:
 (A) cut the compiler's scratch NEED: simplify to single/double-buffer (drop deep prefetch) + the
     ds-immediate addressing -> "correct first, re-add pipeline measured" (recommended next step);
 (B) hand-write the QK/PV inner loop in inline asm to get asm-level register efficiency (true
     asm-copy; larger effort).
Baseline restored to original (address-fold K+V): runs stably, QK = NoPE idx0 only (e2e delta ~1.0-1.7,
~96% mismatch, no NaN, no fault). build.ninja spill flag back to =0.

#### STAGE 1 PASSED (2026-06): QK gemm fully correct (staged dev: QK -> softmax -> PV)
Re-targeted to staged development (don't build the whole MLA at once). Stage 1 = QK only, verified.
Recipe that WORKED (DBG_QK_ONLY in mi35x_v32_fwd_decode_m16x4_bf16_bf16.cuh):
1. Correct ds-immediate addressing in load_k_to_gpr (kImm=kSbFixed&0xFFFF, kAddr=kSbFixed&~0xFFFF;
   `ds_read_b128(p_lds_kv+in_sb+kAddr, kImm)`).
2. `#if !defined(DBG_QK_ONLY)` skip `clobber<p_mfma_ranges>()` + `clobber<o_ranges>()` (don't reserve
   v104-255).
3. After the QK gemm + p_comp dump (DBG_DUMP_PCOMP_BOTH), call `__builtin_amdgcn_endpgm()`
   UNCONDITIONALLY (not under `if constexpr kIsFirstIter`). The intrinsic is noreturn, so the compiler
   DCEs ALL post-QK code (softmax/PV/V/oaccu) in EVERY instantiation -> oaccu's 128 VGPRs are freed ->
   compiler has ample scratch -> the correct addressing no longer corrupts q/kv AGPRs.
   (KEY LESSON: a runtime `if constexpr(kIsFirstIter) s_endpgm` did NOT work -- the kIsFirstIter=false
   instantiation still compiled the PV code and kept oaccu allocated. Must DCE in all instantiations.)
RESULT (DBG_LOAD=pcomp, -n 64,1 -d bf16 -kvd bf16 -b1 -c64 -ms1): full QK p_comp == K.Q^T EXACTLY:
SORTED full=0.00, max_abs_diff=0.0000, mismatch=0.00%, dumped==exp elementwise. No fault.
=> CONFIRMS: root cause was register pressure (oaccu hogging VGPR forcing q/kv AGPR corruption); the
   ds-immediate addressing fix is correct; QK math + layout are correct.
#### STAGE 2 PASSED (2026-06): softmax fully correct
Same recipe (DBG_SOFTMAX_ONLY): free oaccu/p_mfma clobber, run QK + softmax (softmax_scale_p_16 ->
max_16 -> warp_reduce -> softmax_p1_16), dump p_comp (= P) after softmax_p1_16, `__builtin_amdgcn_endpgm()`.
Softmax math: P = exp(score*sm_scale - rowmax[qhead]); v_exp_f32 is exp2, with *log2e => natural exp;
sm_scale = 1/sqrt(qk_head_dim=576); single-tile rescale=1.0. Python `DBG_LOAD=smax` ref:
qk=Kg@Qg.T; scaled=qk*sms; rowmax=scaled.max(dim=0); P=exp(scaled-rowmax). RESULT: SORTED diff=0.00000,
max_abs_diff=0.00000, mismatch=0.00%. So scale + row-max (warp_reduce) + exp are all correct.

NEXT (Stage 3): PV gemm -- re-introduces oaccu (128 VGPR) + V load + bf16 P-pack, so the endpgm-DCE
trick can NO LONGER free oaccu (PV needs it) -> register-pressure fight returns. Plan: oaccu output
chunking (512 -> 2x256, oaccu 128 -> 64 VGPR, frees 64 for scratch) + K AND V ds-immediate addressing
fixes -> should fit. V load (load_transposed_v_to_gpr bf16) needs the SAME ds-immediate fix as K
(currently reverted to address-fold). Sub-step 3a (optional): verify bf16 P-pack (pack_2f32_to_bf16,
p_mfma) in isolation (dump p_mfma + endpgm before PV, oaccu still DCE'd).
#### STAGE 3a PASSED (2026-06): bf16 P-pack correct
DBG_PPACK_ONLY: keep clobber<p_mfma_ranges> (pack writes fixed v104-111), free oaccu; dump p_mfma
(8 vgprs = 16 bf16/lane) after pack_2f32_to_bf16, then `__builtin_amdgcn_endpgm()` after the
kSkipCompute block (unconditional -> DCE PV/oaccu/V in all instantiations). Python DBG_LOAD=ppack:
ref Pb = bf16(exp(score*sms - rowmax)); dumped = attn_logits[:512].view(bf16)[:1024]; p_comp layout.
RESULT: max_abs_diff=0.000000, mismatch=0.00%. The fp32->bf16 P conversion (v_cvt_pk_bf16_f32) is correct.

So the ENTIRE pre-PV path is verified: QK (Stage1) + softmax (Stage2) + bf16 P-pack (Stage3a), all
elementwise-exact. Remaining = Stage 3b: V load (needs ds-immediate fix like K) + PV mma + normalize
(/row_sum) + output. PV needs oaccu (128 VGPR) live -> register fight returns. Plan (correctness-first,
approved): for the single-tile correctness test, oaccu per output-slice can be small (each pv_iter's
64-col slice = 16 VGPR is independent for a single KV tile; only multi-tile online-accumulation needs
all 128 live). So chunk the output / shrink oaccu for the single-tile milestone, apply V ds-immediate
fix, verify end-to-end output. (Multi-tile deep pipeline = perf, later.)
Current file state: DBG_PPACK_ONLY=1; K addressing=ds-immediate (correct); V addressing=address-fold
(still needs the same fix); NOT a real full kernel yet.

#### STAGE 3b IN-PROGRESS (2026-06): PV mma OK, output_to_vram faults (NOT pressure)
DBG_PV_SINGLETILE: full kernel, oaccu reused as ONE 16-VGPR slice (oaccu_base=k_o_begin constant,
not +tile_idx*16; o_ranges NOT clobbered). CONFIRMED via objdump -t .num_vgpr: gqaratio=1 kernel now
num_vgpr=0x80=128 (was 256!), num_agpr=104 -> arch VGPR halved, ~128 VGPR HEADROOM. So oaccu
reduction worked and the kernel is NO LONGER register-starved.
ISOLATION (DBG_PV_NOOUT = endpgm in PV iter0 before the kDoEpilogue output block):
 - PV mma + V load (incl. col-tiles 2,3 via next-tile prefetch) RUN WITH NO FAULT (output garbage,
   100% mismatch, but no memfault).
 - Enabling output_to_vram -> memfault (non-deterministic VA). So the fault is the OUTPUT stage, and
   it is NOT register starvation (128 VGPR headroom). V immediate-fix is NOT the trigger either
   (faults with V=address-fold too).
output_to_vram (OManager16bitsV2, hk_mla_buffer_managers.cuh:1882): address = out_br(base=final_output
+ qo_start*num_qheads*kVoHeadDim) + lane offset + kColOffset(=col_off); data = float_2_bf16_pair(GPR_START..);
kCheckOOB=true (so OOB stores are dropped, shouldn't fault). GPR_START=oaccu_base only feeds DATA, not
the address -> oaccu-slice shouldn't break the address. So the fault is likely a register/SGPR ripple
from the output's regs (out_br SGPRs, b16_pair) under the unrolled 8-iter output, OR the prologue
Q-load buffer resource getting clobbered when output codegen is present (Stage 1/2/3a all had output
DCE'd and ran clean). NEXT ISOLATION: (a) limit to 1 output iter (endpgm after iter0 output) to see
if multi-iter output is the trigger; (b) check if it's the prologue Q-load faulting for real
(rocgdb shows prologue Q-load but that's also the debug-mode artifact). Build state has DBG_PV_NOOUT=1
(runs, no output) + DBG_PV_SINGLETILE=1; V=address-fold (isolation).

REFINED ISOLATION (2026-06): endpgm AFTER iter0's output (1 output iter) -> RUNS, NO FAULT (cols 0-63
written, rest garbage, 99.4% mismatch). endpgm absent (all 8 output iters) -> FAULT. So the trigger is
the MULTI-ITER (8x unrolled) output, NOT a single output write, NOT the PV mma, NOT V-addressing
(faults with both V=immediate and V=address-fold). With oaccu reused (one v128-143 slice) all 8 iters
serialize on oaccu (WAW) AND each adds output temps (b16_pair/offset/out_br); the 8x-unrolled
accumulation faults a global access. HYPOTHESIS: the 8x-unrolled output + oaccu-WAW serialization
spikes register/SGPR usage (out_br is 4 SGPR x8) or the compiler reuses a live reg. NEXT: (a) don't
fully unroll the PV/output (runtime loop over output slices instead of static_for) so output temps
don't 8x-accumulate; (b) or give each iter its own oaccu slice again (the 128-VGPR version) now that
we know upstream is correct -- but that's the original pressure. Likely (a): a runtime output loop
with the oaccu slice. KEY POSITIVE: QK+softmax+P-pack verified exact; PV mma + single output verified
runnable; only the multi-slice output assembly remains.

#### STAGE 3b UPDATE-2 (2026-06): full PV+output RUNS (no fault); output VALUES wrong
endpgm placed AFTER the full 8-iter PV loop (DBG_PV_NOOUT) -> NO FAULT. So the earlier "multi-iter
output faults" conclusion was WRONG: the real fault is the POST-loop code (lse/swap in the OTHER
mla_main instantiations) rippling regalloc, NOT the PV/output. The whole QK->softmax->PV->output
pipeline executes cleanly when the post-loop is endpgm'd away.
BUT e2e output is WRONG (~95% mismatch, |delta|~1.0, not NaN/zero):
 - oaccu single-slice reuse is NOT the cause (full per-tile 128-VGPR oaccu also ~95% wrong).
 - V immediate-fix vs V address-fold give the SAME ~95% -> bug is NOT V addressing.
 => Suspect the bf16 V-load LAYOUT (load_transposed_v_to_gpr: ds_read_b64_tr_b16 + finalize -> mma
    B-operand layout) OR the bf16 output_to_vram (OManager16bitsV2) layout. delta~1.0 structured =>
    layout/transpose mismatch, not value corruption.
REMAINING for a correct single-tile kernel: (1) PV output VALUES -> verify V load (dump kv_0 after
load_transposed_v_to_gpr+finalize vs ref V; sorted-multiset first to separate layout-vs-value) and/or
output_to_vram layout; (2) post-loop register-ripple fault -> reduce that code's pressure / keep PV
endpgm while debugging values.
Build state: DBG_PV_NOOUT=1 (endpgm after PV loop; runs; output wrong); full per-tile oaccu; K+V both
ds-immediate. NOT correct yet -- PV value bug + post-loop fault remain.

#### STAGE 3b-V (2026-06): V VALUES correct, V transpose LAYOUT is the suspect
DBG_DUMP_V: dumped finalized kv_0 (= V[rows0-31,cols0-31] transposed, a72-79) after
finalize_load_transposed_v_to_gpr, endpgm. Python DBG_LOAD=vload: SORTED multiset vs Kg[0:32,0:32]
(V value part) = diff 0.00000, 0% zero. So the V LOAD brings the correct VALUES into the GPRs.
=> Since V-values OK, P OK (Stage 2/3a), but PV output ~95% wrong, the bug is a LAYOUT mismatch:
   the bf16 V load uses ds_read_b64_tr_b16 (16-bit transpose) where fp8 used ds_read_b64_tr_b8
   (8-bit transpose). tr_b16 vs tr_b8 produce DIFFERENT element-to-lane/reg arrangements, so the
   finalize + PV mma B-operand layout (copied from fp8) likely doesn't match the bf16 transpose ->
   mma multiplies mis-arranged V -> wrong O. (sorted matches because all values are present, just
   permuted.) The transpose/finalize for bf16 needs to be made consistent with tr_b16's layout.
NEXT: verify the EXACT kv_0 element layout (not just sorted) vs the mma B-operand expectation; fix
finalize_load_transposed_v_to_gpr / load_transposed_v_to_gpr bf16 path to match tr_b16. Also check
output_to_vram layout. (Reference: fp8 path in mi35x_v32_fwd_decode_m16x4_fp8_fp8.cuh + HK tr docs.)
Plus the separate post-loop register-ripple fault still needs handling for the real (non-endpgm) kernel.

#### STAGE 3b-V ROOT CAUSE (2026-06): bf16 tr_b16 V layout != mma A-operand expectation
Brute-forced the actual kv_0 element->(kvpos,outcol) map (DBG_LOAD=vload, dumped kv_0 a72-79 after
finalize, matched each value into Kg[0:32,0:32]; ignored random float collisions). Clean pattern for
kv_0_top (a72-75, e=0..7 per lane):
  ACTUAL:   kv_0_top[lane][e] = Kg[kvpos=(lane//16)*4 + (e%4)][outcol=((e//4)%2)*16 + (lane%16)]
            (full 16-elem incl bot: kvpos=(e//8)*16+(lane//16)*4+(e%4), outcol=((e//4)%2)*16+lane%16)
  EXPECTED (mma A-operand, == verified QK kin layout for K):
            kv_0_top[lane][e] = Kg[kvpos=(lane//16)*8 + e][outcol=lane%16]   (V transposed: row=outcol,col=kvpos)
They DIFFER -> tr_b16 (16-bit transpose) yields a different lane/reg arrangement than tr_b8 (fp8,
which the downstream layout was copied from). H1(transposed)/H2(direct) both diff ~5 (neither simple).
=> ROOT CAUSE of wrong PV: the bf16 load_transposed_v_to_gpr lane->LDS mapping
   (in_sb=(lane>>2)*64+(lane&3)*8) + ds_read_b64_tr_b16 does NOT produce the mma A-operand layout the
   PV mma needs; the "no finalize swap (tr_b16->mfma A)" assumption is WRONG.
FIX (next): derive the correct bf16 V lane->LDS mapping (and/or a bf16 finalize) so tr_b16 yields
   kv_0_top[lane][e]=V[kvpos=(lane//16)*8+e][outcol=lane%16]. Needs ds_read_b64_tr_b16 transpose
   semantics (cdna4 isa / HK tr docs). Everything else (QK, softmax, P-pack, V VALUES, PV mma, single
   output, oaccu-reduction) is verified/working. This V-layout fix + the post-loop fault are all that
remain for a correct single-tile bf16 kernel.

#### STAGE 3b-V ATTEMPTS (2026-06): tr_b16 layout still not cracked
ISA (mfma.md:312): DS_READ_B64_TR_B16 = 2 issues are K-RANGES of one tile (first K0-3,8-11; second
K4-7,12-15), NOT col-halves. Our bf16 V load issued them as col-halves (+32B). HK reference
(shared_to_register.cuh): st_16x32_s->rt_16x32_s uses 2nd issue offset = offset0 + 4*row_bytes (+256B)
and per-lane addr row_off=(lane%16)/4+(lane/32)*8, col_off=(lane%4)*4+16*((lane%32)/16).
Tried: (1) offset +32->+256: vload SORTED 0->0.64 (some lanes read kvpos>=32, OOB of the 32-row sub-block).
(2) +256 AND HK in_sb (row_off/col_off above): vload SORTED=0 again, but the per-lane map (brute-forced)
shows kv_0 covers only kvpos {0-7,16-23} per lane (MISSING 8-15,24-31) and outcol=lane -> still doesn't
match a clean mma A-layout; e2e still ~95% wrong.
=> The bf16 V LDS layout (written by async_load_kv_tile_bf16, no-pad 16x32 sub-blocks) is likely NOT
   consistent with what HK's tr_b16 path assumes (HK has its own st layout). Matching HK's in_sb alone
   isn't enough; the LDS WRITE layout and the tr_b16 READ must be co-designed.
NEXT (better approaches than blind iteration): (a) GROUND TRUTH: dump the WORKING fp8 kernel's kv_0
   (load_transposed_v_to_gpr tr_b8) element map and make bf16 produce the identical logical arrangement;
   (b) verify bug CLASS first: sorted(e2e output) vs sorted(golden) -- if equal, it's a pure permutation
   (output_to_vram layout or a consistent transpose), meaning V/mma are actually fine and I've been
   chasing the wrong stage; (c) derive ds_read_b64_tr_b16 exact transpose from cdna4 ISA p.98-99 and
   co-design the async_load LDS write + the tr_b16 read.
Build state: V load = HK in_sb + (+256) (sorted=0, NOT mma-correct); DBG_PV_NOOUT=1 (e2e, ~95% wrong).
Recommend approach (b) next (cheap, determines if V is even the bug) then (a).

#### STAGE 3b BREAKTHROUGH (2026-06): it's a LAYOUT/PERMUTATION bug, not a value bug
Approach (b) done: added `DBG e2e: sorted(out) vs sorted(ref)` to the test. RESULT: sorted diff = 0.145
(SMALL vs output range ~[-3,3]; ~bf16 rounding level). e2e elementwise still ~95% mismatch.
=> The output VALUES are essentially CORRECT; they're in the WRONG POSITIONS. This is a
   LAYOUT/PERMUTATION bug, NOT a value bug. So V VALUES are fine (already knew), AND the PV mma is
   computing the right numbers -- the bug is the ORIENTATION/PLACEMENT: oaccu is O^T or a permuted O
   vs what output_to_vram writes (out[qhead][outcol]). i.e. the bf16 V-transpose gives a different
   oaccu orientation than fp8's (fp8 works with the same output_to_vram).
This REFRAMES the whole effort: stop trying to match V VALUES; instead fix the oaccu ORIENTATION so
output_to_vram places them right. NEXT (decisive localization): dump oaccu (fp32) after the PV mma
(before normalize/output) for one 16x16 sub-tile and compare to the python reference O=P@V sub-tile:
 - if oaccu == O sub-tile -> output_to_vram placement is the bug.
 - if oaccu == O^T / permuted -> the V-transpose orientation (kColOffset/kRowOffset roles, or
   swapping which dim is M vs K in the V load) is the bug; flip it so oaccu comes out as O.
Either way it's a contained orientation fix now that values are confirmed correct. (The bf16 V load
currently = HK in_sb + +256.)

#### STAGE 3b FINAL SYNTHESIS (2026-06): permutation bug; needs fp8 ground-truth for V orientation
oaccu dump comparison was INVALID: oaccu_0_a is a PARTIAL kvpos sum (kv_0_top=kvpos0-15 +
kv_0_alt_top=kvpos32-47; kvpos16-31,48-63 go to oaccu_0_b) -- the fp8 interleaved-accumulation
structure -- so oaccu_0_a != full O. Can't localize via a single oaccu sub-tile.
SOLID, CONFIRMED facts:
 - QK, softmax, bf16 P-pack: verified EXACT (Stages 1,2,3a).
 - V VALUES correct (vload sorted=0). PV pipeline RUNS (endpgm after PV loop; post-loop lse/swap in
   other instantiations is a separate register-ripple fault).
 - e2e output: VALUES correct (sorted(out) vs sorted(ref)=0.14 ~ bf16 rounding), POSITIONS wrong
   (~95% elementwise mismatch) => PERMUTATION / ORIENTATION bug in the PV->output mapping.
ROOT: the bf16 V transpose (ds_read_b64_tr_b16) yields a DIFFERENT oaccu orientation than fp8's
(ds_read_b64_tr_b8). fp8 works with the same output_to_vram + the same interleaved oaccu structure;
bf16's V orientation permutes the output. Note fp8 ALSO does a finalize v_swap_b32 (line 1643) that
bf16 skips ("no finalize swap" assumption) -- bf16 likely needs its OWN finalize/lane-mapping to match
the fp8 orientation. Copying HK's GENERIC st_16x32_s tr_b16 load (in_sb/offset) is NOT guaranteed to
match the fp8 MLA V orientation.
RECOMMENDED NEXT (the right way, instead of blind iteration): GROUND-TRUTH against the WORKING fp8 V
load -- build the fp8 m16x4 kernel with the same kv_0 dump (DBG_DUMP_V) + an fp8 test, capture the
fp8 kv_0 element->(kvpos,outcol) map, then derive the bf16 tr_b16 in_sb/offset/finalize that produces
the IDENTICAL logical arrangement. (fp8 tr_b8 lane map: row=(l/16)*4+((l%16)/2)%4,
col=((l%2)+((l%16)/8)*2)*8, with a finalize v_swap.) This is a focused orientation derivation.
Build state: V=HK in_sb+256; DBG_PV_NOOUT=1 (e2e runs, values OK, positions permuted). All DBG dump
modes (qk/smax/ppack/vload/oaccu/kvlds) are in the test (DBG_LOAD=...) for re-verification.

#### opus tr_b16 reference (2026-06, mla_disasm/opus_qh64.s)
opus uses 768x `ds_read_b64_tr_b16` for its V load, in PAIRS: v[220:221],v208 (off 0) +
v[222:223],v208 offset:512 ; v[224:225] off:32 + v[226:227] off:544 ; ... So pair-stride = +512,
tile-step = +32. The +512 is opus's K-range pair offset FOR OPUS'S LDS LAYOUT (padded, row stride
128B -> 4 rows = 512). Our no-pad 16x32 (row 64B) -> 4 rows = 256 (HK's value). So pair-stride is
layout-dependent: ours = +256 (correct for no-pad). opus per-lane addr v208 = v_add3(v1,v2,v3) (3
lane-derived terms) for opus's layout -- not directly transferable.
CONCLUSION on references: ours(no-pad), HK(st_16x32_s), opus(padded), fp8(padded tr_b8) all use
DIFFERENT LDS layouts for the V tr read, so no addr/offset is directly copyable. The tr_b16 APPROACH
(paired reads, transpose) is confirmed correct; the per-lane addr + pair-stride + finalize must be
co-designed with OUR no-pad write layout (async_load_kv_tile_bf16). This is the focused derivation
that remains. Pragmatic options ranked: (1) fp8-ground-truth: dump fp8 kv_0 LOGICAL map (lane->
kvpos,outcol) -- the mma needs the SAME logical A orientation for bf16, independent of dtype/LDS --
then make bf16 produce that logical map; (2) derive ds_read_b64_tr_b16 transpose from cdna4 ISA
p.98-99 for the no-pad layout; (3) change the async_load WRITE layout so the standard HK tr_b16 read
gives the right orientation. STATUS: diagnosed to a precise contained orientation bug; values all
correct; not yet fixed. (Hit diminishing returns on blind in_sb/offset iteration -- needs the
ground-truth/derivation approach, a focused fresh task.)

#### STAGE 3b-V SOLVED-STRUCTURE (2026-06): LDS layout + target + canonical addr all CONFIRMED
Did the static derivation (contract-first) + a clean sentinel measurement. RESULTS (all evidence-backed):
1. LDS WRITE layout CONFIRMED by reading async_load_kv_tile_bf16 (hk_mla_buffer_managers.cuh:1259-
   1278): within a col-tile, byte(kvpos,outcol) = kvpos*64 + outcol*2. Sub-blocks are col-major
   (sub_block(rt,ct)=(ct*4+rt)*1024) so kvpos 0-63 of ct=0 ARE byte-contiguous (rt*1024=(rt*16)*64).
2. kv_0_top REGISTER/ART CONFIRMED (mi35x..bf16.cuh:200-204,229): kv_0_top = regs k_kv_0_begin+0..+3
   (4 VGPR), art rt_16x32_s, kTileM=16, kBlockK=32. mma = __builtin_amdgcn_mfma_f32_16x16x32_bf16
   (mma.cuh:35), so A=16x32 => M=outcol(16), K=kvpos(32). A[m=outcol][k=kvpos]=V[kvpos][outcol].
3. TARGET CONFIRMED = canonical mfma_16x16x32_bf16 A-layout = H1:
   kv_0_top[lane][e] = V[kvpos=(lane//16)*8+e][outcol=lane%16], e=0..7. (lane group L//16 in {0,1,2,3}
   provides K=0-7,8-15,16-23,24-31; lane%16 = the M=outcol.) So kv_0_top spans kvpos 0-31 = TWO
   sub-blocks (0-15 in sub_block(0,0), 16-31 in sub_block(1,0)), outcol 0-15 (col_off range).
4. SENTINEL TOOL added (test_mla_persistent.py): DBG_SENT=col sets V[k][c]=c, DBG_SENT=row sets
   V[k][c]=k; the vload dump then decodes EXACTLY (integers, bf16-exact, no false matches) -- prints
   per-lane e0..7 = the outcol (col) / kvpos (row) each element holds. THE definitive map tool.
CAVEAT (seed variance): the test sets NO manual seed -> KV is random each run, so the e2e
sorted(out)-vs-sorted(ref) ABSOLUTE value varies run-to-run (observed 0.14/0.39/0.67 across runs are
PARTLY seed noise, NOT a clean addr A/B signal). The RELIABLE signal is the DETERMINISTIC sentinel
map + the checkAllclose pass/fail. (Recommend: add torch.manual_seed for comparable e2e numbers.)
5. CANONICAL addr (row_off=(lane%16)/4+(lane/16)*8, col_off=(lane%4)*4) is STRUCTURALLY CORRECT
   (this is the deterministic, seed-independent finding): sentinel showed outcol=lane%16 (EXACT H1)
   and kvpos=base+e contiguous (EXACT H1 structure). The ONLY defect: sentinel kvpos base was
   offset/OOB (groups 0,1 -> kvpos 48-63, groups 2,3 -> 0). ROOT CAUSE: at the load<0,0> read point the current V tile has only 16
   valid kvpos loaded (one sub-block of the iter-0 tile, which for c=64 = abs kvpos 48-63), so the
   canonical reach into the 2nd sub-block (kvpos+16) reads OOB / the wrong tile. i.e. kv_0_top wants
   K=32 (2 sub-blocks) but the per-iter V tile granularity / load-ordering only guarantees 16 kvpos.
   The ORIGINAL addr (row_off uses /32 -> stays in 1 sub-block, 16 kvpos) is in-bounds => pure
   PERMUTATION (sorted 0.14, all values present) but only K=16 distinct kvpos => also fundamentally
   wrong K. Reverted to original (better e2e) pending the tiling fix.
THE REMAINING FIX (precise, contained): make kv_0_top's TWO sub-blocks (kvpos 0-15 AND 16-31 of the
CURRENT tile) both valid/loaded before load_transposed_v_to_gpr<0,0> reads them, then use the
CANONICAL addr (row_off=(lane%16)/4+(lane/16)*8, col_off=(lane%4)*4, 2nd issue +256). Need to read
the outer mla_main tile loop + async tile_start/double-buffer to confirm the V-tile kvpos granularity
(is it 16 or 32 per iter?) and the processing order (sentinel proved iter-0 tile != abs kvpos 0-31).
#### STAGE 3b-V ROOT CAUSE *CONFIRMED* (2026-06): standalone tr_b16 probe, no tile contamination
Wrote /home/mh/aiter/_tr16_probe.cu -- a STANDALONE gfx950 HIP probe (NO MLA pipeline): fill LDS u16
slots with their flat index (= kvpos*32+outcol, kvpos 0-63 contiguous = same as the kernel col-major
sub-block layout for ct=0), run the EXACT 2-issue ds_read_b64_tr_b16 (offset 0 / +256B) with a chosen
per-lane addr, dump each lane's 8 outputs -> each value decodes to its source (kvpos,outcol). Build:
  docker exec pa_bench_mh hipcc --offload-arch=gfx950 -O2 -DVARIANT={0,1} _tr16_probe.cu -o /tmp/t && /tmp/t
RESULT (clean, deterministic, no contamination):
 - VARIANT 0 (ORIGINAL addr: row_off=(l%16)/4+(l/32)*8, col_off=(l%4)*4+16*((l%32)/16)):
   out[lane][e] = V[kvpos=e][outcol=lane]  -> lanes 0-15 happen to match H1, but lane-GROUP (l/16)
   goes to OUTCOL (16,17,..) and kvpos is STUCK at 0-7. Never reads kvpos 8-31. WRONG.
 - VARIANT 1 (CANONICAL addr: row_off=(l%16)/4+(l/16)*8, col_off=(l%4)*4):
   out[lane][e] = V[kvpos=(l/16)*8+e][outcol=l%16] for ALL 64 lanes = EXACTLY H1. CONFIRMED.
   (L0->kv0-7/oc0, L16->kv8-15/oc0, L32->kv16-23/oc0, L48->kv24-31/oc0, L17->kv8-15/oc1, ...)
=> ROOT CAUSE (confirmed, not hypothesis): the bf16 V-read per-lane addr maps the lane-group (lane/16)
   to the OUTPUT COLUMN instead of to KVPOS (the K/contraction dim). Fix = CANONICAL addr (group->kvpos
   via row_off+=(l/16)*8; drop the group->outcol term; col_off=(l%4)*4). This is proven in isolation.
SECONDARY (why canonical alone didn't fix the kernel e2e): the kernel's kv_0_top/kv_0_bot load pair
uses kRowOffset=0/16 assuming each load = 16 kvpos. But kv_0_top(->oaccu_0_a) and kv_0_bot(->oaccu_0_b)
are DIFFERENT OUTCOL (M: 0-15 vs 16-31), SAME kvpos 0-31; and canonical's ONE load already reads
kvpos 0-31 (K=32). So the full kernel fix = (1) kv_0_top canonical addr; (2) kv_0_bot must shift by
+16 OUTCOL (col), NOT kRowOffset=16 (kvpos); (3) confirm the outer mla_main tile loop actually loads
kvpos 0-31 of the current tile into sub_block(0,0)+(1,0) at the read point (earlier sentinel saw
sub_block(0,0) holding a different tile's kvpos -- verify tile-start/double-buffer ordering).
NOTE: VARIANT 0/1 are the SAME row/col formulas tested in the kernel; the probe matches the kernel's
LDS byte layout (kvpos*64+outcol*2, sub_block(rt,0) at rt*1024 = kvpos rt*16) so it is representative.

#### STAGE 3b-V *FIXED* (2026-06): root cause = V/P kvpos-per-slot mismatch; V must match p_mfma
ROOT CAUSE (confirmed): the PV mfma contracts A[lane][i] with B[lane][i] at the SAME hardware slot, so
V (A) and p_mfma (B) must put the SAME physical kvpos at each slot. p_mfma keeps the QK D-layout order
(the P-pack is order-preserving f32->bf16): g_B(lane,i) = (i/4)*16 + (lane/16)*4 + (i%4). The earlier
"canonical/standard mfma A layout" target (H1: kvpos=(lane/16)*8+i) was WRONG -- it doesn't match
p_mfma. fp8 works because fp8's tr_b8 V naturally produces g_B (same order as the shared p_mfma).
WHY sentinel-col (V=outcol) PASSED despite the bug: with V=outcol, A[lane][i]=outcol is kvpos-
INDEPENDENT, so the kvpos mismatch cancels -> output correct. It only tests V->output, not P alignment.
FIX (verified by probe MODE 7 then e2e): solve tr_b16 contract out[lane][e]=M[4e+l/4][l%4] for g_B:
  row_off = (lane/16)*4 + (lane%16)/4 ; col_off = (lane%4)*4 + (kColOffset%32)
  2nd issue offset = +1024B (+16 kvpos rows; g_B i=4..7 are kvpos+16, NOT +4/+256).
RESULT: max_split=1 e2e "golden vs aiter_asm PASSED" (sorted 0.004). Multi-split: "partial_out_ref vs
attn_logits PASSED" (per-split PV correct) but final "golden vs aiter_asm" still FAILS -> the remaining
bug is the split LSE REDUCTION, which lives in the POST-LOOP code that DBG_PV_NOOUT=1 endpgm's away.
NEXT: remove DBG_PV_NOOUT, compute the post-loop LSE, and fix the known post-loop register-ripple fault.

#### STAGE 4 (2026-06): c=64 FULLY CORRECT; post-loop fault = register pressure (rocgdb-localized)
After the V fix: bisected the post-loop fault by moving the DBG_PV_NOOUT endpgm to the END of mla_main
(KEEP swap + LSE live). RESULT: c=64 (max_split=1 AND default split=32) ALL THREE checks PASS
(golden / split_out_ref / partial_out_ref). So the LSE + swap + output logic are CORRECT. endpgm-at-end
is correct for c<=64 (=kBlockN -> exactly ONE mla_main call per wave; reduction is a separate kernel),
but breaks for c>64 with coarse splits (a wave then needs multiple mla_main calls; first endpgm kills it).
TRUE root cause of the post-loop fault (rocgdb, true no-endpgm build): faulting instr =
  `buffer_load_dwordx4 v1, s[8:11], 0 offen lds`  (the async global->LDS KV load in
  async_load_kv_tile_bf16). Its per-lane global offset v1 (= v_off = phys_row*576*2 + ...) is CLOBBERED
  under full-kernel register pressure -> OOB global addr -> memfault. WHY: VGPR map is k_o(oaccu)=128
  regs (v128-255) + p_comp 16 (v112-127) + p_mfma 8 (v104-111) = 152 managed VGPR, leaving only
  num_vgpr(104) for the compiler. When the full kernel (all dispatch instantiations + mla_main RETURN)
  compiles, 104 is insufficient -> spill -> the async-load offset reg is corrupted. endpgm-at-end avoids
  it by making post-return code dead (lower live-range pressure).
=> REMAINING TASK (general c>64): reduce register pressure. Options: (a) shrink the 128-VGPR oaccu via
   per-outcol-tile output (write incrementally instead of accumulating all 512 outcol), freeing VGPRs to
   raise num_vgpr; (b) noinline the dispatch branches / mla_main to cut peak pressure; (c) move managed
   VGPR tiles to the top and raise num_vgpr if footprint allows (152 managed leaves no room now).
   This is a kernel-design/regalloc task, SEPARATE from the (now-fixed) V/PV correctness bug.
Build state: V fix in; DBG_PV_NOOUT = endpgm at END of mla_main (c<=64 fully correct, all 3 checks).

#### STAGE 4b (2026-06): register-usage analysis vs hand-asm (why arch VGPR is maxed)
Measured our bf16 kernel from the GPU code object (extract .hip_fatbin -> clang-offload-bundler unbundle
-> llvm-objdump dev.co): max v255 (256 arch VGPR, MAXED), max a103 (104 AGPR), sgpr 46, 0 spills,
0 v_accvgpr moves (q/kv in a0-103 fed DIRECTLY to MFMA). VGPR map: compiler scratch v0-103 (num_vgpr
104) + managed v104-255 (p_mfma 8, p_comp 16, oaccu 128 = 152).
Compared to the hand-asm reference mla_a16w16_qh64_qseqlen1_gqaratio64_v3.s:
  arch VGPR ~178 (max v177), AGPR ~208 (max a207), accum_offset 256, 0 accvgpr moves. MFMA operand
  kinds (1176 mmas): D/C ALWAYS VGPR (accumulator); A ALWAYS AGPR; B mixed (672 VGPR + 476 AGPR).
  => hand-asm BALANCES the 512-reg file (178 VGPR + 208 AGPR), putting K/V/Q (double-buffered ->208
     AGPR) in AGPR and accumulator+P in VGPR; lean hand-allocated temps (no compiler-scratch bloat).
ROOT of our maxed VGPR: same operand CATEGORIES as hand-asm (K/V/Q->AGPR, P+accum->VGPR), but the C++
compiler needs ~104 scratch where the hand-asm uses ~26 -> 152 managed + 104 scratch = 256 (maxed).
The movable data (K/V/Q) is ALREADY in AGPR; the only VGPR data left (oaccu/p_comp/p_mfma) is VALU-
touched so can't move to AGPR without v_accvgpr copies.
spill-vgpr-to-agpr=1 EXPERIMENT: FAILED (still memfaults) -- the compiler spills from a0 up and
clobbers the art-managed q/kv at a0-103 (it doesn't know they're reserved). That's why it was =0.
=> FIX OPTIONS for general c>64 (free arch VGPR for the compiler):
   (1) oaccu(128 VGPR) -> AGPR (MFMA D/C in AGPR via ACC_CD): frees 128 VGPR (compiler gets 232) but
       needs v_accvgpr_read/mul/write around each of the ~442 rescales (correctness-first; opt later).
   (2) oaccu halving: process output in 2x 256-outcol passes, oaccu=64 VGPR (frees 64) -> ~168 compiler
       VGPR; 2 PV passes (re-read KV, keep P). Less overhead than (1) per-rescale but ~2x PV.
   (3) reduce compiler scratch (simplify dispatch / fewer inlined instantiations).
Current build: spill flag back to 0; DBG_PV_NOOUT = endpgm at end (c<=64 correct).

#### STAGE 4c (2026-06): the bloat is dispatch-inlining scratch, NOT redundant movement
Measured actual SCRATCH (v<104) usage from the GPU disasm (extract .hip_fatbin -> unbundle -> objdump):
 - endpgm-at-end build (c=64 WORKS): max scratch = v27 (28 used). LEAN -- same as the hand-asm (~26).
   NO redundant data movement/rearrangement in the executed path. v_mov breakdown: 372 scratch->scratch,
   132 imm->scratch, 79 sgpr->scratch (count is a perf nit, NOT the pressure cause). The V/K load
   addressing is efficient: 2 VGPRs (v1,v2) + ds `offset:` immediates (1024..7168), reads into AGPR
   a72-103; MFMA A=AGPR, B=v104-107(p_mfma), D=v128+(oaccu).
 - no-endpgm (FULL, faults): max scratch = v103 (104 used, MAXED to the num_vgpr cap). spill=0 in both.
=> The arch VGPR=256 is the OACCU (v128-255, 128 regs, by design) + p_comp(112-127)+p_mfma(104-111).
   The WORKING path needs only 28 scratch; the FULL kernel's jump 28->104 comes from the dispatch
   inlining ~10 mla_main template instantiations (OutputFinal/Split/None x FirstIter x ...,
   -amdgpu-early-inline-all=true) into one function -> all branches' live values stack -> 104 scratch.
   104 + 152 managed = 256, ZERO regalloc slack (no spill) -> the async buffer_load_lds offset reg is
   reused/clobbered -> memfault. So it is NOT redundant movement; it is dispatch-inlining pressure with
   no headroom, on top of the 128-VGPR oaccu.
FIX (ranked): (2-pref) cut dispatch inlining (noinline mla_main / merge branches) -> full-kernel scratch
   falls toward 28 -> 152+~28=180, ample slack. (1) shrink oaccu (->AGPR or halve) -> frees managed.
NOTE: v28-103 is NOT wasted in the full kernel (it uses all 104); it IS unused in the endpgm-at-end build.

#### STAGE 4d (2026-06): post-loop fault is NOT register pressure -- it's a return-path codegen bug
Exhaustively ruled OUT register pressure as the cause of the full-kernel (no-endpgm) memfault:
 - num_vgpr(64): spill=0 (true scratch peak <=64; the 104 was just cap-fill). Still faults.
 - Repacked managed tiles to v64..215 (max arch VGPR 255->216, 40-reg slack v216-255). STILL faults.
 - spill-vgpr-to-agpr=1: faults (clobbers managed q/kv a0-103). Q-load address-fold (below): no help to peak.
 => Neither scratch room (104) nor max-VGPR reduction (216) nor 40-reg slack fixes it. The fault is a
    GENUINE codegen/logic bug that appears ONLY when mla_main is compiled to RETURN (vs the endpgm-at-end
    build which DCEs the return path and works). rocgdb pinned the faulting instr to the async
    buffer_load_lds (offset reg), but it is wrong/clobbered due to the return-path codegen, NOT pressure.
 NEXT (different approach needed, not regalloc): rocgdb the EXACT faulting wave's v_off inputs on the
    no-endpgm build; or diff the async-load codegen between endpgm vs return builds; or check whether a
    SUBSEQUENT mla_main call / the persistent work-loop computes a bad kv_tile_start/phys_row.
KEPT a legit cleanup: load_q_to_gpr_bf16 Phase 2 now folds the per-col-tile sub-block offset (ct*1024)
into the ds OFFSET immediate (one base VGPR) instead of materializing 18 addresses (your "redundant
rearrangement" hunch -- real, but it did not change the peak/fault).
Build state: known-good VGPR layout (num_vgpr 104, managed v104-255); Q-fold cleanup IN; DBG_PV_NOOUT =
endpgm at end (c<=64 fully correct, all 3 checks).

#### STAGE 4e (2026-06): no-endpgm fault = SYSTEMATIC codegen miscompile (whack-a-mole; exhaustive)
rocgdb: faulting addr ~1GB BELOW the (valid) kv_buffer base; traced to a memory-op ADDRESS clobbered in
a VALID work-loop iteration (s60=kv_tile_start=0, s61=kv_end=16, s57=page_stride=1152 ALL correct;
v_off small/valid). Hardening each resource makes the fault MOVE to the next op (proves systematic):
  KV buffer_load_lds (bound g_kv) -> pending split-output buffer_store (kCheckOOB=false => 0xffffffff,
  set true + qo_end=partial_qo_loc+1) -> still faults. So bounding only hides FAULTS, not the address
  clobber (DATA still corrupt) -> NOT a real fix.
Hardening kept (defensive, c<=64 still passes all 3): (1) load_q_to_gpr_bf16 Phase-2 ds-offset fold
  (1 base VGPR vs 18); (2) async_load_kv_tile_bf16 bound g_kv + clamp phys_row<0->0; (3) load_q bound
  g_q; (4) split output_to_vram kCheckOOB=true + qo_end=partial_qo_loc+1.
RULED OUT (none fix no-endpgm): register pressure (num_vgpr 104/64; managed-repack v64-215 = 40-reg
  slack; spill-to-agpr=1), scheduler (enable-post-misched=1), LSR (-disable-lsr), source logic
  (rocgdb-verified). CONCLUSION: an LLVM codegen miscompile from the work-loop back-edge + mla_main
  return over the fixed-register HK art tiles (oaccu pinned v128-255). endpgm-INSIDE-mla_main avoids it.
NEXT (beyond quick fixes): minimal LLVM repro + pass bisect, OR restructure so mla_main isn't inlined
  with a back-edge over art-pinned tiles (process 1 work-item/wave + clean return; or unpin oaccu).

#### tr_b16 CONTRACT fully characterized + V addressing CONFIRMED PREDICTIVE (2026-06)
Ran 6 controlled probe modes (_tr16_probe.cu / cdna4-isa skill). Single-issue transfer function
(verified, groups of 16, g=lane/16, l=lane%16, M[l][0..3]=4 contiguous u16 at lane addr):
   out[lane][e] = M[ 4*e + (l>>2) ][ l & 3 ]          (e=0..3)
offset:imm = pure +bytes; 2-issue pair (off / off+256B) -> 8 K-contiguous kvpos per lane.
Full characterization written to ~/.cursor/skills/cdna4-isa/mfma.md (+ probe in that dir).
CONFIRMED V addressing for BOTH operands (predicted analytically from the formula, then probe-verified
element-by-element). Each load reads kvpos 0-31 (K=32) in ONE call via the group->kvpos mapping:
  kv_0_top (outcol 0-15):  row_off=(l%16)/4+(l/16)*8 ; col_off=(l%4)*4
  kv_0_bot (outcol 16-31): row_off=(l%16)/4+(l/16)*8 ; col_off=(l%4)*4 + 16
  (both: 2 issues at kImm and kImm+256B; same sub_block base.)
=> KERNEL FIX (now fully specified, no more guessing):
  1. load_transposed_v_to_gpr bf16: use the canonical row_off/col_off above. Add a way to pick the
     outcol-16 half for the "bot" operand (e.g. a kColHalf template param adding +16 to col_off),
     INSTEAD of the current kRowOffset=0/16 (which shifted kvpos -- wrong; canonical already spans
     kvpos 0-31 in one call).
  2. Call sites (prologue line ~1005 + body): kv_0_top = top-half, kv_0_bot = +16-outcol half; drop
     the kRowOffset=16 second-call kvpos shift. Same for kv_1 / the _alt (HI) tiles (outcol +32 each).
  3. Re-derive how the 8 PV iters x (kBlockK*2=64 outcol) tile across the 512 vo_head_dim with the new
     col-half scheme, and confirm the V tile for the current kv block is loaded (kvpos 0-31 valid).
  Validate each step with the probe FIRST (predict -> probe), then DBG_SENT in-kernel, then e2e.

CLEAN DETERMINISTIC A/B (DBG_SEED=1, torch.manual_seed(0), same data): original addr sorted=0.559 vs
canonical sorted=0.602 -- BOTH FAIL, canonical NOT a clear win. So fixing the in_sb addr ALONE is
insufficient: the canonical over-reach reads OOB (the iter-0 tile only has 16 valid kvpos) which
cancels its structural correctness. The bug is DEEPER than in_sb -- it's the V sub-block LOADING /
K=32-spanning. Confirms the next task is the tile-granularity fix, not more addr iteration.
VALIDATE with a CLEAN sentinel measurement: use a single 32-kvpos context (or fix the dump to target
a known tile) so the dumped kv_0_top is the clean kvpos 0-31 tile, then confirm DBG_SENT=row gives
base=(lane//16)*8 (0,8,16,24) and DBG_SENT=col gives lane%16. Everything else (QK/softmax/P-pack/V
values/mma/output path) verified. Build state: addr REVERTED to original (in-bounds, sorted 0.14);
DBG_PV_NOOUT=1 + DBG_DUMP_PCOMP_BOTH=1 (e2e runs). DBG_DUMP_V + DBG_SENT off.

#### STAGE 5 (2026-06): no-endpgm fault SOLVED -- it was AGPR aliasing, NOT a loop miscompile
SUPERSEDES STAGE 4d/4e ("return-path codegen miscompile"). That conclusion was WRONG -- it was
behavior-only, never machine-code-confirmed. rocgdb (precise-memory + single-step + per-lane reg
dumps) on the MINIMAL repro nailed the real cause:
- `-c 64 -ms 1` (ONE work-item, ONE tile, NO back-edge, NO middle tiles) ALSO faults -> kills the
  "work-loop back-edge / multi-tile miscompile" theory outright.
- Faulting instr = `buffer_load_dwordx4 v1, s[12:15], 0 offen lds` (KV vmem->LDS gather). Its byte
  offset `v1` is built `v1 = v45 | a7` where v45 = phys_row*1152 (clean) but a7 is read via
  `v_accvgpr_read_b32 v1, a7`. a7 = flat AGPR 263 = INSIDE the pinned q_nope range a0..a63, holding
  Q bf16 data -> offset ~3GB -> OOB. ct0 uses a VGPR (v46, no fault); ct1..3 use a7/a8/a9/a10.
- 2nd independent site (multi-split): LSE store `global_store_dword v[2:3]`, address via a5 (also
  q_nope range) -> base + ~4.27GB -> "write to read-only page".
ROOT CAUSE: under heavy-C++ address pressure LLVM uses the q_nope-pinned AGPRs as memory-address
scratch. HK reservation does NOT hard-reserve: `clobber<>` = `asm volatile("" ::: "aN")` (one-shot),
and the art MFMA/ld ops name AGPRs as HARDCODED asm immediates (`"n"(GPR-256)`, e.g.
`ds_read_b128 a[%0:%1]`), NOT constrained operands -- so the allocator has no virtual reg tied to
a0..a103 and freely reuses them. PROVEN by standalone probe: a one-shot AND a wide (a0..a31) asm
clobber both FAIL to keep the allocator off the register; the allocator fills AGPR BOTTOM-UP (a0,a1..).
No `amdgpu_num_agpr` attr exists; no `-mllvm` agpr flag; `amdgpu-agpr-alloc` is IR-only (not source-
settable). HK's own kernels survive only because they keep compiler-visible C++ scratch tiny
(num_vgpr(29) window) so it never overflows into the asm-named AGPRs.
FIX (two parts, both verified):
  1. KV-load: fold the per-col-tile column offset into the buffer immediate i_off (12-bit MUBUF
     OFFSET, opus.hpp:1691 static_assert) so the offset stays in ONE vgpr (no per-ct AGPR const).
     async_load_kv_tile_bf16.
  2. SYSTEMIC: top-pack the AGPR map. q/kv tiles moved from a0..a103 to a152..a255
     (k_agpr_top_base=408). The allocator's bottom-up AGPR scratch now lands in the now-free
     a0..a151 and never reaches the tiles -- mirrors the WORKING VGPR scheme (managed v104..255,
     compiler v0..103). One-line change: k_q_nope_begin 256 -> 408 (rest chains).
RESULT: ALL configs run fault-free (-c 64 -ms 1, -c 64, -c 1200). Device metadata: agpr_count 256,
total 512/lane, sgpr/vgpr spill_count = 0 (fits without memory spills), occupancy 1 (unchanged),
fp8 kernel untouched. DBG_PV_NOOUT now DISABLED (real path runs). num_vgpr(104) kept.
REMAINING (separate, pre-existing): output is sorted-diff small (~0.29 c64 / 0.73 c1200) = the V/PV
->output PERMUTATION/layout bug from STAGE-4f above, NOT a value or fault bug. That is the next task.
GENERAL LESSON (cross-project): HK clobber<>/art is asm-naming + a hint, NOT a hard register
reservation. If a kernel mixes art tiles with non-trivial C++ (address math, reductions), pin the
art tiles at the TOP of each register bank and cap the compiler LOW (num_vgpr for VGPR; top-placement
+ bottom-up-allocator behavior for AGPR), or the allocator will reuse the "reserved" regs as scratch.

#### STAGE 6 (2026-06): correctness (post-fault) localized with DETERMINISTIC sentinels
Fault is fixed; remaining failure is checkAllclose (max delta 1.58, 95.7% elems; e2e sorted 0.29).
Value-matching perm-finder (DBG_PERM) is UNRELIABLE here -- bf16 outputs aren't unique, argmin gives
spurious matches (only 143 unique targets / 374 matches). Use DETERMINISTIC sentinels instead:
- DBG_SENT=col sets V[kvpos][oc]=oc (constant over kvpos) -> normalized out[qh][oc] MUST ==oc. This
  is BLIND to the kvpos axis; it isolates the output (qhead+outcol) LAYOUT only.
  RESULT: lower outcol half 0..255 = 100% correct; upper half 256..511 = ~144/256 WRONG, pattern =
  even outcols correct, ODD outcols hold their even neighbor (257->256, 259->260, 261->260, ...).
  => an output-LAYOUT bug confined to outcol>=256 (PV tiles 4..7), odd positions (value duplication,
  odd outcols lost). NOT a uniform permutation.
- Real data is 95.7% wrong (>> the ~28% col-sentinel shows) => there is ALSO a kvpos-pairing/value
  component (P-slot vs V-slot order in the PV contraction) on top, only visible when V varies / kvpos.
BISECT DONE: oaccu(tile4) dump under DBG_SENT=col shows the duplication is ALREADY in the PV
accumulator: oaccu_0_a(lane0,r0..3) got outcols [256,256,258,260] vs expected [256,257,258,259]
(stride-2, odd outcols lost). => bug is in the V TRANSPOSE LOAD load_transposed_v_to_gpr (tr_b16
outcol/col addressing) for upper tiles (kColOffset>=256); output_to_vram + PV mma + lower tiles 0..3
are CORRECT. The col_off/kColHalf/kAddr-kImm split in load_transposed_v_to_gpr (hk_mla_buffer_managers
.cuh ~1631-1650) is the suspect; the per-tile difference is kColOffset (0,64,128,192 OK; 256,320,384,
448 broken) -> check the kSb sub-block selection + the 2nd ds_read_b64_tr_b16 immediate
(kImm + 16*kSubBlockCols16*2) for kColOffset>=256 (possible 16-bit ds-immediate wrap or wrong
sub-block stride). NOTE this only explains the col-sentinel ~28%; real-data 95.7% ALSO needs the
DBG_SENT=row check (P/V kvpos slot-order pairing).
DBG scaffolding fix: DBG_DUMP_OACCU endpgm is now INSIDE the per-tile guard (was firing every iter ->
wave died at i=0, dumped all-zero); set the i.value in the guard to pick which tile to dump.

#### STAGE 7 (2026-06): col-sentinel upper-half was a BF16 ARTIFACT; real bug is the QK gemm
CORRECTION to STAGE 6: the "outcol>=256 odd-outcol duplication" is NOT a kernel bug. DBG_SENT=col sets
V[kvpos][oc]=oc; bf16 has 7 mantissa bits so integers 256..511 have ULP=2 -> ODD ints >=256 are
unrepresentable and round to the even neighbor (257->256, 259->260, ...). That EXACTLY produces the
observed pattern. So the OUTPUT LAYOUT (qhead+outcol) is CORRECT; the col-sentinel is simply blind/
broken above 256. (Folding kSb into the V-read base changed nothing == same address, consistent.)
Also: col-sentinel is BLIND to P (normalize cancels it: O=sum_k P[k]*oc/sum_k P[k]=oc for any P!=0),
so a DEAD/permuted QK still passes col-sentinel -> col-sentinel proves NOTHING about QK/softmax/PV.
REAL BUG (DBG_SENT=row, V[k]=k, values 0..63 bf16-exact, ref uses same sentinel): checkAllclose FAILS
hard (sorted 41.97, max delta 65>63, 97.5% wrong) = genuine VALUE bug in the QK->softmax->P->PV chain.
Traced to QK: DBG_QK_ONLY + DBG_DUMP_PCOMP_BOTH, DBG_LOAD=pcomp on real data shows p_comp 98% mismatch:
  nope SORTED diff 9.29 (values ~right as a multiset but per-tile permuted: N1=60, N16=9.3),
  rope SORTED diff 63.68 (multiset itself WRONG -> rope QK genuinely value-wrong, not just permuted).
IDENTICAL with top-pack on (408) vs reverted (256) -> the fault-fix did NOT cause it; QK was already
broken, masked by the fault all session (e2e never passed this session). The i_off async change is
semantically a no-op for LDS contents (same VMEM offset, just immediate vs register) so it's not the
cause either. So this is a PRE-EXISTING QK correctness bug (the STAGE-4f "QK verified" was either a
different state or a dump-layout-assumption match, not an e2e pass).
CAVEAT (verify next): part of the nope "permutation" could be the pcomp TEST's stale p_comp layout
assumption rather than the kernel. Decisive next steps: (1) confirm pcomp test layout vs kernel MFMA
D-layout; (2) the rope multiset being wrong (63.68) is the clearest real defect -> debug the rope QK
path (q_rope a216-223 after top-pack / k_rope load / rope scale) first; (3) then DBG_SOFTMAX_ONLY (P)
and the PV kvpos slot-order. Build state: clean (top-pack 408, all DBG off), fault-free, e2e sorted
0.289 (NOT a pass -- the small value is coincidence, real output is uniform-ish from broken QK).
Diagnostics added (env-gated, harmless): test DBG_SENT=col now reports full 512-outcol layout
correctness + per-qhead identity count; DBG_PERM does robust exact-match + powers-of-2 bit probe.

#### STAGE 8 (2026-06): CORRECTNESS FIXED -- the rope bug was MY OWN i_off "fix" regressing it
ALL configs now PASS checkAllclose (-c 64 -ms 1, -c 64, -c 1200, -c 333; e2e sorted 0.002-0.004 =
bf16 noise). Root cause + fix chain (verified end-to-end with deterministic sentinels):
- DBG_SENT=row (V[k]=k, bf16-exact) hard-failed -> value bug in QK/softmax chain. DBG_LOAD=pcomp:
  nope ~right (4.63 = bf16), rope wrong. DBG_SENT=one (q_rope=k_rope=1 -> every score must gain a
  uniform +64) gave 26.19 -> rope contraction broken. DBG_LOAD=krope -> rope K GPR was random, not 1.
  DBG_LOAD=lds16 (raw LDS sub-block 16 after async) -> ALSO random -> the ASYNC WRITE of rope was
  broken, not the read.
- THE BUG: my STAGE-5 KV-load "fix" folded the per-col-tile offset into the buffer_load_lds OFFSET
  IMMEDIATE (i_off = ct*64). VERIFIED: ct 0-7 (i_off 0-448) worked (the only part the col-sentinel
  covered, outcol 0-255), but ct>=16 (rope, i_off=1024/1088) loaded GARBAGE VMEM -> rope sub-blocks
  wrong. NOTE the earlier "exceeds 12-bit range" guess was WRONG: 1088 < 4095, so simple range
  overflow does NOT explain it -- the buffer_load_lds inst immediate is effectively unusable for
  >=1024 B here (break at 1024; 512-960 untested; mechanism not isolated). The fault masked this all
  along (e2e never ran clean until the AGPR fix), so it looked pre-existing.
- FIX: revert that hack -- put the FULL byte offset back in the voffset VGPR (32-bit, no range limit),
  i_off=0 (async_load_kv_tile_bf16). This is safe now because the AGPR top-pack (STAGE 5) already
  prevents the per-ct address from spilling into the q_nope AGPRs (compiler scratch lands in the free
  low AGPRs a0..a151). So: top-pack fixes the FAULT; full-voffset fixes CORRECTNESS; they compose.
- Confirms the user's recollection: QK/nope was genuinely verified earlier; only MY i_off change had
  regressed the rope (a self-inflicted bug, masked by the unrelated fault). LESSON: an "optimization"
  that moves an offset into a hardware immediate MUST check the immediate's range against the FULL
  index domain, not just the cases a partial sentinel happens to cover.
Build state: top-pack 408, full-voffset async, i_off K-read split restored, all DBG_* off, fault-free,
correct. Next: performance (vs mla_a16w16_qh64_qseqlen1_gqaratio64_v3_ps.co).

#### STAGE 9 (2026-06): PERF baseline + evidence-based gap attribution (HK bf16 vs asm qh64)
Started the optimization loop. Clean min-of-3 on an IDLE gfx950 (MI355X, container pa_bench_mh),
`-n 64,1 -d bf16 -kvd bf16 -c 1200`, HK = AITER_ENABLE_EXPERIMENTAL=1 vs asm = 0 (both correct):
| B | HK us | asm us | HK/asm |
|---|---|---|---|
| 16 | 27.0 | 23.8 | 1.13x |
| 64 | 50.5 | 42.83 | 1.18x |
| 128 | 79.67 | 63.18 | 1.26x |
=> HK bf16 is the BEST HK result to date (OPUS path ~1.3x, fp8 HK 1.2-1.35x): the project's core bet
PAID OFF -- the bf16 HK kernel achieves the asm REGISTER STRATEGY (verified from device code object:
.vgpr_count 512 = 256 arch + 256 AGPR, .agpr_count 256, 0 spills; q/kv in AGPR feeding MFMA directly,
accumulator in VGPR). This is exactly what OPUS structurally COULD NOT do (its 4242 v_accvgpr shuffles).
The gap grows with batch (= with KV-tiles-per-workgroup) -> a PER-TILE overhead, not a fixed cost.

ATTRIBUTION (PMC + ISA-diff, the disciplined part -- REFUTES the obvious "deep prefetch" lever):
1. **vmcnt: asm holds 10 (deep prefetch, 2 tiles in flight); HK drains vmcnt(0) at ~119 sites.** This
   LOOKS like the gap but is NOT: rocprofv3 PMC (B128) shows HK SQ_WAIT_INST_LDS=5.6e7 vs asm 1.28e8
   and SQ_WAIT_INST_ANY=8.3e8 vs asm 1.22e9 -- **HK WAITS LESS than asm** (KV is L2-resident at c1200,
   so HK's vmcnt(0) drains cost little). Deepening prefetch will NOT help. (Classic parity-skill trap:
   the sync COUNT is a hypothesis, not the cause -- PMC falsified it.)
2. **NOT instruction count:** PMC runtime SQ_INSTS_VALU HK 4.72e8 < asm 5.04e8; MFMA_BF16 1.45e8 ~=
   asm 1.38e8; SQ_INSTS_LDS 2.27e8 ~= asm 2.17e8. HK executes <= asm in every class. (The static
   disasm shows HK 2x more instrs, but that is the ~10 inlined mla_main instantiations -- dead in any
   single runtime path; the PMC runtime counts are the truth.)
3. **NOT MFMA-dependency bubbles:** HK has zero large `s_nop N` waitstates (only `s_nop 0`), so
   interleaving independent MFMA chains won't help (consistent with the prior MFMA microbench = 0 gain).
4. **THE GAP = softmax-phase MFMA idle (lower IPC at 1 wave/SIMD).** GRBM_GUI_ACTIVE ratio 1.20-1.22x
   == the perf gap, but SQ_BUSY 1.32x with <= instrs and less wait => more stall cycles/instr = pipeline
   bubbles NOT attributed to s_waitcnt. ISA-confirmed directly: the HK softmax region disassembles to a
   PURE VALU run (3 v_max, 8 v_pk_mul, 16 v_exp, v_add) with ZERO MFMA or ds_read interleaved -- the
   matrix unit sits idle through the whole max-reduce/exp every tile. asm avoids this via INLINE online
   softmax (v_max/v_exp interleaved between QK/PV MFMAs, asm disasm: v_exp scattered, not clustered).
   HK already interleaves the oaccu RESCALE (mul_pair) with PV MFMA, but the max-reduce + exp phase
   between QK and PV is a separate, un-overlapped bubble.

THE ONLY REMAINING LEVER + its BLOCKER: fill the softmax(N) bubble with the next tile's QK MFMA(N+1).
That requires a CROSS-TILE software pipeline holding a 2nd p_comp accumulator (+QK state) live during
softmax(N) -- register-blocked at the 512-reg ceiling (oaccu alone = 128 VGPR). Within-tile there is NO
independent MFMA to hoist (PV needs P from this softmax). To free registers one would output-D-split
(halve oaccu 128->64 VGPR, the lever the OPUS split-D used) -- but unlike the OPUS split-D failure
(which went SHALLOW and lost overlap), here the freed regs would ADD cross-tile overlap. This is a
MAJOR, NaN-prone rewrite (re-run the full QK->softmax->P-pack->PV staged ladder), with the documented
expectation that the residual ~1.2x is near the 1-wave hand-schedule floor. NOT attempted without buy-in.
Evidence files this session: /tmp/hk_gqa1.s, /tmp/asm_qh64.s (disasm), PMC via gpu-profiling gfx950 set.
LESSON (cross-project candidate): for an HK/asm-register-matched kernel, profile (PMC wait + GUI) BEFORE
copying the asm's vmcnt/prefetch -- a kernel can drain vmcnt(0) yet wait LESS than the deep-prefetch asm
when data is cache-resident; the real 1-wave gap is then VALU-phase MFMA-idle, not load latency.

#### STAGE 9b (2026-06): softmax-overlap lever DIAGNOSTIC = only ~3% (cross-tile overlap is the real gap)
Before carving the cross-tile softmax-under-MFMA pipeline, measured its UPPER BOUND cheaply: a throwaway
`DBG_NOSOFTMAX` build (skip max_16 + warp_reduce + softmax_p1_16/exp entirely; breaks correctness, perf
only -- guard kept disabled in mi35x_..._m16x4_bf16_bf16.cuh at the softmax block + top #define).
Clean min-of-3, c1200:
| B | HK baseline | HK no-softmax | savings | gap-to-asm |
|---|---|---|---|---|
| 64 | 50.5 | 48.67 | ~3.6% | 18% |
| 128 | 79.67 | 77.39 | ~2.9% | 26% |
=> Removing the ENTIRE softmax bubble recovers only ~3%, NOT the 18-26% gap. So overlapping softmax
under the next tile's MFMA (the planned cross-tile carve) would pay at most ~3% -- NOT worth the rewrite.
This DIAGNOSTIC (one guarded build) saved a multi-hour register-blocked carve. (karpathy: test the lever,
don't infer from the bubble's visibility.)

WHERE THE REAL GAP IS (read the full PV loop, lines ~1142-1391): the kernel is ALREADY asm-quality WITHIN
each phase -- PV double-buffers V (LO in kv_0/1, HI in kv_*_alt), rescale (mul_pair) is interleaved 1:1
with PV MFMAs, s_setprio-tuned, loads prefetched. The localized inter-phase bubbles (softmax 3%, P-pack a
bit) are small. The residual ~16-20% is CROSS-TILE overlap: mla_main does a FULL drain
`__builtin_amdgcn_s_waitcnt(0)` + `s_barrier` at EVERY tile top (line ~571), serializing PV(N) -> QK(N+1).
asm keeps the pipeline flowing across tiles (PMC-derived MFMA util asm/HK ~1.16x = the gap). Matching it
needs overlapping PV(N) with QK(N+1):
- VGPR is NOT the blocker for that overlap: PV(N) uses oaccu(v128-255)+p_mfma(v104-111); QK(N+1) uses
  p_comp(v112-127) -- DISJOINT, already coexist.
- The real blockers: (a) K(N+1) and V(N) share the SAME kv_0/1/kv_*_alt AGPRs (32 AGPR) -> need a 2nd
  AGPR KV tile set for K(N+1) while V(N) is live (AGPR headroom a0-151 exists but is the compiler's
  scratch from the top-pack fix -> pinning it risks the address-spill fault again); (b) the 2-buffer LDS
  ping-pong (2*73728=147KB, can't 3-buffer in 160KB) forces a barrier before the swap; (c) the per-tile
  phased loop must become a software-pipelined prologue/steady/epilogue holding 2 tiles' state.
=> "Match the asm pipeline" = a full cross-tile deep-pipeline interleave REWRITE (multi-session, NaN-prone)
on the currently-correct kernel, for a ~16-20% target. This is the documented hand-schedule floor; the
localized levers are empirically exhausted (prefetch refuted by PMC, softmax overlap = 3% by diagnostic).
RECOMMENDATION: ship HK bf16 at 1.13-1.26x (Phase-0 GO -- it beat OPUS's 1.3x by matching asm's register
strategy) and either (1) scope the cross-tile rewrite as a dedicated effort if large-B qlen1 matters, or
(2) route pure qlen1 large-B to the asm .co (vendors ship asm for exactly these shapes). Build/tree state:
DBG_NOSOFTMAX reverted, kernel rebuilt + correctness re-verified (c1200 B64 PASS).

#### STAGE 10 (2026-06): ATT INSTALLED -> true bottleneck = KV VMEM-load path (CORRECTS STAGE 9/9b)
Installed the rocprof thread-trace decoder (was missing; rocprofv3 --att produced empty code.json without
it). Steps that worked on this box (Ubuntu 22.04, ROCm 7.2.3, MI355X):
  curl the deb: compute-artifactory.amd.com/.../thread-trace-decoder/rocprof-trace-decoder-manylinux-2.28-0.1.4-Linux.deb
  dpkg -i FAILS (arch "()" != amd64); instead: dpkg-deb -x <deb> /tmp/x && cp /tmp/x/opt/rocm/lib/librocprof-trace-decoder.so /opt/rocm/lib/ && ldconfig
  run: rocprofv3 --att --att-library-path /opt/rocm/lib --att-target-cu 1 --att-simd-select 0xF
       --att-shader-engine-mask 0xF --att-activity 8 --kernel-include-regex "<name>" -d OUT -- <cmd>
  (NOTE: --att-serialize-queue is NOT a valid flag in this rocprofv3; -b1 -ms1 gives an empty CU-1 trace
   -- use a FULL workload e.g. -b 128 -c 1200 so CU 1 runs the steady loop.)
ATT decode + analyze scripts: gpu-profiling/att.md (analyze_att.py classify; top-stall drill-in).

DECISIVE FINDING (ATT code.json, HK vs asm, same workload B128 c1200, same ATT distortion):
HK's stall is MEMORY-bound, dominated by waiting on KV VMEM loads -- NOT MFMA/softmax (the STAGE-9b
cross-tile-MFMA-overlap target was WRONG). Stall breakdown (% of each kernel's stall):
| class | HK | asm |
|---|---|---|
| waitcnt_vmcnt | 23.1% | 0.8%  |   <- HK stalls ~58x more on VMEM-load completion
| waitcnt_lgkmcnt | 25.1% | 18.0% |
| vmem_load | 22.0% | 40.3% |
| mfma | 9.9% | 3.9% |
| valu_softmax | 3.3% | 6.5% |
HK total stall ~2x asm's. Top stalling instrs (HK): s_waitcnt vmcnt(0) drains 86k/84k/67k cycles +
buffer_load_dwordx4...offen lds 52k/27k/20k... vs asm: ONE boundary drain 20.7k, buffer_load_lds only 8.3k.

CORRECTS STAGE 9: "deep prefetch refuted by PMC (HK waits LESS)" was a PMC ACCOUNTING QUIRK -- my PMC set
isolated SQ_WAIT_INST_LDS (lgkmcnt) + SQ_WAIT_INST_ANY but had NO vmcnt-wait counter, so it MISSED HK's
dominant stall (vmcnt). ATT is ground truth (gpu-profiling/att.md cross-check table). The deep-prefetch /
asm-pipeline lever the user pointed at from the start IS correct. (Also: the STAGE-9b softmax=3% diagnostic
was right that softmax-overlap is small -- but the gap isn't there; it's KV-load latency.)

TWO CONCRETE asm TECHNIQUES TO COPY (both in the KV global->LDS load = async_load_kv_tile_bf16):
1. **M0 setup per load.** asm: M0 set once then `s_add_u32 m0, m0, 0x1000` (constant LDS stride) between
   consecutive `buffer_load_dwordx4 ... offen lds`. HK: `v_readfirstlane s22,v4; s_mov m0,s22; s_nop 0`
   PER LOAD (cross-lane readfirstlane + nop) -> serializes, ~150k stall cycles. Fix: set M0 once, advance
   by constant stride across the tile's loads.
2. **vmcnt drains.** asm holds vmcnt(~10) and almost never stalls on it; HK drains vmcnt(0) repeatedly and
   stalls ~280k cycles. Fix: deepen prefetch / partial vmcnt(N) waits so the wave doesn't fully drain
   waiting for KV loads (the asm deep-pipeline). LDS is 2-buffer (147KB/160KB) so depth is via not-draining,
   not a 3rd buffer.
REVISED PLAN: target async_load_kv_tile_bf16 (hk_mla_buffer_managers.cuh) -- (a) constant-stride M0, (b)
relax vmcnt(0)->partial. This is MORE TARGETED + LOWER RISK than the cross-tile MFMA rewrite (R2 reserved
regs p_mfma_b/kv_k + num_vgpr 96 may be unneeded for this path -- revert if so). Verify each with ATT
re-trace (vmcnt% + buffer_load stall down) + correctness + bench. Build/tree: num_vgpr(96) + p_mfma_b/kv_k
reserved (unused, 0 spill, correct) -- pending the revised KV-load work.

#### STAGE 10b (2026-06): M0 fix = correct but PERF-NEUTRAL; root cause = bf16 BURSTS the KV loads
Reverted R2 scaffolding (num_vgpr->104, removed p_mfma_b/kv_k) -> back to clean baseline.
FIX #1 (M0, DONE): async_load_kv_tile_bf16 passed a per-lane LDS dst (`p_lds_kv + sub_block(warp,ct) +
lane_idx*16`) to buffer_load_lds. buffer_load_lds HW auto-fans lanes by lane*size, so the lane_idx term is
redundant for the LOAD (only the OOB ds_write needs it) and FORCED a per-ct `v_readfirstlane+s_mov m0`.
Removed it (pass uniform `p_lds_base`; OOB ds_write keeps +lane_idx*16). RESULT: v_readfirstlane GONE
(verified disasm: now `s_mov m0,sN; s_nop 0; buffer_load_lds`), correctness PASS. But PERF NEUTRAL
(B64 50.5->50.1, B128 79.7->80.1, noise) AND ATT buffer_load stall UNCHANGED (~83k/54k/38k). So the per-load
M0 setup was NOT the binding cost -- it overlapped compute. (Keep the fix: correct, canonical idiom per
mha_native/fused/op_lds.hpp, composes with the real fix; the compiler still emits 18 separate `s_mov m0,sN`
not the asm `s_add m0,m0,0x1000` chain, but that's not the bottleneck.)
TRUE ROOT CAUSE (ATT post-fix top stalls): `buffer_load_dwordx4 ... offen lds` stalls 83k/54k/38k +
`s_waitcnt vmcnt(0)` 96k/62k/57k. The bf16 path BURSTS all kNumColTiles16(18) KV loads at the tile top
(one async_load_kv_tile_bf16 static_for) -> fills the VMEM load FIFO -> the loads stall (FIFO full) AND the
following vmcnt(0) drain waits for the whole burst. The FP8 path (and asm) SPREAD the loads across the QK
MFMA loop ("pass-0 col-block 0 here; pass-1 + col-blocks 1..8 issued across the QK loop", fp8 mla_main
comment) so a few loads are in flight at a time, interleaved with MFMA, FIFO never fills, vmcnt stays ~10
(asm) not 0. The bf16 port took the burst SHORTCUT.
FIX #2 (the real lever, NEXT): spread the bf16 next-tile KV loads across the QK loop like fp8 -- make a
per-col-tile bf16 loader (one ct) and issue ct's load inside QK iter ct (NoPE 16 + RoPE 2 = 18 cts),
interleaved with the MFMAs; drop the burst call at mla_main top; use partial vmcnt as each ct's K/V is
consumed. Mirror the fp8 async_load_k_tile threading (row_kv_ld_next, col offsets). Verify staged: ATT
buffer_load + vmcnt stall down, correctness, bench. Build/tree: M0 fix IN (correct, neutral); clean baseline.

#### STAGE 10c (2026-06): FIX #2 (KV load-spread) IMPLEMENTED -> B128 1.26x->1.15x (real win)
Implemented the spread. hk_mla_buffer_managers.cuh: refactored the bf16 KV loader into
(a) resolve_kv_phys_row_bf16<>() (the p_kv_indices lookup, done ONCE per tile -- NOT per col-tile, else
18x VMEM traffic) and (b) async_load_kv_cols_bf16<kCtStart,kCtCount>(...,phys_row) loading a col-tile
RANGE; async_load_kv_tile_bf16 (full burst) kept for the FIRST tile (prologue, no compute to hide under).
mi35x_..._m16x4_bf16_bf16.cuh mla_main: replaced the top burst with a single resolve into next_phys_row_bf16,
then SPREAD the 18 col-tile loads across the QK loop -- NoPE iter idx issues cols [idx*2, idx*2+2) right
after its load_k_to_gpr (interleaved with that iter's MFMAs), RoPE iter issues cols 16,17. All guarded
sizeof(kv_t)==2 && kIsGlobalLast==false (epilogue/skip tiles have no next tile).
RESULT (clean min-of-3 c1200, vs asm):
| B | HK before | HK spread | asm | before/asm | spread/asm |
|---|---|---|---|---|---|
| 16 | 27.0 | 27.03 | 24.17 | 1.13x | 1.12x  |  (B16: few tiles/wg, prologue-bound -> ~no change)
| 64 | 50.5 | 47.59 | 42.57 | 1.18x | 1.12x  |
|128 | 79.67| 73.96 | 64.17 | 1.26x | 1.15x  |  (-7%)
Correctness PASS all configs incl. boundary (c64/333/1200/5000, b1/8/32/64/128, -ms 1). ATT: wall
3.27M->2.98M; vmcnt still top stall (24%) -> MORE headroom remains (the tile-top s_waitcnt vmcnt(0) drain
still waits for the spread loads). NEXT (fix #2b): relax the tile-top vmcnt(0) full drain to a partial
vmcnt so the wave doesn't fully wait for the prefetch (asm holds vmcnt~10). Build/tree: M0 fix + load-spread
IN, correct.

#### STAGE 10d (2026-06): remaining vmcnt stall = COMPILER-conservative vmcnt(0) before ds_read
ATT top stalls on the spread build: in-loop `s_waitcnt vmcnt(0)` 96k (right after the K `ds_read_b128`),
`buffer_load_dwordx4...lds` 65k/63k, tile-top full drain 64k, prologue 60k. CHECKED: ALL `s_waitcnt
vmcnt(0)` in mla_main source are DBG-gated (disabled) -> the in-loop vmcnt(0) is COMPILER-INSERTED. Root:
buffer_load_lds (writes LDS, async, bumps vmcnt) + ds_read (reads LDS) on the same wave -- the compiler
cannot prove the next-tile prefetch buffer (p_lds_kv_next) does NOT alias the current-tile buffer
(p_lds_kv_curr) being ds_read, so it conservatively drains vmcnt(0) before each ds_read group. My spread
interleaved next-tile buffer_load_lds with current-tile ds_reads -> the conservative vmcnt(0) now also
waits on the spread loads. Net STILL a win (wall down, 1.26->1.15x) but this is the new ceiling.
=> fix #2b needs EXPLICIT inline-asm vmcnt(N) management (issue loads, ds_read with a hand-computed
partial vmcnt that excludes the not-yet-needed next-tile loads) -- the asm hand-schedule. Higher risk
(fragile counts, correctness-critical). Deferred as the next asm-parity step. Current shippable state:
M0 fix + load-spread, all correct, B64 1.12x / B128 1.15x of asm (was 1.18x / 1.26x).

#### STAGE 10e (2026-06): page-index prefetch = NEUTRAL; remaining stall is KV-DATA feed-latency
ATT (STAGE-10d) said the #1 in-loop stall was `s_waitcnt vmcnt(0)` feeding `v_max_i32 v,0,v81` -- the
p_kv_indices page-index lookup (get_kv_ld_row's raw_buffer_load_b32) on the critical path. FIX TRIED:
split resolve into issue_kv_row_raw_bf16 (raw load) + finalize_kv_row_bf16 (use), and ISSUE the raw load
BEFORE the tile-top __builtin_amdgcn_s_waitcnt(0) drain (which already waits for prev KV loads, so it
absorbs the small index load) + FINALIZE after. Disasm CONFIRMS: the page-index use no longer has a
preceding vmcnt(0) (load moved before the barrier). BUT clean min-of-3 = NEUTRAL (B64 47.59->47.49,
B128 73.96->73.8). The page-index vmcnt just folded into the drain (which was already stalling on KV data),
so no net win. ATT: vmcnt still 24.9% of stall -- dominated by the KV DATA loads (buffer_load_lds), NOT
the page index. (Kept the split: correct, cleaner codegen, marginally +; resolve_kv_phys_row_bf16 still
used by the prologue burst.)
=> The remaining ~15% (B128) is KV-data-load feed-latency: the tile-top drain waits for the prev iter's
spread KV loads to complete. Even spread, the last loads' latency isn't fully hidden by softmax+PV before
the next drain. Closing it needs the asm DEEP pipeline (hold vmcnt~10 across tiles via explicit inline-asm
vmcnt mgmt -- the compiler inserts vmcnt(0) before each ds_read because it can't prove the next-tile
buffer_load_lds doesn't alias the current-tile LDS being read) OR >2 LDS buffers (doesn't fit: 2*73728=
147KB/160KB). Both are the documented hard hand-schedule. Levers tried this round: M0-uniform (neutral),
load-spread (the win, -7%@B128), page-index prefetch (neutral).

#### STAGE 10f (2026-06): full ctx x B sweep (HK vs asm) + MEMFAULT REGRESSION found & fixed
Swept the OPUS heatmap grid (qh64 qlen1, ctx{21..65536} x B{1..256}) with HK vs asm to check whether HK
closes the cells where OPUS lags asm. CSV: /home/mh/hk_sweep.csv.
**BUG FOUND DURING SWEEP (now fixed): the STAGE-10d page-index prefetch caused a GPU MEMFAULT** at
large-ctx multi-split configs (ctx8192 B1/16/32, ctx16384 B16; archive baseline did NOT fault -> my
regression). Root cause: issue_kv_row_raw_bf16 did the p_kv_indices buffer_load UNCONDITIONALLY (no bounds
check before the load), unlike get_kv_ld_row which only loads in-bounds. A split's last-tile next-idx
exceeds the real p_kv_indices length; the rsrc num_records=0xffffffff does NOT clamp -> unmapped read ->
memfault. Each fault wrote a ~2-32GB GPU coredump to /aiter/gpucore.* -> filled the shared disk -> cascaded
into spurious "No space left"/NA failures on OTHER cells. FIX: reverted the page-index prefetch (it was
perf-NEUTRAL anyway) back to the bounds-checked resolve_kv_phys_row_bf16. All 4 cells now PASS.
NOTE: the sweep's "OOM" cells (ctx65536 B>=5) were NOT GPU out-of-memory -- they were THIS memfault's
coredumps filling the shared disk -> test crashed "No space left on device" -> NA. After the fix + disk
cleanup they ALL run fine (ctx65536: B5 134us PASS, B16 275us 1.16x, B32 1.14x, B64 1.13x, B128 1.16x,
B256 timing 3607us). LESSON
(cross-project): an "issue load early" split MUST keep the original's bounds guard before the load -- a
buffer_load with num_records=0xffffffff does NOT bounds-clamp to the real array; it faults on a truly OOB
VA. Also: GPU memfaults dump huge coredumps that can fill the disk -- set HSA cleanup / rm gpucore.* in any
sweep that may fault. (HSA_ENABLE_COREDUMP=0 / HSA_COREDUMP_FILE=/dev/null did NOT suppress on this ROCm.)

SWEEP RESULT (HK/asm vs OPUS/asm, qh64 qlen1; >1.0 = slower than asm):
- BOTTOM-RIGHT (large ctx>=3200, mid-large B, where OPUS LAGS asm 1.11-1.53): **HK is BETTER than OPUS in
  ~every cell**, roughly HALVING the OPUS deficit. Examples: ctx16384 B16 OPUS 1.53 -> HK 1.18; ctx8192 B32
  OPUS 1.39 -> HK 1.18; ctx5200 B64 OPUS 1.35 -> HK 1.13; ctx8192 B256 OPUS 1.27 -> HK 1.14. HK lands
  ~1.12-1.21x of asm across the whole red region (vs OPUS 1.27-1.53). A few ~ties at the largest B where
  OPUS was already ~1.1.
- TOP-LEFT / small ctx & B (where OPUS WINS asm 0.29-0.96 via split-KV/flash-decode): **HK LOSES to OPUS**
  (HK 1.0-1.45x, no split-KV). HK is a single persistent kernel; it has no small-batch GPU-fill advantage.
=> HK and OPUS are COMPLEMENTARY: HK is the right kernel for the large-ctx/large-B regime (the OPUS-red
region) where it ~halves the gap to asm; OPUS(split-KV) stays best for small ctx/B. Neither beats asm in
the large regime (HK ~1.12-1.21), but HK is the closest source kernel there. Practical dispatch: HK for
large ctx*B qh64-qlen1, OPUS-splitKV for small, asm if available.

#### STAGE 11 (2026-06): global-load deep-dive -- ds_read "memory"-clobber lever RULED OUT
User priority: optimize the global load (longest latency). Re-profiled current build (ATT B128 c1200):
waitcnt_vmcnt 29.6% of stall (#1), waitcnt_lgkmcnt 24.5%, mfma 11.6%. The #1 vmcnt(0) is the tile-top
drain + the page-index resolve, both waiting on the KV global-load feed.
Global-load design diff HK vs asm (disasm): asm holds vmcnt(10) (continuous streaming, ~1-2 tiles in
flight) + const-stride `s_add m0,m0,0x1000`; HK drains vmcnt(0) at every tile boundary + per-load
`s_mov m0`. HYPOTHESIS: the HK ds_read_b128 macro's blanket `:"memory"` clobber forces the compiler to
treat each ds_read as a full memory fence -> vmcnt(0) drains. TESTED: removed `"memory"` from
ds_read_b128 (macros.cuh), rebuilt. RESULT: correctness PASS, but PERF UNCHANGED (47.9us) and the
vmcnt(0) instr count is IDENTICAL (973 both) -> the clobber was NOT generating the drains. The vmcnt(0)s
are the EXPLICIT tile-top `__builtin_amdgcn_s_waitcnt(0)` (LDS-swap barrier) + the page-index resolve's
real data-dep. REVERTED the clobber change (no-win, risky shared macro).
CONCLUSION: the global load is FEED-BOUND at the 2-LDS-buffer limit. The tile-top vmcnt(0) genuinely waits
for the current tile's KV load (issued the prev iter) -- nothing else is in flight at that point, so it is
NOT over-conservative; the load simply takes longer than one tile's compute. Deeper prefetch (>1 tile
ahead) needs a 3rd LDS buffer (2*73728=147KB fits in 160KB, 3rd doesn't). asm fits the same 2 buffers but
hides it by NOT draining (continuous vmcnt~10 streaming with partial per-ds_read waits) -- a hand-scheduled
pipeline HK's drain+barrier structure can't replicate without removing the ds_read "memory" clobbers AND
manually managing every vmcnt/lgkmcnt (asm-level, fragile -- the documented floor). NEGATIVE recorded so
the clobber lever isn't re-tried. Levers exhausted on the global load short of the full hand-schedule.

#### STAGE 12 (2026-06): BREAKTHROUGH -- manual s_waitcnt vmcnt(K) IS respected -> asm streaming IS achievable from HIP
Built a standalone probe (/home/mh/aiter/_vmcnt_probe.cu) mirroring the pipelined buffer_load_lds(next
tile) -> ds_read(cur tile) loop with 2 compile-time-distinct LDS buffers. Compiled device asm
(hipcc --offload-arch=gfx950 -O3 --cuda-device-only -S -DVARIANT=N) and inspected the loop:
| VARIANT | manual wait written | emitted in loop |
|---|---|---|
| 0 | s_waitcnt vmcnt(0)        | vmcnt(0) |
| 1 | s_waitcnt vmcnt(LPT)     | **vmcnt(LPT)** -- exactly mine, NO extra drain |
| 2 | none                     | NO vmcnt at all |
KEY: an inline-asm `s_waitcnt vmcnt(K)` is emitted VERBATIM and the compiler adds NO extra vmcnt(0) --
EVEN with the ds_read `"memory"` clobber present (tested -DMEMCLOBBER=1: still just vmcnt(K)). And with no
manual wait + no clobber, the compiler inserts NOTHING (VARIANT 2 -> would be incorrect). So:
=> asm-style continuous streaming (hold vmcnt~10, keep ~1 KV tile in flight) IS achievable from HIP/HK by
   MANUALLY managing s_waitcnt. The compiler does not fight it. This overturns the STAGE-11 "documented
   floor / not closable from source" framing for the vmcnt axis.
=> Also reconciles STAGE-11: removing the ds_read "memory" clobber didn't change the kernel's vmcnt count
   because those vmcnt(0) are the EXPLICIT __builtin_amdgcn_s_waitcnt(0) (tile-top drain) + the page-index
   resolve -- both replaceable by manual partial vmcnt. The shared macro does NOT need changing.

RECIPE to stream the kernel KV load (the remaining hand-schedule, now de-risked at the mechanism level):
1. The tile-top `__builtin_amdgcn_s_waitcnt(0)` must become a manual `s_waitcnt vmcnt(K) lgkmcnt(0)` that
   keeps the next tile's loads in flight (the s_barrier only needs lgkmcnt(0)).
2. Issue the next tile's KV loads BEFORE that wait so the wait can hold them (vmcnt(K)) -- i.e. prefetch
   one tile ahead at the tile top.
3. The page-index (get_kv_ld_row) currently forces its own vmcnt(0) (load result used immediately) and
   drains the KV. Pipeline it OFF the tile boundary: carry next_phys_row computed in the PREVIOUS iter
   (2-deep), so the tile-top has no page-index vmcnt(0). (Caching p_kv_indices in LDS does NOT generalize:
   large-ctx single-split needs up to ctx int32 = >LDS.)
4. Tune K (loads-in-flight) empirically toward asm's vmcnt(10). Verify correctness incl. multi-split
   large-ctx (the cells that caught the page-index memfault). The vmcnt count K is fragile -- get it from
   the actual per-tile load count.
Probe file kept at _vmcnt_probe.cu for re-validation. NEXT: apply recipe to mla_main (fragile, staged).

#### STAGE 12b (2026-06): kernel application of steps A+B -- both REVERTED; blocker = compiler false-alias vmcnt(0)
Applied the recipe to mla_main, verified correctness on every config incl. multi-split large-ctx (no
memfault), then clean-idle min-of-3 -- BOTH net-negative, reverted to the load-spread known-good (B64
47.58/1.12x, B128 73.85/1.15x):
- STEP A (carry page-index row 2-deep, mirror fp8 next_next): REGRESSED 1.12->1.16x (B64), 1.15->1.23x
  (B128). The carried-row RECOMPUTE (get_kv_ld_row for kv_tile_start+2*kBlockN) placed in the RoPE loop
  does a vmcnt(0) MID-QK that drains the in-flight next-tile SPREAD. The ORIGINAL top-resolve is actually
  optimal: its vmcnt(0) drains only the CURRENT tile (needed for QK anyway) and runs BEFORE the spread
  starts, so it never drains the spread. => the page-index is already well-placed; don't move it.
- STEP B (burst next tile at tile-top + manual s_waitcnt vmcnt(18)): correct, and vmcnt(18) IS emitted
  (19 sites in ISA). BUT the compiler inserts a FALSE-ALIAS `s_waitcnt vmcnt(0)` IMMEDIATELY AFTER my
  vmcnt(18), before the QK ds_read -> drains the burst -> streaming defeated. ROOT: p_lds_kv_curr /
  p_lds_kv_next are runtime-swapped uintptr; the compiler cannot prove the burst's buffer_load_lds
  (writes next) does not alias the ds_read (reads curr), so it serializes via vmcnt(0). This is
  LDS-ADDRESS-SPACE aliasing on the buffer_load_lds builtin -- NOT the ds_read "memory" clobber
  (removing the clobber did NOT remove the vmcnt(0); re-confirmed). The STAGE-12 probe avoided it only
  because it used a `__shared__ lds[2][]` ARRAY the compiler resolves as provably-distinct.
=> THE REMAINING UNLOCK: make the two KV LDS buffers COMPILE-TIME PROVABLY DISTINCT so the compiler drops
   the false-alias vmcnt(0) -- e.g. a 2-elem `__shared__` base array indexed by `lds_buf[parity]` /
   `lds_buf[parity^1]` (parity ^= 1 each iter, no std::swap of uintptr; the compiler proves i != i^1), OR
   unroll-the-tile-loop-by-2 with fixed compile-time A/B. THEN step B's vmcnt(K) streaming holds. This is
   a contained-but-pervasive change (curr/next threads through every load_k/load_v/async/output site).
   Manual-vmcnt mechanism is proven (probe); the buffer-distinctness is the last gate. Tree: reverted to
   load-spread known-good (steps A+B removed; ds_read clobber restored; HK_STREAM_BURST removed).

#### STAGE 12c (2026-06): vmcnt(0) ISOLATED + step B MEASURED worse -> streaming defeated, floor confirmed
ISOLATED the false-alias vmcnt(0) (re-enabled step B alone, original top-resolve, full unfiltered disasm):
the steady-state first QK K ds_read (`ds_read_b128 a[224:227], v1`) is preceded by my `s_waitcnt vmcnt(18)`
THEN a compiler-inserted `s_waitcnt vmcnt(0)`. (The FIRST vmcnt(18) site is the prologue Q-load into
a152-223 -- a different, legit vmcnt(0); the K-load sites a[224 are the steady-state defeater.) Cause: the
compiler's conservative LDS-alias ordering between the burst (buffer_load_lds -> p_lds_kv_next) and the K
ds_read (-> p_lds_kv_curr), whose addresses are OPAQUE after the loop-carried std::swap. 
PROBE NON-REPRO: _vmcnt_probe2.cu (uintptr dest, swap MODE0 / parity MODE1 / array MODE2) emits ONLY
vmcnt(4) in ALL modes -- the swap does NOT itself trigger the vmcnt(0) in isolation (the probe's pointers
stay analyzable). So the trigger is the full-kernel structure (load_k_to_gpr / async_load_kv_cols_bf16 as
separate inlined fns computing addresses from the swapped opaque p_lds_kv_curr/next), which the probe
doesn't reproduce -> can't validate a fix in isolation.
DECISIVE MEASUREMENT (idle GPU, min-of-3, c1200): step B (burst@top + vmcnt(18)) = B64 55.2us (~1.30x),
B128 88.65us (~1.38x) -- WORSE than the spread's 1.12-1.15x. Two compounding losses: (a) the compiler
vmcnt(0) drains the burst (streaming defeated), AND (b) the burst loses the spread's FIFO-avoidance.
=> CONCLUSION: asm-style KV streaming is mechanistically possible (probe) but DEFEATED in the real kernel
by the compiler's conservative vmcnt(0) (opaque swapped LDS pointers across inlined load/read fns). The
only theoretical fix (compile-time provably-distinct buffers) is unvalidated (probe can't repro the bug)
AND even if it dropped the vmcnt(0) the burst measured worse than the spread -- so NOT worth the large
pervasive refactor. The spread-during-QK (STAGE-10c) remains the best source-level KV pipeline at
1.12-1.15x. This is the measured hard floor for the global-load axis. Reverted to known-good; probes
(_vmcnt_probe.cu, _vmcnt_probe2.cu) kept for future reference.

### NET SESSION RESULT (STAGE 10): HK bf16 qh64 1.13-1.26x -> 1.12-1.15x of asm (ATT-driven)
Installing the ATT decoder was the unlock: it corrected the STAGE-9 PMC mis-attribution (gap is the KV
VMEM-load path, not MFMA/softmax) and led to the load-spread fix. Levers: M0-uniform (correct, neutral
alone), KV load-spread (the win). Remaining: compiler-conservative vmcnt(0) before ds_read -> needs
asm-level explicit vmcnt management. The cross-tile MFMA-overlap rewrite (STAGE 9b plan) was correctly
ABANDONED -- ATT showed MFMA is only ~10% of stall.

#### STAGE 12e (2026-06): compile-time-distinct LDS buffers -> DROPS vmcnt(0) but ~4% SLOWER -> vmcnt(0) confirmed NOT critical-path
GOAL: remove the compiler false-alias vmcnt(0) (STAGE 12c) by making the two KV ping-pong slots
compile-time-constant disjoint addresses instead of runtime std::swap pointers.
EXPERIMENT (no-swap, correctness-be-damned, one-variable bisect, burst+manual vmcnt(18) held fixed):
removing std::swap makes the steady-state K ds_read base resolve to a FIXED VGPR (v68/v81) and the
compiler vmcnt(0) before `ds_read_b128 a[224:227]` DISAPPEARS. swap-on: `vmcnt(18); vmcnt(0); ds_read..v1`.
swap-off: `vmcnt(18); ds_read..v68`. ROOT CAUSE CONFIRMED: runtime std::swap of the uintptr_t LDS pointers
defeats LLVM LDS alias analysis -> conservative vmcnt(0).
REAL FIX IMPLEMENTED (correct, all configs incl 64K/16K multi-tile + partial boundaries 257/1023):
templated the hot body `mla_body<bool kCurrIsB,...>` on a compile-time parity bit (curr/next = fixed
p_lds_kv_a/b), wrapper `mla_main` dispatches `if(pong) body<true> else body<false>` and flips a runtime
`pong` bool instead of swapping pointers. Prologue keeps runtime curr/next driven by pong. vgpr 512,
scratch 0 (no spill). Disasm CONFIRMS vmcnt(0) gone at steady-state K ds_read (fixed base).
A/B MEASURED (same GPU session, 3 reps, both vgpr512/scratch0):
  b16 c65536: knowngood ~272.7us -> candidate ~284us (+4.5%)
  b32 c16384: ~158.5us -> ~165.3us (+4.3%)
  b16 c8192 : ~58.5us  -> ~60.8us  (+3.9%)
=> CONSISTENT ~4% REGRESSION. The body-duplication (if(pong) -> two instantiations + per-tile runtime
branch) breaks the compiler's cross-tile software-pipelining of next-tile prefetch into current compute;
the removed vmcnt(0) was already overlapped under that pipeline, so removing it gained nothing while the
duplication cost ~4%. DECISIVE: the vmcnt(0) is NOT on the critical path. Reverted to known-good spread.
Candidate saved as mi35x_..._bf16.cuh.cand_distinct; backup .knowngood_spread.

#### STAGE 13 (2026-06): ATT wall-util re-profile (corrected) + HK-vs-asm head-to-head diff
RE-PROFILED known-good with ATT wall-util (Latency/Stall/Idle = ground truth, not stall-counters).
PITFALL FOUND: aggregate "vmcnt 72%" was TEST-HARNESS POLLUTION -- a 144-inst `__amd_rocclr_copyBuffer`
(hipMemcpy, load->vmcnt(0)->store, n=56784) captured by --att-consecutive-kernels, NOT the decode op.
FILTER ATT by codeobj_id to the decode kernel (cid=28) only. Real per-call times (--kernel-trace):
  HK decode `kn_mi35x_mla_v32_fwd_decode_m16x4_bf16_bf16`  = 81.8us
  asm decode `aiter::mla_a16w16_qh64_qseqlen1_gqaratio64_v3_ps` = 69.3us  (gap 1.18x)
  shared combine `kn_mla_reduce_v1` = 8.1us (both paths)
HK decode (cid=28) wall-util: MFMA 8.1% (NOT compute-bound), Stall 34%, Idle 7%; stalls SPREAD
(lgkmcnt 22.8%, vmcnt 21.4%, mixed 13.9%, vmem 13.8%, mfma-chain 12.5% of stall) -> no single bottleneck.
HEAD-TO-HEAD DIFF (absolute ATT cycles; walls comparable: HK 7.99M vs asm 6.71M = 1.19x ~ time gap):
  KEY: HK explicit s_waitcnt total 1.59M vs asm 0.30M (+1.30M) BUT offset by HK vmem_load 0.42M vs asm
  1.51M (-1.09M). asm accounts memory latency ON the load (overlapped); HK pays it as scattered
  s_waitcnt drains. NET memory cost nearly equal (HK +206K = only 16% of gap). => CONFIRMS A/B:
  chasing waitcnt/vmcnt is LOW-VALUE.
  REDUCIBLE EXCESS (HK - asm latency): SALU +460K (~36% of gap, ~4.5x asm, almost pure latency);
  MFMA +281K of which +264K is STALL (~21% of gap, HK MFMA chain stalls 4.3x asm). valu -161K and
  vmem_load -1090K HK is "better" (accounting). => TWO real targets: (1) SALU bookkeeping, (2) MFMA
  chain interleave. NEITHER needs the fragile deep-pipeline rewrite.

#### STAGE 13a (2026-06): SALU hot-spot = per-col-tile if(oob) branch inside the KV-load static_for
DRILLED HK SALU (cid=28, 589K lat vs asm 129K). In-loop (hit 200-2000) = 79.5% of SALU lat:
  s_mov_b32 101K, s_cbranch_execz 73K, exec-mask (s_*_saveexec/xor/or/and_b64) ~160K, s_add_u32 33K.
Pattern (ATT context): each KV `buffer_load_dwordx4 ... lds` is wrapped per-col-tile with
  {s_mov s18,s22; s_mov s19,s23; s_mov m0,s6; s_nop 0} (descriptor+M0 reload) +
  {s_*_saveexec; s_cbranch_execz; ...ds_write zero-fill...; s_or exec} (predicated oob branch).
ROOT: `async_load_kv_cols_bf16` (hk_mla_buffer_managers.cuh ~1292) has `if(oob){ds_write zeros}else
{buffer_load_lds}` INSIDE the `static_for<kCtCount>` -> branch + exec-mask + descriptor reload emitted
PER col-tile (x2 spread, x18 burst) though `oob`(=phys_row<0) is uniform-per-call. asm has flat
shape-specialized control flow -> ~4.5x less SALU.
HYPOTHESIS (to test): hoist the oob branch OUTSIDE the static_for: `if(oob){for ct: zero-fill}
else{for ct: load}`. Should remove per-col-tile branch+exec-mask (and ideally hoist the descriptor
s_mov). NEXT: verify oob uniformity, implement hoist, build, re-ATT SALU, A/B time.

UNIFORMITY VERIFIED: phys_row comes from get_kv_ld_row_base_idx16 = warp_idx*kSubBlockRows16 +
(lane_idx>>2) -> PER-LANE, so oob is divergent (matches `s_and_saveexec_b64 vcc` in disasm, not a
scalar branch). BUT oob is CONSTANT across the kCtCount col-tiles -> the per-ct re-emit is pure waste.

FIX IMPLEMENTED (STAGE 13a): templated `async_load_kv_cols_bf16<kCtStart,kCtCount,bool kCheckBoundary>`
(+ forward through async_load_kv_tile_bf16; spread call sites pass mla_body's kCheckBoundaryNext).
Body: `bool do_zero=false; if constexpr(kCheckBoundary) do_zero=(phys_row<0); if(do_zero){for ct:
zero-fill} else {for ct: branchless buffer_load_lds}`. When kCheckBoundaryNext==false (full tile, all
phys_row>=0 guaranteed by dispatcher) the loads are FULLY BRANCHLESS; the rare boundary tile takes one
hoisted divergent branch for the whole unrolled loop. Files: hk_mla_buffer_managers.cuh (func),
mi35x_..._bf16.cuh (3 call sites: prologue burst stays <true>, 2 spread sites pass kCheckBoundaryNext).

RESULT (MEASURED, correct all configs incl boundary 257/1023 + multisplit 65536):
  ATT cid28: SALU latency 522,872 (6.5% of wall) -> 220,484 (2.8%) = -302K (-58%). exec-mask churn
  (s_andn2/and_saveexec, s_xor_b64) GONE; s_cbranch_execz 90K(n602)->6.9K(n334); s_mov_b32 120K->47K.
  A/B (same session, 3 reps each):
    b16 c65536: knowngood ~273.1us -> SALU-hoist ~256.9us  (-5.9%)
    b32 c16384: ~158.0us -> ~149.0us  (-5.7%)
    b16 c8192 : ~58.4us  -> ~56.2us   (-3.8%)
  => REAL consistent ~4-6% speedup. Gap to asm decode ~1.18x -> ~1.11x. PROMOTED to known-good
  (backup .knowngood_spread updated; candidate also at .cand_salu). This removed ~64% of the +460K
  SALU excess vs asm (STAGE 13). Remaining reducible target: MFMA chain stalls (+264K, ~21% of gap).

#### STAGE 13b (2026-06): re-profile post-SALU-fix + CORRECTION: MFMA stall = LDS-FEED, not chain
Re-ATT (hk2.json, cid28) + re-diff vs asm. SALU excess +460K -> +140K (target closed). NEW top
reducible (CORRECTS STAGE 13's "MFMA chain" label):
  MFMA stall attribution: accumulator-chain 0%, **LDS-feed (lgkmcnt just before mfma) 83%**, other 17%.
  lgkmcnt wait gates: v_mfma 370K + ds_read_b64_tr_b16 147K. 1080/2040 MFMAs (53%) have an lgkmcnt
  wait within <=2 insts before them. Disasm shows ds_read a[224:239] issued 1-4 insts before the
  consuming v_mfma -> s_waitcnt lgkmcnt(2) stalls ~21K cyc. ROOT: LDS-read -> MFMA distance too short
  (operands read just-in-time), NOT accumulator chain. asm issues ds_read_b64_tr_b16 far ahead
  (STAGE-2: a[72:207] ~135-AGPR rolling window, ~1:1 interleave -> LDS latency hidden).
TWO LDS axes clarified: (1) global->LDS TILE ping-pong (2 KV tiles): HK (p_lds_kv_a/b) == asm (2 LDS
buffers in 160KB) -- ALREADY MATCHED. (2) LDS->register OPERAND feed (read next mfma group's operands
while current mfmas run): asm YES, HK NO -- THIS is the gap. HK uses dynamic extern __shared__ (kd
group_segment_fixed_size=0); asm uses static 163840.

REGISTER HEADROOM CHECK (CRITICAL, gfx950 cdna4: pool <=512 dword total, <=256 each, waves=floor(512/(V+A))):
  HK .kd: V=256 + A=256 = 512 -> 1 wave/SIMD (asm also 1 wave; occupancy is NOT the lever).
  ACTUAL usage (max-index scan of hk2.json cid28): arch VGPR 232 distinct/256 alloc (accumulators
  v[112:127].. -- nearly full); AGPR used a[152:255]=104, **AGPR FREE a[0:151]=152 contiguous slots**.
  Accumulators live in ARCH VGPR; operands (K/Q/V) in AGPR via ds_read_b128 a[...]. The wave already
  owns all 256 AGPR -> a[0:151] is dead allocation, FREE at ZERO occupancy cost. 152/16 ~= 9 K-groups
  of look-ahead capacity (asm uses ~135 AGPR window; HK has MORE room than asm, just doesn't prefetch).
  => LDS-read look-ahead (technique 1.7) is FEASIBLE register-wise. Constraint is SCHEDULING (issue
  next group's ds_read into a[0:151] before current MFMAs), not budget. Risk: hk::art<> range
  abstraction maps operands to a[152:255]; must declare look-ahead tiles into a[0:151] + verify the
  allocator places them low. NEXT: prototype 1-deep K ds_read look-ahead in QK loop, A/B.

#### STAGE 14 (2026-06): QK upper-half LDS-read look-ahead -> reduces lgkmcnt 33% but PERF-NEUTRAL
Implemented within-iteration upper-half K look-ahead in the QK NoPE loop: issue the upper N-half (rows
32..63) K ds_reads at the TOP (with the lower half, 8 ds_read in flight) so they overlap the lower
MFMAs, then MFMA the upper half from the look-ahead regs. lgkmcnt retuned 6/4/2/0 (was 2/0/2/0).
DEDICATED look-ahead tiles kv_*_la at a136..a151 (top of the free AGPR region) -- NOT reusing kv_*_alt
(PV's V-upper buffers) to avoid a false QK<->PV register dependency (user steer).
SPILL CHECK (the key question -- a0..a151 was reserved as compiler AGPR scratch): build shows
private_segment_fixed_size=0, vgpr_spill_count=0, sgpr_spill_count=0. => Using a136..a151 for tiles
does NOT trigger spill; the scratch reservation was OVER-CONSERVATIVE for this build.
RESULT (correct all configs incl 257/1023/65536): ATT lgkmcnt_wait 642K -> 433K (-33%), BUT mfma_stall
343K -> 354K (unchanged) and mfma-with-lgkmcnt-before count 1080/2040 UNCHANGED. A/B (3-rep): c65536
256.9->255.8, c16384 149.0->148.4, c8192 56.2->56.0 -- all within ~0.4% = NEUTRAL.
=> SAME LESSON AS vmcnt(0) (STAGE 12e): reducing a stall COUNTER (lgkmcnt -33%) does NOT reduce wall
when it's already overlapped. The mfma stall did NOT drop with the lgkmcnt wait -> STAGE-13b's "mfma
stall = 83% LDS-feed" was a CORRELATION, not cause; the mfma stall has another cause (accumulator/p_comp
dependency or MFMA issue latency, or the kernel is HBM-latency bound with everything else overlapped at
occupancy-1). REVERTED to SALU-baseline known-good (neutral + adds ~40 lines). Candidate saved
.cand_lookahead. Valuable byproduct: a136..a151 is spill-free usable for future register experiments.

#### STAGE 15 (2026-06): carried page-index prefetch -> -4.8% (THE win user predicted). gap 1.12x->~1.07x
User insight: the page-index `vmcnt(0)` is not just a dependent-load stall -- because AMD vmcnt is a
SINGLE counter for ALL VMEM, that drain ALSO flushes the in-flight KV prefetch every tile, breaking the
streaming pipeline. ATT (STAGE-13b deep dive) confirmed: at the tile boundary, `buffer_load_dword`
(p_kv_indices, via get_kv_ld_row -> raw_buffer_load_b32) then `s_waitcnt vmcnt(0)` = 391K cyc = 68% of
all vmcnt stall = ~5% of wall, issued-then-immediately-waited (zero overlap), gating the next KV load
address.
FIX: resolve each tile's next phys_row ONE TILE AHEAD and carry in a VGPR (`carried_next_phys_row_bf16`,
captured by-ref in mla_main). mla_main(tile N) uses carried (=resolve(N+1) done last call) for the
spread, then issues resolve(N+2) for the next call. Init in prologue = resolve(tile1). Bounds-checked
(<true> -> OOB lanes -1, no OOB p_kv_indices read -> avoids the STAGE-10d memfault). The load issued at
tile N is consumed FOR FREE by the existing top `__builtin_amdgcn_s_waitcnt(0)` at tile N+1 (no dedicated
drain), latency hidden under a full tile of compute.
RESULT (correct ALL configs incl boundary 257/1023 + multisplit 65536/8192-B1):
  A/B 3-rep: c65536 256.9->245.6 (-4.4%), c16384 149.0->141.9 (-4.8%), c8192 56.2->55.0 (-2.1%).
  Gap to asm (test total, decode+combine): b16 c16384 1.087x, b32 c16384 1.069x (was ~1.12x).
  NOTE: ATT vmcnt(0) total looked HIGHER post-fix (sampling noise -- the tile-boundary barrier absorbs
  warp-skew differently); REAL 3-rep timing is the truth and is clearly faster. (Reconfirms: trust
  wall A/B over ATT stall-counter deltas.)
WHY prior page-index attempts (STAGE-10d, "step A") failed but this didn't: 10d issued the load
UNCONDITIONALLY (memfault) + measured pre-ATG-decoder; this uses the bounds-checked resolve and relies
on the EXISTING top s_waitcnt(0) to consume the carry (no added drain). PROMOTED to known-good
(.knowngood_spread updated; .cand_carried_idx saved).
SESSION ARC: 1.18x -> SALU-hoist 1.12x -> carried-index ~1.07x. Two real WORK/serialization wins;
the stall-counter chases (vmcnt0 STAGE12e, lds-lookahead STAGE14) were all neutral.

#### STAGE 16 (2026-06): 2-way-unroll compile-time-distinct buffers -> ~+2% WORSE (3rd dead-end confirm)
Hypothesis: STAGE 12e (compile-time-distinct, drops the swap vmcnt(0)) was neutral/+4% only because the
per-tile `if(pong)` runtime branch broke cross-tile pipelining. Fix: 2-way-unroll the steady-state middle
loop with COMPILE-TIME parity (mla_body<kCurrIsB>; body<B>;body<A> straight-line, no per-tile branch),
runtime `cur_is_b` only for cold first/last/skip paths (via mla_main wrapper). Fixed A/B buffers, no swap.
IMPLEMENTED + CORRECT all configs (incl 257/1023/65536/b1-8192), no spill. Disasm CONFIRMS mechanism:
steady-state `ds_read_b128 a[224:227]` now has fixed base v50 and NO preceding vmcnt(0) (swap-alias drain
gone). BUT 3-rep A/B vs carried-index baseline: c65536 245.6->249.4 (+1.5%), b32 c16384 141.9->144.6
(+1.9%), c8192 55.0->56.1 (+2.0%) = consistently ~2% WORSE.
=> DEFINITIVE (3rd independent confirm: STAGE 12c manual-vmcnt, 12e if(pong), 16 unroll): the
swap-boundary vmcnt(0) is OVERLAPPED / not-critical-path. The 478K boundary drain (STAGE-ATT) is
dominated by the NECESSARY line-571 KV-prefetch-completion wait (a full tile's KV must land before the
next tile's ds_read; only ~1 tile of compute to hide it at occupancy-1), which compile-time-distinct does
NOT remove. Meanwhile the body duplication (2 instantiations) costs ~2%. STOP pursuing compile-time
buffers / swap-vmcnt removal -- it cannot beat the carried-index baseline. Reverted (.cand_unroll saved).
The real KV-load floor needs DEEPER prefetch (2 tiles ahead, asm vmcnt(10-20)) but that needs a 3rd LDS
buffer which does NOT fit (D=576 -> 2 tiles = ~152KB of 160KB). So ~1.07x is the structural HIP floor for
this 2-LDS-buffer design; closing further = asm's from-scratch deep-pipeline (out of scope).

#### STAGE 17 (2026-06): READ THE ASM. It is NOT split-KV -- same cooperative design, hand-scheduled vmcnt(10)
Disassembled mla_a16w16_qh64_qseqlen1_gqaratio64_v3_ps.co (gfx950). Resources: 4 warps, 1 WG/CU, vgpr
512, FULL 160KB static LDS, sgpr 96, no spill. Kernargs: separate O,Q,K,V,BT(block table),CL,KQ,sclg...
CORRECTS the earlier "asm must be per-warp split-K" scope: asm uses 128x `buffer_load_dwordx4 ... lds`
(cooperative KV->LDS via M0) + broadcast `ds_read` -- the SAME design as HK. NOT split-KV.
THE STREAMING (asm vmcnt dist: vmcnt(10)x11, (20)x1, (0)x7 vs HK 115 vmcnt(0)):
  issue ~9x buffer_load_dwordx4 lds (FIXED descriptor s[20:23]; `s_add_i32 m0,m0,0x3c0` increment, NO
  s_mov/s_nop per load) -> `s_waitcnt vmcnt(10); s_barrier; ds_read a[72:143]` (18 col-tiles) -> MFMA
  loop INTERLEAVES 1 v_mfma : 1 buffer_load_dwordx4 lds (next tile) ~1:1.
KEY: `vmcnt(10); s_barrier` keeps ~10 next-tile loads STREAMING across the barrier (not a vmcnt(0) drain),
correct because loads are ordered so the CURRENT tile completes first; the 10 in flight are the next tile,
loaded into the buffer freed after K was ds_read to AGPRs upfront. asm separates K/V pointers -> reads K
to regs early -> frees the K LDS region -> streams next K while V is independent.
THREE deltas (all SAME algorithm, NO split-KV): (1) vmcnt(10) streaming vs HK vmcnt(0) drain [dominant];
(2) fixed-descriptor + M0-increment issue vs HK per-load s_mov descriptor+m0+s_nop; (3) hand-scheduled
1:1 mfma:load interleave. HK blocker = HIP compiler emits vmcnt(0) where asm hand-writes vmcnt(10).
STAGE-16 compile-time-distinct already removed the compiler ds_read vmcnt(0); remaining = replace the
explicit tile-top __builtin_amdgcn_s_waitcnt(0) with manual vmcnt(K) + have next-tile loads in flight
across the barrier (needs K read to regs upfront so the buffer frees, like asm).

#### STAGE 18 (2026-06): PER-WAVE TEMPORAL TIMELINE analysis (att.md \u00a73b) -- settles vmcnt critical-path doubt
Used the rocprofv3 ATT per-wave timeline (ui_output/se*_sm*_sl*_wv0.json: wave.instructions =
[ts,cat,stall,dur,static_idx]; static_idx -> code.json ISA) to attribute CRITICAL-PATH stall per ISA
line (occupancy-1 -> per-wave stall == wall for that wave). Method added to att.md \u00a73b.
CORRECTION: my earlier "vmcnt is overlapped" was WRONG. Per-wave (carried-index): 63% of wave is stall;
of that vmcnt_wait 27%, vmem_load-issue 24%, lgkmcnt 20%, mfma 11%. vmcnt(0) IS on the per-wave critical
path (one recurring static_idx dominates).
3-BUILD TEMPORAL A/B (wave_dur / MEMORY-stall=vmcnt+lgkmcnt+vmem):
  carried-index: 153848 / 69072 (vmcnt 26416, lgkm 19296, vem-issue 23360)
  cand_unroll (vmcnt0 removed): 147132 / 62112 (vmcnt 22132, vem-issue 19616)
  ceiling (vmcnt8 stream): 148528 / 68060 (vmcnt 18336, vem-issue ROSE to 29016)
TWO DECISIVE FINDINGS:
 (1) Streaming RELOCATES, does not reduce: vmcnt(8) cut vmcnt_wait but RAISED vmem_load-issue stall (wave
     parks ON the buffer_load = FIFO/throughput full). Total MEMORY stall stays ~62-69K in ALL builds ->
     it is a HARD memory floor (KV bytes take the same time at occupancy-1; the wait just moves vmcnt<->
     load-issue). asm loads the SAME KV bytes -> same floor -> asm's edge is NOT memory.
 (2) Per-wave critical path does NOT track kernel wall: cand_unroll has the LOWEST wave_dur (147K) yet is
     +2% SLOWER in 3-rep kernel wall. => kernel is THROUGHPUT-bound (i-cache/issue across waves; the 2x
     body duplication hurts), and that cost is INVISIBLE in a single wave's stall trace.
=> FINAL: both levers blocked. Memory stall = bandwidth/latency floor (streaming can't reduce, only
relocate). Kernel wall = throughput-bound where vmcnt0-removal costs more (duplication) than it saves.
asm's 1.07x edge = compact hand-scheduled throughput (fewer instrs/barriers), unreachable in HIP without
writing asm. CAVEAT learned: per-wave ATT timeline proves per-wave critical path but is NOT a kernel-wall
proxy (throughput effects like i-cache are invisible) -- always confirm with 3-rep wall A/B. Reverted to
carried-index (~1.07x) known-good.

#### STAGE 19 (2026-06): ASM per-wave timeline -> the gap is LDS->MFMA FEED, not vmcnt/global-load
Captured the ASM kernel's ATT per-wave timeline (same cpath.py attribution) and compared 4 builds
(b16 c16384, per-wave critical-path stall cyc):
  build           wave_dur  total  vmcnt  lgkmcnt  vem-issue  mfma
  carried-index   153848    96320  26416  19296    23360      11000
  cand_unroll     147132    87288  22132  20364    19616      10732
  ceiling vmcnt8  148528    91380  18336  20708    29016      10908
  ASM             133520    73680   1948   8464    44724       2656
ASM profile = PURE BANDWIDTH FLOOR: vmcnt~0 (3%), mfma~0 (4%), all stall on vmem_load-ISSUE (61%, wave
parks on buffer_load = HBM BW throttle). asm streams so perfectly the load FIFO is the only bottleneck.
DECISIVE absolute-cycle excess (HK carried - asm):
  vmcnt+vem-issue: HK 49776 vs asm 46672 -> ~EQUAL (global-load is at BW parity; HK's vmcnt drains are
    mostly BW that asm accounts on the load-issue instead -- NOT reducible waste).
  lgkmcnt: HK 19296 vs asm 8464 -> HK +10832
  mfma:    HK 11000 vs asm  2656 -> HK + 8344
=> THE GAP IS THE LDS-READ -> MFMA FEED (~+19K cyc/wave): asm interleaves ds_read 1:1 FAR AHEAD
(a[72:207] ~135-AGPR operand window, reads all 18 K col-tiles upfront) so the matrix core NEVER starves
(mfma stall 2656); HK reads K just-in-time -> mfma stalls 4x (11000) + lgkmcnt 2x (19296). This CORRECTS
the STAGE-12..18 emphasis on vmcnt: the vmcnt/global-load axis is ~at asm parity; the real remaining
~7% is the operand-feed pipeline. STAGE-14 (upper-half look-ahead) was the RIGHT direction but only
1-deep + measured wall-neutral; the asm target (lgkmcnt 8464, mfma 2656) is the metric to drive toward.
PUZZLE answers: (1) cand_unroll wave_dur down but kernel +2% = removing vmcnt0 shortened per-wave path
but 2x-body i-cache hurt kernel throughput (per-wave != wall). (2) stream vmcnt down vem-issue up =
relocation toward asm's load-issue shape (same total memory; asm lives there AND killed mfma/lgkmcnt).
NEXT: deepen QK ds_read-ahead (read more K col-tiles upfront into free a0..a151) to drive per-wave
lgkmcnt+mfma toward asm's 8464/2656; validate with per-wave timeline AND 3-rep kernel wall.

#### STAGE 20 (2026-06): QK upper-half feed look-ahead on carried-index -> per-wave -8.8% but WALL FLAT
Re-applied STAGE-14 upper-half K look-ahead (kv_*_la in free a136..a151) on the carried-index base.
Correct all configs, no spill. Per-wave timeline (b16 c16384): wave_dur 153848->140280 (-8.8%), total
stall 96320->80404, lgkmcnt 19296->13752 (toward asm 8464), vmcnt 26416->19380, vem-issue 23360->19516.
BUT: mfma stall UNCHANGED (11000->11260) -- the look-ahead fixed the lgkmcnt feed but NOT the mfma
starvation (asm 2656); ds_read stall ROSE 3736->8580 (the extra reads). 3-rep WALL: c65536 245.6->246.4,
b32 c16384 141.9->141.3, c8192 55.0->54.4 = FLAT (within noise). 5th confirmation that per-wave critical
-path != kernel wall.

### META-CONCLUSION (STAGE 12-20): this kernel's WALL is THROUGHPUT-bound, not stall-bound
Across every experiment the rule holds: changes that reduce dynamic WORK/instructions/serialization move
the wall; changes that reduce per-wave STALL but add instructions do not.
  WORK-reduction WINS: SALU-hoist (-5%), carried page-index (-4.8%, removes a dependent load).
  STALL-reduction NEUTRAL/WORSE: vmcnt removal via compile-time-distinct (cand_unroll +2%, adds 2x body),
    LDS feed look-ahead (STAGE-14/20, flat, adds ds_reads), vmcnt(K) streaming (relocates), all the
    vmcnt(0) chases (12c/12e/16/18).
WHY: occupancy-1 persistent kernel; wall = instruction-issue throughput across the workload, NOT a single
wave's critical path (per-wave wave_dur can drop 8.8% with zero wall change). asm wins by being COMPACT
(748 vs 2040 static mfma; specialized single path; tight 1:1 interleave) = less work AND less stall, plus
mfma-starvation (asm 2656 vs HK 11000) is accumulator/issue-chain, only fixable by asm's hand-schedule.
=> The lever for further HIP gains is WORK/INSTRUCTION REDUCTION, not stall reduction. Remaining ~7% to
asm needs the compact hand-scheduled rewrite (out of scope). 1.07x carried-index is the HIP floor.
Reverted to carried-index (.cand_feedla saved).

#### STAGE 21 (2026-06): tighten QK global-prefetch interleave (1:1) -> +2% WORSE (6th confirm of the rule)
PMC bound check (carried-index b16 c16384, HK decode): L2 hit = TCC_HIT/(HIT+MISS) = 42.2M/323.7M = 13%
(KV streamed from HBM, no reuse); HBM BW ~26-52% of 8TB/s (NOT saturated). => memory-LATENCY-exposed
streaming, not compute and not BW-saturated. (Retracts the loose "throughput-bound" wording; the per-wave
!= wall observation stands but is most likely sampled-wave != wall-determining work-item, not proven.)
MFMA interleave audit: PV loop is ALREADY asm-like (4 independent accumulators oaccu_0_a/b,1_a/b +
mul_pair softmax-rescale + V-prefetch woven between mfmas). QK loop chains into 2 accumulators
(p_comp_lo/hi, 2-deep) with the next-tile global prefetch spread ~2 cols/iter (coarser than asm's 1:1).
ATTEMPT: split the batched async_load_kv_cols_bf16<tile_idx,2> into <tile_idx,1> + <tile_idx+1,1> woven
between the two lower mma_ABt calls (toward asm continuous 1:1). Correct all configs. 3-rep wall: c65536
245.6->249.7, b32 c16384 141.9->145.6, c8192 55.0->55.5 = +1.7..2.6% WORSE. Per-wave: vem-issue ROSE
(23360->25736), total memory slightly down -- but wall worse.
ROOT: each async_load_kv_cols_bf16 call re-sets m0 + buffer descriptor (compiler lowers
__builtin_amdgcn_raw_ptr_buffer_load_lds to s_mov m0 per call). Splitting 1 call into 2 DOUBLES that
setup = added WORK -> throughput-bound wall punishes it. asm issues finer 1:1 CHEAPLY via `s_add_i32 m0,
+const` increment + FIXED descriptor (~1 SALU/load). HK can't get M0-increment from HIP source.
=> 6th confirmation of the META rule: WORK-reduction moves the wall, WORK-addition (even toward an
asm-like structure) hurts it. "Tighten interleave" only helps if the load ISSUE is also made cheaper
(M0-increment) -- which is asm's hand-written lowering, out of reach in HIP. Reverted (regressed, not saved).

#### STAGE 22 (2026-06): inline-asm cheap-load CEILING -> +15% WORSE -> asm-volatile barrier kills interleave
Replaced the kCtCount==2 spread loads in async_load_kv_cols_bf16 with hand inline-asm matching asm's cheap
issue: FIXED descriptor (uniformized via readfirstlane x4) + `s_add_i32 m0, m0, 0x1000` increment +
i_off immediate (offset:64) for the per-col global offset. Disasm CONFIRMS the cheap pattern landed
exactly (s_mov m0,sX; buffer_load v0,s[28:31],0 offen lds; s_add m0,+0x1000; buffer_load ...offset:64 lds
-- 162 sites, fixed descriptor, M0-increment, asm-style). RESULT: correctness FAILED + 3-rep wall +12..17%
WORSE (c65536 245.6->287.9, b32 c16384 141.9->163.6, c8192 55->61.8).
ROOT: `asm volatile` (REQUIRED -- buffer_load_lds has memory side-effects, can't be non-volatile or the
compiler DCEs it) is a HARD scheduling barrier. It serialized the loads and DESTROYED the compiler's
load<->MFMA interleave (the spread). The interleave loss (+15%) dwarfs the cheap-issue saving; plus
readfirstlane x4/call adds work. => You CANNOT isolate "asm-cheap issue" from "interleave" via piecemeal
inline asm in HIP: the moment loads go into asm volatile, the scheduler is walled off and the interleave
is lost. asm gets BOTH (cheap s_add-m0 issue AND 1:1 mfma:load interleave) ONLY because the ENTIRE compute
loop is one hand-written asm region with no compiler barrier between loads and mfmas. This is the cleanest
evidence yet that the remaining ~7% requires a full asm rewrite, not any HIP-source/inline-asm patch.
Reverted to carried-index (~1.07x). FINAL HIP-SOURCE FLOOR confirmed at 1.07x.

#### STAGE 23 (2026-06): SEGMENT-level timeline diff (one tile period) + COMBINED build -> bubble localized, doesn't close
Built per-wave-timeline SEGMENT analysis (att.md technique): split ONE steady tile period (between
s_barriers) into phases {GLOAD, MFMA, lgkmcnt, KREAD, VREAD, SOFTMAX, BARRIER} and compared HK vs asm.
Per-tile periods comparable (~7.4K HK vs ~7.1K asm/tile -- asm barriers 2x more often). PHASE % (of period):
                GLOAD  MFMA  lgkmcnt  KREAD  (feed=MFMA+lgkmcnt+KREAD)
  HK carried:    39%   15%    14%      4%     33%
  asm:           52%    7%     5%      5%     17%
=> WHERE WE LOSE vs asm: the QK/PV K-OPERAND-FEED phase (feed=33% HK vs 17% asm; ~+16pt). The matrix core
stalls on lgkmcnt-before-mfma waiting for K/V from LDS; asm's never starves (deep ds_read-ahead).
The biggest single BUBBLE in BOTH is buffer_load park (GLOAD) -- shared HBM-BW floor, NOT the differentiator
(asm spends MORE % there, 52 vs 39). So the global load is the floor; the FEED is the gap.
COMBINED build (carried-index + feed look-ahead, no body-dup): wall FLAT (~141.5/~245). Segment: lgkmcnt
14->11% (toward asm) BUT KREAD 3.7->10.5% (ROSE) -> feed total 33->37.7% (WORSE). The look-ahead RELOCATED
the stall lgkmcnt->KREAD (wave parks on the ds_read instead of the wait after) + added ds_read work.
=> DEFINITIVE: the feed bubble is REAL and segment-localized (operand feed, ~2x asm) but is NOT closable in
HIP -- the stall is FUNGIBLE across adjacent instructions (lgkmcnt<->ds_read<->vmcnt) with a FIXED total set
by LDS/HBM latency at occupancy-1. Every HIP lever (alone or combined) relocates it, never reduces it,
because the compiler won't schedule deep-ds_read-ahead + tight-interleave + cheap-M0 together (and inline
asm to force any one breaks the rest, STAGE 22). asm closes it only as a single hand-scheduled asm loop.
1.07x is the HIP-source floor; the remaining ~7% = operand-feed phase, asm-rewrite-only.

#### STAGE 24 (2026-06): ABSTRACT QK-GEMM PIPELINE COMPARISON (HK vs asm) -- the structural diff, recorded
Q is loaded ONCE per work-item into registers and is RESIDENT across all KV tiles (NOT streamed). Q goes
VRAM->LDS->reg only in the prologue (LDS = transpose/coalesce vehicle: VRAM row-major -> MFMA reg layout);
one-time, not the hot path. The inner QK loop is the D=576 contraction (18 col-tiles of 32); per col-tile
it reads K from LDS and MFMAs against the resident Q. Only K (and V) stream tile-by-tile.

HK QK pipeline (from disasm, carried-index):
  Q global -> LDS -> reg                  # once, resident all tiles (LDS = transpose vehicle)
  K[0] global -> LDS                       # prologue burst into bufA
  for tile t:
    s_waitcnt(0); s_barrier                # cross-warp LDS-reuse drain; SOURCE __builtin (orig, line 587)
    for col c in 0..17 (~2 cols/step):
      K[t][c]   LDS -> reg                  # JUST-IN-TIME (0-ahead)  <-- the feed gap
      K[t+1][c] global -> LDS (bufB)        # prefetch, spread ~2/iter (vmcnt, separate counter)
      s_waitcnt lgkmcnt(2)                  # hand-placed asm volatile; MFMA STALLS here on the LDS read
      MFMA P += Q[c] x K[t][c]
    swap bufA <-> bufB                       # runtime pointer swap (opaque -> STAGE-12c false-alias vmcnt0)

asm QK pipeline (from /tmp/asm_full.s -- CORRECTED in STAGE 25; earlier "1-ahead" draw was wrong):
  Q resident                               # once
  K[0],K[1] global -> LDS                   # prologue: fill the ring >=2 tiles deep before first read
  for tile t:
    s_waitcnt vmcnt(10); s_barrier         # cap in-flight at ~10 = the LAST ~1 iter of GLOADs (FUTURE tiles)
    K[t-L] LDS -> reg (all 18 col-tiles)    # read a tile prefetched L>=2 iters ago -> its loads long DRAINED
    for col c in 0..17:
      MFMA P += Q[c] x K[t-L][c]            # operand already resident -> mfma stall ~0
      <issue part of K[t]'s 9 GLOADs>       # prefetch for a tile read ~L iters LATER (1:1 w/ MFMA, s_add-m0)

  CORRECTNESS (resolves the "vmcnt(10) reads incomplete tile?" paradox):
  vmcnt(10) does NOT guarantee the read tile via "10 in flight". The read tile (t-L) was issued >=2 iters
  ago and a full tile's MFMA hid its HBM latency -> it has 0 loads in flight -> complete. The ~10 kept in
  flight are always the freshly-issued FUTURE-tile GLOADs, never the tile being read. So vmcnt(10) is a
  STREAMING CAP, not a completion-wait for the consumed tile. (HK's 1-ahead + vmcnt would be unsafe -- which
  is exactly why HK uses vmcnt(0) drain instead; see STAGE 25.)
  MEASURED (one steady iter, asm_full.s): 9 buffer_load_dwordx4 lds/iter, 36 ds_read_b128/iter, 18 MFMA/iter,
  read-base reg cycles v21->v18->v19->v20 (>=4 LDS read slots = ring deeper than double-buffer), 1 vmcnt(10).

THE STRUCTURAL DIFF = LDS->reg feed depth:
  HK    = 0-ahead (just-in-time per col)  -> MFMA waits lgkmcnt every group (feed phase 33% of tile)
  asm   = full-tile-upfront (all 18 read) -> MFMA never waits                (feed phase 17%)
(a standard "1-ahead double-buffer" would sit between; HK is BEHIND it, asm is BEYOND it.)

CLARIFICATIONS (correct earlier loose claims):
- HK's waitcnts are HAND-PLACED (asm volatile lgkmcnt(2)/(0), explicit vmcnt(0)), NOT naive compiler
  auto-inserts. The tile-boundary s_waitcnt(0) is the ORIGINAL author's __builtin (source), not mine/
  compiler. What IS compiler-controlled (and where the gap lives): instruction SCHEDULING/interleave,
  M0 mgmt (s_mov vs s_add increment), and the EXTRA conservative vmcnt(0) the compiler adds before the
  swap-aliased ds_read (STAGE 12c).
- Descriptor is already SINGLE/fixed in HIP (s[24:27], built once per tile, reused) -- matches asm; the
  STAGE-17/21 "per-load descriptor reload" claim was a stale pre-SALU-hoist observation.

WHY HK CAN'T REPLICATE asm's full-upfront feed (the crux, 3 resets): at occupancy-1 the LDS-read latency
must be hidden under compute. asm hides tile N's K reads under tile N-1's MFMA tail in ONE barrier-free
hand-scheduled asm region. HK can't, because the overlap is reset by (a) the per-tile cross-warp s_barrier
(tile N reads can't start until after it), (b) the compiler won't schedule reads into the previous tile,
(c) forcing it via inline asm makes asm volatile a barrier that kills the interleave (STAGE 22). Every
HIP front-load attempt (STAGE 14/20) just RELOCATES the stall lgkmcnt->KREAD (the ds_read parks instead),
total feed unchanged.

ONLY UNTRIED DIRECTION: cross-tile K read-prefetch -- read tile N's K (LDS->reg) into a cross-boundary
register set DURING tile N-1's PV/tail, overlapping its latency with N-1 compute. BLOCKER/RISK: tile N's
K in LDS is loaded cooperatively by all 4 warps; reading it before the cross-warp s_barrier reads
partially-written LDS -> cross-warp RACE. asm avoids this via vmcnt(10)+barrier ordering in one asm region.
High risk; not yet attempted. This is the last theoretical lever and the precise reason it's hard.

ASM BUFFER MGMT = COMPUTED ADDRESS, NOT POINTER-SWAP (disasm-confirmed): asm has NO bufA/bufB swap. It
selects the LDS half by ADDRESS ARITHMETIC per tile: writes via `s_add_u32 m0,0,s56` + `s_add m0,+0x3c0`
increment (M0 base s56 recomputed per-tile from kv-position); reads via `v_add_u32 v12,s56,v10` (base =
s56 + per-lane). The two physical LDS halves alternate because the kv-index-derived address naturally
lands in different halves -- no `std::swap` of opaque pointers anywhere.
=> This is the ROOT of HK's STAGE-12c vmcnt(0): HK's `std::swap(p_lds_kv_curr,p_lds_kv_next)` makes the
two uintptr_t OPAQUE -> compiler can't prove curr/next don't alias -> conservative vmcnt(0) before the K
ds_read. asm's COMPUTED addresses are TRANSPARENT -> no false-alias -> no swap, no vmcnt(0), and no body
duplication (STAGE-16 paid +2% duplication precisely to make buffers compile-time-distinct; asm gets it
free via address math + single loop).
IMPLICATION (correct but NOT a perf lever): HK could compute the LDS base per tile from (tile_idx&1)
instead of swapping pointers -- cleaner than STAGE-16 (no duplication), drops the swap-alias vmcnt(0). But
STAGE-12e/16/18 proved that vmcnt(0) is overlapped/relocates (memory floor) -> wall would not move. So
"computed-address instead of swap" is the right EXPLANATION of asm + a cleaner pattern, but not a new win.

CEILING EXPERIMENT (cand_unroll + tile-top __builtin_amdgcn_s_waitcnt(0) -> `s_waitcnt vmcnt(8) lgkmcnt(0)`,
correctness ignored): disasm CONFIRMS the streaming applied -- steady K `ds_read a[224:227]` has NO
preceding vmcnt(0) (`vmcnt(8); s_barrier; ... ds_read`), 64x vmcnt(8) emitted. 3-rep timing: c65536 248.1,
b32 c16384 144.4, c8192 55.6 = SAME as cand_unroll (~249/144.6/56.1), still ~2% WORSE than carried-index
(245.6/141.9/55.0). => 4th INDEPENDENT CONFIRMATION: NOT waiting for the QK KV loads (vmcnt(8) keeps 8 in
flight) gives ZERO speedup -> the QK tile-boundary KV wait is OVERLAPPED, NOT critical-path. Combined with
STAGE 12e/16 (removing it neutral/+2%), the QK vmcnt drain is conclusively a non-issue for wall time.
DEFINITIVE CONCLUSION on the asm gap: it is NOT any single vmcnt drain (4 experiments prove that). asm's
~7% edge is its HOLISTIC hand-schedule -- fewer static instrs (748 vs 2040 mfma), tight fixed-descriptor+
M0-increment load issue, 1:1 mfma:load interleave, compact single-specialized body -- none replicable by
patching individual waits in HIP (the compiler owns the schedule). ~1.07x is the HIP floor for this
kernel; further requires writing asm (out of scope). REVERTED to carried-index known-good. Candidates
saved: .cand_salu .cand_carried_idx .cand_unroll .cand_lookahead .cand_distinct.

#### STAGE 25 (2026-06): CORRECTION -- asm prefetch is >=2-iter-ahead w/ multi-slot LDS ring, NOT 1-ahead
The STAGE-17/24 pseudocode + the loose claim "vmcnt(10) ... the 10 in flight are the NEXT tile, current tile
completes first" implied a 1-AHEAD model (issue K[t+1] in tile t's loop, read K[t+1] at t+1 after vmcnt(10)).
That model is SELF-INCONSISTENT and was wrong: vmcnt(10) keeps 10 in flight, so if those 10 were the very tile
about to be read, the read would hit an INCOMPLETE tile. (User caught this.)

RE-READ asm_full.s, one steady iter (hard counts): 9 `buffer_load_dwordx4 ... lds`/iter, 36 `ds_read_b128`/
iter, 18 MFMA/iter, ONE `s_waitcnt vmcnt(10)`/iter, and the ds_read BASE register cycles v21->v18->v19->v20
across 4 consecutive iters => >=4 LDS read slots (a RING deeper than double-buffer, not bufA/bufB).

CORRECTED MODEL: prefetch leads the read by L>=2 iterations. Arithmetic: 9 GLOADs issued/iter, vmcnt(10) caps
in-flight at ~10 = the most-recent ~1 iter of GLOADs. The tile being READ was issued >=2 iters earlier, had a
full tile's MFMA to hide its HBM latency, so it has ZERO loads in flight (drained) -> safe. The ~10 kept in
flight are always FUTURE tiles. So vmcnt(10) is a STREAMING CAP on look-ahead depth, NOT a completion-wait for
the consumed tile. [verified: the 4 counts above; extrapolation: exact L (2 vs 3) not traced via M0->read-base
address map, but L>=2 follows from 9-loads/iter + vmcnt(10) + the 4-reg read-base cycle.]

WHY HK USES vmcnt(0) INSTEAD (the real contrast): HK's shallow ~1-ahead ring cannot keep the consumed tile
drained while streaming the next -- with only ~1 tile of look-ahead, the tile about to be read IS the one with
loads still in flight, so HK must vmcnt(0)-drain to be correct. asm's deep ring decouples "tile being read"
(old, drained) from "tile being loaded" (new, in flight), which is what lets it hold vmcnt(10) instead of
vmcnt(0). The depth (>=4-slot ring) is the enabler of the streaming wait, not the wait itself.
This does NOT change the 1.07x verdict (STAGE 12e/16/24 proved the QK vmcnt drain is overlapped/non-critical);
it corrects the MECHANISM description so the pipeline model is internally consistent.

PROLOGUE-PRIMING EVIDENCE (grounds L~3, asm_full.s this session): 29 `buffer_load_dwordx4 ...lds` are issued
BEFORE the first `ds_read` (line 315) -- i.e. ~3 tiles' worth (29 ~= 3x9) primed into the ring before any read.
Whole-kernel totals: 128 dwordx4-lds, 554 ds_read_b128, 1248 ds_read_b64_tr_b16, 32 ds_read_b64, 64 ds_write,
11x vmcnt(10). So L (lead) ~= 3 and ring S >= 4 slots (4 read-base regs v18-v21). A tile's K load = ~9 dwordx4
issued within ONE iter (spread ~1 load : 2 MFMA across its 18 MFMA), but takes ~L iters of HBM latency to LAND;
the ring is L-deep precisely to cover that latency.

ABSTRACT DEEP-RING PIPELINE (asm; verified counts annotated):
  # RING = S>=4 LDS slots (read base cycles v18/v19/v20/v21); LEAD L~3 (prologue primes 29~=3x9)
  prologue: issue K[0..L-1] global->LDS into slots 0..L-1     # 29 dwordx4-lds, NO read yet (fill the pipe)
  for tile t = 0,1,2,...:
    s_waitcnt vmcnt(10)                 # cap in-flight VMEM @~10 = most-recent ~1 tile of GLOADs (FUTURE tiles)
    s_barrier                           # publish cooperatively-filled slot cross-warp (4 warps)
    K[t] LDS->reg from slot[t % S]      # issued L iters ago -> drained (0 in flight) -> complete & safe
    for col c in 0..17:
      MFMA P += Q[c] x K[t][c]          # operand resident -> ~0 feed stall
      issue ~half of K[t+L]'s 9 GLOADs  # spread 1:2 w/ MFMA -> slot[(t+L) % S]
  # INVARIANT that makes vmcnt(10) correct: "tile being READ (t)" and "tiles being LOADED (t+1..t+L)" are ALWAYS
  # different slots, L apart. vmcnt(10) only ever holds the LOAD side (future); it NEVER gates tile t.

lgkmcnt IS NOT THE K-READ GATE (verified this session, corrects a chat-only mis-claim): across 4 consecutive
steady barriers the K `ds_read` is preceded by `s_waitcnt lgkmcnt(0)` only ONCE (barrier #3); the other 3 read
K immediately after `s_barrier` (or after 2 MFMAs), no lgkmcnt. A real per-tile correctness gate would appear
EVERY tile. => K[t] readiness = vmcnt (the buffer_load_lds fill, VMEM-counted) + s_barrier (cross-warp), NOT
lgkmcnt. The lgkmcnt(0)/(2)/(4) waits (76/12/20) are scheduler-placed to drain the OTHER LDS traffic -- the 64
ds_write + 1248 ds_read_b64_tr_b16 (V / P-pack / transposed operand feed) -- unrelated to K's global->LDS fill.
(My earlier chat pseudocode wrote `vmcnt(N) lgkmcnt(0)` at the tile top with a "prev-tile ds_read -> slot reuse"
rationale; that rationale was unverified and WRONG. The STAGE-24 corrected asm block above correctly omits it.)

=== SESSION HANDOFF (for next session) ===
Status: NO code change this session -- pure mechanism-correction of the knowledge model (kernel still at the
carried-index known-good, ~1.07x of asm; that perf verdict is UNCHANGED). What was fixed: the asm QK-pipeline
MENTAL MODEL had two errors, both now corrected in STAGE 24/25:
  (1) asm prefetch is a deep multi-slot ring (S>=4, lead L~3), NOT 1-ahead double-buffer. vmcnt(10) is a
      look-ahead THROTTLE on future-tile loads, not a completion-wait on the consumed tile.
  (2) lgkmcnt(0) is NOT the K-read correctness gate (vmcnt+barrier is); it drains V/P-pack/operand LDS traffic.
Both backed by asm_full.s counts (in-container /tmp/asm_full.s; helper scripts /tmp/asmtrace.sh, asmcount.sh,
asmring.sh). Open (low priority): exact L (2 vs 3) not pinned -- would need to trace the M0-write-address ->
read-base-register map per tile. No pending build/bench. The 1.07x HIP floor + "asm-rewrite-only" conclusion
(STAGE 24) stands; this session only made the WHY internally consistent.

#### STAGE 26 (2026-06): LDS-CAPACITY "contradiction" RESOLVED -- it was a tile-GRANULARITY mix-up (apples vs oranges)
The STAGE 24/25 model (asm = S>=4 deep ring) appeared to contradict Risk #2 (D=576 -> 72-76 KB/tile, "can't fit
>2 tiles in 160 KB"): 4 * 76 KB = 304 KB does not fit. RESOLVED by reading BOTH sides' real tile size. They are
NOT the same "tile" -- HK uses a COARSE tile, asm a FINE one, ~4x apart. No capacity contradiction exists.

EVIDENCE (HK, in-repo, mi35x_..._m16x4_bf16_bf16.cuh):
- kTileM=16, kBlockN=64, kBlockK=32 (lines 193,197,377). One KV tile = kBlockN=64 KV-pos x 576 D bf16 =
  64*576*2 = 73728 = 72 KB. Double-buffered: p_lds_kv_curr + p_lds_kv_next = 2 slots (lines 318-320,427).
  => HK = 2 COARSE slots of 64-KV-pos, ~144 KB.

EVIDENCE (asm, /tmp/asm_full.s, one steady iter lines 633-1314):
- 18x v_mfma_f32_16x16x32_bf16 into ONE accumulator v[38:41] (lines 635-664) = one 16(qh) x 16(KV-pos) output,
  18 MFMAs * k32 = 576 = D_qk contraction. => asm iter processes 16 KV-pos (N=16), NOT 64.
- K read = 18x ds_read_b128, base v21, offset 0..17408 step 1024 (lines 637-664) -> span 18432 B = 18 KB =
  16*576*2. So one asm ring slot = 18 KB (16-KV-pos FINE tile).
- K write = 9x buffer_load_dwordx4 lds/iter, FIXED descriptor s[20:23], s_add_i32 m0,+0x3c0 (960 B) per load
  (lines ~1024-1188) -- cheap M0-increment streaming into the 18 KB slot.
- Prologue primes 29 dwordx4-lds (~3.2 * 9/tile) before first ds_read -> lead L~3 (matches STAGE 25).
- .group_segment_fixed_size = 160 KB (STAGE 2): 160/18 ~= 8.9 -> asm can ring ~8 FINE tiles in the SAME LDS.

RECONCILIATION ARITHMETIC: 4 asm fine-tiles (4*18 KB = 72 KB) = exactly ONE HK coarse-tile. asm's "deep ring"
and HK's "2 coarse slots" use COMPARABLE LDS (~72-144 KB); the difference is GRANULARITY (16 vs 64 KV-pos),
not bytes. Risk #2's "needs a more compact KV layout" is WRONG: a deeper ring needs a SMALLER kBlockN, which the
existing 160 KB already affords.

IMPLICATION (new, untried, distinct from STAGE 14/20 scheduling tweaks): reduce HK kBlockN 64 -> 16 (or 32) so
the coarse double-buffer becomes a FINE multi-slot ring whose per-tile 18 ds_reads pair 1:1 with 18 MFMA and
overlap naturally (asm-isomorphic). This is a TILING change, not a schedule hint.
PREDICTION (honest, from the META throughput rule): MODERATE risk, likely flat-to-negative. Finer tiling ADDS
work (more iters/barriers/index math per 16-pos) which the throughput-bound wall punishes (cf STAGE 21 +2%);
HK m16x4's 4 warps cooperatively fill the 64-pos tile, so kBlockN=16 pushes toward per-warp KV ownership (the
STAGE-24 (C) cross-warp question); and hipcc may still not interleave reads across the per-tile s_barrier
(STAGE 22). Net: the only STRUCTURAL lever left, worth ONE measured try, but predicted to not beat 1.07x.
This does NOT change the 1.07x verdict; it corrects WHY a deep ring looked impossible (capacity) -> it's
granularity + compiler scheduling, same root as STAGE 22/24.

#### STAGE 26b (2026-06): DETAILED abstract software pipeline (supersedes STAGE 24's pseudocode)
Read a full steady period of both kernels line-by-line. KEY new fact vs STAGE 24: asm is NOT "one tile does
QK->softmax->PV then next tile". It OVERLAPS multiple sub-tiles' DIFFERENT stages in ONE vmcnt(10) period -- a
single period (asm_full.s:633-970) holds THREE independent QK accumulators (v[38:41], v[42:45], v[46:49]) in
flight PLUS the prior tile's PV (v[50:53]..v[122:125]). HK has no such cross-tile overlap.

--- A. asm pipeline (FUSED, deep overlap; evidence asm_full.s:633-970) ---
Model as 6 parallel STREAMS, each leading its consumer by several tiles:
  S1 global->LDS K load (lead L~3 tiles): 9x buffer_load_dwordx4 lds, s_add m0,+0x3c0, fixed desc s[20:23]  (:797-837)
  S2 LDS->reg K read (lead ~1 tile):      18x ds_read_b128, base from s56, offset step 1024                 (:637-664,804-848)
  S3 QK MFMA (multi-accumulator):         18x v_mfma -> v[38:41] / v[42:45] / v[46:49]                       (:635-664,789-835,851-900)
  S4 softmax + P-pack:                    v_max3 / permlane*_swap / v_exp / v_fma + v_cvt_pk_bf16            (:725-784)
  S5 LDS->reg V read (transposed):        ds_read_b64_tr_b16 a[144:207]                                      (:852-968)
  S6 PV MFMA -> oaccu:                     v_mfma v[50:53]..v[122:125], a[144:207], v[34:37](bf16 P)          (:911-969)
Steady period (one vmcnt(10) -> next, ~337 instr):
  prologue: issue K[0..L-1] global->LDS into ring slots         # 29 dwordx4-lds ~= 3x9 -> lead L~3
  loop:
    s_waitcnt vmcnt(10)        # THROTTLE: cap in-flight global loads ~10 (= most-recent ~1 tile, FUTURE tiles)
    s_barrier                  # publish cooperatively-filled LDS slot across 4 warps
    # 6 streams interleaved, each hiding the others' latency:
    #   S3 QK(A) 18 mfma  <-1:1->  S2 read-ahead K(B)
    #   S4 softmax(A)+P-pack(A) -> v[34:37]
    #   S3 QK(B,C) 18 mfma each  <->  S1 prefetch K(t+L) 9 loads  <->  S2 read K
    #   S5 V(A) tr_b16 read  <->  S6 PV(prev tile) mfma -> oaccu
    # LDS slot chosen by COMPUTED address (s56 = kv-pos derived) -> lands in alternating half -> no pointer swap
Facts (all verified): granularity = 16 KV-pos/iter (one 16x16 out, 18 mfma = D=576), 18 KB/slot, ring ~8 in 160 KB.
vmcnt(10)=throttle not wait (read tile drained L~3 ago). K feed = read-ahead 1:1 with mfma -> mfma ~never stalls.
lgkmcnt(4)/(0) (:788,850,908,965) drains S5/S6 V/tr_b16 traffic, NOT a K gate. Load issue cheap: s_add-m0 + fixed desc.

--- B. HK pipeline (SHALLOW, per-tile sequential; evidence cuh:571-790) ---
  prologue: Q VRAM->LDS->reg (resident); K[0] global->LDS (bufA)
  for each KV tile (64 KV-pos):
    __builtin_amdgcn_s_waitcnt(0)          # FULL DRAIN (not throttle)                       cuh:587
    s_barrier; sched_barrier(0)            # cross-warp                                      cuh:588-589
    resolve next-tile phys_row (carried, no dependent load)                                  cuh:613-628
    for col-iter idx in 0..num_nope_iter:                # QK over 18 col-tiles
      load_k_to_gpr(kv_0/kv_1 top/bot) LDS->reg          # JIT, right before the mma         cuh:670-677
      async_load_kv_cols_bf16<idx,2> global->LDS(bufB)   # spread 2 col/iter                 cuh:685
      s_waitcnt lgkmcnt(2)                               # mma STALLS here on K              cuh:713
      mma_ABt(p_comp_lo, kv_0, q_0); s_setprio(3)                                            cuh:737-738
      s_waitcnt lgkmcnt(0); mma_ABt(p_comp_lo, kv_1, q_1)                                    cuh:744-745
      load_k_to_gpr(upper N-half, LDS rows 32..63)       # reload SAME vgprs: 64-pos = 2x32  cuh:748-755
      ... upper-half mma ...
    softmax + P-pack -> p_mfma                                                               cuh:759+
    for V-iter: ds_read V (tr_b16) + mma_ABt(oaccu, p_mfma, V)    # PV
    std::swap(p_lds_kv_curr, p_lds_kv_next)               # OPAQUE swap -> vmcnt(0) false-alias

--- C. Structural diff (where the ~7% lives) ---
  axis            asm                                  HK                                   consequence
  tile gran.      16-pos FINE (18 KB)                  64-pos COARSE (72 KB, read 2x32)     asm ring deep, feed natural 1:1
  ring depth      ~8 slots (computed addr)             2 slots (pointer swap)               swap opaque -> extra vmcnt(0) (12c)
  tile-bdy wait   vmcnt(10) THROTTLE                   s_waitcnt(0) FULL DRAIN              HK drains+restarts every tile
  K feed          read-ahead ~1 tile (S2 1:1 mfma)     JIT (load_k_to_gpr->lgkmcnt(2)->mma) HK mfma stalls/group (feed 33 vs 17%, STAGE 23)
  cross-tile      PV(t-1)||softmax(t)||QK(t+1,t+2)     QK(t)->softmax(t)->PV(t) sequential   asm hides LDS/HBM latency in other tiles
                  multi-accumulator                    (only global prefetch spread)
  load issue      s_add-m0 increment + fixed desc      per-call s_mov m0 reset              asm cheap (STAGE 21/22)
  overall         ONE hand-scheduled barrier-free      segmented asm volatile/__builtin,    STAGE 22: forcing any one breaks the rest
                  asm region                           compiler owns the schedule
One line: asm = fine-gran + deep ring + computed addr + multi-accum cross-tile overlap -> every stream's latency
hidden under another's compute. HK = coarse-gran + double-buffer + per-tile sequential + JIT feed -> drains and
restarts at every tile boundary, mfma stalls in the feed phase. The 7% IS this structural delta (asm-rewrite-only).

#### STAGE 27 (2026-06): RE-VERIFIED FROM SCRATCH + the proper rolling double-buffer EXPERIMENT (definitive)
Re-derived everything from own measurement (distrust prior numbers). Tooling: build via AITER_REBUILD=1 in
container pa_bench_mh (GPU7, idle); wall = test_mla_persistent.py "golden vs aiter_asm" us (EXP=1 HK, EXP=0 asm),
3-rep min; ISA via llvm-objcopy .hip_fatbin -> clang-offload-bundler -> llvm-objdump (hk.s); ATT via
rocprofv3 --att --att-library-path /opt/decoder/lib/ + /tmp/cpath.py.

RE-VERIFIED BASELINE (3-rep min, own): HK/asm = b16c16384 90.02/84.76=1.062, b32c16384 144.87/137.50=1.054,
b16c65536 246.58/233.63=1.055. ~1.057x (slightly better than the doc's earlier 1.07; same regime).

CORRECTION to a STAGE-9..24 framing ("HK QK has 2 accumulators"): DISASM shows the bf16 QK runs FOUR p_comp
accumulators (v[112]/v[116]/v[120]/v[124]); source p_comp_lo/hi each = 2 N-tiles. Active ILP is 2-way (lo pair
then hi pair). asm uses ONE accumulator per QK sub-tile chain (18 mfma -> v[38:41]). => "add accumulators" is the
WRONG lever (HK already has more); the real lever is OPERAND-WINDOW DEPTH. HK K window = 16 AGPR (a[224:239])
reloaded every 4 mfma, ds_read issued ~2 mfma before its consumer; asm window ~72 AGPR read a full sub-tile (~18
mfma) ahead.

FEED-LOOKAHEAD A/B (cand_feedla = STAGE-20 1-deep upper-half, own 3-rep): 89.72 / 146.31 / 243.12 = NEUTRAL
(-0.3/+1.0/-1.4%). Confirms the doc: shallow feed look-ahead is wall-neutral.

THE EXPERIMENT (proper rolling 1-ahead operand double-buffer -- what feedla SHOULD have been; user-directed):
restructured the QK NoPE loop into a true software pipeline. Added a 2nd lo-N K buffer kv_0_b/kv_1_b at AGPR
a136..a151 (free region, spill-checked OK). Compile-time idx%2 alternates cur=buf[i%2]/nxt=buf[(i+1)%2]:
prologue loads iter0 lo into A; each iter PREFETCHES iter(i+1)'s lo into nxt while doing iter i's mma from cur
(lo prefetched a full iter earlier); up reloads into cur (still JIT). lgkmcnt(4) before mma_lo (cur ready, 4 nxt
outstanding), lgkmcnt(2)/(0) for the up mma_hi (in-order: nxt completes before up). Saved .cand_rolling_dbuf.
  CORRECTNESS: PASS all configs (b16 c8192/16384, b32 c16384, b16 c65536). No spill (private_segment=0), still
    512 vgpr / 256 agpr (occ-1).
  ISA VERIFIED the read-ahead actually deepened (hk2.s): buffer-A lo ds_read at line 561, consumed by mma at
    line 645 -- a FULL B-iter (~8 mfma + global loads) between = ~8-mfma read-ahead vs known-good's ~2. 4x deeper.
  PER-WAVE (own ATT, cpath): wave_dur 149604 -> 136380 (-9%, vs asm 134816!), total_stall 90976 -> 76764 (-16%),
    LGKMCNT 19956->18760, MFMA 10960->11584 (relocated slightly), DS_READ 3572->4888 (relocated). A REAL per-wave
    win, bigger than feedla/STAGE20, nearly closing the per-wave gap to asm.
  WALL (3-rep min): 89.37 / 145.96 / 247.27 = NEUTRAL (-0.7/+0.8/+0.3%). ZERO wall change.

DEFINITIVE CONCLUSION (by direct construction, not a proxy): a CORRECT rolling double-buffer that ISA-provably
deepens the K read-ahead 4x and ATT-provably cuts per-wave wave_dur ~9% (to within ~1% of asm's wave_dur) yields
ZERO wall improvement. This is the cleanest possible proof that HK's QK feed/mfma stall is NOT on the wall's
critical path: the wall is bound by a SHARED throughput / memory-latency floor (per-wave gap ~1.11x vs wall gap
~1.057x = ~2x dilution; asm itself spends 34.8% wall parked on buffer_load = the HBM-latency floor, BW only
26-52% per STAGE 21). The JIT-feed->mfma "problem" is real per-wave but UNSOLVABLE into wall gains via HIP operand
pipelining (feed look-ahead, rolling double-buffer, deeper windows all relocate/overlap, never move the wall).
~1.057x is the HIP-source floor for this kernel; the remaining gap needs asm's holistic hand-schedule (fewer
static instrs + fine-gran ring + cheap M0 issue), out of scope. Reverted to known-good; .cand_rolling_dbuf kept.

#### STAGE 28 (2026-06): FULL lo+up double-buffer -> per-wave BELOW asm, wall slightly WORSE (the clincher)
Extended STAGE 27's rolling buffer (lo only) to BOTH N-halves: added UP-N K buffers A_up/B_up at AGPR a104..a135
(free region; no spill -- private_segment=0, still 512vgpr/256agpr). Now each iter prefetches the NEXT iter's FULL
K (lo+up = 8 ds_read) 1-ahead into the alternate buffer and consumes cur's full K with NO JIT reload -- the
complete asm-style 1-ahead operand pipeline. lgkmcnt(8) at iter top. Correctness PASS all configs. Saved
.cand_full_dbuf.
  PER-WAVE (own ATT, cpath, b16 c16384):
    metric         known-good   rolling(lo)   FULL(lo+up)   asm
    wave_dur          149604       136380        132300      134816   <- FULL is BELOW asm
    LGKMCNT_WAIT       19956        18760          9336        8684    <- feed wait crushed to asm parity
    MFMA               10960        11584         13844        2544
    DS_READ             3572         4888         12212        6292    <- stall RELOCATED here (STAGE 23 pattern)
    total_stall        90976        76764         70448       75540   <- FULL total stall BELOW asm
  WALL (3-rep min): b16c16384 90.41 / b32c16384 147.56 / b16c65536 248.93 = +0.4/+1.9/+1.0% vs known-good = SLIGHTLY WORSE.

THE CLINCHER: the FULL double-buffer ELIMINATED the feed wait (lgkmcnt 19956->9336 = asm's 8684) and drove per-wave
wave_dur (132300) and total_stall (70448) BELOW asm's (134816 / 75540). If the wall tracked the per-wave critical
path, HK would now be FASTER than asm. Instead the WALL got slightly WORSE (+1% avg). => DEFINITIVE: the wall is
throughput-bound (total instruction ISSUE), NOT per-wave critical path. The full pipeline ADDS work (8 ds_read/iter
+ 32 extra AGPR mgmt vs known-good's 4+4 with reload) which the throughput-bound wall punishes, exactly the META
rule (STAGE 21): WORK-addition hurts the wall even when it removes stalls. The feed/mfma stall is real per-wave but
provably NON-CAUSAL for the wall. CONCLUSION STANDS AND IS NOW PROVEN BY CONSTRUCTION: no HIP operand-pipelining
(shallow feed-LA / rolling-lo / full lo+up) moves the wall; ~1.057x is the HIP floor; the asm gap is its holistic
hand-schedule (compact 748 vs 2040 static mfma + fine-gran ring + s_add-m0 issue), asm-rewrite-only / out of scope.
Reverted to known-good (rebuilt, pass). Candidates kept: .cand_rolling_dbuf (lo), .cand_full_dbuf (lo+up).

#### STAGE 29 (2026-06): CORRECTION -- "2040 vs 1176 static mfma" is CODE SIZE, NOT dynamic work
A STAGE-19..28 framing said asm wins by "fewer static mfma (748/1176 vs 2040)" implying HK does MORE WORK. That is
WRONG / misleading. Measured the steady-state instruction MIX (own disasm count):
  asm  one sub-tile period (asm_full.s:970-1314): 345 instr / 68 mfma = 5.07 instr per mfma
  HK   steady QK+PV chunk   (hk.s:1090-1450):     361 instr / 72 mfma = 5.01 instr per mfma
=> In STEADY STATE HK and asm have the SAME instr/mfma ratio (~5.0). HK does NOT issue more instructions per unit
compute, and dynamic mfma WORK is equal (same KV tiles -> QK = 72 mfma / 64-KV-pos for both; user's intuition right).
The static totals (HK 15291 instr / 2040 mfma vs asm 7453 / 1176) differ because HK emits MANY SPECIALIZED loop-body
COPIES (kIsFirstIter x kSkipCompute x kEpilogueType x kCheckBoundaryNext template combos x unroll) -- a 2x larger
BINARY (code size), not 2x dynamic work.
IMPLICATION: the ~6-7% wall gap is NOT "HK computes more". It is asm's COMPACTNESS (half the code -> lighter
i-cache/issue across the whole dispatch) + its tighter hand-scheduled steady loop. STAGE-28's full-dbuf corroborates:
it drove steady per-wave BELOW asm yet the wall got slightly WORSE -- because it enlarged the code, and the residual
wall cost lives at the code-size/dispatch level, invisible to a single steady wave's ATT (cf STAGE 16 i-cache note).
Also CLOSED a lever: gfx950 has NO ds_read_b128_tr (only b64_tr_b16/b8/b4, b96_tr_b6 -- llvm-mc verified), so V's
b64_tr_b16 is already the max transpose width; "wider V read" is structurally impossible.
NEW (untried) direction this implies: REDUCE code size / specialization count (cut i-cache pressure), NOT operand
pipelining. Different axis from everything tried; matches "wall = throughput/dispatch-bound". Bigger refactor though.
The phrase "748/2040 static mfma = more work" in STAGE 27/28 should be read as CODE SIZE per this correction.

#### STAGE 30 (2026-06): DIRECT throughput test -- wall IS compute(mfma)-sensitive (corrects "pure memory floor")
User was skeptical of the loose "throughput-bound / memory-latency floor" claim. Ran a DIRECT test: known-good +
7x REDUNDANT QK mfma per NoPE iter accumulated into p_comp (correctness ignored), with NO extra memory (same
buffer_load / ds_read). Total mfma/tile ~272 -> ~720 (~2.65x). Build OK.
  RESULT (b16 c16384): wall 90.02 -> 152.34 us = +69% (1.69x).
=> The wall IS strongly COMPUTE(mfma)-SENSITIVE. It is NOT a pure memory-latency floor where compute is free/hidden.
This CORRECTS the STAGE-26b..28 "memory-latency floor / ~2x dilution" wording: compute (mfma issue) is a REAL,
major wall component.
RECONCILED MODEL (now consistent with ALL experiments):
  - The wall = mfma-issue-throughput OVERLAPPED WITH memory-latency. Both are real co-bottlenecks.
  - Sub-linear (2.65x mfma -> 1.69x wall) => at baseline there IS some slack (memory-latency idle the first added
    mfma fill cheaply) THEN it becomes mfma-bound. So baseline sits in the overlap region, not a pure floor.
  - Feed/operand-pipelining (STAGE 27/28 rolling/full double-buffer) is wall-NEUTRAL because the feed ds_read is
    OVERLAPPED UNDER the mfma+memory work -- removing feed stalls just exposes the compute/memory you already pay.
    (This is WHY feed doesn't move the wall -- not a vague "floor"; it's hidden under the real co-bottleneck.)
  - The asm ~6% gap: dynamic mfma EQUAL (72 QK/64-KV-pos both) + memory equal -> the gap is asm's EXPOSED-overhead
    reduction (tighter hand-schedule + half the code size/i-cache, STAGE 29), on top of the shared mfma+memory cost.
CAVEAT (label): the redundant mfma chain into p_comp is dependent; on CDNA accumulator-forwarding dependent
same-acc mfma issue at ~throughput rate, so +69% ~= issue-throughput cost [extrapolation, not directly isolated].
CLEAN follow-ups to fully decompose (untried): (a) redundant mfma into INDEPENDENT accumulators (pure issue-rate,
no dep chain); (b) a memory-side test (extra KV re-load, same compute) to quantify the memory component;
(c) smaller redundant factors to map the slack->mfma-bound knee. Reverted to known-good (clean).

#### STAGE 31 (2026-06): BOTTLENECK DECOMPOSITION study (redundant-resource injection) -- the full picture
Ran the STAGE-30 follow-ups: inject N redundant ops of ONE resource type per NoPE iter, hold everything else fixed,
measure wall slope (correctness ignored). Isolates WHICH hardware resource binds the wall. (own builds, b16, idle GPU)

  injected resource (per iter)        +instr/tile   c16384 wall (Δ vs 87-90 base)   c65536 wall (Δ vs 247 base)
  -- baseline (known-good) --              -          ~90.0                            ~246.6
  +1x mfma block (2x QK compute)        +64 mfma      95.6  (+6%)                      274.5 (+11%)
  +7x mfma block (8x QK compute)       +448 mfma     152.1  (+69%)                     475.4 (+93%)
  +56 VALU/iter (v_add, issue pipe)    +448 VALU     104.4  (+16%)                     305.7 (+24%)
  +7x global KV load (8x HBM traffic)  +7x KV bytes  140.9  (+56%)                     456.2 (+85%)

FINDINGS (decisive, by construction):
1. The wall is NOT a single floor. It is the OVERLAP of THREE resources: MATRIX-UNIT (mfma) + HBM (memory) + a
   smaller VALU/issue component. Adding ANY of the three raises the wall => all three are near-critical (tightly
   overlapped, little slack).
2. MATRIX-UNIT is the HEAVIEST: same instruction count (448), mfma +69% vs VALU +16% => mfma ~4x more expensive per
   instruction. The compute bottleneck is the MFMA pipe specifically, NOT general instruction issue (VALU is ~4x
   cheaper but non-zero). => cutting non-mfma instrs / code size helps only a little; the matrix unit is the wall.
3. HBM is co-major (+56%/+85%), and GROWS with ctx (longer ctx -> more memory-bound). At c65536 mem and compute are
   comparably heavy.
4. mfma marginal cost RISES with amount (0->64 mfma = 0.088us/mfma; 64->448 = 0.147us/mfma) => baseline matrix unit
   is NOT saturated (small additions partially hide under HBM); large additions saturate it. Confirms overlap/slack.
5. RECONCILES the whole project: FEED (LDS-read latency, all of STAGE 14/20/27/28's targets) is HIDDEN under the
   matrix+HBM work, so feed/operand-pipelining is wall-neutral BY DESIGN -- the binding resources are mfma-unit and
   HBM, neither of which feed pipelining touches. Dynamic mfma and HBM bytes are EQUAL HK-vs-asm, so the residual
   ~6% asm edge is exposed-overhead (schedule tightness + code size/i-cache, STAGE 29), not a resource HK lacks.
=> To actually move THIS wall you must reduce mfma-unit work or HBM bytes (algorithmic: fewer mfma / less KV traffic
   / lower precision), NOT operand pipelining. For bf16 qh64 decode both are fixed by the math, so ~1.06x is the
   HIP floor; asm's edge is hand-schedule only. Reverted to known-good (rebuilt, pass). Candidates from STAGE 27/28
   kept; the STAGE 30/31 injection variants were diagnostic-only (not saved).

#### STAGE 32 (2026-06): STAGE-12c + full-dbuf COMBINED test -> slower, and a thermal-drift methodology trap
Tested the user's hypothesis: combine the two per-wave levers (full-dbuf feed removal + STAGE-12c swap-alias vmcnt(0)
removal) -- maybe non-additive and finally moves the wall. Implemented a LIGHT 12c (no 2x-unroll): replaced
`std::swap(p_lds_kv_curr,p_lds_kv_next)` with compile-time-const slots a/b + a runtime `pong` bool, recomputing
curr/next = select(pong, const, const) so AA should prove curr!=next. Built on top of cand_full_dbuf. Correctness PASS.
  RESULT 1 -- the 12c-light did NOT work: compiler still did NOT disambiguate the select; bf16-qlen1 static vmcnt(0)
    = 108 (vs full-dbuf 103), NOT reduced. A true drop needs the 2x-unroll compile-time-distinct buffers
    (`.cand_distinct`) -- which is now STALE (won't compile: async_load_kv_cols_bf16 signature changed since it was saved).
  RESULT 2 -- wall (BACK-TO-BACK 3-rep, same thermal state): known-good 85.69/242.45 vs 12c+full-dbuf 88.08/246.03
    = +2.8% / +1.5% SLOWER. The combination does NOT help; slightly worse (extra full-dbuf regs + 12c select overhead).
METHODOLOGY TRAP (important, distilled): a first single-run showed 12c+full-dbuf 88.2 "faster" than the STAGE-28
known-good 90.0 -- but that was THERMAL DRIFT, not a real win. Over this ~3h session the SAME known-good kernel
drifted 90.0 -> 87.4 -> 85.7us (~5% faster as the box warmed/clocked up). Cross-time comparisons are invalid at the
1-3% level. RULE: every A/B must rebuild+measure BOTH arms BACK-TO-BACK in the same minutes; never compare against a
number measured earlier in the session. (This retro-validates STAGE 27/28's same-session 3-rep protocol.)
CONCLUSION: consistent with STAGE 31 -- the swap-alias vmcnt(0) and feed are both NON-binding; removing them (or
trying to) does not move the wall (binding = mfma-unit + HBM). Even a TRUE vmcnt(0) removal is predicted flat-to-worse
(STAGE 12e 2x-unroll distinct was +2-4%). 12c+full-dbuf not saved (slower). Reverted to known-good (rebuilt, pass).

#### STAGE 33 (2026-06): EXPLICIT MFMA SCHEDULING (sched_group_barrier) -- HK-upstream technique, case-specific
CONTEXT: HK *upstream* (3rdparty/HipKittens/kernels/attn/gqa/kernel.cpp:41-55) ships an explicit-schedule recipe and
USES it in its main attention loop -- so this is a real, endorsed HK technique, not a dead end. The recipe is:
  #define SCHED_BARRIER(mask,cnt,grp) __builtin_amdgcn_sched_group_barrier(mask,cnt,grp)   // MFMA_MASK=0x08 VALU=0x02 EXP=0x400
  sched_barrier_pairs<Pairs,VALU_CNT,Group>  = Pairs x (1 MFMA, VALU_CNT VALU)             // interleave matrix+vector
  sched_barrier_exp_pairs<Pairs,EXP_CNT,Group> = Pairs x (1 MFMA, EXP_CNT EXP)             // interleave matrix+transcendental
  placed in-loop bracketed by s_setprio, Group ids 1/2 (e.g. sched_barrier_exp_pairs<6,3,1>(); sched_barrier_pairs<10,5,1>()).
KEY POINT (the axis I got wrong first): upstream interleaves MFMA with **VALU / EXP (the softmax work)** to co-issue
the matrix unit and the vector/transcendental pipes -- NOT MFMA:DS_READ. ds_read is memory and already overlaps.

ATTEMPT 1 (naive, MFMA:DS_READ on full-dbuf QK loop): clustered 8x(1 MFMA,1 DS_READ) in the QK feed region.
RESULT: mfma stall did NOT drop toward asm; wall flat-to-worse. WRONG AXIS -- it targeted feed (ds_read), which
STAGE 31 already proved non-binding (hidden under mfma+HBM). Reverted.

ATTEMPT 2 = STAGE 33b (faithful, upstream MFMA:VALU pattern): see below.

#### STAGE 33b: faithful upstream recipe (hk_sched_barrier_pairs<4,1,2>) in the PV loop -> FLAT within noise
Replicated upstream's helper verbatim (`__builtin_amdgcn_sched_group_barrier(MFMA=0x08,1,grp)` then `(VALU=0x02,VALU_CNT,grp)`,
Pairs deep) and placed `hk_sched_barrier_pairs<4,1,2>()` after BOTH interleaved PV blocks (2.4 _lo and 2.8 _hi), which
already hand-interleave 4 mma_ABt with 4 mul_pair (the oaccu rescale VALU) under s_setprio. Correctness PASS.
  A/B (BACK-TO-BACK, same thermal window, 3-rep each, idle GPU):
    ctx       STAGE 33b (3-rep)              known-good (3-rep)            verdict
    c16384    84.82 / 85.69 / 88.27          87.90 / 87.17 / 86.81        FLAT (spreads overlap; 33b worst 88.27 > KG worst 87.90)
    c65536    240.46 / 244.84 / 247.86       246.93 / 245.10 / 244.70     FLAT (means 244.4 vs 245.6 ~0.5%, within noise)
  Best-of-3 looks like a ~2% edge but the per-rep spreads fully overlap and 33b's worst rep is slower than KG's worst =>
  NOT a real win (STAGE-32 rule: 1-3% with overlapping spreads = noise). Reverted to known-good (rebuilt + installed, PASS).

WHY upstream benefits but THIS decode does not (now evidence-backed, not just hypothesis): the PV loop ALREADY
hand-interleaves MFMA:VALU (mma_ABt + mul_pair) under s_setprio -- it is upstream's technique applied MANUALLY. Forcing
the SAME clustering via sched_group_barrier adds no NEW overlap the manual schedule didn't already have, so the wall is
flat. Upstream's recipe wins in GQA prefill where the compiler would otherwise NOT cluster softmax VALU/EXP against the
mma; here the kernel author pre-clustered it by hand. CONCLUSION: explicit sched_group_barrier is a real, endorsed HK
technique, but it only helps when there is UN-clustered VALU/EXP co-resident with binding mma; for an already-hand-
scheduled mfma+HBM-bound kernel it is redundant. Keep it in the toolbox (STAGE 30), do not apply it here.
