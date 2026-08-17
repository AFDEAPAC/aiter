# Kimi-K3 a16w4 (bf16 × MXFP4, SiTUv2) fused MoE on gfx942

Kimi-K3's MoE is weight-only MXFP4 with a SiTUv2 activation. On gfx942 (MI300X /
MI325X) none of AITER's native MXFP4 MoE backends accept SiTUv2 — selecting one
raises

```
ValueError: Mxfp4 MoE backend 'AITER_MXFP4_MXFP4' does not support the
deployment configuration since kernel does not support MoEActivation.SITU activation.
```

so vLLM fell back to `EMULATION`, which dequantizes the weights to bf16 **on every
decode step**. This change gives gfx942 a real a16w4 kernel instead, at roughly
**8× the output throughput** of that emulation path.

The kernel is a gfx942 port of FlyDSL's `moe_2stage_a16wmix`, vendored here so it
runs against the `flydsl==0.2.4` that AITER pins. Nothing in the FlyDSL package or
repo needs to change.

---

## 1. What is in this repo

| Path | What it is |
|---|---|
| `aiter/ops/flydsl/kernels/moe_2stage_a16wmix/{__init__,utils,gemm1,gemm2}.py` | The kernel. Vendored from FlyDSL `kernels/moe/moe_2stage_a16wmix` (PR #948, commit `47ed57a`) plus the gfx942 and flydsl-0.2.4 changes in §5. |
| `aiter/ops/flydsl/moe_a16wmix_host.py` | Host launcher (tile-config resolution, JIT). From FlyDSL `tests/kernels/moe_a16wmix_host.py`, imports repointed at AITER's own `layout_utils` / `tensor_shim`. |
| `aiter/ops/flydsl/a16wmix_fused_moe.py` | `fused_moe_a16wmix(...)` — runs sorting + gemm1 + gemm2 as one unit (see §6). |
| `aiter/fused_moe.py` | One dispatch branch in `fused_moe_` (40 lines) routing the gfx942 a16w4 SiTUv2 case to the above. |
| `op_tests/test_moe_a16w4_gfx942.py` | Correctness gate. Self-contained. |

## 2. What you must patch OUTSIDE this repo

Both of these are required. With either one missing the path silently does not run.

### 2.1 vLLM: relax the gfx950-only gate

`vllm/model_executor/layers/quantization/mxfp4.py`, in `_use_k3_situ_aiter()`:

```python
     return (
         rocm_aiter_ops.is_fused_moe_enabled()
-        and on_gfx950()
+        and True  # gfx942 is supported via AITER's a16wmix kernel
         and moe.activation == MoEActivation.SITU
         and moe.activation_situ_linear_beta is not None
```

Without it `_use_k3_situ_aiter` returns `False` on gfx942, vLLM never selects
`AITER_MXFP4_BF16`, and backend selection lands on a backend that rejects SiTUv2.

Verify from the server log:

```
INFO [mxfp4.py:520] Using AITER_MXFP4_BF16 for Kimi-K3 SiTU MXFP4 MoE.
```

### 2.2 Environment: `AITER_SITUV2_A8W4=0`

```bash
export AITER_SITUV2_A8W4=0
```

This flag selects between two different kernels, not two settings of one:

| Value | Activation dtype | Weight layout vLLM produces | Kernel |
|---|---|---|---|
| `0` (this work) | bf16 → **a16w4** | separated (`gate_up=False`) | `a16wmix_fused_moe` |
| `1` | fp8 → a8w4 | gate/up interleaved (`gate_up=True`) | the existing tuned flydsl afp8_wfp4 path |

Setting it to `1` makes `fused_moe_` quantize activations to fp8, so `q_dtype_a`
is no longer bf16, the dispatch branch in §1 does not match, and this kernel is
bypassed entirely. Several older Kimi-K3 launch scripts export `1` — check yours.

## 3. Reproducing correctness

```bash
python3 op_tests/test_moe_a16w4_gfx942.py
```

Expected (gfx942):

```
### tokens=128 model_dim=1024 inter_dim=256 E=8 topk=2
  a16w4 SiTUv2: cos=0.999995 rel_fro=2.980e-03 best_fit_gain=0.999966
  PASS
### tokens=128 model_dim=4096 inter_dim=384 E=8 topk=2
  a16w4 SiTUv2: cos=0.999996 rel_fro=2.987e-03 best_fit_gain=1.000018
  PASS
```

The test asserts three things beyond the tolerance:

- **The a16wmix path actually ran.** It spies on `fused_moe_a16wmix`; a silent
  fall-back to another backend fails the test rather than passing quietly.
- **Best-fit gain `⟨out,ref⟩/⟨ref,ref⟩` is within 2 % of 1.0.** Cosine is blind to a
  uniform scale error, which is the classic MXFP4 failure (a dropped or
  mis-applied E8M0 scale). Gain is not.
- **The gate can fail.** Scoring the same output against a SiLU reference must
  produce a gain far from 1.0. It scores cosine ≈ 0.96 there while gain collapses
  to ≈ 0.48 — which is why the gain assertion carries the weight.

`inter_dim=384` is not decorative: that is Kimi-K3's per-shard inter dim at TP=8,
and it is not a multiple of 256, so it exercises the adaptive stage tiles.

## 4. Reproducing the serving numbers

Two nodes, TP=8 + PP=2 (Kimi-K3 does not fit on 8 GPUs — a single node OOMs at
186 GiB/GPU). Both ranks need §2.1 and §2.2 applied.

```bash
# rank 0 (API server)                       # rank 1 adds --headless --node-rank 1
export VLLM_ROCM_USE_AITER=1
export AITER_SITUV2_A8W4=0
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
vllm serve <KIMI_K3_PATH> --trust-remote-code --moe-backend aiter \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --nnodes 2 --node-rank 0 --master-addr <RANK0_IP> \
  --gpu-memory-utilization 0.95 --max-num-seqs 128 --max-num-batched-tokens 4096 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["+fused_rms_norm_gated"]}'
```

Confirm the kernel is live (one line per worker, first MoE call):

```
[aiter.ops.flydsl.a16wmix_fused_moe] A16WMIX_ACTIVE gfx942 a16w4 SiTUv2 fused MoE:
E=896 model_dim=3584 inter_dim=384 topk=16 block_m=32 act=situv2 w1_layout=standard
```

Benchmark (repeat with 3 distinct seeds per side):

```bash
vllm bench serve --backend openai-chat --base-url http://localhost:8000 \
  --endpoint /v1/chat/completions --model <KIMI_K3_PATH> --trust-remote-code \
  --dataset-name random --random-input-len 2048 --random-output-len 256 \
  --num-prompts 32 --max-concurrency 8 --request-rate inf \
  --ignore-eos --seed <SEED> --percentile-metrics ttft,tpot,itl
```

To measure the baseline, flip §2.1 back to `and False` and add
`--moe-backend emulation`. Leaving §2.1 off *without* `--moe-backend emulation`
does not give you a baseline — it gives you the SiTUv2 `ValueError` above.

### Measured (2048 in / 256 out, concurrency 8, 32 prompts)

| Metric | a16w4 (n=3) | EMULATION (n=5) | Ratio |
|---|---|---|---|
| Output throughput | 41.95 tok/s | 5.24 tok/s | **8.01×** |
| Mean TPOT | 157.2 ms | 1474.9 ms | **9.38× lower** |
| Median TTFT | 8302 ms | 13132 ms | 1.58× lower |

Throughput and TPOT are trustworthy: run-to-run spread is under 5 % on both sides
against an 8–9× gap. **The TTFT ratio is not** — the emulation side ranged
8.7–18.8 s across its five runs, so three-to-five runs cannot resolve a TTFT
difference at `request_rate=inf`. Do not quote it as a prefill result.

## 5. What was changed in the kernel, and why

Against FlyDSL `47ed57a`, three files, 47 added lines.

**gfx942 fp4 upconvert (the actual porting work).** Upstream arch-gates the K=32
MFMA split, the A-tile VGPR staging and the int4 dequant for gfx942 already, but
the MXFP4 branch calls `v_cvt_scalef32_pk_bf16_fp4` unconditionally, and that
instruction is gfx950-only:

```
error: instruction not supported on this GPU (gfx942): v_cvt_scalef32_pk_bf16_fp4
```

`utils.py` adds `_fp4_nibble_to_bf16x8_sw`, an arithmetic E2M1 decode (sign /
2-bit exponent / 1-bit mantissa, with the subnormal case selected out), and
`gemm1.py` / `gemm2.py` take it under `use_k16`. This half is architecture work
and would apply upstream unchanged.

**flydsl 0.2.4 compatibility (local to this vendored copy).** Upstream targets
flydsl 0.3.1; AITER pins 0.2.4. Two API differences, both expressible in 0.2.4
without touching the flydsl package:

- `rocdl.s_waitcnt(lgkmcnt=0)` — named counters are a 0.3.1 addition. Replaced
  with the raw immediate `LGKMCNT_0 = 0xC07F`. `llvm-mc` assembles both to
  `[0x7f,0xc0,0x8c,0xbf]` on gfx942. Note this is *not* `s_waitcnt(0)`, which is
  `vmcnt(0) lgkmcnt(0) expcnt(0)` and would also stall the global loads the
  kernel deliberately keeps in flight.
- `from kernels.common import buffer_ops` → `from flydsl.expr import buffer_ops`.
  0.2.4 ships this module inside the package; the three symbols used
  (`buffer_store`, `create_buffer_resource_from_addr`, `get_element_ptr`) have
  byte-identical signatures. **This line breaks on 0.3.1**, where
  `flydsl.expr.buffer_ops` no longer exists — keep it out of any upstream PR.

## 6. Why a monolithic entry point

`fused_moe_`'s two-stage path allocates the stage1 → stage2 intermediate as
`(token_num, topk, inter_dim)`, indexed by `(token, slot)`. The a16wmix pair both
writes (`gemm1`) and reads (`gemm2`) a **sorted-position** buffer, and
`sorted_size` exceeds `token_num * topk` because each expert's run is padded up to
`block_m`. Bridging the two would mean re-addressing the gemm1 epilogue and the
gemm2 A-gather; running the pair as one unit does not.

The weight layout mapping was settled by measurement, not by naming. vLLM's
default (`AITER_SITUV2_A8W4=0` → `gate_up=False`) corresponds to the kernel's
`w1_layout="standard"`; a 2×2 sweep of the two flags against the torch reference
scored 0.999995 on the diagonal and ≈0.008 off it.

## 7. Environment these numbers came from

```
amd-aiter  0.1.17.dev395+g68e42f5f4      vllm   0.1.dev19253+g5f76ae224.d20260727.rocm723
flydsl     0.2.4                          torch  2.11.0+gitd0c8b1f
ROCm       7.2.3                          arch   gfx942 (MI300X)
image      rocm/ali-private:test-20260804
```

## 8. Limits — not verified

- **Only `AITER_SITUV2_A8W4=0` / bf16 activations.** The a8w4 path is untouched.
- **Serving measured at one operating point** (2048/256, concurrency 8). Other
  context lengths and concurrencies are not measured; do not extrapolate.
- **No accuracy evaluation.** Tensor-level agreement is established; no GSM8K /
  MMLU run was done, so downstream model quality is unverified.
- **Correctness tested up to E=64** in the harness (E=896 OOMs building fp32
  reference weights). E=896 ran in the live server and produced coherent output,
  but was not checked against a reference.
- **The software fp4 decode costs VALU.** It replaces one hardware instruction
  with roughly 8 arithmetic ops per nibble. It wins hugely against emulation; it
  has not been compared against a hypothetical hardware-decode gfx950 run.
