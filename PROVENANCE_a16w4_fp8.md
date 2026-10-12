# What /opt/aiter actually contains

`git log` in this repo is **not** a description of the installed code. Read this first.

## Composition

| Layer | Source |
|---|---|
| Base tree | `aiter` `main` @ `c1ada4884` (what `git log` shows) |
| a16w4 fp8 MoE work | branch `kimi-k3-a16w4-gfx942` @ `adbdc39` of `github.com/AFDEAPAC/aiter`, applied as file content, not as commits |
| flydsl import rewrite | applied on top, see below |
| Two patches that exist only here | see below |

The seven a16w4 files are byte-identical to `adbdc39` (modulo the import rewrite):

```
aiter/ops/flydsl/kernels/moe_2stage_a16wmix/{gemm1,gemm2,utils}.py
aiter/ops/flydsl/{moe_a16wmix_host,a16wmix_fused_moe,a16wmix_fp8_prep}.py
op_tests/test_moe_a16w4_gfx942.py
```

## Why the branch is not simply checked out here

The fork branched from an **older upstream aiter** than this image ships. `git diff c1ada4884 adbdc39`
is **1475 files, +53,866 / -130,534 lines**, and 1468 of those files have nothing to do with a16w4.
Checking out `adbdc39` therefore rolls the whole tree back — `aiter/fused_moe.py` alone drops from 4085
to 3500 lines and loses the `_is_a16w4_situv2` dispatch guard this image's serving path relies on.

So the two lineages are deliberately kept apart: this image keeps the newer upstream base, and only the
a16w4 files ride on top. Making `git log` "honest" by checking out the branch would break the image.

## The flydsl import rewrite

The commits target flydsl 0.2.4, which exports `buffer_ops` from `flydsl.expr`. This image has
**flydsl 0.3.1, which does not** — aiter vendors it instead. Every a16w4 file therefore has

    from flydsl.expr import ..., buffer_ops, ...   ->   from flydsl.expr import ...
                                                       from aiter.ops.flydsl.kernels import buffer_ops

Without this the kernels do not import at all.

## The two patches that exist only in this image

- `aiter/fused_moe.py` — `AITER_A16W4_ALLOW_EP=1` escape on the `expert_mask is not None` guard, so the
  a16w4 path is reachable under expert parallelism. Not committed upstream or on the fork.
- `aiter/ops/triton/quant/fused_fp8_quant.py` — circular-import workaround for `get_fp8_e4m3_dtype()`.

## Verifying this claim

    cd /opt/aiter
    git fetch --no-tags https://github.com/AFDEAPAC/aiter.git kimi-k3-a16w4-gfx942
    git diff FETCH_HEAD -- aiter/ops/flydsl/kernels/moe_2stage_a16wmix/ \
        aiter/ops/flydsl/a16wmix_fp8_prep.py

Expect **only** the `buffer_ops` import lines to differ. Anything else means the tree has drifted.

## Running it

    /opt/scripts/start_kimi_k3_fp8.sh          # both stages on fp8
    /opt/scripts/check_fp8_arm.sh              # confirm the active arm from the server log

Arms are selected by `AITER_A16WMIX_FP8` (stage1) and `AITER_A16WMIX_FP8_S2` (stage2); the server logs
`stage1_a=... stage2_a=...` once per shape. Both default to off, i.e. bf16.
