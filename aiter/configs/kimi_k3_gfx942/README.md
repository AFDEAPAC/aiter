# Kimi-K3 tuned tables for gfx942 (MI300X)

Used by the Kimi-K3 TP8 x PP2 serving setup (`kimi_k3_deploy/` in the SGLang branch of the same name). This
directory is outside `model_configs/`, so aiter does not merge these files by default (`aiter/jit/core.py`
`get_config_file` only globs `configs/model_configs/`). Select them with:

| File | Env var |
|---|---|
| `a16wmix_tuned.csv` | `FLYDSL_A16WMIX_TUNED_CSV` (a16wmix bf16 x MXFP4 MoE tiles, swept with `tune_a16wmix_moe.py`) |
| `kimik3_bf16_tuned_gemm_gfx942.csv` | `AITER_CONFIG_GEMM_BF16` |
| `a8w8_bpreshuffle_tuned_gfx942.csv` | `AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE` |
| `moe_fp8_table_mi300x.csv` | `FLYDSL_A16WMIX_FP8_TUNED_CSV` (a16wmix MoE tiles for the fp8 kernels, 1-8192 tokens; read only when `AITER_A16WMIX_FP8` / `AITER_A16WMIX_FP8_S2` are on) |

The MoE tables and the `cu_num` 304 rows of the GEMM tables were tuned on MI300X. With the fp8 table
in place, the fp8 path does not read `a16wmix_tuned.csv`.
