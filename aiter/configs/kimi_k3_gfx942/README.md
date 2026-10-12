# Kimi-K3 tuned tables for gfx942 (MI308X)

Used by the Kimi-K3 TP8 x PP2 serving setup (sglang `kimi` branch, `kimi_k3_deploy/`). This directory is
outside `model_configs/`, so aiter does not merge these files by default (`aiter/jit/core.py`
`get_config_file` only globs `configs/model_configs/`). Select them with:

| File | Env var |
|---|---|
| `a16wmix_tuned.csv` | `FLYDSL_A16WMIX_TUNED_CSV` (a16wmix bf16 x MXFP4 MoE tiles) |
| `kimik3_bf16_tuned_gemm_gfx942.csv` | `AITER_CONFIG_GEMM_BF16` |
| `a8w8_bpreshuffle_tuned_gfx942.csv` | `AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE` |
