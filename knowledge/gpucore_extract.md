# GPU core dump extract (dumps deleted 2026-09-20)

The 20 `gpucore.*.gpu` files under `/home/mh` held 50.6 GB on a `/home` that was
99% full with 58 GB free -- the dumps were essentially all the remaining
headroom. This file is the text recovered from them before they were deleted.

## What is and is not recoverable from an AMDGPU core dump

Each file is an ELF core (`Type: CORE`, `Machine: AMD GPU`, `e_machine 0xe0`)
with no section headers and 17-58 `LOAD` segments carrying the GPU memory image.
Because that image includes the loaded code object, three things survive as plain
strings and are captured below:

- the target triple (every dump: `amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-`),
- the full kernel symbol table of whatever modules were resident,
- module paths, where any appeared.

**The faulting wave's PC does not survive in readable form.** The wave and queue
state sits in an `AMDGPU` note of type `0x21` that `readelf` will not decode, and
turning it into a PC requires `rocgdb` loaded against the exact code object that
produced the dump. Those `.so` files have been rebuilt many times since. So these
dumps could tell you *which kernels were loaded*, never *which line faulted* --
which is most of why they were not worth 50.6 GB.

## Inventory and classification

Classified by which kernel family appears in each dump. The split is clean and
mutually exclusive.

| dump | size | mtime | path |
|---|---|---|---|
| `aiter-kimi/gpucore.28380.gpu` | 31.14 GB | 2026-07-21 03:11:53 | MXFP4 MoE |
| `aiter-kimi/gpucore.28882.gpu` | 15.75 GB | 2026-07-21 03:14:36 | MXFP4 MoE |
| `aiter/gpucore.3768.gpu` | 0.33 GB | 2026-06-30 07:51:12 | neither |
| `aiter-topk/gpucore.823.gpu` | 0.19 GB | 2026-09-17 17:19:19 | AVO |
| `aiter-topk/gpucore.1502.gpu` | 0.19 GB | 2026-09-17 17:19:28 | AVO |
| `aiter-topk/gpucore.10666.gpu` | 0.19 GB | 2026-09-17 17:21:08 | AVO |
| `aiter-topk/gpucore.10942.gpu` | 0.19 GB | 2026-09-17 17:21:13 | aiter mb/ob |
| `aiter-topk/gpucore.822.gpu` | 0.19 GB | 2026-09-17 17:24:37 | AVO |
| `aiter-topk/gpucore.1501.gpu` | 0.19 GB | 2026-09-17 17:24:47 | AVO |
| `aiter-topk/gpucore.10665.gpu` | 0.19 GB | 2026-09-17 17:26:31 | AVO |
| `aiter-topk/gpucore.10941.gpu` | 0.19 GB | 2026-09-17 17:26:36 | aiter mb/ob |
| `aiter-topk/gpucore.11338.gpu` | 0.19 GB | 2026-09-17 18:26:59 | AVO |
| `aiter-topk/gpucore.11614.gpu` | 0.19 GB | 2026-09-17 18:27:05 | aiter mb/ob |
| `aiter-topk/gpucore.13883.gpu` | 0.19 GB | 2026-09-18 02:11:13 | AVO |
| `aiter-topk/gpucore.14159.gpu` | 0.19 GB | 2026-09-18 02:11:19 | aiter mb/ob |
| `aiter-topk/gpucore.3280.gpu` | 0.25 GB | 2026-09-18 05:40:42 | AVO |
| `aiter-topk/gpucore.4362.gpu` | 0.25 GB | 2026-09-18 05:40:58 | AVO |
| `aiter-topk/gpucore.7.gpu` | 0.25 GB | 2026-09-18 06:00:25 | AVO |
| `aiter-topk/gpucore.13863.gpu` | 0.19 GB | 2026-09-18 07:06:38 | aiter mb/ob |
| `aiter-topk/gpucore.13865.gpu` | 0.19 GB | 2026-09-19 13:38:40 | aiter mb/ob |

The 17 `aiter-topk` dumps split 11 AVO / 6 aiter mb/ob. AVO dumps carry
`phase_a_threshold`, `phase_b_filter_coop`, `phase_b_filter_waveseg`,
`phase_b_filter_wavestage`, `phase_c_select_contig` and
`phase_c_select_waveseg`; aiter dumps carry
`aiter::ob::radix_topk_one_block_kernel` and
`aiter::mb::radix_kernel_persistent`. No dump carries both.

**[inference, not established]** Five of those dumps pair up 5-6 seconds apart
with one AVO and one aiter member (17:21:08/17:21:13, 17:26:31/17:26:36,
18:26:59/18:27:05, 02:11:13/02:11:19). That is the shape
`bench/aiter_contract_audit.py` produces, since it runs each term on the `avo`
entry and then the `aiter` entry -- and it matches
`knowledge/aiter_contract_audit.md:256`, which records that the
`rowend_past_stride0` term originally "faults on both". Consistent with, not
proof of, that attribution: the PC is gone, so the term cannot be confirmed.

Note the count in `bench/stress_topk.py`'s docstring ("nine `gpucore.*.gpu`
dumps") was written when there were nine; 17 had accumulated by the time they
were removed.

The `aiter-kimi` pair is unrelated to top-k: `aiter::mxfp4_moe_sort_kernel`,
`aiter::opus_moe_sorting_entry`, `aiter::moe_smooth_per_token_scaled_quant_kernel`
and `aiter::topksoftmax_4x256x8_bf16` identify it as the Kimi MXFP4 MoE work of
2026-07-21.

## Per-dump detail

What follows is the raw per-file extract.

---


## `/mh/aiter-kimi/gpucore.28380.gpu`

- size: 33432875456 bytes (31.14 GB)
- mtime: 2026-07-21 03:11:53
- LOAD segments: 58
- strings scanned: 3454084 lines

### target triple
    amdgcn-amd-amdhsa--gfx950
    amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-

### module / code-object paths

### kernels present (distinct name, instantiation count)
       4069 rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel
       2096 at::native::vectorized_elementwise_kernel
       1542 at::native::elementwise_kernel_manual_unroll
        558 at::native::indexFuncLargeIndex
        524 at::native::unrolled_elementwise_kernel
        456 at::native::reduce_kernel
        320 at::native::indexFuncSmallIndex
        225 at::native::
        102 at::native::index_elementwise_kernel
         74 aiter::opus_moe_sorting_entry
         71 
         54 at::native::vectorized_templated_elementwise_kernel
         54 aiter::smooth_per_token_scaled_quant_kernel
         48 aiter::moe_smooth_per_token_scaled_quant_kernel_v1
         24 aiter::dynamic_per_group_scaled_quant_kernel
         13 at::cuda::cub::calc_block_sums
         12 at::native::_assert_async_cuda_kernel
         12 aiter::moe_smooth_per_token_scaled_quant_kernel_v2
         12 aiter::dynamic_per_token_scaled_quant_kernel
          8 aiter::partial_transpose_kernel
          8 aiter::mxfp4_moe_sort_kernel
          5 aiter::fused_mx_quant_moe_sort_kernel
          4 rocprim::ROCPRIM_400200_NS::detail::init_lookback_scan_state_kernel
          2 aiter::data_to_scale_kernel
          1 aiter::topksoftmax_4x256x8_bf16
          1 aiter::scaled_quant_kernel
          1 aiter::initializeScale
          1 _Zmd]\$4
          1 _ZN5aiter43moe_smooth_per_token_scaled_quant_kernel_v2IDhDB8_Li512ELi16EEEvPT0_PfPT_S4_PiS7_S7_iiiiiiiiiibb
          1 _ZN5aiter43moe_smooth_per_token_scaled_quant_kernel_v2IDhDB8_Li256ELi8EEEvPT0_PfPT_S4_PiS7_S7_iiiiiiiiiibb

### notable runtime strings
    hipError

## `/mh/aiter-kimi/gpucore.28882.gpu`

- size: 16911421488 bytes (15.75 GB)
- mtime: 2026-07-21 03:14:36
- LOAD segments: 51
- strings scanned: 7991230 lines

### target triple
    amdgcn-amd-amdhsa--gfx950
    amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-

### module / code-object paths

### kernels present (distinct name, instantiation count)
       4069 rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel
       2096 at::native::vectorized_elementwise_kernel
       1542 at::native::elementwise_kernel_manual_unroll
        558 at::native::indexFuncLargeIndex
        524 at::native::unrolled_elementwise_kernel
        456 at::native::reduce_kernel
        320 at::native::indexFuncSmallIndex
        225 at::native::
        102 at::native::index_elementwise_kernel
         74 aiter::opus_moe_sorting_entry
         71 
         54 at::native::vectorized_templated_elementwise_kernel
         54 aiter::smooth_per_token_scaled_quant_kernel
         48 aiter::moe_smooth_per_token_scaled_quant_kernel_v1
         24 aiter::dynamic_per_group_scaled_quant_kernel
         13 at::cuda::cub::calc_block_sums
         12 at::native::_assert_async_cuda_kernel
         12 aiter::moe_smooth_per_token_scaled_quant_kernel_v2
         12 aiter::dynamic_per_token_scaled_quant_kernel
          8 aiter::partial_transpose_kernel
          8 aiter::mxfp4_moe_sort_kernel
          5 aiter::fused_mx_quant_moe_sort_kernel
          4 rocprim::ROCPRIM_400200_NS::detail::init_lookback_scan_state_kernel
          2 aiter::data_to_scale_kernel
          1 aiter::topksoftmax_4x256x8_bf16
          1 aiter::scaled_quant_kernel
          1 aiter::initializeScale
          1 _Zmd]\$4
          1 _ZN5aiter43moe_smooth_per_token_scaled_quant_kernel_v2IDhDB8_Li512ELi16EEEvPT0_PfPT_S4_PiS7_S7_iiiiiiiiiibb
          1 _ZN5aiter43moe_smooth_per_token_scaled_quant_kernel_v2IDhDB8_Li256ELi8EEEvPT0_PfPT_S4_PiS7_S7_iiiiiiiiiibb

### notable runtime strings
    hipError

## `/mh/aiter-topk/gpucore.10665.gpu`

- size: 199178432 bytes (0.19 GB)
- mtime: 2026-09-17 17:26:31
- LOAD segments: 17
- strings scanned: 15333 lines

### target triple
    amdgcn-amd-amdhsa--gfx950
    amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-

### module / code-object paths

### kernels present (distinct name, instantiation count)
         84 at::native::vectorized_elementwise_kernel
         63 at::native::elementwise_kernel_manual_unroll
         21 at::native::unrolled_elementwise_kernel
         16 at::native::
          8 phase_c_select_waveseg
          4 phase_small_n_topk
          4 phase_d_fallback
          4 phase_c_select_contig
          4 phase_a_threshold
          2 phase_b_filter_wavestage
          2 phase_b_filter_waveseg
          2 phase_b_filter_coop
          2 phase_ab_fused
          1 fill_random_fp32
          1 fill_identity_rows
          1 d_xorwow_sequence_jump_matrices
          1 d_xorwow_jump_matrices
          1 d_mrg31k3p_A2P72
          1 d_mrg31k3p_A2P134
          1 d_mrg31k3p_A2
          1 d_mrg31k3p_A1P72
          1 d_mrg31k3p_A1P134
          1 d_mrg31k3p_A1
          1 d_lfsr113_sequence_jump_matrices
          1 d_lfsr113_jump_matrices
          1 d_A2P76
          1 d_A2P127
          1 d_A2
          1 d_A1P76
          1 d_A1P127

### notable runtime strings

## `/mh/aiter-topk/gpucore.10666.gpu`

- size: 199178432 bytes (0.19 GB)
- mtime: 2026-09-17 17:21:08
- LOAD segments: 17
- strings scanned: 15333 lines

### target triple
    amdgcn-amd-amdhsa--gfx950
    amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-

### module / code-object paths

### kernels present (distinct name, instantiation count)
         84 at::native::vectorized_elementwise_kernel
         63 at::native::elementwise_kernel_manual_unroll
         21 at::native::unrolled_elementwise_kernel
         16 at::native::
          8 phase_c_select_waveseg
          4 phase_small_n_topk
          4 phase_d_fallback
          4 phase_c_select_contig
          4 phase_a_threshold
          2 phase_b_filter_wavestage
          2 phase_b_filter_waveseg
          2 phase_b_filter_coop
          2 phase_ab_fused
          1 fill_random_fp32
          1 fill_identity_rows
          1 d_xorwow_sequence_jump_matrices
          1 d_xorwow_jump_matrices
          1 d_mrg31k3p_A2P72
          1 d_mrg31k3p_A2P134
          1 d_mrg31k3p_A2
          1 d_mrg31k3p_A1P72
          1 d_mrg31k3p_A1P134
          1 d_mrg31k3p_A1
          1 d_lfsr113_sequence_jump_matrices
          1 d_lfsr113_jump_matrices
          1 d_A2P76
          1 d_A2P127
          1 d_A2
          1 d_A1P76
          1 d_A1P127

### notable runtime strings

## `/mh/aiter-topk/gpucore.10941.gpu`

- size: 199854272 bytes (0.19 GB)
- mtime: 2026-09-17 17:26:36
- LOAD segments: 17
- strings scanned: 14380 lines

### target triple
    amdgcn-amd-amdhsa--gfx950
    amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-

### module / code-object paths

### kernels present (distinct name, instantiation count)
         84 at::native::vectorized_elementwise_kernel
         63 at::native::elementwise_kernel_manual_unroll
         21 at::native::unrolled_elementwise_kernel
         16 at::native::
         16 aiter::ob::radix_topk_one_block_kernel
          6 aiter::mb::radix_kernel_persistent
          1 d_xorwow_sequence_jump_matrices
          1 d_xorwow_jump_matrices
          1 d_mrg31k3p_A2P72
          1 d_mrg31k3p_A2P134
          1 d_mrg31k3p_A2
          1 d_mrg31k3p_A1P72
          1 d_mrg31k3p_A1P134
          1 d_mrg31k3p_A1
          1 d_lfsr113_sequence_jump_matrices
          1 d_lfsr113_jump_matrices
          1 d_A2P76
          1 d_A2P127
          1 d_A2
          1 d_A1P76
          1 d_A1P127
          1 d_A1

### notable runtime strings

## `/mh/aiter-topk/gpucore.10942.gpu`

- size: 199854272 bytes (0.19 GB)
- mtime: 2026-09-17 17:21:13
- LOAD segments: 17
- strings scanned: 14374 lines

### target triple
    amdgcn-amd-amdhsa--gfx950
    amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-

### module / code-object paths

### kernels present (distinct name, instantiation count)
         84 at::native::vectorized_elementwise_kernel
         63 at::native::elementwise_kernel_manual_unroll
         21 at::native::unrolled_elementwise_kernel
         16 at::native::
         16 aiter::ob::radix_topk_one_block_kernel
          6 aiter::mb::radix_kernel_persistent
          1 d_xorwow_sequence_jump_matrices
          1 d_xorwow_jump_matrices
          1 d_mrg31k3p_A2P72
          1 d_mrg31k3p_A2P134
          1 d_mrg31k3p_A2
          1 d_mrg31k3p_A1P72
          1 d_mrg31k3p_A1P134
          1 d_mrg31k3p_A1
          1 d_lfsr113_sequence_jump_matrices
          1 d_lfsr113_jump_matrices
          1 d_A2P76
          1 d_A2P127
          1 d_A2
          1 d_A1P76
          1 d_A1P127
          1 d_A1

### notable runtime strings

## `/mh/aiter-topk/gpucore.11338.gpu`

- size: 199182528 bytes (0.19 GB)
- mtime: 2026-09-17 18:26:59
- LOAD segments: 17
- strings scanned: 15329 lines

### target triple
    amdgcn-amd-amdhsa--gfx950
    amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-

### module / code-object paths

### kernels present (distinct name, instantiation count)
         84 at::native::vectorized_elementwise_kernel
         63 at::native::elementwise_kernel_manual_unroll
         21 at::native::unrolled_elementwise_kernel
         16 at::native::
          8 phase_c_select_waveseg
          4 phase_small_n_topk
          4 phase_d_fallback
          4 phase_c_select_contig
          4 phase_a_threshold
          2 phase_b_filter_wavestage
          2 phase_b_filter_waveseg
          2 phase_b_filter_coop
          2 phase_ab_fused
          1 fill_random_fp32
          1 fill_identity_rows
          1 d_xorwow_sequence_jump_matrices
          1 d_xorwow_jump_matrices
          1 d_mrg31k3p_A2P72
          1 d_mrg31k3p_A2P134
          1 d_mrg31k3p_A2
          1 d_mrg31k3p_A1P72
          1 d_mrg31k3p_A1P134
          1 d_mrg31k3p_A1
          1 d_lfsr113_sequence_jump_matrices
          1 d_lfsr113_jump_matrices
          1 d_A2P76
          1 d_A2P127
          1 d_A2
          1 d_A1P76
          1 d_A1P127

### notable runtime strings

## `/mh/aiter-topk/gpucore.11614.gpu`

- size: 199854272 bytes (0.19 GB)
- mtime: 2026-09-17 18:27:05
- LOAD segments: 17
- strings scanned: 14388 lines

### target triple
    amdgcn-amd-amdhsa--gfx950
    amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-

### module / code-object paths

### kernels present (distinct name, instantiation count)
         84 at::native::vectorized_elementwise_kernel
         63 at::native::elementwise_kernel_manual_unroll
         21 at::native::unrolled_elementwise_kernel
         16 at::native::
         16 aiter::ob::radix_topk_one_block_kernel
          6 aiter::mb::radix_kernel_persistent
          1 d_xorwow_sequence_jump_matrices
          1 d_xorwow_jump_matrices
          1 d_mrg31k3p_A2P72
          1 d_mrg31k3p_A2P134
          1 d_mrg31k3p_A2
          1 d_mrg31k3p_A1P72
          1 d_mrg31k3p_A1P134
          1 d_mrg31k3p_A1
          1 d_lfsr113_sequence_jump_matrices
          1 d_lfsr113_jump_matrices
          1 d_A2P76
          1 d_A2P127
          1 d_A2
          1 d_A1P76
          1 d_A1P127
          1 d_A1

### notable runtime strings

## `/mh/aiter-topk/gpucore.13863.gpu`

- size: 199854272 bytes (0.19 GB)
- mtime: 2026-09-18 07:06:38
- LOAD segments: 17
- strings scanned: 14384 lines

### target triple
    amdgcn-amd-amdhsa--gfx950
    amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-

### module / code-object paths

### kernels present (distinct name, instantiation count)
         84 at::native::vectorized_elementwise_kernel
         63 at::native::elementwise_kernel_manual_unroll
         21 at::native::unrolled_elementwise_kernel
         16 at::native::
         16 aiter::ob::radix_topk_one_block_kernel
          6 aiter::mb::radix_kernel_persistent
          1 d_xorwow_sequence_jump_matrices
          1 d_xorwow_jump_matrices
          1 d_mrg31k3p_A2P72
          1 d_mrg31k3p_A2P134
          1 d_mrg31k3p_A2
          1 d_mrg31k3p_A1P72
          1 d_mrg31k3p_A1P134
          1 d_mrg31k3p_A1
          1 d_lfsr113_sequence_jump_matrices
          1 d_lfsr113_jump_matrices
          1 d_A2P76
          1 d_A2P127
          1 d_A2
          1 d_A1P76
          1 d_A1P127
          1 d_A1

### notable runtime strings

## `/mh/aiter-topk/gpucore.13865.gpu`

- size: 199854272 bytes (0.19 GB)
- mtime: 2026-09-19 13:38:40
- LOAD segments: 17
- strings scanned: 14369 lines

### target triple
    amdgcn-amd-amdhsa--gfx950
    amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-

### module / code-object paths

### kernels present (distinct name, instantiation count)
         84 at::native::vectorized_elementwise_kernel
         63 at::native::elementwise_kernel_manual_unroll
         21 at::native::unrolled_elementwise_kernel
         16 at::native::
         16 aiter::ob::radix_topk_one_block_kernel
          6 aiter::mb::radix_kernel_persistent
          1 d_xorwow_sequence_jump_matrices
          1 d_xorwow_jump_matrices
          1 d_mrg31k3p_A2P72
          1 d_mrg31k3p_A2P134
          1 d_mrg31k3p_A2
          1 d_mrg31k3p_A1P72
          1 d_mrg31k3p_A1P134
          1 d_mrg31k3p_A1
          1 d_lfsr113_sequence_jump_matrices
          1 d_lfsr113_jump_matrices
          1 d_A2P76
          1 d_A2P127
          1 d_A2
          1 d_A1P76
          1 d_A1P127
          1 d_A1

### notable runtime strings

## `/mh/aiter-topk/gpucore.13883.gpu`

- size: 199182528 bytes (0.19 GB)
- mtime: 2026-09-18 02:11:13
- LOAD segments: 17
- strings scanned: 15329 lines

### target triple
    amdgcn-amd-amdhsa--gfx950
    amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-

### module / code-object paths

### kernels present (distinct name, instantiation count)
         84 at::native::vectorized_elementwise_kernel
         63 at::native::elementwise_kernel_manual_unroll
         21 at::native::unrolled_elementwise_kernel
         16 at::native::
          8 phase_c_select_waveseg
          4 phase_small_n_topk
          4 phase_d_fallback
          4 phase_c_select_contig
          4 phase_a_threshold
          2 phase_b_filter_wavestage
          2 phase_b_filter_waveseg
          2 phase_b_filter_coop
          2 phase_ab_fused
          1 fill_random_fp32
          1 fill_identity_rows
          1 d_xorwow_sequence_jump_matrices
          1 d_xorwow_jump_matrices
          1 d_mrg31k3p_A2P72
          1 d_mrg31k3p_A2P134
          1 d_mrg31k3p_A2
          1 d_mrg31k3p_A1P72
          1 d_mrg31k3p_A1P134
          1 d_mrg31k3p_A1
          1 d_lfsr113_sequence_jump_matrices
          1 d_lfsr113_jump_matrices
          1 d_A2P76
          1 d_A2P127
          1 d_A2
          1 d_A1P76
          1 d_A1P127

### notable runtime strings

## `/mh/aiter-topk/gpucore.14159.gpu`

- size: 199854272 bytes (0.19 GB)
- mtime: 2026-09-18 02:11:19
- LOAD segments: 17
- strings scanned: 14395 lines

### target triple
    amdgcn-amd-amdhsa--gfx950
    amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-

### module / code-object paths

### kernels present (distinct name, instantiation count)
         84 at::native::vectorized_elementwise_kernel
         63 at::native::elementwise_kernel_manual_unroll
         21 at::native::unrolled_elementwise_kernel
         16 at::native::
         16 aiter::ob::radix_topk_one_block_kernel
          6 aiter::mb::radix_kernel_persistent
          1 d_xorwow_sequence_jump_matrices
          1 d_xorwow_jump_matrices
          1 d_mrg31k3p_A2P72
          1 d_mrg31k3p_A2P134
          1 d_mrg31k3p_A2
          1 d_mrg31k3p_A1P72
          1 d_mrg31k3p_A1P134
          1 d_mrg31k3p_A1
          1 d_lfsr113_sequence_jump_matrices
          1 d_lfsr113_jump_matrices
          1 d_A2P76
          1 d_A2P127
          1 d_A2
          1 d_A1P76
          1 d_A1P127
          1 d_A1

### notable runtime strings

## `/mh/aiter-topk/gpucore.1501.gpu`

- size: 202516784 bytes (0.19 GB)
- mtime: 2026-09-17 17:24:47
- LOAD segments: 19
- strings scanned: 24463 lines

### target triple
    amdgcn-amd-amdhsa--gfx950
    amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-

### module / code-object paths

### kernels present (distinct name, instantiation count)
        180 at::native::vectorized_elementwise_kernel
        141 at::native::elementwise_kernel_manual_unroll
         78 
         45 at::native::unrolled_elementwise_kernel
         18 at::native::vectorized_templated_elementwise_kernel
         16 at::native::
          8 phase_c_select_waveseg
          4 phase_small_n_topk
          4 phase_d_fallback
          4 phase_c_select_contig
          4 phase_a_threshold
          2 phase_b_filter_wavestage
          2 phase_b_filter_waveseg
          2 phase_b_filter_coop
          2 phase_ab_fused
          1 fill_random_fp32
          1 fill_identity_rows
          1 d_xorwow_sequence_jump_matrices
          1 d_xorwow_jump_matrices
          1 d_mrg31k3p_A2P72
          1 d_mrg31k3p_A2P134
          1 d_mrg31k3p_A2
          1 d_mrg31k3p_A1P72
          1 d_mrg31k3p_A1P134
          1 d_mrg31k3p_A1
          1 d_lfsr113_sequence_jump_matrices
          1 d_lfsr113_jump_matrices
          1 d_A2P76
          1 d_A2P127
          1 d_A2

### notable runtime strings

## `/mh/aiter-topk/gpucore.1502.gpu`

- size: 202516784 bytes (0.19 GB)
- mtime: 2026-09-17 17:19:28
- LOAD segments: 19
- strings scanned: 24463 lines

### target triple
    amdgcn-amd-amdhsa--gfx950
    amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-

### module / code-object paths

### kernels present (distinct name, instantiation count)
        180 at::native::vectorized_elementwise_kernel
        141 at::native::elementwise_kernel_manual_unroll
         78 
         45 at::native::unrolled_elementwise_kernel
         18 at::native::vectorized_templated_elementwise_kernel
         16 at::native::
          8 phase_c_select_waveseg
          4 phase_small_n_topk
          4 phase_d_fallback
          4 phase_c_select_contig
          4 phase_a_threshold
          2 phase_b_filter_wavestage
          2 phase_b_filter_waveseg
          2 phase_b_filter_coop
          2 phase_ab_fused
          1 fill_random_fp32
          1 fill_identity_rows
          1 d_xorwow_sequence_jump_matrices
          1 d_xorwow_jump_matrices
          1 d_mrg31k3p_A2P72
          1 d_mrg31k3p_A2P134
          1 d_mrg31k3p_A2
          1 d_mrg31k3p_A1P72
          1 d_mrg31k3p_A1P134
          1 d_mrg31k3p_A1
          1 d_lfsr113_sequence_jump_matrices
          1 d_lfsr113_jump_matrices
          1 d_A2P76
          1 d_A2P127
          1 d_A2

### notable runtime strings

## `/mh/aiter-topk/gpucore.3280.gpu`

- size: 266603136 bytes (0.25 GB)
- mtime: 2026-09-18 05:40:42
- LOAD segments: 25
- strings scanned: 160168 lines

### target triple
    amdgcn-amd-amdhsa--gfx950
    amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-

### module / code-object paths

### kernels present (distinct name, instantiation count)
       1028 at::native::vectorized_elementwise_kernel
        733 at::native::elementwise_kernel_manual_unroll
        558 at::native::indexFuncLargeIndex
        320 at::native::indexFuncSmallIndex
        257 at::native::unrolled_elementwise_kernel
        208 at::native::
        149 
         36 at::native::vectorized_templated_elementwise_kernel
         12 at::native::_assert_async_cuda_kernel
          8 phase_c_select_waveseg
          4 phase_small_n_topk
          4 phase_d_fallback
          4 phase_c_select_contig
          4 phase_a_threshold
          2 phase_b_filter_wavestage
          2 phase_b_filter_waveseg
          2 phase_b_filter_coop
          2 phase_ab_fused
          1 fill_random_fp32
          1 fill_identity_rows
          1 d_xorwow_sequence_jump_matrices
          1 d_xorwow_jump_matrices
          1 d_mrg31k3p_A2P72
          1 d_mrg31k3p_A2P134
          1 d_mrg31k3p_A2
          1 d_mrg31k3p_A1P72
          1 d_mrg31k3p_A1P134
          1 d_mrg31k3p_A1
          1 d_lfsr113_sequence_jump_matrices
          1 d_lfsr113_jump_matrices

### notable runtime strings

## `/mh/aiter-topk/gpucore.4362.gpu`

- size: 266603136 bytes (0.25 GB)
- mtime: 2026-09-18 05:40:58
- LOAD segments: 25
- strings scanned: 160168 lines

### target triple
    amdgcn-amd-amdhsa--gfx950
    amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-

### module / code-object paths

### kernels present (distinct name, instantiation count)
       1028 at::native::vectorized_elementwise_kernel
        733 at::native::elementwise_kernel_manual_unroll
        558 at::native::indexFuncLargeIndex
        320 at::native::indexFuncSmallIndex
        257 at::native::unrolled_elementwise_kernel
        208 at::native::
        149 
         36 at::native::vectorized_templated_elementwise_kernel
         12 at::native::_assert_async_cuda_kernel
          8 phase_c_select_waveseg
          4 phase_small_n_topk
          4 phase_d_fallback
          4 phase_c_select_contig
          4 phase_a_threshold
          2 phase_b_filter_wavestage
          2 phase_b_filter_waveseg
          2 phase_b_filter_coop
          2 phase_ab_fused
          1 fill_random_fp32
          1 fill_identity_rows
          1 d_xorwow_sequence_jump_matrices
          1 d_xorwow_jump_matrices
          1 d_mrg31k3p_A2P72
          1 d_mrg31k3p_A2P134
          1 d_mrg31k3p_A2
          1 d_mrg31k3p_A1P72
          1 d_mrg31k3p_A1P134
          1 d_mrg31k3p_A1
          1 d_lfsr113_sequence_jump_matrices
          1 d_lfsr113_jump_matrices

### notable runtime strings

## `/mh/aiter-topk/gpucore.7.gpu`

- size: 266603136 bytes (0.25 GB)
- mtime: 2026-09-18 06:00:25
- LOAD segments: 25
- strings scanned: 160168 lines

### target triple
    amdgcn-amd-amdhsa--gfx950
    amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-

### module / code-object paths

### kernels present (distinct name, instantiation count)
       1028 at::native::vectorized_elementwise_kernel
        733 at::native::elementwise_kernel_manual_unroll
        558 at::native::indexFuncLargeIndex
        320 at::native::indexFuncSmallIndex
        257 at::native::unrolled_elementwise_kernel
        208 at::native::
        149 
         36 at::native::vectorized_templated_elementwise_kernel
         12 at::native::_assert_async_cuda_kernel
          8 phase_c_select_waveseg
          4 phase_small_n_topk
          4 phase_d_fallback
          4 phase_c_select_contig
          4 phase_a_threshold
          2 phase_b_filter_wavestage
          2 phase_b_filter_waveseg
          2 phase_b_filter_coop
          2 phase_ab_fused
          1 fill_random_fp32
          1 fill_identity_rows
          1 d_xorwow_sequence_jump_matrices
          1 d_xorwow_jump_matrices
          1 d_mrg31k3p_A2P72
          1 d_mrg31k3p_A2P134
          1 d_mrg31k3p_A2
          1 d_mrg31k3p_A1P72
          1 d_mrg31k3p_A1P134
          1 d_mrg31k3p_A1
          1 d_lfsr113_sequence_jump_matrices
          1 d_lfsr113_jump_matrices

### notable runtime strings

## `/mh/aiter-topk/gpucore.822.gpu`

- size: 202516784 bytes (0.19 GB)
- mtime: 2026-09-17 17:24:37
- LOAD segments: 19
- strings scanned: 24463 lines

### target triple
    amdgcn-amd-amdhsa--gfx950
    amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-

### module / code-object paths

### kernels present (distinct name, instantiation count)
        180 at::native::vectorized_elementwise_kernel
        141 at::native::elementwise_kernel_manual_unroll
         78 
         45 at::native::unrolled_elementwise_kernel
         18 at::native::vectorized_templated_elementwise_kernel
         16 at::native::
          8 phase_c_select_waveseg
          4 phase_small_n_topk
          4 phase_d_fallback
          4 phase_c_select_contig
          4 phase_a_threshold
          2 phase_b_filter_wavestage
          2 phase_b_filter_waveseg
          2 phase_b_filter_coop
          2 phase_ab_fused
          1 fill_random_fp32
          1 fill_identity_rows
          1 d_xorwow_sequence_jump_matrices
          1 d_xorwow_jump_matrices
          1 d_mrg31k3p_A2P72
          1 d_mrg31k3p_A2P134
          1 d_mrg31k3p_A2
          1 d_mrg31k3p_A1P72
          1 d_mrg31k3p_A1P134
          1 d_mrg31k3p_A1
          1 d_lfsr113_sequence_jump_matrices
          1 d_lfsr113_jump_matrices
          1 d_A2P76
          1 d_A2P127
          1 d_A2

### notable runtime strings

## `/mh/aiter-topk/gpucore.823.gpu`

- size: 202516784 bytes (0.19 GB)
- mtime: 2026-09-17 17:19:19
- LOAD segments: 19
- strings scanned: 24463 lines

### target triple
    amdgcn-amd-amdhsa--gfx950
    amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-

### module / code-object paths

### kernels present (distinct name, instantiation count)
        180 at::native::vectorized_elementwise_kernel
        141 at::native::elementwise_kernel_manual_unroll
         78 
         45 at::native::unrolled_elementwise_kernel
         18 at::native::vectorized_templated_elementwise_kernel
         16 at::native::
          8 phase_c_select_waveseg
          4 phase_small_n_topk
          4 phase_d_fallback
          4 phase_c_select_contig
          4 phase_a_threshold
          2 phase_b_filter_wavestage
          2 phase_b_filter_waveseg
          2 phase_b_filter_coop
          2 phase_ab_fused
          1 fill_random_fp32
          1 fill_identity_rows
          1 d_xorwow_sequence_jump_matrices
          1 d_xorwow_jump_matrices
          1 d_mrg31k3p_A2P72
          1 d_mrg31k3p_A2P134
          1 d_mrg31k3p_A2
          1 d_mrg31k3p_A1P72
          1 d_mrg31k3p_A1P134
          1 d_mrg31k3p_A1
          1 d_lfsr113_sequence_jump_matrices
          1 d_lfsr113_jump_matrices
          1 d_A2P76
          1 d_A2P127
          1 d_A2

### notable runtime strings

## `/mh/aiter/gpucore.3768.gpu`

- size: 356580704 bytes (0.33 GB)
- mtime: 2026-06-30 07:51:12
- LOAD segments: 38
- strings scanned: 206469 lines

### target triple
    amdgcn-amd-amdhsa--gfx950
    amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-

### module / code-object paths

### kernels present (distinct name, instantiation count)
       3068 rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel
       1460 at::native::vectorized_elementwise_kernel
       1077 at::native::elementwise_kernel_manual_unroll
        365 at::native::unrolled_elementwise_kernel
        285 at::native::
        117 rocprim::ROCPRIM_400200_NS::detail::device_merge_sort_compile_time_verifier_arch
        102 at::native::index_elementwise_kernel
         90 mla_decode_splitkv_s2_kernel
         90 at::native::vectorized_templated_elementwise_kernel
         87 
         60 at::native::reduce_kernel
         51 mla_decode_splitkv_s1_16mx8_kernel
         51 mla_decode_16mx8_32nx1_kernel
         34 mla_decode_splitkv_s1_16mx1_kernel
         34 mla_decode_16mx1_16nx4_kernel
         12 at::native::_assert_async_cuda_kernel
          9 rocprim::ROCPRIM_400200_NS::block_radix_sort
          5 mla_decode_qpack_s1_16mx8_kernel
          5 mla_decode_qpack_16mx8_kernel
          2 opus::numeric_limits
          2 at::native::vectorized_gather_kernel
          1 mla_decode_qpack_h8_kernel
          1 _Z34mla_decode_splitkv_s1_16mx8_kernelI32mla_decode_rope_16mx8_nw8_traitsILi9EDF16bEEv21mla_decode_opus_kargs
          1 _Z34mla_decode_splitkv_s1_16mx8_kernelI32mla_decode_rope_16mx8_nw8_traitsILi8EDF16bEEv21mla_decode_opus_kargs
          1 _Z34mla_decode_splitkv_s1_16mx8_kernelI32mla_decode_rope_16mx8_nw8_traitsILi7EDF16bEEv21mla_decode_opus_kargs
          1 _Z34mla_decode_splitkv_s1_16mx8_kernelI32mla_decode_rope_16mx8_nw8_traitsILi6EDF16bEEv21mla_decode_opus_kargs
          1 _Z34mla_decode_splitkv_s1_16mx8_kernelI32mla_decode_rope_16mx8_nw8_traitsILi5EDF16bEEv21mla_decode_opus_kargs
          1 _Z34mla_decode_splitkv_s1_16mx8_kernelI32mla_decode_rope_16mx8_nw8_traitsILi4EDF16bEEv21mla_decode_opus_kargs
          1 _Z34mla_decode_splitkv_s1_16mx8_kernelI32mla_decode_rope_16mx8_nw8_traitsILi3EDF16bEEv21mla_decode_opus_kargs
          1 _Z34mla_decode_splitkv_s1_16mx8_kernelI32mla_decode_rope_16mx8_nw8_traitsILi2EDF16bEEv21mla_decode_opus_kargs

### notable runtime strings
    hipError

