# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Every environment variable MegaMoEV2 reads, in one place.

Read through this module (``envs.AITER_MEGA_...``) rather than ``os.environ``.
A value is read when it is accessed, so the caller decides when it takes effect:
at import, when a kernel is compiled, or when an instance is built.
"""

import os
from collections.abc import Callable
from typing import Any


def _flag(value: str) -> bool:
    return value == "1"


# name -> (parser, default)
environment_variables: dict[str, tuple[Callable[[str], Any], str]] = {
    # Largest MTPR still dispatched fixed-slot (direct expert slots, no count
    # exchange). 511 / 1023 / 2047 admit the 256 / 512 / 1024-token capacities,
    # validated on the EP8 v4_pro layout. Read once, at import.
    "AITER_MEGA_FIXED_SLOT_MAX_MTPR": (int, "255"),
    # Default of forward(mask_invalid_slots=...): combine skips top-k slots with
    # id -1. Off because the id check costs prefill time; callers that pad with
    # -1 ask for it per call.
    "AITER_MEGA_COMBINE_MASK": (_flag, "0"),
    # Load the AOT Stage1/Stage2 bundles when an instance is built.
    "AITER_MEGA_MOE_PRELOAD": (_flag, "0"),
    # Fallbacks for optimizations that are on by default (read per Stage1 compile):
    # GEMM1 quant epilogue columns per lane, 8 or 2.
    "AITER_MEGA_S1_EPI_EVEC": (int, "8"),
    # Paired K loop for fixed-slot GEMM1 tiles with LDS-DMA A copies.
    "AITER_MEGA_S1_FIXED_KPAIR": (_flag, "1"),
    # Compact dispatch producers keep two rows in flight.
    "AITER_MEGA_DISPATCH_FAST_COPY": (_flag, "1"),
    # Record a per-workgroup Stage1 timeline (see trace.py). Read per Stage1
    # compile and when an instance is built; off adds no code to the kernel.
    "AITER_MEGA_S1_TRACE": (_flag, "0"),
    # Debug timing only: Stage1 keeps dispatch and every readiness wait but
    # skips the GEMM1 tile body (its output is garbage).
    "AITER_MEGA_S1_TRACE_SKIP_GEMM": (_flag, "0"),
    # gfx942 GEMM1 with per-token FP8 activations and FP8 MFMA (instead of the
    # per-1x32 bf16 dequant path). Read per Stage1 compile and per instance.
    "AITER_MEGA_GFX942_A_PER_TOKEN": (_flag, "1"),
    # gfx942 GEMM2 on FP8 MFMA: the GEMM1 epilogue shares one E8M0 exponent per
    # 128 intermediate columns and GEMM2 scales each 128-K step per row.
    "AITER_MEGA_GFX942_A2_FP8": (_flag, "1"),
    # gfx942 SBM16 + Stage2 BM16 for buckets <= 128 tokens/rank (needs per-token A).
    "AITER_MEGA_GFX942_SBM16": (_flag, "1"),
    # gfx942 fixed-slot dispatch (no prepare kernel) for MTPR <= the fixed-slot max
    # at EP <= 8 and up to 128 experts per rank.
    "AITER_MEGA_GFX942_FIXED_SLOT": (_flag, "1"),
}


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        parse, default = environment_variables[name]
        return parse(os.environ.get(name, default))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(environment_variables)
