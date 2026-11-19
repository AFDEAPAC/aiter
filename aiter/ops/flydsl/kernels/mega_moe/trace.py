# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Opt-in Stage1 timeline (AITER_MEGA_S1_TRACE=1).

Each workgroup appends fixed-size events ``(kind, t0, t1, info)`` of int64 to
``events[block, slot]``, where ``t0``/``t1`` are ``s_memrealtime`` ticks (a
constant 100 MHz clock on CDNA). Kernels emit none of this unless the flag is
set at compile time, so the default build is unchanged.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import T

from .. import communication_ops_utils as comm_ops
from ..tensor_shim import ptr_buf_tensor

TRACE_EVENTS_SLOT = 53
TRACE_COUNT_SLOT = 54
MAX_EVENTS = 1024
FIELDS = 4

# info layouts: CTA -> is_producer; PRODUCER -> producer slot;
# TASK -> segment | chunk << 16 | rows << 32; WAIT / TILE -> flat work id.
KIND_CTA = 0
KIND_PRODUCER = 1
KIND_TASK = 2
KIND_WAIT = 3
KIND_TILE = 4


def now():
    return fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))


@flyc.jit
def record(addr_disp, kind, t0, t1, info):
    """Append one event for this workgroup; call from a single thread."""
    table = ptr_buf_tensor(addr_disp, fx.Int64)
    events = table[fx.Int32(TRACE_EVENTS_SLOT)]
    counts = table[fx.Int32(TRACE_COUNT_SLOT)]
    block = fx.Int32(fx.block_idx.x)
    slot = fx.Int32(
        comm_ops.atomic_add_agent(counts + fx.Int64(block) * fx.Int64(4), fx.Int32(1))
    )
    if slot < fx.Int32(MAX_EVENTS):
        buf = ptr_buf_tensor(events, fx.Int64)
        base = (block * fx.Int32(MAX_EVENTS) + slot) * fx.Int32(FIELDS)
        buf[base] = fx.Int64(kind)
        buf[base + fx.Int32(1)] = fx.Int64(t0)
        buf[base + fx.Int32(2)] = fx.Int64(t1)
        buf[base + fx.Int32(3)] = fx.Int64(info)
