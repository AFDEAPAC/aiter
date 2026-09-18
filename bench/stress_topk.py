#!/usr/bin/env python3
"""Stress the AVO top-k entry with boundary and hostile inputs.

Why this is not just more grid points
-------------------------------------
`bench/verify_grid.py` covers 587 shapes x 5 distributions, but every one of
those shapes is well-formed, and it could not be otherwise: the extents come
from `bench/grid.py` and `benchmark_topk` clamps them again on the way in
(`benchmark_topk.hip.cpp:1531-1533`), so a `rowEnd` past the pitch cannot be
expressed through that harness at all. `bench/aiter_contract_audit.py` covers
the DISPATCH layer, term by term, against aiter. Neither asks the question this
file asks: given a hostile argument, does the GPU survive?

That question is not hypothetical. The HIP 700 fault on unaligned row bases
(v5 Stage 2) was found by accident while scoping an unrelated tuning stage --
`knowledge/aiter_contract_audit.md` says so in its first paragraph -- and
`/home/mh/aiter-topk` still carries nine `gpucore.*.gpu` dumps from that period.
A fault found by accident is a fault that shipped.

Three techniques do the work
----------------------------
1. **+inf poison.** Everything outside `[rowStart, rowEnd)` is set to `+inf`,
   including one whole extra row past the logical matrix. `+inf` outranks every
   real logit, so an out-of-range read surfaces as an index outside the slice
   rather than as a fault we might not get. That distinction matters: the
   caching allocator usually backs a 3-float over-read with mapped memory, so
   NOT faulting is not evidence of staying in bounds. This is why the Stage 2
   bug only faulted when the slice ran all the way to the pitch.

   NaN would be the wrong poison. The AVO key
   `(u & 0x80000000) ? ~u : (u ^ 0x80000000)` sends -NaN below -inf and drops
   it (`knowledge/aiter_contract_audit.md`), so a NaN poison could be read and
   never seen. vLLM's `test_deep_select_topk` can use NaN only because
   DeepSelect ships `abort_when_nan_found=True`; we have no such trap.

2. **One subprocess per case.** A memory fault poisons the HIP context, so
   without isolation one bad case reports a wall of false failures after it.
   Borrowed from `aiter_contract_audit.py`, along with its two import helpers.
   A child that dies without printing a verdict IS the failure signal: that is
   what "must not core dump" means operationally.

3. **Clamped-extent oracle.** Every expectation is computed from
   `start = clamp(rowStarts[r], 0, pitch)` and `end = clamp(rowEnds[r], 0,
   pitch)`. That is a no-op for a well-formed row and is the DEFINED behaviour
   for a hostile one, so one oracle serves both and the file does not need a
   second notion of correctness for the hostile half.

Scope
-----
The AVO entry itself (`top_k_per_row_prefill_avo`), not the dispatcher. Routing
is already covered by `aiter_contract_audit.py`, and pushing these cases through
`top_k_per_row_prefill` would mostly measure aiter's mb/ob path instead: that
dispatch needs `stride0 >= 32768` AND `topk_avo_supports()` (`aiter/ops/topk.py`),
and most hostile shapes fail one of the two.

Run inside the correctness image with both repos mounted:

  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    --ipc=host --shm-size 16G \
    -v /home/mh/aiter-topk:/aiter -v /home/mh/topk-prefill-avo:/topk -w /aiter \
    rocm/ali-private:ubuntu22.04_rocm10.0.0_cp313_torch2.12.0_vllm_dsv4_20260916 \
    python /topk/bench/stress_topk.py --smoke
"""

import argparse
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aiter_contract_audit import AITER_ROOT, stub_flydsl, use_mounted_aiter  # noqa: E402

# csrc/topk_shape.hip.hpp: k above this is declined by topk_avo_supports().
PHASE_C_CAP_MAX = 8192
INT32_MAX = 2**31 - 1

# Keep a case under ~256 MB of logits so the largest widths still fit alongside
# the poison row and the workspace.
MAX_ELEMS = 64 << 20


# --------------------------------------------------------------------------
# Case table. expect is "serve" (in contract: must produce a correct answer) or
# "reject" (out of contract: must raise, or return a defined clamped answer --
# either is acceptable, a fault is not).
# --------------------------------------------------------------------------
# name, m, n, k, starts, ends, dist, ws, values, stride1, expect
BOUNDARY = [
    # --- shape boundaries -------------------------------------------------
    ("m_zero",           0,   65536, 2048, "zero", "pitch",      "uniform",  "auto", 1, 1, "serve"),
    ("m_one",            1,   65536, 2048, "zero", "pitch",      "uniform",  "auto", 1, 1, "serve"),
    ("m_three",          3,   65536, 2048, "zero", "pitch",      "uniform",  "auto", 1, 1, "serve"),
    ("m_255",          255,   65536, 2048, "zero", "pitch",      "uniform",  "auto", 1, 1, "serve"),
    ("m_4097",        4097,   32768, 2048, "zero", "pitch",      "uniform",  "auto", 1, 1, "serve"),
    ("n_one",           64,       1,    1, "zero", "pitch",      "uniform",  "auto", 1, 1, "serve"),
    ("n_three",         64,       3,    1, "zero", "pitch",      "uniform",  "auto", 1, 1, "serve"),
    ("n_four",          64,       4,    2, "zero", "pitch",      "uniform",  "auto", 1, 1, "serve"),
    ("n_2047",          64,    2047, 1024, "zero", "pitch",      "uniform",  "auto", 1, 1, "serve"),
    ("n_2049",          64,    2049, 1024, "zero", "pitch",      "uniform",  "auto", 1, 1, "serve"),
    ("n_odd_small",     64,   12289, 2048, "zero", "pitch",      "uniform",  "auto", 1, 1, "serve"),
    ("n_odd_large",     64,  131073, 2048, "zero", "pitch",      "uniform",  "auto", 1, 1, "serve"),
    ("n_max",            8, 1048576, 2048, "zero", "pitch",      "uniform",  "auto", 1, 1, "serve"),
    ("n_zero",          64,       0, 2048, "zero", "pitch",      "uniform",  "auto", 1, 1, "reject"),
    # --- k boundaries -----------------------------------------------------
    ("k_zero",          64,   65536,    0, "zero", "pitch",      "uniform",  "auto", 1, 1, "reject"),
    ("k_neg",           64,   65536,   -1, "zero", "pitch",      "uniform",  "auto", 1, 1, "reject"),
    ("k_one",           64,   65536,    1, "zero", "pitch",      "uniform",  "auto", 1, 1, "serve"),
    ("k_three",         64,   65536,    3, "zero", "pitch",      "uniform",  "auto", 1, 1, "serve"),
    ("k_at_cap",        64,   65536, 8192, "zero", "pitch",      "uniform",  "auto", 1, 1, "serve"),
    ("k_over_cap",      64,   65536, 8193, "zero", "pitch",      "uniform",  "auto", 1, 1, "reject"),
    ("k_over_n",        64,     512, 2048, "zero", "pitch",      "uniform",  "auto", 1, 1, "serve"),
    # --- hostile rowStarts -------------------------------------------------
    ("start_ramp1",     64,  131072, 2048, "ramp1",      "pitch", "uniform", "auto", 1, 1, "serve"),
    ("start_ramp65",    64,  131072, 2048, "ramp65",     "pitch", "uniform", "auto", 1, 1, "serve"),
    ("start_neg",       64,  131072, 2048, "neg",        "pitch", "uniform", "auto", 1, 1, "reject"),
    ("start_at_pitch",  64,  131072, 2048, "at_pitch",   "pitch", "uniform", "auto", 1, 1, "reject"),
    ("start_past",      64,  131072, 2048, "past_pitch", "pitch", "uniform", "auto", 1, 1, "reject"),
    # --- hostile rowEnds ---------------------------------------------------
    ("end_past_pitch",  64,  131072, 2048, "zero",  "past_pitch", "uniform", "auto", 1, 1, "reject"),
    ("end_intmax",      64,  131072, 2048, "zero",  "intmax",     "uniform", "auto", 1, 1, "reject"),
    ("end_neg",         64,  131072, 2048, "zero",  "neg",        "uniform", "auto", 1, 1, "reject"),
    ("end_eq_start",    64,  131072, 2048, "ramp4", "eq_start",   "uniform", "auto", 1, 1, "serve"),
    ("end_lt_start",    64,  131072, 2048, "ramp4", "lt_start",   "uniform", "auto", 1, 1, "serve"),
    ("end_one",         64,  131072, 2048, "zero",  "one",        "uniform", "auto", 1, 1, "serve"),
    ("end_half",        64,  131072, 2048, "zero",  "half",       "uniform", "auto", 1, 1, "serve"),
    ("end_short_k",     64,  131072, 2048, "zero",  "kdiv2",      "uniform", "auto", 1, 1, "serve"),
    ("end_ramp_past",   64,  131072, 2048, "ramp65", "past_pitch", "uniform", "auto", 1, 1, "reject"),
    # --- value pathologies -------------------------------------------------
    ("dist_gaussian",   64,  131072, 2048, "zero", "half", "gaussian", "auto", 1, 1, "serve"),
    ("dist_equal",      64,  131072, 2048, "zero", "half", "equal",    "auto", 1, 1, "serve"),
    ("dist_inf",        64,  131072, 2048, "zero", "half", "inf",      "auto", 1, 1, "serve"),
    ("dist_nan_pos",    64,  131072, 2048, "zero", "half", "nan_pos",  "auto", 1, 1, "serve"),
    ("dist_nan_neg",    64,  131072, 2048, "zero", "half", "nan_neg",  "auto", 1, 1, "serve"),
    ("dist_denorm",     64,  131072, 2048, "zero", "half", "denorm",   "auto", 1, 1, "serve"),
    ("dist_zeros",      64,  131072, 2048, "zero", "half", "zeros",    "auto", 1, 1, "serve"),
    ("dist_adversarial",64,  131072, 2048, "zero", "half", "adversarial", "auto", 1, 1, "serve"),
    # --- tail controls: the row max hidden where only a clamped read finds it
    ("tail_max_odd",     8,  131073, 2048, "zero", "pitch", "tail_max", "auto", 1, 1, "serve"),
    ("tail_max_short",  64,  131072, 2048, "zero", "half",  "tail_max_end", "auto", 1, 1, "serve"),
    # --- workspace ---------------------------------------------------------
    ("ws_dirty",        64,  131072, 2048, "zero", "pitch", "uniform", "dirty", 1, 1, "serve"),
    ("ws_dirty_ragged", 64,  131072, 2048, "ramp65", "half", "uniform", "dirty", 1, 1, "serve"),
    ("ws_short",        64,  131072, 2048, "zero", "pitch", "uniform", "short", 1, 1, "reject"),
    # --- misc flags --------------------------------------------------------
    ("no_values",       64,  131072, 2048, "zero", "pitch", "uniform", "auto", 0, 1, "serve"),
    ("stride1_two",     64,  131072, 2048, "zero", "pitch", "uniform", "auto", 1, 2, "reject"),
]

SMOKE_NAMES = {
    "m_zero", "m_one", "n_one", "n_zero", "k_zero", "k_over_cap", "k_over_n",
    "start_ramp65", "start_neg", "start_past", "end_past_pitch", "end_intmax",
    "end_neg", "end_eq_start", "end_lt_start", "dist_inf", "dist_nan_neg",
    "tail_max_odd", "tail_max_short", "ws_dirty", "ws_short", "stride1_two",
    "no_values", "n_odd_large", "k_at_cap",
}

STARTS_KINDS = ["zero", "ramp1", "ramp4", "ramp65", "neg", "at_pitch", "past_pitch"]
ENDS_KINDS = ["pitch", "past_pitch", "intmax", "neg", "eq_start", "lt_start",
              "one", "half", "kdiv2"]
DISTS = ["uniform", "gaussian", "equal", "inf", "nan_pos", "nan_neg", "denorm",
         "zeros", "adversarial"]
# A start or end that leaves the slice out of [0, pitch] is out of contract.
HOSTILE_STARTS = {"neg", "at_pitch", "past_pitch"}
HOSTILE_ENDS = {"past_pitch", "intmax", "neg"}


def fuzz_cases(n_cases, seed=20260918):
    """Seeded so any failure reproduces from its case id alone."""
    rnd = random.Random(seed)
    out = []
    ms = [1, 2, 3, 7, 8, 16, 63, 64, 65, 256, 1024, 4096]
    ns = [1, 3, 4, 7, 512, 2047, 2048, 2049, 4096, 12289, 32768, 32833,
          65536, 65537, 131072, 131073, 131075, 262144, 524288, 1048573]
    ks = [1, 2, 3, 17, 512, 1024, 2048, 4096, 8192]
    for i in range(n_cases):
        m = rnd.choice(ms)
        n = rnd.choice(ns)
        while m * n > MAX_ELEMS:
            n = rnd.choice(ns)
        k = rnd.choice(ks)
        s = rnd.choice(STARTS_KINDS)
        e = rnd.choice(ENDS_KINDS)
        d = rnd.choice(DISTS)
        ws = rnd.choice(["auto", "auto", "auto", "dirty"])
        vals = rnd.choice([1, 1, 0])
        hostile = (s in HOSTILE_STARTS) or (e in HOSTILE_ENDS) or k > PHASE_C_CAP_MAX
        out.append(("fuzz_%03d" % i, m, n, k, s, e, d, ws, vals, 1,
                    "reject" if hostile else "serve"))
    return out


def all_cases(tier):
    if tier == "smoke":
        return [c for c in BOUNDARY if c[0] in SMOKE_NAMES]
    if tier == "boundary":
        return list(BOUNDARY)
    return list(BOUNDARY) + fuzz_cases(450)


# --------------------------------------------------------------------------
# Input construction
# --------------------------------------------------------------------------
def make_extents(kind_s, kind_e, m, n, k, torch):
    dev = "cuda"
    ar = torch.arange(m, dtype=torch.int32, device=dev)
    full = lambda v: torch.full((m,), v, dtype=torch.int32, device=dev)  # noqa: E731
    starts = {
        "zero": torch.zeros(m, dtype=torch.int32, device=dev),
        "ramp1": ar * 1,
        "ramp4": ar * 4,
        "ramp65": ar * 65,
        "neg": full(-8),
        "at_pitch": full(n),
        "past_pitch": full(n + 16),
    }[kind_s]
    ends = {
        "pitch": full(n),
        "past_pitch": full(n + 64),
        "intmax": full(INT32_MAX),
        "neg": full(-1),
        "eq_start": starts.clone(),
        "lt_start": starts - 16,
        "one": full(1),
        "half": full(n // 2),
        "kdiv2": full(max(1, k // 2)),
    }[kind_e]
    return starts.contiguous(), ends.contiguous()


def make_logits(dist, m, n, starts, ends, torch):
    """Poisoned logits: (m+1, n), everything outside each row's slice is +inf.

    The extra row is the part that matters. Row m-1's slice ends at the end of
    the logical matrix, so an over-read past it would leave the tensor entirely;
    giving it a poisoned row to land in turns "maybe faults, maybe reads mapped
    garbage" into "always detected by value".
    """
    dev = "cuda"
    g = torch.Generator(device=dev)
    g.manual_seed(42)
    buf = torch.empty((m + 1, n), dtype=torch.float32, device=dev)
    buf.fill_(float("inf"))
    if m == 0 or n == 0:
        return buf
    body = buf[:m]

    if dist == "equal":
        body.fill_(1.25)
    elif dist == "zeros":
        body.zero_()
        body[:, ::3] = -0.0
    elif dist == "denorm":
        # Smallest normal is ~1.18e-38; scaling below it lands in subnormals.
        body.copy_(torch.randn((m, n), generator=g, device=dev) * 1e-42)
    elif dist == "gaussian":
        body.copy_(torch.randn((m, n), generator=g, device=dev) * 7.0)
    elif dist == "adversarial":
        # Near-ties at the K-th boundary: the case where a wrong comparison or
        # a wrong tie-break is visible at all.
        body.copy_(torch.randint(0, 3, (m, n), generator=g, device=dev,
                                 dtype=torch.int32).float())
    else:
        body.copy_(torch.randn((m, n), generator=g, device=dev))

    if dist == "inf":
        body[:, n // 3] = float("-inf")
        if n > 5:
            body[:, n // 5] = float("-inf")
    elif dist == "nan_pos":
        body[:, n // 3] = float("nan")
    elif dist == "nan_neg":
        body[:, n // 3] = -float("nan")

    # Poison outside each row's clamped slice. Doing it per row rather than with
    # a single mask keeps it correct for ramped starts.
    cols = torch.arange(n, device=dev)
    s = starts.to(torch.int64).clamp(0, n).unsqueeze(1)
    e = ends.to(torch.int64).clamp(0, n).unsqueeze(1)
    outside = (cols.unsqueeze(0) < s) | (cols.unsqueeze(0) >= e)
    if dist in ("tail_max", "tail_max_end"):
        # Plant each row's maximum at the last element the kernel is allowed to
        # read. It is reachable only through the partial vector that
        # load_row_f4 clamps, so dropping the tail loses the largest entry.
        last = (e.squeeze(1) - 1).clamp(min=0)
        rows = torch.arange(m, device=dev)
        keep = e.squeeze(1) > s.squeeze(1)
        body[rows[keep], last[keep]] = 1e6
    body[outside] = float("inf")
    return buf


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------
def sample_rows(m):
    return sorted({0, 1, m // 2, m - 2, m - 1} & set(range(m)))


def order_keys(t, torch):
    """Independent Python model of the kernel's ordering key.

    csrc/topk_common.hip.hpp:258 is
    `(u & 0x80000000u) ? ~u : (u ^ 0x80000000u)`, a total order over all bit
    patterns. It therefore ranks -NaN BELOW -inf and +NaN above +inf, which is
    where both AVO and aiter put them and where torch.topk does not
    (knowledge/aiter_contract_audit.md: "Matching aiter is the contract").

    Reimplemented here rather than shared with the kernel so it is still an
    independent oracle, and used only as a fallback when torch disagrees on a
    slice that actually contains NaN -- otherwise torch stays the primary
    reference.
    """
    u = t.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return torch.where((u & 0x80000000) != 0, (~u) & 0xFFFFFFFF, u ^ 0x80000000)


def check(logits_cpu, starts, ends, idx, vals, m, n, k, torch):
    """Problems list against the clamped-extent oracle. Empty means correct."""
    problems = []
    # NaN != NaN makes equality fail on a correct answer, so every multiset
    # comparison gets a normalised retry, exactly as aiter_contract_audit does.
    def nz(t):
        return torch.nan_to_num(t, 1e30, 1e30, -1e30)

    for r in sample_rows(m):
        s = min(max(int(starts[r]), 0), n)
        e = min(max(int(ends[r]), 0), n)
        ln = max(0, e - s)
        want = min(k, ln)
        row = idx[r]
        if want < k and not bool((row[want:] == -1).all().item()):
            problems.append("row %d: padding is not -1" % r)
        if vals is not None and want < k:
            pad = vals[r][want:]
            if not bool((pad == float("-inf")).all().item()):
                problems.append("row %d: value padding is not -inf" % r)
        if want == 0:
            continue
        got = row[:want]
        inrange = ((got >= s) & (got < e))
        if not bool(inrange.all().item()):
            bad = got[~inrange][:4].tolist()
            problems.append("row %d: index outside [%d,%d): %s" % (r, s, e, bad))
            continue
        if len(set(got.tolist())) != want:
            problems.append("row %d: duplicate indices" % r)
            continue
        gv = torch.sort(logits_cpu[r][got]).values
        sl = logits_cpu[r][s:e]
        rv = torch.sort(torch.topk(sl, want).values).values
        if not torch.equal(gv, rv) and not torch.equal(nz(gv), nz(rv)):
            if not bool(torch.isnan(sl).any().item()):
                problems.append("row %d: value multiset != torch.topk" % r)
                continue
            # torch ranks every NaN highest; the kernel's bitwise key drops -NaN
            # below -inf. On a NaN-bearing slice the two are allowed to differ,
            # so re-judge against the key order, comparing keys (integers)
            # rather than values so the comparison is not itself NaN-poisoned.
            kref = torch.sort(order_keys(sl, torch),
                              descending=True).values[:want]
            kgot = order_keys(logits_cpu[r][got], torch)
            if not torch.equal(torch.sort(kgot).values, torch.sort(kref).values):
                problems.append("row %d: multiset != torch AND != key order" % r)
                continue
        if vals is not None:
            vv = torch.sort(vals[r][:want]).values
            if not torch.equal(vv, gv) and not torch.equal(nz(vv), nz(gv)):
                problems.append("row %d: values disagree with logits[idx]" % r)
    return problems


# --------------------------------------------------------------------------
# Child: one case, one process
# --------------------------------------------------------------------------
def run_one(case):
    name, m, n, k, kind_s, kind_e, dist, ws, want_vals, stride1, expect = case
    use_mounted_aiter()
    stub_flydsl()
    import torch
    import aiter
    from aiter.ops.topk import (  # noqa: F401
        _top_k_per_row_prefill_avo,
        top_k_per_row_prefill_avo,
        topk_avo_supports,
        topk_avo_workspace_size,
    )

    res = {"case": name, "m": m, "n": n, "k": k, "starts": kind_s,
           "ends": kind_e, "dist": dist, "ws": ws, "expect": expect}
    if not aiter.__file__.startswith(AITER_ROOT):
        res["outcome"] = "wrong_aiter"
        res["error"] = aiter.__file__
        print(json.dumps(res))
        return 0

    starts, ends = make_extents(kind_s, kind_e, m, n, k, torch)
    buf = make_logits(dist, m, n, starts, ends, torch)
    logits = buf[:m] if m > 0 else buf[:0]
    idx = torch.full((max(m, 1), max(k, 1)), -2, dtype=torch.int32, device="cuda")
    vals = (torch.zeros((max(m, 1), max(k, 1)), dtype=torch.float32, device="cuda")
            if want_vals else None)

    try:
        res["supports"] = bool(topk_avo_supports(m, n, k))
    except Exception as e:
        res["supports"] = "raised: %s" % str(e)[:60]

    try:
        if ws == "auto":
            top_k_per_row_prefill_avo(logits, starts, ends, idx, vals,
                                      m, n, stride1, k)
        else:
            size = 1 if ws == "short" else int(topk_avo_workspace_size(m, n, k))
            w = torch.empty(max(size, 1), dtype=torch.uint8, device="cuda")
            # 0xFF everywhere: if any consumer reads a counter before its
            # producer writes it, the value is maximal rather than plausibly
            # zero, so the failure is loud instead of accidentally correct.
            # The entry's own comment says this is safe because Phase A clears
            # the counters it shares and Phase B assigns rather than
            # accumulates; that was an untested assumption until now.
            w.fill_(0xFF)
            # Through the PUBLIC wrapper, not the raw binding: the wrapper is
            # where the workspace-size check lives, and a short buffer must
            # raise rather than reach C++ and abort.
            top_k_per_row_prefill_avo(logits, starts, ends, idx, vals,
                                      m, n, stride1, k, w)
        torch.cuda.synchronize()
    except Exception as e:
        res["outcome"] = "raised"
        res["error"] = str(e).strip().splitlines()[-1][:200] if str(e).strip() else repr(e)[:200]
        print(json.dumps(res))
        return 0

    res["outcome"] = "ran"
    if m == 0 or k <= 0 or n <= 0:
        res["problems"] = []
        print(json.dumps(res))
        return 0
    try:
        res["problems"] = check(buf[:m].cpu(), starts.cpu(), ends.cpu(),
                                idx.cpu().to(torch.int64),
                                vals.cpu() if vals is not None else None,
                                m, n, k, torch)
    except Exception as e:
        res["problems"] = ["checker raised: %s" % str(e)[:120]]
    print(json.dumps(res))
    return 0


# --------------------------------------------------------------------------
# Parent
# --------------------------------------------------------------------------
def verdict(case, rec):
    """PASS / DECL / FAIL / CRASH.

    A crash is always a failure -- that is the whole point of the file.

    Raising is NOT unconditionally a pass. An exception is a defined, survivable
    refusal, so it is correct for a hostile input and correct for a shape
    `topk_avo_supports()` declines. But a shape that supports() claims AND the
    entry then refuses is a contract inconsistency, and reporting it green would
    hide exactly the kind of bug this file is for.
    """
    expect = case[10]
    out = rec.get("outcome")
    supported = rec.get("supports") is True
    if out == "crashed":
        return "CRASH", rec.get("error", "")[:90]
    if out == "wrong_aiter":
        return "FAIL", "imported %s" % rec.get("error", "")[:70]
    if out == "raised":
        err = rec.get("error", "")[:70]
        if expect == "reject" or not supported:
            return ("PASS" if expect == "reject" else "DECL"), "raised: %s" % err
        return "FAIL", "supports()=True but entry refused: %s" % err
    probs = rec.get("problems") or []
    if probs:
        return "FAIL", probs[0][:90]
    if expect == "reject":
        # Served a hostile input without faulting and with a clamped-correct
        # answer. That is the behaviour we want, not a surprise.
        return "PASS", "served (clamped)"
    return "PASS", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=("smoke", "boundary", "full"), default="smoke")
    ap.add_argument("--smoke", action="store_const", const="smoke", dest="tier")
    ap.add_argument("--full", action="store_const", const="full", dest="tier")
    ap.add_argument("--case-id", type=int, default=None)
    ap.add_argument("--only", default=None, help="substring filter on case name")
    ap.add_argument("--json-out", default="/topk/log/stress_topk.json")
    args = ap.parse_args()

    cases = all_cases(args.tier)
    if args.case_id is not None:
        return run_one(cases[args.case_id])

    sel = [(i, c) for i, c in enumerate(cases)
           if args.only is None or args.only in c[0]]
    print("tier=%s cases=%d" % (args.tier, len(sel)))
    print("%-5s %-18s %-7s %-8s %-6s %s"
          % ("id", "case", "expect", "outcome", "verd", "detail"))
    out, n_fail, n_crash, n_decl = [], 0, 0, 0
    for i, c in sel:
        p = subprocess.run([sys.executable, __file__, "--tier", args.tier,
                            "--case-id", str(i)],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True)
        rec = None
        for line in p.stdout.splitlines():
            if line.strip().startswith("{"):
                try:
                    rec = json.loads(line.strip())
                except ValueError:
                    pass
        if rec is None:
            tail = (p.stderr.strip().splitlines() or ["(no stderr)"])[-1]
            rec = {"case": c[0], "outcome": "crashed", "rc": p.returncode,
                   "error": tail[:200]}
        v, detail = verdict(c, rec)
        rec["verdict"] = v
        if v == "CRASH":
            n_crash += 1
        elif v == "FAIL":
            n_fail += 1
        elif v == "DECL":
            n_decl += 1
        print("%-5d %-18s %-7s %-8s %-6s %s"
              % (i, c[0], c[10], rec.get("outcome"), v, detail))
        out.append(rec)

    print("\n  cases    %d" % len(sel))
    print("  declined %d   (topk_avo_supports() said no; kernel never ran)" % n_decl)
    print("  failed   %d" % n_fail)
    print("  crashed  %d   <-- must be 0" % n_crash)
    try:
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
        json.dump(out, open(args.json_out, "w"), indent=1)
        print("WROTE %s" % args.json_out)
    except OSError as e:
        print("could not write json: %s" % e)
    print("\nSTRESS GATE %s" % ("PASS" if (n_fail + n_crash) == 0 else "FAIL"))
    return 1 if (n_fail + n_crash) else 0


if __name__ == "__main__":
    raise SystemExit(main())
