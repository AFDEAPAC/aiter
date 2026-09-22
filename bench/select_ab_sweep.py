#!/usr/bin/env python3
"""Measure `aiter.topk_select` over the (M, N) grid at three widths, two ways.

Two modes, because two different questions need two different tables.

  --mode backends   For each cell, force each backend that can serve it and time
                    it, and time the `sampled` kernel alongside. This is the
                    table a routing rule gets fitted to. `topk_select.py:90` and
                    `:367` say to re-fit with `topk_backend_fit.py` over
                    `topk_backend_sweep.py`; neither file exists anywhere on this
                    machine, so this mode replaces them.
                    Two arms always run beside the forced ones: `routed`, the
                    real rule, so a forced number has something from the SAME
                    run to be compared against; and `amax`, one pass of the
                    input, which is the floor a cell should be reported in and
                    the only check this table can make on its own numbers.
                    `--backends` restricts the forced arms; it does not turn
                    those two off.

  --mode ab         Time the real router twice, once with `sampled` available and
                    once without. This is the before/after report.

Why not `bench/ceiling_sweep.py`: that file calls the kernel entry directly and
says so -- `AITER_DISABLE_TOPK_SAMPLED` never reaches it. It also answers a
different question, "how far from the ideal-selector floor". This one answers
"what does the router pick, and what does that cost".

Three things are inherited from `ceiling_sweep.py` because they were right there
and are right here:

  - aiter's own `@perftest()`, which sizes an argument rotation from the measured
    input so the working set defeats L2, and reads GPU kernel time from a
    profiler trace rather than wall clock. A hand-written loop over one input
    measures a warm cache.
  - All three widths built and warmed BEFORE any of them is timed, then
    interleaved rounds. Timing them one after another charges run-order drift to
    whichever width went first, which is how an earlier sweep reported a +5.60%
    parity effect that was +0.34% once interleaved.
  - Full uniform rows [0, N). The earlier heatmap used full rows, so cells stay
    comparable with it. Ragged rows read up to 25% fewer bytes at large M and
    small N.

Forcing a backend: `topk_select` has no override parameter, so this replaces
`topk_select_backend` and clears `_choose`, which is `@lru_cache(maxsize=1024)`
and would otherwise keep handing back the name chosen before the patch. The
whole `topk_select` path still runs -- argument rejection, allocation, dispatch
-- so a forced number is comparable with a routed one.

Run inside the correctness image with both repos mounted:

  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \\
    --ipc=host --shm-size 16G -e PYTHONPATH=/aiter \\
    -v /home/mh/aiter-topk:/aiter -v /home/mh/topk-prefill-avo:/topk -w /aiter \\
    rocm/ali-private:ubuntu22.04_rocm10.0.0_cp313_torch2.12.0_vllm_dsv4_20260916 \\
    python /topk/bench/select_ab_sweep.py --mode backends
"""
import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.profiler as tpf  # noqa: E402

import aiter  # noqa: E402

# `python /topk/bench/x.py` puts sys.path[0] at this file's directory, not the
# cwd, so `import aiter` can silently resolve to the site-packages copy and
# produce a green run against a module nobody changed. Caught exactly that once.
if not aiter.__file__.startswith("/aiter/"):
    raise SystemExit(
        "WRONG AITER: imported %s, expected one under /aiter/. "
        "Re-run with -e PYTHONPATH=/aiter." % aiter.__file__
    )

import grid  # noqa: E402
from aiter.ops import topk_select as ts  # noqa: E402
from aiter.test_common import perftest  # noqa: E402

MS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
BASES = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
WIDTH_OFFSETS = (0, 1, 2)
TOPK = 2048
ROUNDS = 3


# Above this many bytes of input, time with CUDA events instead of the
# torch.profiler trace `@perftest()` uses by default. Two reasons, and both are
# about this size class rather than about preferring one timer:
#
#   the trace teardown segfaults here. Deterministically, at the largest cell of
#   a run and only when it is not the first cell: ROCTracer warns "duplicate flow
#   start", then `torch/profiler/profiler.py:331 stop_trace` dies with SIGSEGV
#   through `test_common.py:96`. The sweep loses the process, not just the cell.
#
#   rotation has nothing left to buy. The profiler path rotates through copies
#   of the input so consecutive iterations cannot reuse cache; the event path
#   calls with one buffer and does not. That distinction is real below the LLC
#   and empty above it -- a 1 GiB input cannot sit in this part's cache whether
#   it is rotated or not.
#
# So the two timers should agree above the bound, and `--timer-ab` checks that
# they do rather than leaving it asserted.
EVENT_ABOVE_BYTES = 1 << 30


# Pinned rather than left to the decorator's defaults. These ARE the defaults
# (`aiter/test_common.py:242`), which is also what `op_tests/test_topk_select.py`
# runs and what the published PR 5686 report means by "warmup 2 + 101 launches".
# Written out so a later change to aiter's defaults cannot silently move this
# sweep off the ruler its numbers are compared against.
NUM_ITERS = 101
NUM_WARMUP = 2

# A cell whose profiler capture falls below this fraction of the events it
# should have produced is reported, not silently averaged. See `_trace_time`.
MIN_CAPTURE = 0.10


def _device_rows(prof):
    """(name, self_device_time_total) for every GPU event in a trace."""
    out = []
    for el in prof.events():
        if str(getattr(el, "device_type", "")).split(".")[-1] != "CUDA":
            continue
        out.append((el.name, float(el.self_device_time_total)))
    return out


def _counts_per_iter(fn, probe=3):
    """How many times each kernel fires in ONE call, from a short profiled run.

    Measured rather than assumed because it is backend-specific: `decode` issues
    seven kernels per call, `sampled` three to five depending on which phases the
    shape takes, `stream` one or two. `_trace_time` needs it to turn a per-kernel
    mean back into a per-call time.
    """
    with tpf.profile(
        activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA],
        profile_memory=False, with_stack=False, with_modules=True,
    ) as prof:
        for _ in range(probe):
            fn()
        torch.cuda.synchronize()
    counts = {}
    for name, _ in _device_rows(prof):
        counts[name] = counts.get(name, 0) + 1
    return {n: max(1, int(round(c / probe))) for n, c in counts.items()}


# How much GPU work one profiler session may cover. A session that spans much
# more than this dies in `stop_trace` on the largest cells -- m=4096 n=1048576
# at 303 calls of a 3ms selector is a reliable SIGSEGV, the same teardown crash
# the sweep hit before, while the same cell at 101 calls survives.
#
# Cutting the count is safe because the estimator is a per-kernel MEAN, so it
# needs enough samples rather than a fixed number, and a 3ms kernel is stable:
# measured at m=4096 n=1048576, iters of 10/30/60/101/150/303 read 3047.2 /
# 3047.8 / 3049.2 / 3045.7 / 3046.4 / 3045.8 us -- a 0.1% spread over a 30x
# range. Small cells keep the full count, which is where the published
# protocol's 101 launches actually matters.
MAX_PROFILED_US = 60_000
MIN_PROFILED_ITERS = 20
# And a bound on EVENTS, which is the one that actually predicts the crash.
# m=128 n=1048576 is the worst cell in the grid for this: `decode` issues seven
# kernels and runs 387us, so a 154-call session is 1078 events -- that cell both
# loses 87% of its events and takes the teardown down with it, while m=4096
# n=1048576 at five kernels and 3ms per call is only ~95 events and is fine.
# Bounding GPU time alone does not catch it, because the offender is a cell that
# is fast per call and rich in kernels.
MAX_PROFILED_EVENTS = 600


def _profiled_iters(fn, rounds, counts=None):
    """How many calls one profiler session should cover for this shape."""
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    fn()
    e.record()
    e.synchronize()
    est_us = max(s.elapsed_time(e) * 1000.0, 1e-3)
    want = int(MAX_PROFILED_US / est_us)
    if counts:
        want = min(want, MAX_PROFILED_EVENTS // max(1, sum(counts.values())))
    return max(MIN_PROFILED_ITERS, min(NUM_ITERS * rounds, want))


def _trace_time(fn, counts, iters=NUM_ITERS):
    """Device time for one call, in us, from a profiler trace.

    NOT `aiter.test_common.get_trace_perf`. That function sums every captured
    kernel and divides by `num_iters` -- the count it INTENDED to run, not the
    count the trace actually holds. On this box the ROCm profiler drops events
    under load, and it does so silently: at m=128 n=1048576 a 101-iteration run
    captured 16 to 21 events per kernel instead of 101, so the sum was divided by
    100 and the cell read 70.9us against a true 387.5. That is a 5.5x
    under-report with no warning, and it is what put `decode` below the physical
    read floor in an earlier sweep.

    Per-kernel MEAN times survive the dropping -- only the counts are wrong. So
    the estimator is the sum over kernels of (mean time) x (calls per iteration),
    which reproduced the published PR 5686 figure for that cell to 0.03%
    (387.4us against 387.52).

    Returns (us, capture_ratio). A ratio near 1.0 means the trace was complete;
    anything lower means the number came from a sample, which is still correct
    but is recorded so a reader can see it.
    """
    with tpf.profile(
        activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA],
        profile_memory=False, with_stack=False, with_modules=True,
    ) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()

    agg = {}
    for name, dt in _device_rows(prof):
        s, c = agg.get(name, (0.0, 0))
        agg[name] = (s + dt, c + 1)

    total_us, seen, expected = 0.0, 0, 0
    for name, per_iter in counts.items():
        expected += per_iter * iters
        if name not in agg:
            continue
        s, c = agg[name]
        seen += c
        total_us += (s / c) * per_iter
    return total_us, (seen / expected if expected else 0.0)


def _arms(use_event):
    """The three timed calls, bound to one timer.

    `use_event` is kept for `--timer-ab` only. The scoring path is `_trace_time`,
    which reads the same profiler trace `@perftest()` does but divides by the
    events it actually captured rather than the ones it meant to run.
    """
    p = perftest(
        num_iters=NUM_ITERS, num_warmup=NUM_WARMUP, use_cuda_event=use_event
    )

    def select(inp, k, out_idx):
        return aiter.topk_select(inp, k, output_idx=out_idx)

    def sampled(logits, row_starts, row_ends, indices, values,
                num_rows, stride_row, stride_col, k):
        return aiter.top_k_per_row_prefill_sampled(
            logits, row_starts, row_ends, indices, values,
            num_rows, stride_row, stride_col, k=k)

    def amax(logits):
        """The read-only reference: one pass of the input, one scalar per row.

        Not a selector and not an alternative to one. It carries a cell's cache
        and buffer placement the way a selector does, which is what makes it the
        unit to report a cell in, and it is stable to +-0.5% across runs where a
        multi-launch backend spreads 55%.

        It is a REFERENCE, not a proof of a lower bound: torch's reduction is
        not itself at peak bandwidth everywhere, and at m=4096 n=524288 it
        measured 1841.0us for 8.59GB, which is 4.67TB/s against the 6.61TB/s
        this box reaches. A selector timed below it there is not necessarily
        mistimed -- compare against the analytic floor before concluding that.
        """
        return torch.amax(logits, dim=1)

    return {
        "select": p(select), "sampled": p(sampled), "amax": p(amax),
        # The undecorated callables, for `_trace_time` to drive itself.
        "raw": {"select": select, "sampled": sampled, "amax": amax},
    }


ARMS = {"profiler": _arms(False), "event": _arms(True)}
RAW = ARMS["profiler"]["raw"]


def force(name):
    """Pin `topk_select` to one backend. Pass None to restore the real rule."""
    if name is None:
        ts.topk_select_backend = force._real
    else:
        ts.topk_select_backend = lambda rows, width, k, available: name
    ts._choose.cache_clear()


force._real = ts.topk_select_backend


def available_for(m, width, k):
    """Backends that can serve this geometry, as `topk_select` computes it."""
    try:
        wave = ts.wave_size_of(torch.device("cuda", 0).index or 0)
    except Exception:
        wave = 64
    avail = ts._available(width, k, wave, False, True)
    allowed = frozenset(ts._BACKENDS_BY_TIE[None])
    return sorted(avail & allowed)


def check(inp, idx, k, rows_to_check=(0,)):
    """Index set against torch.topk on the same row. Values, not positions:
    ties are common at these widths and two backends may break them differently
    while both being correct."""
    want = min(k, inp.shape[1])
    for r in rows_to_check:
        g = idx[r][:want].to(torch.int64)
        if not bool(((g >= 0) & (g < inp.shape[1])).all().item()):
            return "row %d: index out of range" % r
        if len(set(g.tolist())) != want:
            return "row %d: duplicate indices" % r
        got = torch.sort(inp[r][g]).values
        ref = torch.sort(torch.topk(inp[r], want).values).values
        if not torch.equal(got, ref):
            return "row %d: value multiset != torch.topk" % r
    return None


def build(m, widths, topk, seed=0):
    """One tensor set per width, all built before any is timed.

    The seed is reset through the GLOBAL generators, not a local
    `torch.Generator`, and defaults to 0, because that is what the published PR
    5686 report did and its numbers are the baseline these are compared against.
    It is not cosmetic: `sampled` sends a row whose candidate set comes out
    unusable to `phase_d_fallback`, so which rows fall back is a function of the
    data. At m=4096 n=524288 the two seeds measured 27% apart -- 1684.9 us at
    seed 42 against the report's 2140.95 at seed 0 -- while the cells either side
    of it agreed to 5%.
    """
    args = {}
    for w in widths:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        lg = torch.randn((m, w), dtype=torch.float32, device="cuda")
        assert lg.stride(0) == w, (lg.stride(0), w)
        idx = torch.empty((m, topk), dtype=torch.int32, device="cuda")
        rs = torch.zeros(m, dtype=torch.int32, device="cuda")
        re = torch.full((m,), w, dtype=torch.int32, device="cuda")
        args[w] = {"logits": lg, "idx": idx, "rs": rs, "re": re}
    return args


def _time_arm(args, widths, rounds, call, verify_with, timer="profiler",
              raw=None):
    """Warm every width, verify the first, then time all widths interleaved.

    Two timers live here. `timer="profiler"` -- the scoring path -- drives `raw`
    (the undecorated callable) through `_trace_time`, which divides by the
    events the trace actually holds. `timer="event"` keeps the old
    `@perftest(use_cuda_event=True)` route and exists for `--timer-ab`.
    """
    # Warm with the RAW callable. Warming through the decorated one makes every
    # "warmup" a full 101-iteration profiled run, which is what pushed the
    # session count high enough to bring back the teardown SIGSEGV at 16 GiB.
    warm = raw if raw is not None else call
    for w in widths:
        for _ in range(NUM_WARMUP):
            warm(args[w], w)
    torch.cuda.synchronize()
    bad = None
    if verify_with is not None:
        bad = verify_with(args[widths[0]])

    if timer != "profiler":
        samples = {w: [] for w in widths}
        for _ in range(rounds):
            for w in widths:                   # interleaved
                samples[w].append(call(args[w], w)[1])
        return samples, bad, {w: 1.0 for w in widths}

    # Kernels-per-call, measured once per width: the same backend can issue a
    # different number on a different width (a `stream` split, a `sampled` phase
    # that a shape skips), so it cannot be taken from one width and reused.
    counts = {w: _counts_per_iter(lambda: raw(args[w], w)) for w in widths}
    # ONE profiler session per width, covering all rounds. The rounds are still
    # run -- they are what the published protocol means by "3 rounds" -- but
    # they share a session, because every extra session is another teardown and
    # the teardown is what crashes on the largest cells. The estimator is a
    # per-kernel mean either way, so merging the rounds only makes it an average
    # over more calls.
    samples, capture = {}, {}
    for w in widths:
        fn = (lambda ww: (lambda: raw(args[ww], ww)))(w)
        # A session that captures NOTHING gives a sum of zero, and zero divided
        # by anything is still zero -- so an empty trace reads as an
        # infinitely fast kernel rather than as a failure. Measured 9 times in
        # 2730 rows. Retry, and if it stays empty leave the sample out rather
        # than let a 0 into the table.
        us = ratio = None
        for _ in range(3):
            us, ratio = _trace_time(fn, counts[w],
                                    iters=_profiled_iters(fn, rounds,
                                                          counts[w]))
            torch.cuda.synchronize()
            if ratio > 0:
                break
        samples[w] = [us] if ratio else []
        capture[w] = ratio
    return samples, bad, capture


def _emit(rows, m, base, topk, widths, samples, bad, backend, timer,
          capture=None, extra=None):
    for w in widths:
        empty = not samples[w]
        row = {"m": m, "base": base, "width": w, "topk": topk,
               "backend": backend,
               "us": None if empty else st.median(samples[w]),
               "runs": [round(x, 3) for x in samples[w]],
               "correct": bad is None,
               "detail": "profiler captured no events" if empty else bad,
               "timer": timer}
        if capture is not None:
            row["capture"] = round(capture.get(w, 1.0), 4)
        if extra:
            row.update(extra.get(w, {}))
        rows.append(row)


def pick_timer(m, base, topk, event_above=EVENT_ABOVE_BYTES):
    """Which timer this cell gets. Recorded on every row it produces, because a
    table that mixes two timers and does not say which is which cannot be read
    back."""
    if event_above < 0:
        return "event"
    if event_above == 0:
        return "profiler"
    return "event" if m * (base + max(WIDTH_OFFSETS)) * 4 >= event_above \
        else "profiler"


def time_backends(m, base, topk, rounds, verify, wanted=None,
                  event_above=EVENT_ABOVE_BYTES, seed=0):
    """Force each serving backend in turn, plus the router, the sampled kernel
    direct, and a pure-read floor.

    `routed` and `amax` are measured HERE rather than read back from an earlier
    run, and that is the point of them being here at all: this file's own
    docstring says a number from another harness is not comparable to one from
    this one, and the same holds for one from another RUN. A forced arm and the
    router only compare if one process, one set of tensors and one timer
    produced both.

    `amax` is the read-only reference, not a selector; see the arm's own
    docstring for why it is not a hard lower bound either.

    `wanted` filters the FORCED backend arms only. `routed` and `amax` always
    run: they are the units the forced arm is reported in, not alternatives to
    it, and a table without them cannot be read.
    """
    widths = tuple(base + o for o in WIDTH_OFFSETS)
    args = build(m, widths, topk, seed)
    rows = []

    timer = pick_timer(m, base, topk, event_above)
    arm = ARMS[timer]
    cands = available_for(m, base, topk)
    if wanted is not None:
        cands = [c for c in cands if c in wanted]
    supported = bool(aiter.topk_sampled_supports(m, base, topk))
    verify_with = (
        (lambda a: check(a["logits"], a["idx"], topk)) if verify else None
    )

    def select_call(a, w):
        return arm["select"](a["logits"], topk, a["idx"])

    def select_raw(a, w):
        return RAW["select"](a["logits"], topk, a["idx"])

    def sampled_call(a, w):
        return arm["sampled"](a["logits"], a["rs"], a["re"], a["idx"], None,
                              m, w, 1, topk)

    def sampled_raw(a, w):
        return RAW["sampled"](a["logits"], a["rs"], a["re"], a["idx"], None,
                              m, w, 1, topk)

    def arm_time(call, raw, verify_it):
        return _time_arm(args, widths, rounds, call, verify_it, timer,
                         (lambda a, w: raw(a, w)) if timer == "profiler"
                         else None)

    try:
        for name in cands:
            force(name)
            samples, bad, cap = arm_time(select_call, select_raw, verify_with)
            _emit(rows, m, base, topk, widths, samples, bad, name, timer, cap)

        if supported and (wanted is None or "sampled" in wanted):
            force(None)
            samples, bad, cap = arm_time(sampled_call, sampled_raw, verify_with)
            # `sampled_direct`, not `sampled`: the forced-backend loop above
            # already emits a row called `sampled` for the same kernel reached
            # through `topk_select`, and two rows of one name in one cell cannot
            # be told apart afterwards. The published PR 5686 log uses the same
            # two names for the same two things.
            _emit(rows, m, base, topk, widths, samples, bad, "sampled_direct",
                  timer, cap)

        # The real rule, last, so the forced arms cannot inherit a warm cache
        # that only it would have produced.
        force(None)
        picked = {}
        for w in widths:
            try:
                picked[w] = ts._choose(m, w, topk, ts.wave_size_of(0),
                                       False, None, False, True)
            except Exception:                  # noqa: BLE001 -- recorded as "?"
                picked[w] = "?"
        samples, bad, cap = arm_time(select_call, select_raw, verify_with)
        _emit(rows, m, base, topk, widths, samples, bad, "routed", timer, cap,
              {w: {"routed_pick": picked[w]} for w in widths})

        samples, _, cap = arm_time(
            lambda a, w: arm["amax"](a["logits"]),
            lambda a, w: RAW["amax"](a["logits"]), None)
        # `correct: None`, not True: this arm selects nothing, so there is no
        # claim to be right about. The run's "must be 0" filter tests `is False`
        # and so passes it over either way; None is what stops a reader of the
        # JSON from counting it as a verified selector.
        _emit(rows, m, base, topk, widths, samples, None, "amax", timer, cap,
              {w: {"correct": None, "detail": "read floor, not a selector"}
               for w in widths})
    finally:
        force(None)
        del args
        torch.cuda.empty_cache()
    return rows


def time_ab(m, base, topk, rounds, verify, event_above=EVENT_ABOVE_BYTES,
            seed=0):
    """The real router, with and without `sampled` in the available set."""
    widths = tuple(base + o for o in WIDTH_OFFSETS)
    args = build(m, widths, topk, seed)
    rows = []
    timer = pick_timer(m, base, topk, event_above)
    run_select = ARMS[timer]["select"]
    try:
        for side in ("before", "after"):
            if side == "before":
                os.environ["AITER_DISABLE_TOPK_SAMPLED"] = "1"
            else:
                os.environ.pop("AITER_DISABLE_TOPK_SAMPLED", None)
            ts._choose.cache_clear()
            ts._available.cache_clear()

            picked = {}
            for w in widths:
                a = args[w]
                run_select(a["logits"], topk, a["idx"])
                try:
                    picked[w] = ts._choose(
                        m, w, topk, ts.wave_size_of(0), False, None, False, True)
                except Exception:
                    picked[w] = "?"
            bad = None
            if verify:
                a = args[widths[0]]
                bad = check(a["logits"], a["idx"], topk)
            samples = {w: [] for w in widths}
            for _ in range(rounds):
                for w in widths:
                    a = args[w]
                    samples[w].append(run_select(a["logits"], topk, a["idx"])[1])
            for w in widths:
                rows.append({"m": m, "base": base, "width": w, "topk": topk,
                             "side": side, "backend": picked[w],
                             "us": st.median(samples[w]),
                             "runs": [round(x, 3) for x in samples[w]],
                             "correct": bad is None, "detail": bad,
                             "timer": timer})
    finally:
        os.environ.pop("AITER_DISABLE_TOPK_SAMPLED", None)
        ts._choose.cache_clear()
        ts._available.cache_clear()
        del args
        torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("backends", "ab"), default="backends")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--ms", help="comma list, overrides the M axis")
    ap.add_argument("--bases", help="comma list, overrides the N axis")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the torch.topk check (timing only)")
    ap.add_argument("--backends",
                    help="comma list restricting the FORCED backend arms in "
                         "--mode backends; routed and amax always run. "
                         "Default: every backend that serves the cell.")
    ap.add_argument("--event-above-bytes", type=int,
                    default=EVENT_ABOVE_BYTES,
                    help="input bytes past which a cell is timed with CUDA "
                         "events instead of the profiler trace. 0 = always "
                         "the trace, -1 = always events. Default %d."
                         % EVENT_ABOVE_BYTES)
    ap.add_argument("--seed", type=int, default=0,
                    help="reset torch.manual_seed and torch.cuda."
                         "manual_seed_all to this before each width. 0 is the "
                         "published report's seed; change it only to measure "
                         "data sensitivity, never to score.")
    ap.add_argument("--timer-ab", action="store_true",
                    help="time every cell BOTH ways and report the spread, "
                         "instead of picking one. Use on small cells only: "
                         "the trace teardown segfaults on large ones.")
    ap.add_argument("--out", default="/topk/reports/select_ab.json")
    args = ap.parse_args()

    ms = [int(x) for x in args.ms.split(",")] if args.ms else MS
    bases = [int(x) for x in args.bases.split(",")] if args.bases else BASES
    verify = not args.no_verify
    wanted = set(args.backends.split(",")) if args.backends else None

    print("aiter under test: %s" % aiter.__file__)
    print("mode=%s topk=%d rounds=%d cells=%d forced=%s event_above=%d "
          "seed=%d iters=%d warmup=%d"
          % (args.mode, args.topk, args.rounds, len(ms) * len(bases),
             "all" if wanted is None else ",".join(sorted(wanted)),
             args.event_above_bytes, args.seed, NUM_ITERS, NUM_WARMUP))
    print("%6s %9s | %s" % ("M", "base", "result"))

    out, skipped = [], []
    for m in ms:
        for base in bases:
            if not grid.fits_in_vram(m, base + max(WIDTH_OFFSETS), args.topk):
                skipped.append((m, base))
                print("%6d %9d | oom_skip" % (m, base))
                continue
            try:
                if args.timer_ab:
                    # Both timers on the same cell, same tensors, back to back.
                    # This is what says the bound may be crossed without a
                    # change of units; it is not a measurement of the kernel.
                    rows = []
                    for bound in (0, -1):
                        rows.extend(time_backends(m, base, args.topk,
                                                  args.rounds, verify, wanted,
                                                  bound, args.seed))
                elif args.mode == "backends":
                    rows = time_backends(m, base, args.topk, args.rounds,
                                         verify, wanted, args.event_above_bytes,
                                         args.seed)
                else:
                    rows = time_ab(m, base, args.topk, args.rounds, verify,
                                   args.event_above_bytes, args.seed)
            except Exception as e:  # noqa: BLE001 -- record, do not abort the sweep
                print("%6d %9d | ERROR %s" % (m, base, str(e)[:60]))
                out.append({"m": m, "base": base, "error": str(e)[:200]})
                continue
            out.extend(rows)
            # `amax` is excluded from "fastest": it is the floor, so it would
            # win every cell and the column would stop saying anything.
            timed = [r for r in rows
                     if r.get("us") and r.get("backend") != "amax"]
            best = min(timed, key=lambda r: r["us"], default=None)
            floor = min((r["us"] for r in rows
                         if r.get("backend") == "amax" and r["width"] == base),
                        default=None)
            wrong = [r for r in rows if r.get("correct") is False]
            # A selector under the read floor was mistimed, not fast. Printed
            # per cell so the run says so while it is still running.
            impossible = [r for r in timed
                          if floor and r["width"] == base and r["us"] < floor]
            thin = [r for r in rows
                    if r.get("capture") is not None
                    and r["capture"] < MIN_CAPTURE]
            # flush: stdout to a pipe is block-buffered, and this sweep has
            # already lost eight cells' worth of progress lines to a SIGSEGV
            # that made it look as though it died on the first cell.
            print("%6d %9d | %2d rows, fastest %s %.1fus, ref %s%s%s%s"
                  % (m, base, len(rows),
                     best["backend"] if best else "-",
                     best["us"] if best else float("nan"),
                     "-" if floor is None else "%.1fus" % floor,
                     "  WRONG:%d" % len(wrong) if wrong else "",
                     "  UNDER-REF:%s" % ",".join(
                         r["backend"] for r in impossible) if impossible else "",
                     "  THIN-TRACE:%s" % ",".join(
                         sorted({r["backend"] for r in thin})) if thin else ""),
                  flush=True)
            # Written every cell, so a killed run is resumable by hand and a
            # long sweep never loses everything to one bad shape.
            with open(args.out, "w") as f:
                json.dump({"mode": args.mode, "topk": args.topk,
                           "rounds": args.rounds, "ms": ms, "bases": bases,
                           "width_offsets": list(WIDTH_OFFSETS),
                           "forced_backends":
                               None if wanted is None else sorted(wanted),
                           "event_above_bytes": args.event_above_bytes,
                           "timer_ab": args.timer_ab,
                           "seed": args.seed,
                           "num_iters": NUM_ITERS,
                           "num_warmup": NUM_WARMUP,
                           "oom_skipped": skipped, "rows": out}, f, indent=1)

    print()
    print("cells skipped for VRAM: %d %s" % (len(skipped), skipped if skipped else ""))
    wrong = [r for r in out if r.get("correct") is False]
    print("rows with a wrong result: %d   <-- must be 0" % len(wrong))
    for r in wrong[:10]:
        print("   m=%(m)s width=%(width)s backend=%(backend)s %(detail)s" % r)
    print("WROTE %s (%d rows)" % (args.out, len(out)))
    return 1 if wrong else 0


if __name__ == "__main__":
    raise SystemExit(main())
