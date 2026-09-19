#!/usr/bin/env python3
"""Gate the customer spec's low-N corner, which no other harness reaches.

The spec is: logits [M, N] fp32, M in 1, 4, 8, ... 4k, N in 512, 1024, ... 1M,
topk = 2048. The N axis therefore starts at 512, and nothing here tested that.
`bench/grid.py` bottoms out at N = 2048, `bench/verify_grid.py` inherits that
floor, and `bench/stress_topk.py` touches N = 512 in a single boundary case. So
the two smallest columns the customer will actually call -- N = 512 and
N = 1024 -- had no coverage at all.

Those two columns are not a smaller version of the rest of the grid. At
topk = 2048 they are the k >= N regime, where every element of the row is
selected and the correct answer is a permutation of the whole row followed by a
-1 tail. Nothing in the N >= 2048 grid exercises that, and the dispatch there
lands on aiter's one-block path rather than AVO (AVO needs stride0 >= 32768).

Driven through the production dispatcher `aiter.top_k_per_row_prefill` with full
rows, which is also the only way to express M > N -- `op_tests` builds ragged
rows of width num_prefix + num_rows and cannot reach M > N at these widths.

Latency is reported but NOT gated: this region is host-bound (about 16 us of
wall time around a 2.4 us kernel, see aiter commit 76e94f4df), so the number
tracks Python overhead and would make a flaky gate. Correctness is gated.

Run inside the correctness image with both repos mounted:

  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \\
    --ipc=host --shm-size 16G -e PYTHONPATH=/aiter \\
    -v /home/mh/aiter-topk:/aiter -v /home/mh/topk-prefill-avo:/topk -w /aiter \\
    rocm/ali-private:ubuntu22.04_rocm10.0.0_cp313_torch2.12.0_vllm_dsv4_20260916 \\
    python /topk/bench/spec_low_n.py

PYTHONPATH=/aiter is load-bearing, not decoration -- see the guard below.
"""
import argparse
import json
import statistics as st
import time

import torch

import aiter

# `python /topk/bench/spec_low_n.py` puts sys.path[0] at /topk/bench, not at the
# cwd, so `import aiter` silently resolves to whatever copy is installed in
# site-packages instead of the mounted repo. That produced a full green sweep
# against a module we had not changed. Fail loudly rather than measure a stranger.
_EXPECT = "/aiter/"
if not aiter.__file__.startswith(_EXPECT):
    raise SystemExit(
        "WRONG AITER: imported %s, expected one under %s. "
        "Re-run with -e PYTHONPATH=/aiter." % (aiter.__file__, _EXPECT)
    )

MS = [1, 4, 8, 16, 64, 256, 1024, 4096]
NS = [512, 1024, 2048, 4096]
K = 2048


def check(logits, idx, m, n, k):
    """Compare against torch.topk on three rows: first, middle, last.

    Checks the index set, not the values, because ties at these widths are
    common and the kernel may break them differently from torch. The value
    multiset must still match exactly.
    """
    want = min(k, n)
    for r in (0, m // 2, m - 1):
        row = idx[r]
        if want < k and not bool((row[want:] == -1).all().item()):
            return "row %d: padding past %d is not -1" % (r, want)
        g = row[:want]
        if not bool(((g >= 0) & (g < n)).all().item()):
            return "row %d: index outside [0, %d)" % (r, n)
        if len(set(g.tolist())) != want:
            return "row %d: duplicate indices" % r
        got = torch.sort(logits[r][g]).values
        ref = torch.sort(torch.topk(logits[r], want).values).values
        if not torch.equal(got, ref):
            return "row %d: value multiset != torch.topk" % r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write per-cell results here")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument(
        "--inject-fault", type=int, default=0, choices=[0, 1, 2, 3],
        help="corrupt the result to prove the gate can go red: "
             "1 = duplicate an index, 2 = index out of range, "
             "3 = wrong value multiset (swap in a non-top element)",
    )
    args = ap.parse_args()

    print("aiter under test: %s" % aiter.__file__)
    print("customer spec low-N corner, topk=%d, full rows" % K)
    print("%6s %7s %6s | %9s %9s | %s"
          % ("M", "N", "k>=N", "enqueue", "e2e_us", "correctness"))

    rows, failed = [], 0
    for m in MS:
        for n in NS:
            g = torch.Generator(device="cuda")
            g.manual_seed(42)
            lg = torch.randn((m, n), generator=g, dtype=torch.float32, device="cuda")
            rs = torch.zeros(m, dtype=torch.int32, device="cuda")
            re = torch.full((m,), n, dtype=torch.int32, device="cuda")
            idx = torch.full((m, K), -2, dtype=torch.int32, device="cuda")
            call_args = (lg, rs, re, idx, None, m, n, 1, K)

            try:
                aiter.top_k_per_row_prefill(*call_args)
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001 -- a raise here is a failure
                failed += 1
                print("%6d %7d %6s | %9s %9s | RAISED: %s"
                      % (m, n, "yes" if K >= n else "no", "-", "-", str(e)[:50]))
                rows.append({"M": m, "N": n, "verdict": "RAISED", "detail": str(e)})
                continue

            if args.inject_fault:
                want = min(K, n)
                if args.inject_fault == 1 and want > 1:
                    idx[0, 1] = idx[0, 0]          # duplicate index
                elif args.inject_fault == 2:
                    idx[0, 0] = n                  # index past the row
                elif args.inject_fault == 3 and want < n:
                    # Swap in an element that is genuinely not in the top-k:
                    # torch.topk's own (want)-th pick, which by definition ranks
                    # below every index the kernel should have returned.
                    idx[0, 0] = int(torch.topk(lg[0], want + 1).indices[want])

            bad = check(lg.cpu(), idx.cpu().to(torch.int64), m, n, K)
            if bad:
                failed += 1

            for _ in range(5):
                aiter.top_k_per_row_prefill(*call_args)
            torch.cuda.synchronize()
            enq, e2e = [], []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                for _ in range(args.iters):
                    aiter.top_k_per_row_prefill(*call_args)
                t1 = time.perf_counter()
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                enq.append((t1 - t0) / args.iters * 1e6)
                e2e.append((t2 - t0) / args.iters * 1e6)

            me, m2 = st.median(enq), st.median(e2e)
            print("%6d %7d %6s | %9.2f %9.2f | %s"
                  % (m, n, "yes" if K >= n else "no", me, m2, bad or "PASS"))
            rows.append({"M": m, "N": n, "enqueue_us": me, "e2e_us": m2,
                         "verdict": "FAIL" if bad else "PASS", "detail": bad})

    print()
    print("  cells    %d" % len(rows))
    print("  failed   %d   <-- must be 0" % failed)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print("WROTE %s" % args.json)
    print()
    print("SPEC LOW-N GATE %s" % ("PASS" if failed == 0 else "FAIL"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
