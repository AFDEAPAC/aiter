"""N+0 / N+1 / N+2 / N+3 across the whole customer grid.

Every base width the spec lists, each with the four offsets, against the M values
the spec lists. Three invariants per case -- value multiset against torch.topk,
index uniqueness, index range -- on every row where that is affordable and on a
spread of rows where it is not.

tail_is_best is included at the widths that route to `sampled`: it puts the three
largest values in exactly the columns a truncated n4 would skip, so a failure to
read the tail is certain rather than probabilistic.
"""

import os
import sys

import aiter
import torch

K = 2048
MAXELEM = 3 << 30  # 12 GB of fp32
FAILS = []
CASES = [0]
ROWS = [0]


def call(x, k):
    r = aiter.topk_select(x, k)
    if isinstance(r, (tuple, list)):
        r = [t for t in r if t is not None][-1]
    return r


def make(M, N, dist):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    if dist == "gaussian":
        return torch.randn(M, N, device="cuda", dtype=torch.float32)
    if dist == "all_equal":
        return torch.full((M, N), 1.25, device="cuda", dtype=torch.float32)
    if dist == "tail_is_best":
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        x[:, -3:] = 1e6
        return x
    raise SystemExit(dist)


def check(M, N, dist):
    if M * N > MAXELEM:
        return
    CASES[0] += 1
    k = min(K, N)
    try:
        x = make(M, N, dist)
        idx = call(x, k)
    except Exception as exc:
        FAILS.append((M, N, dist, "raised " + type(exc).__name__, str(exc)[:70]))
        return
    rows = range(M) if M <= 128 else [0, 1, M // 3, M // 2, 2 * M // 3, M - 2, M - 1]
    for r in rows:
        ROWS[0] += 1
        gi = idx[r][:k].to(torch.int64)
        if int(gi.min()) < 0 or int(gi.max()) >= N:
            FAILS.append((M, N, dist, "index outside [0,%d) row %d" % (N, r), ""))
            return
        if int(torch.unique(gi).numel()) != k:
            FAILS.append((M, N, dist, "duplicate index row %d" % r, ""))
            return
        got = x[r][gi].sort().values
        ref = torch.topk(x[r].float(), k).values.sort().values
        if not torch.equal(got, ref):
            FAILS.append(
                (
                    M,
                    N,
                    dist,
                    "value mismatch row %d" % r,
                    "%d of %d" % (int((got != ref).sum()), k),
                )
            )
            return


BASES = [
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
]
MS = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
print("AITER:", aiter.__file__, flush=True)
only = os.environ.get("BASE")
for base in BASES:
    if only and int(only) != base:
        continue
    for off in (0, 1, 2, 3):
        N = base + off
        for M in MS:
            check(M, N, "gaussian")
            if base >= 131072:
                check(M, N, "tail_is_best")
            if M <= 64:
                check(M, N, "all_equal")
    print(
        "  base %-8d done: %d cases, %d rows checked, %d failures"
        % (base, CASES[0], ROWS[0], len(FAILS)),
        flush=True,
    )

print()
print("=" * 76)
print(
    "N+0/N+1/N+2/N+3 over the customer grid: %d cases, %d rows, %d FAILURES"
    % (CASES[0], ROWS[0], len(FAILS))
)
for f in FAILS[:40]:
    print("   m=%-5d n=%-9d %-14s %s %s" % f)
print("=" * 76)
sys.exit(1 if FAILS else 0)
