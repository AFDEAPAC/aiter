"""Exhaustive correctness gate for the sampled top-k dispatch.

Three invariants per (M, N, k, distribution, dispatch path):
  values   the multiset of selected values must equal torch.topk's -- the right
           invariant under ties, where WHICH equal element comes back may differ
           but what it is worth may not.
  unique   the returned indices must be distinct. A value-multiset check cannot
           see a duplicate index and a duplicate is a real defect.
  range    every index in [0, row_len).

The path matters as much as the shape: the regression this gate exists for was
correct on the ragged path and wrong on the plain one at the same width.
"""
import os
import sys

import torch

import aiter

FAILS = []
CASES = [0]


def call(x, k, end=None):
    r = aiter.topk_select(x, k, end=end) if end is not None else aiter.topk_select(x, k)
    if isinstance(r, (tuple, list)):
        r = [t for t in r if t is not None][-1]
    return r


def make(M, N, dist, seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if dist == "gaussian":
        return torch.randn(M, N, device="cuda", dtype=torch.float32)
    if dist == "uniform":
        return torch.rand(M, N, device="cuda", dtype=torch.float32)
    if dist == "all_equal":
        return torch.full((M, N), 1.25, device="cuda", dtype=torch.float32)
    if dist == "two_values":
        return torch.randint(0, 2, (M, N), device="cuda").float() * 3.0 - 1.0
    if dist == "ascending":
        return torch.arange(N, device="cuda", dtype=torch.float32).repeat(M, 1)
    if dist == "descending":
        return torch.arange(N - 1, -1, -1, device="cuda", dtype=torch.float32).repeat(M, 1)
    if dist == "negative":
        return -torch.rand(M, N, device="cuda", dtype=torch.float32) - 1.0
    if dist == "with_inf":
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        x[:, ::9999] = float("inf")
        x[:, 3::9999] = float("-inf")
        return x
    if dist == "tiny_spread":
        return 1.0 + torch.randn(M, N, device="cuda", dtype=torch.float32) * 1e-7
    raise SystemExit("unknown dist " + dist)


def check(M, N, k, dist, ragged):
    CASES[0] += 1
    x = make(M, N, dist)
    end = None
    lens = [N] * M
    if ragged:
        e = torch.full((M,), N, device="cuda", dtype=torch.int32)
        if M >= 4:
            e[0] = max(1, k // 2)
            e[1] = k
            e[2] = min(N, 4096)
        end = e
        lens = [int(v) for v in e]
    try:
        idx = call(x, k, end)
    except Exception as exc:
        FAILS.append((M, N, k, dist, ragged, "raised " + type(exc).__name__, str(exc)[:80]))
        return
    if idx.dtype is not torch.int32:
        FAILS.append((M, N, k, dist, ragged, "dtype", str(idx.dtype)))
    for r in range(M):
        L = lens[r]
        take = min(k, L)
        gi = idx[r][:take].to(torch.int64)
        if int(gi.min()) < 0 or int(gi.max()) >= L:
            FAILS.append((M, N, k, dist, ragged, "index outside [0,%d) row %d" % (L, r),
                          "min %d max %d" % (int(gi.min()), int(gi.max()))))
            return
        if int(torch.unique(gi).numel()) != take:
            FAILS.append((M, N, k, dist, ragged, "duplicate index row %d" % r,
                          "%d distinct of %d" % (int(torch.unique(gi).numel()), take)))
            return
        got = x[r][gi].sort().values
        ref = torch.topk(x[r][:L].float(), take).values.sort().values
        if not torch.equal(got, ref):
            FAILS.append((M, N, k, dist, ragged, "value mismatch row %d" % r,
                          "%d of %d" % (int((got != ref).sum()), take)))
            return


def banner(t):
    print("\n=== %s ===" % t, flush=True)
    print("   %d cases, %d failures so far" % (CASES[0], len(FAILS)), flush=True)


print("AITER:", aiter.__file__, flush=True)
S = os.environ.get("STAGE", "all")

if S in ("all", "1"):
    banner("1. every residue mod 4 near each routing-relevant width, 9 distributions")
    for N in (131072, 131073, 131074, 131075, 131076, 131077, 131078, 131079,
              262144, 262147, 262151, 524288, 524291, 1048576, 1048579):
        for d in ("gaussian", "uniform", "all_equal", "two_values", "ascending",
                  "descending", "negative", "with_inf", "tiny_spread"):
            check(8, N, 2048, d, False)

if S in ("all", "2"):
    banner("2. widths nowhere near a power of two")
    for N in (130000, 131071, 133337, 150001, 199999, 262143, 300007, 500009,
              524287, 700001, 999983, 1000000, 1048575):
        for d in ("gaussian", "all_equal", "with_inf"):
            check(8, N, 2048, d, False)

if S in ("all", "3"):
    banner("3. M coverage, odd and boundary row counts")
    for M in (1, 2, 3, 5, 8, 16, 33, 64, 127, 128, 255, 256, 512, 1023, 1024):
        for N in (131072, 131075, 262147):
            check(M, N, 2048, "gaussian", False)

if S in ("all", "4"):
    banner("4. k coverage")
    for k in (1, 2, 15, 16, 512, 1000, 1024, 2048):
        for N in (131072, 131075, 262144, 262147):
            check(8, N, k, "gaussian", False)
            check(8, N, k, "all_equal", False)

if S in ("all", "5"):
    banner("5. the ragged path, with short rows, at the same widths")
    for N in (131072, 131075, 131077, 262144, 262147, 524288, 1048576):
        for d in ("gaussian", "all_equal", "with_inf"):
            check(8, N, 2048, d, True)

if S in ("all", "6"):
    banner("6. widths below the sampled routing gate, where another backend serves")
    for N in (2048, 4096, 8192, 8193, 32768, 32771, 65536, 65539, 98304, 131071):
        for d in ("gaussian", "all_equal"):
            check(8, N, 2048, d, False)

if S in ("all", "7"):
    banner("7. large M, the shapes the customer grid uses")
    for M, N in ((1024, 131072), (1024, 131075), (2048, 131072), (2048, 262147),
                 (4096, 131072), (4096, 131075), (512, 524288), (512, 524291),
                 (256, 1048576), (256, 1048579), (128, 1048576)):
        for d in ("gaussian", "all_equal"):
            check(M, N, 2048, d, False)

if S in ("all", "8"):
    banner("8. repeatability: the same call twice must return the same SELECTION")
    # Deliberately not torch.equal on the indices. With sorted=False the order is
    # unspecified -- the gather races for output slots -- and even with
    # sorted=True two exactly equal values may come back in either order. Both
    # behaviours predate this branch: the upstream tip returns 4 and 22 differing
    # index positions at m=8 n=131072 and m=64 n=262144, every one of them a tie.
    # What must be stable is the set of columns and the multiset of values.
    for M, N in ((8, 131075), (64, 262147), (512, 131072), (8, 131072)):
        x = make(M, N, "gaussian")
        a = call(x, 2048).clone().to(torch.int64)
        b = call(x, 2048).clone().to(torch.int64)
        CASES[0] += 1
        if not torch.equal(a.sort(dim=1).values, b.sort(dim=1).values):
            FAILS.append((M, N, 2048, "gaussian", False, "column set not stable",
                          "%d rows differ" % int((a.sort(dim=1).values
                                                  != b.sort(dim=1).values).any(1).sum())))
            continue
        va = torch.gather(x, 1, a).sort(dim=1).values
        vb = torch.gather(x, 1, b).sort(dim=1).values
        if not torch.equal(va, vb):
            FAILS.append((M, N, 2048, "gaussian", False, "value multiset not stable",
                          "%d of %d differ" % (int((va != vb).sum()), va.numel())))

print()
print("=" * 78)
print("%d cases checked. %d FAILURES." % (CASES[0], len(FAILS)))
for f in FAILS[:40]:
    print("   m=%-5d n=%-9d k=%-5d %-11s ragged=%-5s %s | %s" % f)
print("=" * 78)
sys.exit(1 if FAILS else 0)
