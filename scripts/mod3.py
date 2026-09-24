"""Focused stress on N % 4 == 3, the longest tail the plain path has to pick up.

ncols = N % FP32_EPT, so this residue is the only one that exercises a
three-column tail -- one lane inactive in the ballot, three columns staged, and
the largest chance of an off-by-one at either end of the range. The 390-cell
performance grid never reaches it: its width offsets are +0, +1 and +2 only.
"""
import torch

import aiter

K = 2048
FAILS = []
CASES = [0]


def call(x, k, end=None):
    r = aiter.topk_select(x, k, end=end) if end is not None else aiter.topk_select(x, k)
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
    if dist == "uniform":
        return torch.rand(M, N, device="cuda", dtype=torch.float32)
    if dist == "tail_is_best":
        # The three columns the old code dropped hold the three LARGEST values,
        # so any failure to read them is a guaranteed, not a probabilistic, miss.
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        x[:, -3:] = 1e6
        return x
    if dist == "tail_is_worst":
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        x[:, -3:] = -1e6
        return x
    if dist == "with_inf":
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        x[:, ::9999] = float("inf")
        return x
    raise SystemExit(dist)


def check(M, N, dist, ragged=False):
    CASES[0] += 1
    x = make(M, N, dist)
    end = None
    lens = [N] * M
    if ragged:
        e = torch.full((M,), N, device="cuda", dtype=torch.int32)
        e[0] = N - 3
        e[1] = N - 1
        end = e
        lens = [int(v) for v in e]
    idx = call(x, K, end)
    for r in range(M):
        L = lens[r]
        take = min(K, L)
        gi = idx[r][:take].to(torch.int64)
        if int(gi.min()) < 0 or int(gi.max()) >= L:
            FAILS.append((M, N, dist, ragged, "index outside [0,%d) row %d" % (L, r)))
            return
        if int(torch.unique(gi).numel()) != take:
            FAILS.append((M, N, dist, ragged, "duplicate index row %d" % r))
            return
        got = x[r][gi].sort().values
        ref = torch.topk(x[r][:L].float(), take).values.sort().values
        if not torch.equal(got, ref):
            FAILS.append((M, N, dist, ragged,
                          "value mismatch row %d, %d of %d" % (r, int((got != ref).sum()), take)))
            return


print("AITER:", aiter.__file__, flush=True)
W3 = [n for n in
      (131075, 131079, 131083, 131199, 132099, 139267, 163839, 196611,
       262147, 262151, 262271, 299999, 393215, 524291, 524299, 699999,
       786431, 999999, 1048579, 1048583)
      if n % 4 == 3]
print("widths with N %% 4 == 3 under test: %d" % len(W3), flush=True)
for N in W3:
    for d in ("gaussian", "all_equal", "tail_is_best", "tail_is_worst"):
        check(8, N, d)
print("  after width sweep: %d cases, %d failures" % (CASES[0], len(FAILS)), flush=True)
for M in (1, 2, 3, 5, 8, 16, 33, 64, 127, 128, 255, 256, 512, 1024, 2048):
    for N in (131075, 262147, 524291):
        check(M, N, "gaussian")
        check(M, N, "tail_is_best")
print("  after M sweep: %d cases, %d failures" % (CASES[0], len(FAILS)), flush=True)
for k in (1, 2, 16, 512, 1024, 2048):
    for N in (131075, 262147):
        CASES[0] += 1
        x = make(8, N, "tail_is_best")
        gi = call(x, k)[0][:k].to(torch.int64)
        got = x[0][gi].sort().values
        ref = torch.topk(x[0].float(), k).values.sort().values
        if not torch.equal(got, ref):
            FAILS.append((8, N, "tail_is_best k=%d" % k, False, "value mismatch"))
print("  after k sweep: %d cases, %d failures" % (CASES[0], len(FAILS)), flush=True)
for N in (131075, 262147, 524291):
    for d in ("gaussian", "tail_is_best", "with_inf"):
        check(8, N, d, ragged=True)
print("  after ragged sweep: %d cases, %d failures" % (CASES[0], len(FAILS)), flush=True)
print()
print("=" * 70)
print("N %% 4 == 3 focused stress: %d cases, %d FAILURES" % (CASES[0], len(FAILS)))
for f in FAILS[:30]:
    print("   m=%-5d n=%-9d %-14s ragged=%-5s %s" % f)
print("=" * 70)
