#!/usr/bin/env python3
"""Differential contract audit: the AVO op against the aiter entry it replaces.

This op is dispatched in place of `top_k_per_row_prefill`, so every term of that
contract aiter honours and AVO silently does not is a functional bug waiting for
the first caller who uses it. The shipped HIP-700 fault on unaligned row bases
was found by accident; this finds the rest by construction.

Each case runs in its OWN subprocess. A memory fault poisons the HIP context, so
one failing case would otherwise take every later case down with it and report a
wall of false failures.

Run inside the correctness image with both repos mounted:

  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    --ipc=host --shm-size 16G \
    -v /home/mh/aiter-topk:/aiter -v /home/mh/topk-prefill-avo:/topk -w /aiter \
    rocm/ali-private:ubuntu22.04_rocm10.0.0_cp313_torch2.12.0_vllm_dsv4_20260916 \
    python /topk/bench/aiter_contract_audit.py

Delete aiter/jit/module_top_k_per_row.so first: the JIT reuses a stale .so and
reports the pre-change verdict (knowledge/known_bad.md, "aiter's JIT does not
rebuild when a generated source changes").
"""

import argparse
import json
import os
import subprocess
import sys

# Each case: name -> (what the contract term is, whether aiter is expected to
# serve it). AVO's verdict is what we are measuring.
CASES = [
    ("baseline_uniform",      "rowStarts=0, aligned, pow2 N: the shape the grid already covers"),
    ("rowstart_unaligned_1",  "rowStarts = r*1, so most row bases are not 4-aligned"),
    ("rowstart_unaligned_65", "rowStarts = r*65, the aiter-like stride that is not 4-aligned"),
    ("rowstart_aligned_4",    "rowStarts = r*4: aligned, isolates alignment from row length"),
    ("stride0_odd",           "stride0 not a multiple of 4, residue 1 (served since v5 Stage 7)"),
    ("stride0_odd_2",         "stride0 not a multiple of 4, residue 2: a 2-element clamped tail"),
    ("stride0_odd_3",         "stride0 not a multiple of 4, residue 3: a 3-element clamped tail"),
    ("stride0_odd_tail_max",  "odd stride0 with each row's max planted in the clamped tail"),
    ("stride0_nonpow2",       "stride0 = 2^k + num_rows, aiter's real width"),
    ("stride1_not_one",       "stride1 != 1: aiter ignores it, AVO asserts"),
    ("nan_positive",          "+NaN in the data"),
    ("nan_negative",          "-NaN in the data: sign-dependent under a bitwise key"),
    ("inf_mixed",             "+inf and -inf in the data"),
    ("k_small",               "k = 3"),
    ("k_at_cap",              "k = 8192, the Phase C LDS cap"),
    ("rowlen_zero",           "rowEnds == rowStarts: zero-length rows"),
    ("rowlen_negative",       "rowEnds < rowStarts"),
    ("rowend_past_stride0",   "rowEnds > stride0"),
    ("rowlen_short",          "row_len < k: identity emit and the padding value"),
    ("numrows_one",           "numRows = 1"),
    ("workspace_exact",       "workspace sized exactly at topk_sampled_workspace_size"),
    ("stable_true",           "stable=True must never route to AVO"),
]


AITER_ROOT = os.environ.get("AITER_ROOT", "/aiter")


def use_mounted_aiter():
    """Force the mounted checkout ahead of the image's own /opt/aiter.

    The correctness image ships its own aiter, and `import aiter` picks that one
    up, so the audit would measure a different build than the one this repo
    exports into.
    """
    sys.path.insert(0, AITER_ROOT)


def stub_flydsl():
    """Make `import aiter` work despite an unrelated FlyDSL version mismatch.

    The installed flydsl is a different version than aiter expects, and topk does
    not touch it, so every module under `aiter.ops.flydsl` is answered by a stub.
    Done with a meta_path finder rather than a fixed name list because the import
    chain reaches several modules deep and grows; chasing the names one
    ModuleNotFoundError at a time is not a terminating process.
    """
    import importlib.abc
    import importlib.machinery
    import types

    PKG = "aiter.ops.flydsl"

    class Stub(types.ModuleType):
        __path__ = []

        def __getattr__(self, n):
            if n.startswith("__"):
                raise AttributeError(n)
            # Any attribute is either a submodule or a callable; a module object
            # satisfies `from X import name` for both.
            m = Stub(self.__name__ + "." + n)
            m.__spec__ = importlib.machinery.ModuleSpec(m.__name__, None)
            sys.modules[m.__name__] = m
            return m

        def __call__(self, *a, **k):
            return None

    class Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, name, path=None, target=None):
            if name == PKG or name.startswith(PKG + "."):
                return importlib.machinery.ModuleSpec(name, self, is_package=True)
            return None

        def create_module(self, spec):
            return Stub(spec.name)

        def exec_module(self, module):
            module.is_flydsl_available = lambda: False
            module.wave_size_of = lambda *a, **k: 64
            module._run_compiled = lambda *a, **k: None

    sys.meta_path.insert(0, Finder())


def build_case(case, torch):
    """Return (logits, row_starts, row_ends, numRows, stride0, stride1, k, stable)."""
    dev = "cuda"
    M, N, K = 64, 65536, 2048
    stride1 = 1
    stable = False

    if case == "stride0_odd":
        N = 65537
    elif case == "stride0_odd_2":
        N = 65538
    elif case == "stride0_odd_3":
        N = 65539
    elif case == "stride0_odd_tail_max":
        N = 65539
    elif case == "stride0_nonpow2":
        N = 65536 + 64
    elif case == "k_small":
        K = 3
    elif case == "k_at_cap":
        K = 8192
    elif case == "numrows_one":
        M = 1
    elif case == "stable_true":
        stable = True
    elif case == "stride1_not_one":
        stride1 = 2

    torch.manual_seed(42)
    logits = torch.randn((M, N), dtype=torch.float32, device=dev)

    if case == "nan_positive":
        logits[:, N // 3] = float("nan")
    elif case == "nan_negative":
        logits[:, N // 3] = -float("nan")
    elif case == "inf_mixed":
        logits[:, N // 3] = float("inf")
        logits[:, N // 5] = -float("inf")
    elif case == "stride0_odd_tail_max":
        # The last element of an odd row is only reachable through the partial
        # vector that load_row_f4 clamps, so planting each row's maximum there
        # is a positive control: if the tail were dropped, the top-k would be
        # missing its largest entry and the multiset check would catch it.
        logits[:, N - 1] = 1e6

    starts = torch.zeros(M, dtype=torch.int32, device=dev)
    ends = torch.full((M,), N, dtype=torch.int32, device=dev)

    if case == "rowstart_unaligned_1":
        starts = (torch.arange(M, dtype=torch.int32, device=dev) * 1)
    elif case == "rowstart_unaligned_65":
        starts = (torch.arange(M, dtype=torch.int32, device=dev) * 65)
    elif case == "rowstart_aligned_4":
        starts = (torch.arange(M, dtype=torch.int32, device=dev) * 4)
    elif case == "rowlen_zero":
        ends = starts.clone()
    elif case == "rowlen_negative":
        starts = torch.full((M,), 16, dtype=torch.int32, device=dev)
        ends = torch.zeros(M, dtype=torch.int32, device=dev)
    elif case == "rowend_past_stride0":
        ends = torch.full((M,), N + 64, dtype=torch.int32, device=dev)
    elif case == "rowlen_short":
        ends = torch.full((M,), K // 2, dtype=torch.int32, device=dev)

    return logits, starts.contiguous(), ends.contiguous(), M, N, stride1, K, stable


def run_one(case, which):
    """Run a single case under a single backend. Returns a result dict."""
    use_mounted_aiter()
    stub_flydsl()
    import torch
    import aiter
    from aiter.ops.topk import topk_sampled_supports

    if not aiter.__file__.startswith(AITER_ROOT):
        print(json.dumps({"case": case, "backend": which, "outcome": "wrong_aiter",
                          "error": "imported %s, expected under %s"
                                   % (aiter.__file__, AITER_ROOT)}))
        return 0

    if which == "aiter":
        os.environ["AITER_DISABLE_TOPK_SAMPLED"] = "1"
    else:
        os.environ["AITER_DISABLE_TOPK_SAMPLED"] = "0"

    logits, starts, ends, M, N, stride1, K, stable = build_case(case, torch)
    idx = torch.full((M, K), -1, dtype=torch.int32, device="cuda")
    vals = torch.full((M, K), 0.0, dtype=torch.float32, device="cuda")

    res = {"case": case, "backend": which, "m": M, "n": N, "k": K}
    try:
        res["sampled_supports"] = bool(topk_sampled_supports(M, N, K))
    except Exception as e:
        res["sampled_supports"] = "raised: %s" % str(e)[:80]

    try:
        aiter.top_k_per_row_prefill(logits, starts, ends, idx, vals,
                                    M, N, stride1, K, stable)
        torch.cuda.synchronize()
    except Exception as e:
        res["outcome"] = "raised"
        res["error"] = str(e)[:240]
        print(json.dumps(res))
        return 0

    res["outcome"] = "ran"
    # Summarise the output so the parent can diff two backends without shipping
    # whole tensors between processes.
    i = idx.detach().to(torch.int64).cpu()
    v = vals.detach().cpu()
    res["idx_sum"] = int(i.sum().item())
    res["idx_min"] = int(i.min().item())
    res["idx_max"] = int(i.max().item())
    res["n_neg1"] = int((i == -1).sum().item())
    finite = v[torch.isfinite(v)]
    # Order-INDEPENDENT digest. A raw fp32 sum is not one: AVO and aiter emit the
    # same multiset in different orders, so torch.sum accumulates in a different
    # order and lands on a different rounding. Using the plain sum as an equality
    # key reported the baseline shape as diverging when all 64 rows were in fact
    # bit-identical multisets. Sort first, accumulate in float64.
    res["val_digest"] = (round(float(finite.double().sort().values.sum().item()), 6)
                         if finite.numel() else None)
    res["n_neg_inf"] = int((v == float("-inf")).sum().item())
    res["n_pos_inf"] = int((v == float("inf")).sum().item())
    res["n_nan"] = int(torch.isnan(v).sum().item())
    res["n_zero"] = int((v == 0.0).sum().item())

    # Value multiset against torch, on row 0 only, and only where the row is a
    # well-formed slice. torch is the independent oracle; aiter and AVO are both
    # candidates against it.
    s0, e0 = int(starts[0].item()), int(ends[0].item())
    if 0 <= s0 <= e0 <= N:
        rl = e0 - s0
        n = min(K, rl)
        if n > 0:
            g = i[0, :n]
            if bool(((g >= s0) & (g < e0)).all().item()):
                gv = torch.sort(logits[0].cpu()[g], descending=True).values
                rv = torch.sort(torch.topk(logits[0, s0:e0].cpu(), n).values,
                                descending=True).values
                res["topk_match"] = bool(torch.equal(gv, rv))
                res["topk_match_nan_eq"] = bool(
                    torch.equal(torch.nan_to_num(gv, 1e30, 1e30, -1e30),
                                torch.nan_to_num(rv, 1e30, 1e30, -1e30)))
            else:
                res["topk_match"] = "index out of [start,end)"
    print(json.dumps(res))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None)
    ap.add_argument("--backend", choices=("sampled", "aiter"), default="sampled")
    ap.add_argument("--json-out", default="/topk/log/aiter_contract_audit.json")
    args = ap.parse_args()

    if args.case:
        return run_one(args.case, args.backend)

    out = []
    print("%-24s %-9s %-8s %-9s %s" % ("case", "backend", "supports", "outcome", "detail"))
    for case, desc in CASES:
        row = {"case": case, "term": desc}
        for which in ("sampled", "aiter"):
            p = subprocess.run([sys.executable, __file__, "--case", case,
                                "--backend", which],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)
            rec = None
            for line in p.stdout.splitlines():
                line = line.strip()
                if line.startswith("{"):
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        pass
            if rec is None:
                rec = {"outcome": "crashed", "rc": p.returncode,
                       "error": (p.stderr.strip().splitlines() or ["(no stderr)"])[-1][:200]}
            row[which] = rec
            detail = rec.get("error", "")
            if not detail:
                detail = "match=%s neg1=%s -inf=%s nan=%s" % (
                    rec.get("topk_match"), rec.get("n_neg1"),
                    rec.get("n_neg_inf"), rec.get("n_nan"))
            rec["term"] = desc
            print("%-24s %-9s %-8s %-9s %s"
                  % (case, which, rec.get("sampled_supports"), rec.get("outcome"),
                     detail[:88]))
        out.append(row)

    try:
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
        json.dump(out, open(args.json_out, "w"), indent=1)
        print("\nWROTE %s" % args.json_out)
    except OSError as e:
        print("could not write json: %s" % e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
