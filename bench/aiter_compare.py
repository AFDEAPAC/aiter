#!/usr/bin/env python3
"""Head-to-head: this repo's top-k kernel vs aiter's top_k_per_row_prefill.

Run inside an image whose Python matches aiter's prebuilt .so (python3.10 for
/home/mh/aiter), with both repos mounted:

  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    --ipc=host --shm-size 16G \
    -v /home/mh/aiter:/aiter -v /home/mh/topk-prefill-avo:/topk -w /aiter \
    lmsysorg/sglang-rocm:v0.5.18-rocm720-mi35x-20260822 \
    python /topk/bench/aiter_compare.py

Why this exists before any integration work: if aiter's own gfx950 kernel is
already faster, registering ours into aiter is pointless. This measures that
first, on shapes BOTH implementations accept.

aiter's contract (csrc/include/topk_per_row.h, verified 2026-09-17):
    top_k_per_row_prefill(logits, rowStarts, rowEnds, indices, values,
                          numRows, stride0, stride1, k=2048)
  - logits  fp32 [numRows, maxN], padded with -inf beyond each row's end
  - valid slice of row i is [rowStarts[i], rowEnds[i]); row_len may be < k
  - indices int32 [numRows, k], -1 for slots a short row cannot fill
  - the asm `_fast` path exists only for gfx942; hsa/gfx950 has no topk .co,
    so on MI355X aiter has only the generic HIP kernel. That empty slot is
    what this kernel would fill.
"""

import argparse
import importlib
import json
import os
import statistics
import subprocess
import sys
import time
import types


def stub_flydsl():
    """Make `import aiter` work despite an unrelated FlyDSL version mismatch.

    The installed flydsl is a different version than aiter/ops/flydsl expects
    (`cannot import name 'vector'/'buffer_ops' from flydsl.expr`). topk does not
    touch flydsl, so the whole aiter.ops.flydsl subpackage is replaced BEFORE
    import so its __init__ never runs. Stubbing flydsl itself does not work:
    aiter calls importlib.util.find_spec("flydsl"), which needs a real __spec__.
    """
    class Stub(types.ModuleType):
        def __getattr__(self, n):
            if n.startswith("__"):
                raise AttributeError(n)
            m = Stub(self.__name__ + "." + n)
            m.__spec__ = importlib.machinery.ModuleSpec(m.__name__, None)
            sys.modules[m.__name__] = m
            return m

    for name in ("aiter.ops.flydsl", "aiter.ops.flydsl.utils"):
        m = Stub(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)
        m.__path__ = []
        sys.modules[name] = m
    sys.modules["aiter.ops.flydsl.utils"].is_flydsl_available = lambda: False
    sys.modules["aiter.ops.flydsl"].is_flydsl_available = lambda: False


def time_us(fn, warmup, iters, repeats):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    meds = []
    for _ in range(repeats):
        s = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            s.append((time.perf_counter() - t0) * 1e6)
        meds.append(statistics.median(s))
    return statistics.median(meds), meds


def topk_set_match(logits, got, ref_k, row_len):
    """Set-of-values comparison, the same tolerance aiter's own test uses:
    ties may resolve to different indices, so compare the value multiset."""
    import torch
    n = min(ref_k, row_len)
    g = got[:n].long()
    if (g < 0).any() or (g >= row_len).any():
        return False, "index out of range"
    gv = torch.sort(logits[g], descending=True).values
    rv = torch.sort(torch.topk(logits[:row_len], n).values, descending=True).values
    return bool(torch.equal(gv, rv)), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk-bin", default="/topk/benchmark_topk")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json-out", default="/topk/reports/aiter_compare.json")
    args = ap.parse_args()

    stub_flydsl()
    import torch
    import aiter
    from aiter.jit.utils.chip_info import get_gfx

    gfx = get_gfx()
    dev = torch.cuda.get_device_name(0)
    print("device %s  gfx %s  torch %s" % (dev, gfx, torch.__version__))
    print("aiter has _fast prefill path for this arch: %s"
          % ("yes" if gfx == "gfx942" else "NO (hsa/%s has no topk .co)" % gfx))
    print()

    # Uniform-length rows: the only shapes both implementations accept, since
    # this repo's kernel assumes every row is exactly N long.
    shapes = [(1, 65536), (1, 262144), (1, 1048576),
              (8, 65536), (64, 65536), (256, 65536),
              (64, 131072), (256, 131072), (1024, 131072), (4096, 131072),
              (64, 262144), (1024, 1048576)]
    K = 2048
    out = []

    for (M, N) in shapes:
        if M * N * 4 > 40 * 1024 ** 3:
            continue
        torch.manual_seed(42)
        logits = torch.randn((M, N), dtype=torch.float32, device="cuda")
        row_starts = torch.zeros(M, dtype=torch.int32, device="cuda")
        row_ends = torch.full((M,), N, dtype=torch.int32, device="cuda")
        idx = torch.full((M, K), -1, dtype=torch.int32, device="cuda")

        def run_aiter():
            aiter.top_k_per_row_prefill(logits, row_starts, row_ends, idx, None,
                                        M, logits.stride(0), logits.stride(1), K)

        try:
            run_aiter()
            torch.cuda.synchronize()
        except Exception as e:
            print("M=%-5d N=%-8d aiter FAILED: %s" % (M, N, str(e)[:110]))
            out.append({"m": M, "n": N, "aiter_us": None, "err": str(e)[:200]})
            continue

        ok, why = topk_set_match(logits[0], idx[0], K, N)
        a_us, _ = time_us(run_aiter, args.warmup, args.iters, args.repeats)

        # This repo, same shape, via its own binary (own timing loop).
        mine_us = None
        if os.path.exists(args.topk_bin):
            p = subprocess.run([args.topk_bin, "--mode", "time", "--m", str(M), "--n", str(N),
                                "--topk", str(K), "--warmup", "20", "--iters", "100",
                                "--repeats", "5"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)
            import re
            mm = re.search(r"wall_ms_median=([0-9.]+)", p.stdout)
            if mm:
                mine_us = float(mm.group(1)) * 1000.0

        rec = {"m": M, "n": N, "topk": K, "aiter_us": round(a_us, 2),
               "aiter_correct": ok, "mine_us": round(mine_us, 2) if mine_us else None}
        if mine_us:
            rec["speedup_mine_over_aiter"] = round(a_us / mine_us, 3)
        out.append(rec)
        print("M=%-5d N=%-8d  aiter %9.2f us (correct=%s)   mine %9.2f us   %s"
              % (M, N, a_us, ok, mine_us if mine_us else float("nan"),
                 ("%.2fx faster" % (a_us / mine_us)) if mine_us else ""))

    good = [r for r in out if r.get("mine_us") and r.get("aiter_us")]
    if good:
        import math
        g = math.exp(sum(math.log(r["speedup_mine_over_aiter"]) for r in good) / len(good))
        print("\ngeomean speedup of this kernel over aiter over %d shapes: %.2fx" % (len(good), g))
        print("shapes where aiter is faster: %s"
              % ([(r["m"], r["n"]) for r in good if r["speedup_mine_over_aiter"] < 1.0] or "none"))
    try:
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
        json.dump(out, open(args.json_out, "w"), indent=1)
        print("WROTE %s" % args.json_out)
    except OSError as e:
        print("could not write json: %s" % e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
