#!/usr/bin/env python3
"""Measure deepseek-ai/DeepSelect's TopK on this repo's grid, fp32 k=2048.

  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    --ipc=host --shm-size 16G \
    -v /home/mh/DeepSelect:/ds -v /home/mh/topk-prefill-avo:/topk -w /ds \
    rocm/ali-private:ubuntu22.04_rocm10.0.0_cp313_torch2.12.0_vllm_dsv4_20260916 \
    python /topk/bench/deepselect_compare.py

DeepSelect is the reference TopK for DeepSeek Sparse Attention. Its prebuilt
deep_select_cuda.cpython-313 .so in this tree is a ROCm build (links
libamdhip64 / libc10_hip) with gfx950 baked in, so it runs here directly.

Why it matters for the aiter work: DeepSelect ALREADY implements the one thing
this repo's kernel does not -- per-row variable length via `end`, with short
rows padded by `idx_oob_fill_value`. `begin` is "CURRENTLY NOT SUPPORTED" there
too, so rowStarts is an open gap in both. If DeepSelect is also faster, it is
the better thing to wire into aiter and this kernel is not needed.

Declared scope of DeepSelect (README): topk <= 4096; fp32 "Sampling" case is
stated for vocab_size around 128K, so large N is outside what it advertises --
measure it, do not assume either way.
"""

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time


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
    return statistics.median(meds)


def value_multiset_ok(row, got_idx, k, row_len):
    """Same tolerance the vendors' own tests use: ties may pick different
    indices, so compare the multiset of selected VALUES."""
    import torch
    n = min(k, row_len)
    g = got_idx[:n].long()
    if (g < 0).any() or (g >= row_len).any():
        return False
    gv = torch.sort(row[g], descending=True).values
    rv = torch.sort(torch.topk(row[:row_len], n).values, descending=True).values
    return bool(torch.equal(gv, rv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk-bin", default="/topk/benchmark_topk")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json-out", default="/topk/reports/deepselect_compare.json")
    ap.add_argument("--ragged", action="store_true",
                    help="also test variable-length rows via DeepSelect's `end`")
    args = ap.parse_args()

    import torch
    sys.path.insert(0, "/ds")
    import deep_select

    print("device %s  torch %s" % (torch.cuda.get_device_name(0), torch.__version__))
    print("DeepSelect stride requirement (in/out bytes): %s"
          % (deep_select.get_stride_requirement(),))
    print()

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

        def run_ds():
            deep_select.topk(logits, K, sorted=False, indices_type=torch.int32,
                             return_value=False)

        try:
            _, idx = deep_select.topk(logits, K, sorted=False,
                                      indices_type=torch.int32, return_value=False)
            torch.cuda.synchronize()
        except Exception as e:
            print("M=%-5d N=%-8d DeepSelect FAILED: %s: %s"
                  % (M, N, type(e).__name__, str(e)[:110]))
            out.append({"m": M, "n": N, "ds_us": None, "err": "%s: %s" % (type(e).__name__, str(e)[:200])})
            continue

        ok = value_multiset_ok(logits[0], idx[0], K, N)
        ds_us = time_us(run_ds, args.warmup, args.iters, args.repeats)

        mine_us = None
        if os.path.exists(args.topk_bin):
            p = subprocess.run([args.topk_bin, "--mode", "time", "--m", str(M), "--n", str(N),
                                "--topk", str(K), "--warmup", "20", "--iters", "100",
                                "--repeats", "5"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)
            mm = re.search(r"wall_ms_median=([0-9.]+)", p.stdout)
            if mm:
                mine_us = float(mm.group(1)) * 1000.0

        rec = {"m": M, "n": N, "topk": K, "ds_us": round(ds_us, 2), "ds_correct": ok,
               "mine_us": round(mine_us, 2) if mine_us else None}
        if mine_us:
            rec["speedup_mine_over_ds"] = round(ds_us / mine_us, 3)
        out.append(rec)
        print("M=%-5d N=%-8d  DeepSelect %9.2f us (correct=%s)   mine %9.2f us   %s"
              % (M, N, ds_us, ok, mine_us if mine_us else float("nan"),
                 ("mine %.2fx" % (ds_us / mine_us)) if mine_us else ""))

    if args.ragged:
        print("\n=== variable-length rows (DeepSelect `end`), which this repo cannot do ===")
        for (M, N) in [(4096, 4096), (1024, 65536), (4096, 131072)]:
            torch.manual_seed(7)
            logits = torch.randn((M, N), dtype=torch.float32, device="cuda")
            ends = torch.arange(N - M + 1, N + 1, dtype=torch.int32, device="cuda")
            try:
                _, idx = deep_select.topk(logits, K, sorted=False, end=ends,
                                          indices_type=torch.int32, return_value=False)
                torch.cuda.synchronize()
                oks = all(value_multiset_ok(logits[i], idx[i], K, int(ends[i]))
                          for i in range(0, M, max(1, M // 16)))
                us = time_us(lambda: deep_select.topk(logits, K, sorted=False, end=ends,
                                                      indices_type=torch.int32,
                                                      return_value=False),
                             args.warmup, args.iters, args.repeats)
                print("  M=%-5d N=%-8d ragged  %9.2f us  correct=%s" % (M, N, us, oks))
                out.append({"m": M, "n": N, "ragged": True, "ds_us": round(us, 2),
                            "ds_correct": oks})
            except Exception as e:
                print("  M=%-5d N=%-8d ragged FAILED: %s" % (M, N, str(e)[:100]))

    good = [r for r in out if r.get("mine_us") and r.get("ds_us") and not r.get("ragged")]
    if good:
        g = math.exp(sum(math.log(r["speedup_mine_over_ds"]) for r in good) / len(good))
        print("\ngeomean: this kernel is %.2fx vs DeepSelect over %d uniform shapes" % (g, len(good)))
        loses = [(r["m"], r["n"], r["speedup_mine_over_ds"]) for r in good
                 if r["speedup_mine_over_ds"] < 1.0]
        print("shapes where DeepSelect is faster: %s" % (loses or "none"))
    bad = [r for r in out if r.get("ds_us") is None]
    if bad:
        print("DeepSelect could not run: %s" % [(r["m"], r["n"]) for r in bad])
    try:
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
        json.dump(out, open(args.json_out, "w"), indent=1)
        print("WROTE %s" % args.json_out)
    except OSError as e:
        print("could not write json: %s" % e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
