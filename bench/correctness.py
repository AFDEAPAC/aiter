#!/usr/bin/env python3
"""Matrix fuzz correctness for generalized top-k paths."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark_topk"

# Regime boundary shapes: both sides of N_lds, large-N, small-M/decode.
MATRIX_CASES = [
    {"name": "small_n_512", "m": 64, "n": 512, "topk": 256, "path": "small_n"},
    {"name": "small_n_8192", "m": 64, "n": 8192, "topk": 2048, "path": "small_n"},
    {"name": "n_lds_boundary", "m": 64, "n": 8192, "topk": 2048, "path": "auto"},
    {"name": "n_lds_plus1", "m": 64, "n": 8196, "topk": 2048, "path": "auto"},
    {"name": "prefill_main", "m": 4096, "n": 131072, "topk": 2048, "path": "prefill"},
    {"name": "large_n_1m", "m": 64, "n": 1048576, "topk": 2048, "path": "auto"},
    {"name": "decode_m1", "m": 1, "n": 1048576, "topk": 2048, "path": "decode"},
    {"name": "decode_m128", "m": 128, "n": 1048576, "topk": 2048, "path": "decode"},
    {"name": "coop_g4", "m": 64, "n": 262144, "topk": 2048, "path": "decode", "coop_g": 4},
    {"name": "coop_g16", "m": 16, "n": 262144, "topk": 2048, "path": "decode", "coop_g": 16},
]

DISTRIBUTIONS = ("uniform", "gaussian", "equal", "inf", "adversarial")


def make_input(m, n, dist, seed):
    if not HAS_TORCH:
        return None
    if dist == "gaussian":
        return torch.randn(m, n, dtype=torch.float32)
    if dist == "equal":
        return torch.ones(m, n, dtype=torch.float32)
    if dist == "inf":
        x = torch.rand(m, n, dtype=torch.float32) * 1e-3
        x[:, ::257] = float("inf")
        return x
    if dist == "adversarial":
        x = torch.rand(m, n, dtype=torch.float32) * 1e-3
        x[:, -min(3000, n) :] = 100.0 + torch.arange(min(3000, n), dtype=torch.float32).unsqueeze(0) * 1e-3
        return x
    return torch.rand(m, n, dtype=torch.float32) * 2.0 - 1.0


def run_bench(case, dist, seed, inject_fault=0, pipeline="fused", verify_oracle="gpu"):
    m, n, topk = case["m"], case["n"], case["topk"]
    x = make_input(m, n, dist, seed)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as inf:
        input_path = Path(inf.name)
    if x is not None:
        x.contiguous().numpy().tofile(input_path)
    else:
        try:
            input_path.unlink()
        except OSError:
            pass
        input_path = None
    cmd = [
        str(BENCH),
        "--mode",
        "verify",
        "--m",
        str(m),
        "--n",
        str(n),
        "--topk",
        str(topk),
        "--dist",
        dist,
        "--seed",
        str(seed),
        "--inject-fault",
        str(inject_fault),
        "--pipeline",
        pipeline,
        "--verify-oracle",
        verify_oracle,
        "--verify-sample-rows",
        str(min(m, 64)),
    ]
    if input_path is not None:
        cmd += ["--input-bin", str(input_path)]
    if case.get("path"):
        cmd += ["--path", case["path"]]
    if case.get("coop_g"):
        cmd += ["--coop-g", str(case["coop_g"])]

    dump_path = None
    if inject_fault == 0:
        fd, dump_name = tempfile.mkstemp(suffix=".bin")
        import os

        os.close(fd)
        dump_path = Path(dump_name)
        cmd += ["--dump-indices", str(dump_path)]

    proc = subprocess.run(cmd, cwd=ROOT, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if input_path is not None:
        try:
            input_path.unlink()
        except OSError:
            pass
    return x, proc, dump_path


def load_idx(path, m, topk):
    raw = path.read_bytes()
    idx = torch.frombuffer(bytearray(raw), dtype=torch.int32).reshape(m, topk).clone()
    try:
        path.unlink()
    except OSError:
        pass
    return idx


def run_gates(x, idx, topk):
    if not HAS_TORCH:
        return
    ref_vals, _ = torch.topk(x, topk, dim=1, largest=True, sorted=False)
    got_vals = torch.gather(x, 1, idx.long())
    got_sorted, _ = torch.sort(got_vals, dim=1, descending=True)
    ref_sorted, _ = torch.sort(ref_vals, dim=1, descending=True)
    if not torch.equal(got_sorted, ref_sorted):
        raise AssertionError("gate1 value multiset mismatch vs torch.topk")
    for r in range(idx.shape[0]):
        row = idx[r]
        if torch.unique(row).numel() != row.numel():
            raise AssertionError("gate2 duplicate indices row=%d" % r)
    if (idx < 0).any() or (idx >= x.shape[1]).any():
        raise AssertionError("gate3 index out of range")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-inject", action="store_true")
    parser.add_argument("--quick", action="store_true", help="only primary + small_n cases")
    args = parser.parse_args()

    if not BENCH.exists():
        print("ERROR: missing %s; run make first" % BENCH, file=sys.stderr)
        return 2

    cases = MATRIX_CASES
    if args.quick:
        cases = [c for c in MATRIX_CASES if c["name"] in ("small_n_512", "prefill_main", "large_n_1m")]

    for case in cases:
        x, proc, dump = run_bench(case, "uniform", args.seed)
        if proc.returncode != 0 or "VERDICT PASS" not in proc.stdout:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            return 2
        if HAS_TORCH and dump is not None:
            run_gates(x, load_idx(dump, case["m"], case["topk"]), case["topk"])
        elif dump is not None:
            try:
                dump.unlink()
            except OSError:
                pass

    # Path-specific: direct oracle must match torch on a sample row.
    for path in ("small_n", "prefill", "decode"):
        case = {"m": 8, "n": 65536, "topk": 512, "path": path}
        if path == "small_n":
            case["n"] = 4096
        x, proc, dump = run_bench(case, "uniform", args.seed + 1, pipeline="direct")
        if proc.returncode != 0 or "VERDICT PASS" not in proc.stdout:
            print(proc.stdout)
            return 2
        if HAS_TORCH and dump is not None:
            run_gates(x, load_idx(dump, case["m"], case["topk"]), case["topk"])
        elif dump is not None:
            try:
                dump.unlink()
            except OSError:
                pass

    for i, dist in enumerate(DISTRIBUTIONS):
        case = {"m": 32, "n": 16384, "topk": 512, "path": "auto"}
        x, proc, dump = run_bench(case, dist, args.seed + 10 + i)
        if proc.returncode != 0 or "VERDICT PASS" not in proc.stdout:
            print(proc.stdout)
            return 2
        if HAS_TORCH and dump is not None:
            run_gates(x, load_idx(dump, case["m"], case["topk"]), case["topk"])
        elif dump is not None:
            try:
                dump.unlink()
            except OSError:
                pass

    if not args.skip_inject:
        case = MATRIX_CASES[0]
        _, iproc, _ = run_bench(case, "uniform", args.seed + 99, inject_fault=1)
        if iproc.returncode == 0:
            raise AssertionError("inject-fault should fail")

    print("CORRECTNESS PASS matrix=%d paths=3 inject_red=ok" % len(cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
