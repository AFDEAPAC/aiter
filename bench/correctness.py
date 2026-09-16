#!/usr/bin/env python3
"""Correctness oracle for topk-prefill AVO (five hard gates)."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark_topk"


def make_input(m: int, n: int, dist: str, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
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
        x[:, -3000:] = 100.0 + torch.arange(3000, dtype=torch.float32).unsqueeze(0) * 1e-3
        return x
    return torch.rand(m, n, dtype=torch.float32) * 2.0 - 1.0


def run_bench(m: int, n: int, topk: int, dist: str, seed: int, inject_fault: int = 0) -> tuple[torch.Tensor | None, subprocess.CompletedProcess, Path | None]:
    dump_path = None
    x = None if inject_fault else make_input(m, n, dist, seed)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as inf:
        input_path = Path(inf.name)
    if x is None:
        x = make_input(m, n, dist, seed)
    x.contiguous().numpy().tofile(input_path)
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
        "--input-bin",
        str(input_path),
        "--inject-fault",
        str(inject_fault),
    ]
    if inject_fault == 0:
        fd, dump_name = tempfile.mkstemp(suffix=".bin")
        import os

        os.close(fd)
        dump_path = Path(dump_name)
        cmd += ["--dump-indices", str(dump_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    input_path.unlink(missing_ok=True)
    return x, proc, dump_path


def load_idx(path: Path, m: int, topk: int) -> torch.Tensor:
    raw = path.read_bytes()
    idx = torch.frombuffer(bytearray(raw), dtype=torch.int32).reshape(m, topk).clone()
    path.unlink(missing_ok=True)
    return idx


def run_gates(x: torch.Tensor, idx: torch.Tensor, topk: int) -> None:
    ref_vals, _ = torch.topk(x, topk, dim=1, largest=True, sorted=False)
    got_vals = torch.gather(x, 1, idx.long())
    got_sorted, _ = torch.sort(got_vals, dim=1, descending=True)
    ref_sorted, _ = torch.sort(ref_vals, dim=1, descending=True)
    if not torch.equal(got_sorted, ref_sorted):
        raise AssertionError("gate1 value multiset mismatch vs torch.topk")

    for r in range(idx.shape[0]):
        row = idx[r]
        if torch.unique(row).numel() != row.numel():
            raise AssertionError(f"gate2 duplicate indices row={r}")

    if idx.numel() != x.shape[0] * topk:
        raise AssertionError("gate3 output count != M*k")
    if (idx < 0).any() or (idx >= x.shape[1]).any():
        raise AssertionError("gate3 index out of range")

    if not torch.equal(got_sorted, ref_sorted):
        raise AssertionError("gate4 value[i]==input[index[i]] multiset mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=32768)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--full-shape", action="store_true")
    parser.add_argument("--skip-inject", action="store_true")
    args = parser.parse_args()

    if not BENCH.exists():
        print(f"ERROR: missing {BENCH}; run make first", file=sys.stderr)
        return 2

    m = 4096 if args.full_shape else args.m
    n = 131072 if args.full_shape else args.n
    topk = 2048 if args.full_shape else args.topk

    x, proc, dump = run_bench(m, n, topk, "uniform", args.seed)
    if proc.returncode != 0 or "VERDICT PASS" not in proc.stdout:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        return 2
    assert x is not None and dump is not None
    idx = load_idx(dump, m, topk)
    run_gates(x, idx, topk)

    # guard shape (correctness-only)
    gx, gproc, gdump = run_bench(64, 32768, 512, "uniform", args.seed + 1)
    if gproc.returncode != 0 or gdump is None:
        print(gproc.stdout)
        return 2
    run_gates(gx, load_idx(gdump, 64, 512), 512)

    # gate5 distributions
    for i, dist in enumerate(("gaussian", "equal", "inf", "adversarial")):
        dx, dproc, ddump = run_bench(min(m, 128), min(n, 65536), min(topk, 512), dist, args.seed + 10 + i)
        if dproc.returncode != 0 or ddump is None:
            print(dproc.stdout)
            return 2
        run_gates(dx, load_idx(ddump, dx.shape[0], min(topk, 512)), min(topk, 512))

    if not args.skip_inject:
        _, iproc, _ = run_bench(min(m, 64), min(n, 32768), min(topk, 512), "uniform", args.seed + 99, inject_fault=1)
        if iproc.returncode == 0:
            raise AssertionError("inject-fault should fail")

    print("CORRECTNESS PASS gates=5 inject_red=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
