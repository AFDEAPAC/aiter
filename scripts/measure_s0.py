#!/usr/bin/env python3
"""S0 baseline measurements: HBM BW + torch.topk + seed kernel smoke."""

import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd, **kw):
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, **kw)


def parse_bw(stdout: str) -> dict:
    m = re.search(r"bandwidth_GiB_s=([0-9.]+)", stdout)
    ms = re.search(r"median_ms=([0-9.]+)", stdout)
    sd = re.search(r"stddev_pct=([0-9.]+)", stdout)
    return {
        "bandwidth_gib_s": float(m.group(1)) if m else None,
        "median_ms": float(ms.group(1)) if ms else None,
        "stddev_pct": float(sd.group(1)) if sd else None,
        "raw": stdout.strip(),
    }


def parse_timing(stdout: str) -> dict:
    m = re.search(r"wall_ms_median=([0-9.]+)", stdout)
    sd = re.search(r"stddev_pct=([0-9.]+)", stdout)
    return {
        "wall_ms_median": float(m.group(1)) if m else None,
        "stddev_pct": float(sd.group(1)) if sd else None,
        "raw": stdout.strip(),
    }


def torch_topk(m=4096, n=131072, k=2048, warmup=20, iters=100, repeats=5):
    script = f"""
import torch, statistics, time
m,n,k={m},{n},{k}
warmup,iters,repeats={warmup},{iters},{repeats}
x=torch.rand(m,n,dtype=torch.float32,device='cuda')
for _ in range(warmup):
    torch.topk(x,k,dim=1)
torch.cuda.synchronize()
run_medians=[]
for _ in range(repeats):
    ts=[]
    for _ in range(iters):
        t0=time.perf_counter()
        torch.topk(x,k,dim=1)
        torch.cuda.synchronize()
        ts.append((time.perf_counter()-t0)*1e3)
    run_medians.append(statistics.median(ts))
med=statistics.median(run_medians)
sd=100.0*(statistics.pstdev(run_medians)/med if len(run_medians)>1 and med>0 else 0.0)
print(f'TORCH wall_ms_median={{med:.4f}} stddev_pct={{sd:.3f}}')
"""
    proc = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)
    return parse_timing(proc.stdout)


def main():
    subprocess.check_call(["make", "-C", str(ROOT), "all"])
    bw = run(["./bw_kernel"])
    if bw.returncode != 0:
        print(bw.stderr, file=sys.stderr)
        raise SystemExit(bw.returncode)
    bw_info = parse_bw(bw.stdout)
    print(bw.stdout)

    torch_info = torch_topk()
    seed = run([
        "./benchmark_topk", "--mode", "verify_and_time", "--m", "4096", "--n", "131072", "--topk", "2048",
        "--warmup", "20", "--iters", "100", "--repeats", "5",
    ])
    print(seed.stdout)
    if seed.returncode != 0:
        print(seed.stderr, file=sys.stderr)
        raise SystemExit(seed.returncode)
    seed_info = parse_timing(seed.stdout)

    out = {
        "hbm_stream_bw": bw_info,
        "torch_topk": torch_info,
        "seed_kernel_x0": seed_info,
    }
    path = ROOT / "knowledge" / "s0_baseline.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print("WROTE", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
