# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MegaMoE accuracy + latency regression gate against a recorded baseline.

Runs test_mega_moe_v2.py under torchrun and compares every batch size with the
baseline JSON (relL2 and graph-timed stage1 / stage2 / e2e). Exits 1 if relL2
exceeds --max-rel or any latency is more than --max-slowdown (and --min-delta-ms) slower.

  python op_tests/multigpu_tests/mega_moe_regress.py                       # compare
  python op_tests/multigpu_tests/mega_moe_regress.py --update-baseline     # record
"""

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LINE = re.compile(
    r"\[MEGA-V2\] bs=(\d+) relL2=([\d.]+).*?SBM=(\d+) G2_BM=(\d+) "
    r"stage1=([\d.]+)/[\d.]+ms stage2=([\d.]+)/[\d.]+ms e2e=([\d.]+)/[\d.]+ms"
)


def run(args):
    cmd = ["torchrun", f"--nproc_per_node={args.nproc}", os.path.join(HERE, "test_mega_moe_v2.py"),
           "--network", args.network, "--quant", "a8w4", "--max-tok-per-rank", str(args.mtpr),
           "--bs-list", args.bs_list, "--accuracy-max-bs", str(args.mtpr), "--iters", str(args.iters)]
    env = dict(os.environ)
    env.setdefault("MORI_SHMEM_HEAP_SIZE", "16G")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    rows = {}
    for m in LINE.finditer(proc.stdout + proc.stderr):
        bs, rel, sbm, bm, s1, s2, e2e = m.groups()
        rows[int(bs)] = dict(relL2=float(rel), SBM=int(sbm), G2_BM=int(bm),
                             stage1=float(s1), stage2=float(s2), e2e=float(e2e))
    if proc.returncode or not rows:
        sys.stderr.write(proc.stdout[-4000:] + proc.stderr[-4000:])
        raise SystemExit(f"test_mega_moe_v2 failed (exit {proc.returncode})")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--network", default="v41_flash")
    p.add_argument("--nproc", type=int, default=4)
    p.add_argument("--mtpr", type=int, default=8192)
    p.add_argument("--bs-list", default="1,16,128,512,2048,8192")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--baseline", default=os.path.join(HERE, "mega_moe_gfx942_baseline.json"))
    p.add_argument("--max-rel", type=float, default=0.06)
    p.add_argument("--max-slowdown", type=float, default=0.05)
    # Standalone Stage2 at 1 token jitters by about 0.06 ms between runs; a
    # slowdown only counts when it also exceeds this absolute delta.
    p.add_argument("--min-delta-ms", type=float, default=0.08)
    p.add_argument("--update-baseline", action="store_true")
    args = p.parse_args()
    rows = run(args)
    key = f"{args.network}_ep{args.nproc}_mtpr{args.mtpr}"
    base_all = json.load(open(args.baseline)) if os.path.exists(args.baseline) else {}
    if args.update_baseline:
        base_all[key] = {str(k): v for k, v in sorted(rows.items())}
        json.dump(base_all, open(args.baseline, "w"), indent=1)
        print(f"baseline {key} updated in {args.baseline}")
    base = base_all.get(key, {})
    failed = 0
    print(f"{'bs':>6} {'relL2':>8} {'stage1':>18} {'stage2':>18} {'e2e':>18}  verdict")
    for bs, r in sorted(rows.items()):
        b = base.get(str(bs))
        bad = r["relL2"] >= args.max_rel
        cells = []
        for f in ("stage1", "stage2", "e2e"):
            if b:
                ratio = r[f] / b[f]
                bad |= ratio > 1 + args.max_slowdown and r[f] - b[f] > args.min_delta_ms
                cells.append(f"{r[f]:.3f} ({ratio:.2f}x)")
            else:
                cells.append(f"{r[f]:.3f}")
        failed += bad
        print(f"{bs:>6} {r['relL2']:>8.4f} {cells[0]:>18} {cells[1]:>18} {cells[2]:>18}  "
              f"{'FAIL' if bad else 'ok'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
