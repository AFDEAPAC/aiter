#!/usr/bin/env python3
"""Negative control for the torch gates in correctness.py.

A green correctness run only counts as evidence if the gate can go red. This
asserts torch is actually present (otherwise run_gates() returns early and every
run is vacuously green) and then feeds it three corrupted index sets that it
must reject.

Must run where torch is available, e.g.

  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    -v $PWD:/work -w /work <rocm-pytorch-image> python bench/gate_selftest.py
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_correctness():
    spec = importlib.util.spec_from_file_location("cor", ROOT / "bench" / "correctness.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def expect_red(label, fn):
    try:
        fn()
    except Exception as e:  # AssertionError from the gate, or torch range error
        print("  %-14s -> RED: %s" % (label, str(e)[:58]))
        return True
    print("  %-14s -> PASS  *** GATE IS DEAD ***" % label)
    return False


def main():
    cor = load_correctness()
    if not cor.HAS_TORCH:
        print("FAIL: torch missing, so run_gates() is a no-op and every run is "
              "vacuously green. Run this inside a torch-enabled image.", file=sys.stderr)
        return 2

    case = {"m": 16, "n": 8192, "topk": 2048, "path": "small_n"}
    x, proc, dump = cor.run_bench(case, "uniform", 0)
    if proc.returncode != 0 or dump is None:
        print(proc.stdout)
        return 2
    idx = cor.load_idx(dump, case["m"], case["topk"])
    k = case["topk"]

    import torch

    cor.run_gates(x, idx, k)
    print("  %-14s -> PASS (expected)" % "clean")

    ok = True
    # An index outside the true top-k: the value multiset must stop matching.
    _, order = torch.sort(x[0], descending=True)
    bad = idx.clone()
    bad[0, 0] = int(order[-1])
    ok &= expect_red("value-corrupt", lambda: cor.run_gates(x, bad, k))

    # A repeated index: K slots filled but one true member is missing.
    dup = idx.clone()
    dup[0, 1] = int(dup[0, 0])
    ok &= expect_red("duplicate", lambda: cor.run_gates(x, dup, k))

    # Out of range.
    oor = idx.clone()
    oor[0, 0] = case["n"] + 5
    ok &= expect_red("out-of-range", lambda: cor.run_gates(x, oor, k))

    if not ok:
        print("GATE SELFTEST FAILED: at least one corruption was accepted", file=sys.stderr)
        return 2
    print("GATE SELFTEST PASS: torch present, 3/3 corruptions rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
