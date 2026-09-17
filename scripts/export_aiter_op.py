#!/usr/bin/env python3
"""Generate the aiter-side sources for the AVO top-k op from this repo.

The kernels have to live inside aiter for aiter to build them, and a hand copy
drifts the moment either side is touched. So the aiter files are generated and
this script is the only thing allowed to write them: `--check` re-generates into
memory and diffs, which is what tells you the two trees have diverged.

  python3 scripts/export_aiter_op.py --aiter /path/to/aiter          # write
  python3 scripts/export_aiter_op.py --aiter /path/to/aiter --check  # verify

What ships is benchmark_topk.hip.cpp up to the AITER_EXPORT_END marker (kernels
plus dispatch, no harness) with csrc/topk_aiter_entry.inc.hip appended, plus the
three headers it includes.
"""

import argparse
import difflib
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
HEADERS = ("topk_common.hip.hpp", "topk_shape.hip.hpp", "topk_generalize.hip.hpp")
MARKER = "// ---- AITER_EXPORT_END ----"

# Relative to the aiter root. The headers keep their bare cross-includes and so
# must land in one directory together; only the .cu, which sits elsewhere, needs
# its include paths rewritten.
HDR_DIR = "csrc/include/topk_avo"
CU_PATH = "csrc/kernels/topk_per_row_avo_kernels.cu"

BANNER = """// SPDX-License-Identifier: MIT
// GENERATED FILE -- DO NOT EDIT.
//
// Source of truth: the topk-prefill-avo repo. Regenerate with
//   python3 scripts/export_aiter_op.py --aiter <this aiter checkout>
// and verify an existing tree with the same command plus --check.
//
// This is benchmark_topk.hip.cpp up to its AITER_EXPORT_END marker (the kernels
// and their dispatch) followed by csrc/topk_aiter_entry.inc.hip (the aiter op
// entry). The harness half of that file -- CPU/GPU verification oracles, timing,
// CLI -- is deliberately not here.
"""


def kernel_region() -> str:
    src = (REPO / "benchmark_topk.hip.cpp").read_text()
    cut = src.find(MARKER)
    if cut < 0:
        sys.exit(f"{MARKER} not found in benchmark_topk.hip.cpp; nothing to export")
    return src[:cut]


def generate():
    """Map aiter-relative path -> file content."""
    out = {}
    for h in HEADERS:
        out[f"{HDR_DIR}/{h}"] = BANNER + (REPO / "csrc" / h).read_text()

    body = kernel_region()
    for h in HEADERS:
        # The .cu lives in csrc/kernels while the headers live under
        # csrc/include/topk_avo, so the bare includes it inherited from the
        # harness would resolve to nothing.
        body = body.replace(f'#include "{h}"', f'#include "topk_avo/{h}"')
    entry = (REPO / "csrc" / "topk_aiter_entry.inc.hip").read_text()
    out[CU_PATH] = BANNER + body + entry
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aiter", required=True, type=pathlib.Path)
    ap.add_argument("--check", action="store_true", help="diff instead of write")
    args = ap.parse_args()

    if not (args.aiter / "csrc" / "include").is_dir():
        sys.exit(f"{args.aiter} does not look like an aiter checkout")

    files = generate()
    stale = []
    for rel, content in sorted(files.items()):
        dst = args.aiter / rel
        if args.check:
            have = dst.read_text() if dst.exists() else ""
            if have != content:
                stale.append(rel)
                diff = difflib.unified_diff(
                    have.splitlines(True),
                    content.splitlines(True),
                    fromfile=f"aiter/{rel}",
                    tofile=f"generated/{rel}",
                    n=1,
                )
                sys.stdout.writelines(list(diff)[:40])
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(content)
            print(f"wrote {rel} ({len(content.splitlines())} lines)")

    if args.check:
        if stale:
            print(f"\nSTALE: {len(stale)} file(s) differ: {', '.join(stale)}")
            return 1
        print(f"up to date: {len(files)} file(s) match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
