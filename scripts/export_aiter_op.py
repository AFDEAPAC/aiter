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
import os
import pathlib
import shutil
import subprocess
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
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// GENERATED FILE -- DO NOT EDIT.
//
// Source of truth: the topk-prefill-avo repo. Regenerate with
//   python3 scripts/export_aiter_op.py --aiter <this aiter checkout>
// and verify an existing tree with the same command plus --check.
//
// This is benchmark_topk.hip.cpp up to its AITER_EXPORT_END marker (the kernels
// and their dispatch) followed by csrc/topk_aiter_entry.inc.hip (the aiter op
// entry). The harness half of that file -- CPU/GPU verification oracles, timing,
// CLI -- is deliberately not here. The source repo indents at 2; what you are
// reading was reformatted to aiter's .clang-format on the way in, so this file
// does not line up line-for-line with the source.
//
// Formatted by: {formatter}

"""

# The ROCm toolchain ships a clang-format and a dev box for this repo has ROCm
# by definition, so this is the default rather than a fallback. See Formatter.
CLANG_FORMAT_FALLBACK = "/opt/rocm/llvm/bin/clang-format"


def kernel_region() -> str:
    src = (REPO / "benchmark_topk.hip.cpp").read_text()
    cut = src.find(MARKER)
    if cut < 0:
        sys.exit(f"{MARKER} not found in benchmark_topk.hip.cpp; nothing to export")
    return src[:cut]


class Formatter:
    """clang-format, resolved once, with its version on the record.

    The version matters and is written into the banner because two formatters
    disagree on this code: upstream clang-format 14.0.6, 16.0.6, 18.1.8, 20.1.0
    and 21.1.2 all produce byte-identical output, and the AMD build shipped in
    /opt/rocm (22.0.0git) produces something different. Without the version in
    the file, a `--check` failure showing nothing but formatting churn is
    indistinguishable from a real source change; with it, the banner says which
    formatter wrote the tree you are diffing against.
    """

    def __init__(self, explicit=None):
        # ROCm's build is preferred over whatever is on PATH, which is the
        # opposite of the usual order and is a measured choice: the two disagree
        # on short function bodies, and the AMD build keeps `{ return x; }` on
        # one line the way aiter's own sources do (20 occurrences under csrc/,
        # e.g. csrc/kernels/mla/reduce.cu:124) while upstream expands it to
        # three lines. Preferring PATH would make an upstream install silently
        # produce code that deviates from the repo it is being formatted for.
        for cand in (explicit, os.environ.get("CLANG_FORMAT"),
                     CLANG_FORMAT_FALLBACK, shutil.which("clang-format")):
            if cand and (shutil.which(cand) or os.path.isfile(cand)):
                self.exe = cand
                break
        else:
            sys.exit(
                "clang-format not found. The generated sources are formatted to "
                "aiter's .clang-format, so this script will not emit unformatted "
                "output and silently hand a reviewer 2-space-indented code.\n"
                "Pass --clang-format <path>, set CLANG_FORMAT, or install one "
                "(`pip install clang-format`)."
            )
        # stdout=PIPE rather than capture_output=, universal_newlines= rather
        # than text=: this runs on the host's python 3.6.
        v = subprocess.run([self.exe, "--version"], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True)
        # Drop the " (https://... <hash>)" tail the ROCm build appends: it wraps
        # over two more banner lines and the part that distinguishes the
        # formatters that actually disagree here is already in what is kept.
        self.version = (v.stdout.strip() or "unknown").split(" (")[0]

    def __call__(self, content, dest_rel, aiter_root):
        # --assume-filename so clang-format finds aiter's .clang-format by
        # walking up from where the file will land, and picks the language from
        # the extension. The content is formatted on stdin; nothing is written
        # into the aiter tree until the caller decides to.
        p = subprocess.run(
            [self.exe, "--style=file", "--assume-filename", str(aiter_root / dest_rel)],
            input=content, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if p.returncode != 0:
            sys.exit(f"clang-format failed on {dest_rel}:\n{p.stderr}")
        return p.stdout


def generate(fmt, aiter_root):
    """Map aiter-relative path -> file content."""
    banner = BANNER.format(formatter=fmt.version)
    raw = {}
    for h in HEADERS:
        raw[f"{HDR_DIR}/{h}"] = banner + (REPO / "csrc" / h).read_text()

    body = kernel_region()
    for h in HEADERS:
        # The .cu lives in csrc/kernels while the headers live under
        # csrc/include/topk_avo, so the bare includes it inherited from the
        # harness would resolve to nothing.
        body = body.replace(f'#include "{h}"', f'#include "topk_avo/{h}"')
    entry = (REPO / "csrc" / "topk_aiter_entry.inc.hip").read_text()
    raw[CU_PATH] = banner + body + entry

    return {rel: fmt(content, rel, aiter_root) for rel, content in raw.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aiter", required=True, type=pathlib.Path)
    ap.add_argument("--check", action="store_true", help="diff instead of write")
    ap.add_argument("--clang-format", default=None,
                    help="path to clang-format; see Formatter on why it matters")
    args = ap.parse_args()

    if not (args.aiter / "csrc" / "include").is_dir():
        sys.exit(f"{args.aiter} does not look like an aiter checkout")

    fmt = Formatter(args.clang_format)
    files = generate(fmt, args.aiter)
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
