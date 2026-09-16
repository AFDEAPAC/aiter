#!/usr/bin/env bash
# AVO bench loop entry point. Layers, in order:
#   1. Correctness gate (must pass before any timing).
#   2. (E2E only) Stage 1 accuracy smoke (32 samples minimum) if
#      bench/accuracy_stages/ exists.
#   3. Either the shared rocprofv3 wrapper (simple/fallback) or a hand-written
#      bench/harness.py escape hatch.
#
# Default path (simple fallback for HIP / Triton / FlyDSL / asm):
#   - Edit bench/spec.json to point at your candidate/best commands and
#     declare warmup, repeat, kernel_name_regex, rule, and unit.
#   - The wrapper drives rocprofv3 --kernel-trace, parses a small trace,
#     and emits a higher-is-better score.json conforming to
#     skills/agentic-kernel-evolution/score-schema.md.
#
# Escape hatch: if you need to emit score.json yourself (custom timing,
# multi-kernel pipelines, frameworks that own dispatch), set
#   AVO_USE_CUSTOM_HARNESS=1
# and implement bench/harness.py end-to-end.
#
# E2E layout (created by evo_init.py --e2e):
#   bench/accuracy_stages/stage1_smoke.py   # always run here (32 samples min)
#   bench/accuracy_stages/stage2_pinned.py  # supervisor invokes on green Tier A
#   bench/accuracy_stages/stage3_gsm8k.py   # supervisor invokes on new-best/final
set -euo pipefail

python bench/correctness.py

if [ -f bench/accuracy_stages/stage1_smoke.py ]; then
    python bench/accuracy_stages/stage1_smoke.py
fi

if [ "${AVO_USE_CUSTOM_HARNESS:-0}" = "1" ]; then
    python bench/harness.py "$@"
else
    python scripts/score_from_rocprofv3.py bench/spec.json --output score.json
fi
