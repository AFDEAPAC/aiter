# Fresh Agent Start

This is the first file to read when resuming this AVO run.

## Run Snapshot

- mode: single_kernel
- kernel_or_pipeline: topk_indices_kernel
- kernel_type: generic
- target_arch: gfx950
- score_rule: geomean
- latest_logic_review: `log/logic_review.md`
- attempts_log: `log/attempts.tsv`
- evolution_log: `log/evolution.tsv`

## Required First Reads

- `.evo/config.yaml`
- `knowledge/shapes.json`
- `knowledge/known_bad.md`
- `log/logic_review.md`
- `log/supervisor.md`
- `agentic-kernel-evolution/variation-step.md`
- `agentic-kernel-evolution/supervision.md`

## Resume Checklist

- Run `python $CC/skills/agentic-kernel-evolution/scripts/validate_evo.py . --runner-ready` before unattended work.
- If latest logic review is blocking, do not run the worker until resolved.
- If E2E mode has no bottleneck classification yet, perform Loop 0 before patching.
- This packet is project-local; do not use server-global transcripts or other evo repos as resume state.
