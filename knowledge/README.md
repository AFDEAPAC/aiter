# Knowledge Base K

This directory contains run-local evidence and references for `topk_indices_kernel`.

General AMD AVO policy:
- Keep generic workflow docs small and stable.
- Read arch-specific ISA docs only after the target arch is known or the
  current disassembly/profiler evidence needs that semantic detail.
- Read kernel-specific pattern docs only after the kernel type or measured
  bottleneck points there.
- Put failed directions in `known_bad.md` with the command/log/profiler
  evidence that disproved them.

Required before unattended evolution:
- `shapes.json` contains the production shape distribution.
- `known_bad.md` starts with any inherited failed directions.
- `refs/` contains the seed and any behavior/layout/algorithmic twins.
- `.evo/hw.json` is captured on the machine that will run the benchmarks.
