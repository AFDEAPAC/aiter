#!/usr/bin/env python3
"""Backend-agnostic AVO bench wrapper.

Runs each (shape, command) pair under `rocprofv3 --kernel-trace`, parses the
generated kernel_trace.csv to collect per-dispatch durations, takes the median
+ stddev of `repeat` runs after `warmup` warmups, and emits a `score.json`
file matching the canonical schema in `agentic-kernel-evolution/score-schema.md`.

This script is written so HIP, Triton, FlyDSL, and asm bench harnesses all
share the same wrapper instead of each backend re-implementing rocprofv3
parsing. The bench command is whatever launches your kernel — a hipcc binary,
`python my_triton_bench.py --shape ...`, a `flydsl ...` invocation, etc.

This is the simple/fallback timing path. Keep `warmup=20` and `repeat=100` for
benchmark acceptance unless the user explicitly documents a smaller diagnostic
trace path.

Spec file format (JSON, both candidate and best required):

    {
      "shapes": [
        {"name": "BF16_4Kx8",  "candidate_argv": ["./kernel_test", "--shape", "4096x8"],
                                 "best_argv":      ["./kernel_best", "--shape", "4096x8"]},
        {"name": "BF16_8Kx4",  "candidate": "./kernel_test --shape 8192x4",
                                 "best":      "./kernel_best --shape 8192x4"}
      ],
      "kernel_name_regex": "my_kernel.*",
      "warmup": 20,
      "repeat": 100,
      "rule": "geomean",
      "unit": "us",
      "weights": {}
    }

Prefer `candidate_argv` / `best_argv` for new specs. The legacy string fields
`candidate` / `best` are still supported and run through the shell.

Each command is launched once per shape. It must run the target workload long
enough to emit at least `warmup + repeat` matching kernel dispatches in that
same host process; this wrapper drops the warmups and aggregates the following
dispatches. It does not re-launch the command once per sample.

`unit` is what your harness *measured* (the "raw unit"). The wrapper converts
it to a higher-is-better score because `avo_step.py` acceptance rules
(`geomean` / `pareto` / `weighted`) all assume higher-is-better:

    raw `unit`        score.json `unit`     score.json `raw_unit`
    --------------    ------------------    ---------------------
    "us"  / "ns"      "ops_per_sec"         "us" / "ns"
    "tflops"          "TFLOPS"              "tflops"
    "gbps"            "GB/s"                "gbps"

So if you read `score.json` directly, the values are always higher-is-better
under `unit`, and `raw_unit` records what the harness actually sampled.

Usage:

    python score_from_rocprofv3.py spec.json --output score.json

Or as a module:

    from score_from_rocprofv3 import write_score
    write_score("spec.json", "score.json")

This script depends only on the Python standard library + `rocprofv3` on PATH.
"""

import argparse
import csv
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


SUPPORTED_RULES = {"geomean", "pareto", "weighted", "pareto_with_floor"}
SUPPORTED_UNITS = {"us", "ns", "tflops", "gbps"}


def find_kernel_trace_csv(prof_dir):
    """rocprofv3 writes <hostname>/<pid>_kernel_trace.csv; find the first one."""
    for path in Path(prof_dir).rglob("*_kernel_trace.csv"):
        return path
    return None


def parse_kernel_durations(csv_path, kernel_name_regex):
    """Return list of (kernel_name, duration_ns) from a rocprofv3 kernel_trace.

    Drops dispatches whose name doesn't match the supplied regex (so vendor
    fillBuffer / RCCL / etc. are excluded). Empty regex matches everything.
    """
    pattern = re.compile(kernel_name_regex) if kernel_name_regex else None
    out = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Kernel_Name", "").strip().strip('"')
            if pattern and not pattern.search(name):
                continue
            try:
                start = int(row["Start_Timestamp"])
                end = int(row["End_Timestamp"])
            except (KeyError, ValueError):
                continue
            duration = end - start
            if duration > 0:
                out.append((name, duration))
    return out


def format_command(command):
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    return str(command)


def shape_command(shape, label):
    argv_key = f"{label}_argv"
    if argv_key in shape:
        command = shape[argv_key]
        if not isinstance(command, list) or not command or not all(isinstance(part, str) for part in command):
            raise SystemExit(f"shape {shape.get('name')!r} {argv_key} must be a non-empty list of strings")
        return command
    command = shape.get(label)
    if not command:
        raise SystemExit(f"shape {shape.get('name')!r} missing {label!r} command")
    if not isinstance(command, str):
        raise SystemExit(f"shape {shape.get('name')!r} {label!r} must be a string")
    return command


def run_one_shape(command, kernel_name_regex, warmup, repeat, env=None):
    """Run `command` once under rocprofv3 with warmup+repeat iterations baked in.

    The bench `command` is responsible for launching the kernel
    `warmup + repeat` times (the wrapper does NOT loop the command itself —
    re-launching the host process per iteration would dwarf the kernel time).
    The wrapper drops the first `warmup` matching dispatches and uses the rest.
    """
    with tempfile.TemporaryDirectory(prefix="avo_prof_") as prof_dir:
        if isinstance(command, list):
            rocprof_cmd = [
                "rocprofv3",
                "--kernel-trace",
                "--output-format",
                "csv",
                "-d",
                prof_dir,
                "--",
            ] + command
            shell = False
        else:
            rocprof_cmd = f"rocprofv3 --kernel-trace --output-format csv -d {prof_dir} -- {command}"
            shell = True
        proc = subprocess.run(
            rocprof_cmd,
            shell=shell,
            env=env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout)
            raise SystemExit(
                f"rocprofv3 invocation failed (exit {proc.returncode}); command was: {format_command(command)}"
            )
        csv_path = find_kernel_trace_csv(prof_dir)
        if csv_path is None:
            raise SystemExit(f"no kernel_trace CSV under {prof_dir}; rocprofv3 produced no output")
        durations = parse_kernel_durations(csv_path, kernel_name_regex)
    if not durations:
        raise SystemExit(
            f"no dispatches matched kernel_name_regex={kernel_name_regex!r}; "
            "check the regex against rocprofv3's Kernel_Name column"
        )
    if len(durations) <= warmup:
        raise SystemExit(
            f"only {len(durations)} matching dispatches but warmup={warmup}; "
            "have the bench command launch the kernel at least warmup+repeat times"
        )
    keep = durations[warmup : warmup + repeat] if repeat > 0 else durations[warmup:]
    if not keep:
        raise SystemExit(
            f"no dispatches left after dropping warmup={warmup} from {len(durations)} total"
        )
    if repeat > 0 and len(keep) < repeat:
        raise SystemExit(
            f"only {len(keep)} measured dispatches after warmup={warmup}, but repeat={repeat}; "
            "reduce repeat for profiling traces or have the bench command emit more dispatches"
        )
    return [duration_ns for _, duration_ns in keep]


def aggregate(durations_ns, unit, flops=None, byte_count=None):
    """Convert a list of dispatch durations to (value, stddev, output_unit).

    The returned `value` is always **higher-is-better** so avo_step.py's
    acceptance rules (geomean / pareto / weighted) work uniformly:
      - For raw timing units (`ns`, `us`) the duration is inverted to
        `ops_per_sec = 1e9 / median_ns`. Faster kernel → bigger score.
      - For `tflops` / `gbps` the value passes through unchanged.

    `output_unit` is the honest label that goes into score.json's `unit`
    field. Callers should also stamp the original (raw) unit into
    score.json's `raw_unit` so debuggers can recover what was sampled.
    """
    median_ns = statistics.median(durations_ns)
    stddev_ns = statistics.stdev(durations_ns) if len(durations_ns) > 1 else 0.0
    rel_err = stddev_ns / median_ns if median_ns > 0 else 0.0

    if unit in ("ns", "us"):
        ops_per_sec = 1.0e9 / median_ns
        return ops_per_sec, ops_per_sec * rel_err, "ops_per_sec"
    if unit == "tflops":
        if not flops:
            raise SystemExit("unit=tflops requires per-shape `flops` field")
        tflops = flops / median_ns / 1e3
        return tflops, tflops * rel_err, "TFLOPS"
    if unit == "gbps":
        if not byte_count:
            raise SystemExit("unit=gbps requires per-shape `bytes` field")
        gbps = byte_count / median_ns
        return gbps, gbps * rel_err, "GB/s"
    raise SystemExit(f"unsupported unit: {unit!r} (expected one of {sorted(SUPPORTED_UNITS)})")


def geomean(values):
    if not values or any(v <= 0 for v in values):
        return 0.0
    return math.exp(sum(math.log(v) for v in values) / len(values))


def measure_spec(spec):
    rule = spec.get("rule", "geomean")
    if rule not in SUPPORTED_RULES:
        raise SystemExit(f"rule must be one of {sorted(SUPPORTED_RULES)}, got {rule!r}")
    unit = spec.get("unit", "us")
    if unit not in SUPPORTED_UNITS:
        raise SystemExit(f"unit must be one of {sorted(SUPPORTED_UNITS)}, got {unit!r}")
    warmup = int(spec.get("warmup", 20))
    repeat = int(spec.get("repeat", 100))
    kregex = spec.get("kernel_name_regex", "")

    candidate = {}
    best = {}
    output_unit = None
    for shape in spec["shapes"]:
        name = shape["name"]
        flops = shape.get("flops")
        byte_count = shape.get("bytes")
        for label, sink in (("candidate", candidate), ("best", best)):
            cmd = shape_command(shape, label)
            durations_ns = run_one_shape(cmd, kregex, warmup, repeat)
            value, stderr, shape_output_unit = aggregate(durations_ns, unit, flops, byte_count)
            sink[name] = [value, stderr]
            if output_unit is None:
                output_unit = shape_output_unit
            elif output_unit != shape_output_unit:
                raise SystemExit(
                    f"output unit drift across shapes: {output_unit!r} vs {shape_output_unit!r}; "
                    "all shapes in one spec must aggregate to the same unit"
                )

    cand_gm = geomean([v[0] for v in candidate.values()])
    best_gm = geomean([v[0] for v in best.values()])
    delta_pct = 100.0 * (cand_gm - best_gm) / best_gm if best_gm > 0 else 0.0

    out = {
        "candidate": candidate,
        "best": best,
        "rule": rule,
        "delta_pct": round(delta_pct, 6),
        "unit": output_unit or unit,
        "raw_unit": unit,
    }
    weights = spec.get("weights")
    if weights:
        out["weights"] = weights
    return out


def write_score(spec_path, output_path):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    score = measure_spec(spec)
    Path(output_path).write_text(json.dumps(score, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(score, indent=2))
    return score


def require_rocprofv3():
    if shutil.which("rocprofv3") is None:
        raise SystemExit("rocprofv3 not found on PATH; install ROCm or extend PATH")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="JSON spec file (see module docstring)")
    parser.add_argument("--output", type=Path, default=Path("score.json"))
    args = parser.parse_args()
    require_rocprofv3()
    write_score(args.spec, args.output)


if __name__ == "__main__":
    main()
