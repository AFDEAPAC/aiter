#!/usr/bin/env python3
"""Generate the commit message body/trailer for one accepted candidate."""

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--score", default="score.json")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--tolerance", default="from .evo/config.yaml")
    args = parser.parse_args()

    score = json.loads(Path(args.score).read_text(encoding="utf-8"))
    candidate = score.get("candidate", {})
    shapes = list(candidate)
    medians = [candidate[name][0] for name in shapes]
    unit = score.get("unit", "score")

    print(args.title)
    print()
    print(f"Parent: {args.parent}")
    print(f"Attempt: {args.attempt}")
    print(f"Plan: {args.plan}")
    print("Shapes: " + ", ".join(shapes))
    print("Score:  " + ", ".join(str(x) for x in medians))
    print(f"ScoreUnit: {unit}")
    print(f"Tolerance: {args.tolerance}")
    print(f"Commit-Reason: {args.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
