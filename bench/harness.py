#!/usr/bin/env python3
"""Performance harness template.

Real implementations should:
1. Run the candidate and current best in the same invocation.
2. Run 20 warmups + 100 measured benchmark iterations and aggregate locally.
   Keep rocprofv3 / ATT traces small and use them for diagnosis artifacts.
3. Emit score.json with candidate/best shape vectors:
   {"candidate": {"shape": [median, stddev]}, "best": {...}, ...}
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="candidate")
    parser.add_argument("--also-measure", default="best")
    parser.add_argument("--output", default="score.json")
    args = parser.parse_args()

    if os.environ.get("AVO_ALLOW_STUB_BENCH") != "1":
        print(
            "ERROR: bench/harness.py is still a stub. Implement rocprofv3 "
            "timing before running AVO.",
            file=sys.stderr,
        )
        return 2

    score = {
        "candidate": {"smoke": [1.0, 0.0]},
        "best": {"smoke": [1.0, 0.0]},
        "unit": "score",
        "rule": "geomean",
        "delta_pct": 0.0,
        "notes": ["stub score; do not commit real work with this"],
    }
    Path(args.output).write_text(json.dumps(score, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(score, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
