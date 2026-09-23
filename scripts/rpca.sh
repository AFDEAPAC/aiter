#!/bin/bash
cd /topk
timeout 1200 rocprof-compute analyze -p /topk/workloads/pa8/MI355 \
  -b 2 6 7 10 11 16 17 2>&1 | grep -vE "^\s*$" | head -110
