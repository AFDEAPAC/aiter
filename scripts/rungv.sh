#!/bin/bash
cd /topk
python3 -u scripts/gridver.py 2>&1 | grep -vE "^\[aiter\]|WARNING|warn|NUMA"
