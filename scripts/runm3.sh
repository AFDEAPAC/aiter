#!/bin/bash
cd /topk
python3 -u scripts/mod3.py 2>&1 | grep -vE "^\[aiter\]|WARNING|warn|NUMA"
