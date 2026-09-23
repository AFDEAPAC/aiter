#!/bin/bash
cd /topk
python3 -u scripts/nonpow2.py 2>&1 | grep -vE "^\[aiter\]|WARNING|warn" | tail -30
